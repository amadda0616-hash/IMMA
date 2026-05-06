"""
src/stage3_numerical.py

Stage 3-N — Donut Numerical VLM (fine-tuned)

Purpose
-------
Fine-tune Donut on engineering-drawing annotation patches (Measure / GDT /
Roughness, output by Stage 2's perspective-warped crop) and emit
schema-defined JSON.

Decisions referenced
--------------------
- D-001 / D-018  : architecture = Donut, paper-faithful
- D-005 / configs/donut_numerical.yaml :
    epoch 30 / AdamW / cosine / lr 1e-6 / batch 4 / FP16 (or BF16)
    + gradient_checkpointing for RTX 5080 16 GB
- D-024          : group-aware train/val/test split

Input data layout
-----------------
::

    data/vlm/numerical/
    ├── <id>.jpg         # de-rotated patch (Stage 2 crop)
    └── <id>.json        # schema ground truth

Where ``<id>`` typically encodes the source drawing + view + class index
(e.g. ``mydrawing__View_00__Measure_03``).

Schema (HANDOFF §5.4)
---------------------
::

    Measure   : {"type": "Measure",   "nominal": 25.0,
                 "tolerance": {"upper": 0.05, "lower": -0.05}, "unit": "mm"}
    GDT       : {"type": "GDT", "symbol": "⏤",
                 "tolerance": 0.02, "datum": ["A", "B"]}
    Roughness : {"type": "Roughness", "Ra": 1.6, "unit": "μm"}

CLI
---
::

    # Fine-tune
    python src/stage3_numerical.py train \\
        --cfg configs/donut_numerical.yaml --device 0

    # Predict for a single warped patch
    python src/stage3_numerical.py predict \\
        --image outputs/crops/<drawing>/annotations/Measure/foo.jpg \\
        --region-class Measure

    # Batch over a Stage 2 crop folder
    python src/stage3_numerical.py batch \\
        --input-dir outputs/crops/<drawing>/annotations \\
        --out-dir outputs/<drawing>/numerical
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "donut_numerical.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "donut_numerical"

# Region-class → task token (registered into Donut's tokenizer)
TASK_TOKENS = {
    "Measure":   "<s_measure>",
    "GDT":       "<s_gdt>",
    "Roughness": "<s_roughness>",
}

# Reverse map (lowercased) — for inference dispatch
TASK_TOKEN_REVERSE = {v: k for k, v in TASK_TOKENS.items()}

# Field tokens we register so Donut decoder produces well-formed XML.
# Closing tokens are auto-derived as </s_xxx>.
SCHEMA_FIELD_NAMES = [
    # Measure
    "nominal", "tolerance", "upper", "lower", "unit",
    "diameter", "radius", "thread", "depth",
    # GDT
    "symbol", "datum", "modifier",
    # Roughness
    "Ra", "Rz", "Rmax",
    # Common
    "type",
]

# Separator for list values (e.g. datum=[A, B])
LIST_SEP = "<sep/>"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage3_numerical")


# ---------------------------------------------------------------------------
# YAML config helpers
# ---------------------------------------------------------------------------
def load_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# JSON ↔ Donut token-string conversion
# ---------------------------------------------------------------------------
def json_to_donut(obj: Any) -> str:
    """Convert a Python dict/list/scalar into Donut's XML-like token string.

    Examples
    --------
    >>> json_to_donut({"nominal": 25.0, "unit": "mm"})
    '<s_nominal>25.0</s_nominal><s_unit>mm</s_unit>'

    >>> json_to_donut({"datum": ["A", "B"]})
    '<s_datum>A<sep/>B</s_datum>'
    """
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if k == "type":
                continue   # task token handles 'type'
            inner = json_to_donut(v)
            parts.append(f"<s_{k}>{inner}</s_{k}>")
        return "".join(parts)
    if isinstance(obj, list):
        return LIST_SEP.join(json_to_donut(x) for x in obj)
    if obj is None:
        return ""
    if isinstance(obj, bool):
        return "true" if obj else "false"
    return str(obj)


_TOKEN_PATTERN = re.compile(r"<s_([^>/]+)>(.*?)</s_\1>", re.DOTALL)
_TASK_TOKEN_PATTERN = re.compile(
    r"^<s_(?:" + "|".join(re.escape(t.strip("<>").replace("s_", ""))
                          for t in TASK_TOKENS.values()) + r")>"
)


def donut_to_json(seq: str, strip_task_prefix: bool = True) -> Any:
    """Reverse json_to_donut: token sequence → dict.

    Recursive. Non-greedy regex with backreference correctly handles nested
    tokens of *different* names (which is always the case in our schema).
    """
    seq = seq.strip()
    if strip_task_prefix:
        seq = _TASK_TOKEN_PATTERN.sub("", seq, count=1).strip()

    if not seq:
        return {}

    out: Dict[str, Any] = {}
    pos = 0
    for m in _TOKEN_PATTERN.finditer(seq):
        if m.start() < pos:
            continue
        key = m.group(1)
        inner = m.group(2)
        if LIST_SEP in inner:
            out[key] = [donut_to_json_value(x.strip())
                        for x in inner.split(LIST_SEP)]
        elif "<s_" in inner:
            out[key] = donut_to_json(inner, strip_task_prefix=False)
        else:
            out[key] = donut_to_json_value(inner)
        pos = m.end()
    return out


def donut_to_json_value(raw: str) -> Any:
    """Coerce raw token text to a Python scalar."""
    s = raw.strip()
    if not s:
        return None
    if "<s_" in s:
        return donut_to_json(s)
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Token registration
# ---------------------------------------------------------------------------
def register_special_tokens(processor) -> List[str]:
    """Add task + schema tokens to Donut's tokenizer. Returns the token list."""
    new_tokens: List[str] = list(TASK_TOKENS.values())
    for name in SCHEMA_FIELD_NAMES:
        new_tokens.append(f"<s_{name}>")
        new_tokens.append(f"</s_{name}>")
    new_tokens.append(LIST_SEP)
    added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    )
    log.info("Added %d special tokens to tokenizer", added)
    return new_tokens


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    image_path: Path
    json_data: Dict[str, Any]
    region_class: str
    target_seq: str
    group_key: str


def discover_samples(root: Path) -> List[Sample]:
    """Scan data/vlm/numerical/ for paired (.jpg, .json) samples."""
    if not root.exists():
        return []
    samples: List[Sample] = []
    for img in sorted(root.rglob("*.jpg")):
        json_path = img.with_suffix(".json")
        if not json_path.exists():
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load %s: %s", json_path, e)
            continue

        region_class = data.get("type") or _infer_class_from_path(img)
        if region_class not in TASK_TOKENS:
            log.warning("Unknown region class %s for %s", region_class, img.name)
            continue

        target_seq = TASK_TOKENS[region_class] + json_to_donut(data)

        # Group key for D-024: strip Stage 1/2 suffixes back to original drawing
        stem = img.stem
        gk = stem.split(".rf.")[0]
        for marker in ("__View_", "__Measure_", "__GDT_", "__Roughness_"):
            if marker in gk:
                gk = gk.split(marker)[0]
                break
        samples.append(Sample(
            image_path=img,
            json_data=data,
            region_class=region_class,
            target_seq=target_seq,
            group_key=gk,
        ))
    return samples


def _infer_class_from_path(p: Path) -> Optional[str]:
    """Derive region class from parent folder name if JSON lacks 'type'."""
    parents = [x.name for x in p.parents]
    for name in parents[:3]:
        if name in TASK_TOKENS:
            return name
    return None


def split_samples(samples: List[Sample],
                  ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
                  seed: int = 42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Group-aware train/val/test split (D-024)."""
    if not samples:
        return [], [], []
    try:
        from sklearn.model_selection import GroupShuffleSplit  # noqa: PLC0415
    except ImportError:
        log.warning("scikit-learn not available, falling back to random split")
        import random
        rng = random.Random(seed)
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        return (shuffled[:n_train],
                shuffled[n_train:n_train + n_val],
                shuffled[n_train + n_val:])

    groups = [s.group_key for s in samples]
    train_val_split = GroupShuffleSplit(
        n_splits=1, test_size=ratios[2], random_state=seed,
    )
    tv_idx, test_idx = next(train_val_split.split(samples, groups=groups))
    tv_samples = [samples[i] for i in tv_idx]
    tv_groups = [groups[i] for i in tv_idx]

    val_frac = ratios[1] / (ratios[0] + ratios[1])
    train_val_split2 = GroupShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed + 1,
    )
    train_idx, val_idx = next(train_val_split2.split(tv_samples, groups=tv_groups))
    return (
        [tv_samples[i] for i in train_idx],
        [tv_samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def build_torch_dataset(samples: List[Sample], processor, max_length: int):
    """PyTorch Dataset wrapping samples with Donut tokenization."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from torch.utils.data import Dataset  # noqa: PLC0415

    class _NumDS(Dataset):
        def __init__(self, items: List[Sample]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int):
            s = self.items[i]
            image = Image.open(s.image_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values[0]

            tokenized = processor.tokenizer(
                s.target_seq,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = tokenized.input_ids[0].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "decoder_input_ids": _shift_right(
                    tokenized.input_ids[0], processor
                ),
            }

    return _NumDS(samples)


def _shift_right(input_ids, processor):
    import torch  # noqa: PLC0415
    decoder_start = processor.tokenizer.convert_tokens_to_ids(
        list(TASK_TOKENS.values())[0]
    )
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[1:] = input_ids[:-1].clone()
    shifted[0] = decoder_start
    shifted[shifted == -100] = processor.tokenizer.pad_token_id
    return shifted


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg_path: Path,
          device: Optional[str] = None,
          resume: Optional[Path] = None,
          load_in_8bit: bool = False) -> Path:
    """Fine-tune Donut on data/vlm/numerical/. Returns checkpoint dir."""
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        DonutProcessor,
        VisionEncoderDecoderModel,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )

    # ★ D-053 fix (2026-05-06): transformers 5.x 가 compute_loss 에서
    # `num_items_in_batch` kwargs 를 model.forward 에 전달.
    # DonutSwinModel.forward() 는 미지원 → TypeError.
    # Custom Trainer subclass 로 compute_loss override 하여 kwargs 흡수.
    class DonutTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    cfg = load_cfg(cfg_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    out_cfg = cfg.get("output", {})

    pretrained = model_cfg.get("pretrained", "naver-clova-ix/donut-base")
    input_size = model_cfg.get("input_size", [960, 1280])
    max_length = model_cfg.get("max_length", 768)

    data_root = Path(data_cfg.get("root", PROJECT_ROOT / "data" / "vlm" / "numerical"))
    split_ratios = (
        data_cfg.get("split", {}).get("train", 0.70),
        data_cfg.get("split", {}).get("val", 0.20),
        data_cfg.get("split", {}).get("test", 0.10),
    )
    seed = data_cfg.get("shuffle_seed", 42)

    log.info("Loading processor + model: %s", pretrained)
    processor = DonutProcessor.from_pretrained(pretrained)
    processor.image_processor.size = {
        "height": input_size[0], "width": input_size[1],
    }
    processor.image_processor.do_align_long_axis = False

    quant_kwargs: Dict[str, Any] = {}
    if load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
            log.info("Loading model in 8-bit (bitsandbytes)")
        except ImportError:
            log.warning("bitsandbytes not available; using FP16/BF16 instead")
            load_in_8bit = False

    model = VisionEncoderDecoderModel.from_pretrained(
        pretrained, **quant_kwargs,
    )

    # Register new tokens then resize embedding
    register_special_tokens(processor)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Set decoder start
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        list(TASK_TOKENS.values())[0]
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # ---- Discover & split samples --------------------------------
    log.info("Discovering samples under %s", data_root)
    samples = discover_samples(data_root)
    if not samples:
        raise RuntimeError(
            f"No (.jpg, .json) pairs under {data_root}. "
            "Run prepare_vlm_dataset.py first (Step 4)."
        )
    log.info("Found %d samples", len(samples))

    train_s, val_s, test_s = split_samples(samples, split_ratios, seed=seed)
    log.info("Split: train=%d  val=%d  test=%d (group-aware, D-024)",
             len(train_s), len(val_s), len(test_s))

    train_ds = build_torch_dataset(train_s, processor, max_length)
    val_ds = build_torch_dataset(val_s, processor, max_length)

    # ---- TrainingArguments ---------------------------------------
    ckpt_dir = Path(out_cfg.get("ckpt_dir", DEFAULT_CKPT_DIR))
    log_dir = Path(out_cfg.get("log_dir", ckpt_dir / "logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    precision = train_cfg.get("precision", "fp16").lower()
    use_bf16 = precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = (precision == "fp16") and not use_bf16

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        logging_dir=str(log_dir),
        num_train_epochs=train_cfg.get("epochs", 30),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(train_cfg.get("learning_rate", 1e-6)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        max_grad_norm=train_cfg.get("grad_clip", 1.0),
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("log_every_n_steps", 20),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=train_cfg.get("save_top_k", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = DonutTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.tokenizer,
        # ★ D-052 fix (2026-05-06): Donut batch (pixel_values/labels/decoder_input_ids)
        # 는 input_ids 키 없음 → HF default DataCollatorWithPadding 의 tokenizer.pad()
        # 호출 시 ValueError. default_data_collator 는 단순 stack 만 수행해 호환.
        # ★ D-053 fix (2026-05-06): DonutTrainer subclass 로 compute_loss override
        # → transformers 5.x 의 num_items_in_batch kwargs 흡수.
        data_collator=default_data_collator,
    )

    log.info("Starting training (%d epochs, batch=%d, lr=%.2e, %s)",
             args.num_train_epochs, args.per_device_train_batch_size,
             args.learning_rate, "BF16" if use_bf16 else "FP16" if use_fp16 else "FP32")
    if resume:
        trainer.train(resume_from_checkpoint=str(resume))
    else:
        trainer.train()

    # Save final
    final_dir = ckpt_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    log.info("Saved final model to %s", final_dir)
    return final_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_inference_model(ckpt_dir: Path,
                         device: Optional[str] = None) -> Tuple[Any, Any, str]:
    """Load processor + model for inference. Returns (processor, model, device)."""
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        DonutProcessor,
        VisionEncoderDecoderModel,
    )
    processor = DonutProcessor.from_pretrained(str(ckpt_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(ckpt_dir))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        model = model.to(device).half()
    else:
        model = model.to(device)
    model.eval()
    return processor, model, device


def predict_one(image_path: Path,
                region_class: str,
                processor=None,
                model=None,
                device: Optional[str] = None,
                ckpt_dir: Path = DEFAULT_CKPT_DIR / "final",
                max_length: int = 768) -> Dict[str, Any]:
    """Run fine-tuned Donut on one warped patch. Returns parsed schema dict."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if region_class not in TASK_TOKENS:
        raise ValueError(f"Unknown region_class: {region_class!r} "
                         f"(expected one of {list(TASK_TOKENS)})")
    if processor is None or model is None:
        processor, model, device = load_inference_model(ckpt_dir, device)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    if device.startswith("cuda"):
        pixel_values = pixel_values.to(device).half()
    else:
        pixel_values = pixel_values.to(device)

    task_token = TASK_TOKENS[region_class]
    decoder_input_ids = processor.tokenizer(
        task_token, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "")
    seq = seq.replace(processor.tokenizer.pad_token, "")

    parsed = donut_to_json(seq)
    if isinstance(parsed, dict):
        parsed.setdefault("type", region_class)
    return {
        "type": region_class,
        "source": str(image_path),
        "parsed": parsed,
        "raw_seq": seq,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    final = train(
        cfg_path=Path(args.cfg),
        device=args.device,
        resume=Path(args.resume) if args.resume else None,
        load_in_8bit=args.load_in_8bit,
    )
    log.info("Training complete. Final checkpoint: %s", final)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    rec = predict_one(
        image_path=Path(args.image),
        region_class=args.region_class,
        device=args.device,
        ckpt_dir=Path(args.ckpt),
    )
    out = Path(args.out) if args.out else (
        DEFAULT_OUTPUT_DIR / f"{Path(args.image).stem}.num.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log.info("Saved: %s", out)
    log.info("Parsed: %s", json.dumps(rec["parsed"], ensure_ascii=False))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """Run on a Stage 2 crop folder: <input-dir>/{Measure,GDT,Roughness}/*.jpg"""
    root = Path(args.input_dir)
    if not root.exists():
        log.error("Input dir not found: %s", root)
        return 2

    items: List[Tuple[str, Path]] = []
    for cls in TASK_TOKENS:
        d = root / cls
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((cls, p))

    if not items:
        log.warning("No patches found under %s/{Measure,GDT,Roughness}", root)
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else (root / "numerical")
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = load_inference_model(Path(args.ckpt), args.device)
    log.info("Loaded model. Processing %d patches.", len(items))

    summary: List[Dict[str, Any]] = []
    for i, (cls, p) in enumerate(items, 1):
        try:
            rec = predict_one(
                image_path=p, region_class=cls,
                processor=processor, model=model, device=device,
                ckpt_dir=Path(args.ckpt),
            )
        except Exception as e:  # noqa: BLE001
            log.error("Failed on %s: %s", p.name, e)
            rec = {"type": cls, "source": str(p), "error": str(e)}

        out_path = out_dir / f"{p.stem}.num.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        summary.append({"image": str(p), "class": cls, "json": str(out_path)})

        if i % 25 == 0 or i == len(items):
            log.info("[%d/%d] %s", i, len(items), p.name)

    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Done. Manifest: %s", manifest)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3-N — Donut Numerical VLM (fine-tune + predict)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train -------------------------------------------------------
    pt = sub.add_parser("train", help="Fine-tune Donut on data/vlm/numerical/.")
    pt.add_argument("--cfg", type=str, default=str(DEFAULT_CFG_PATH))
    pt.add_argument("--device", type=str, default=None)
    pt.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint dir.")
    pt.add_argument("--load-in-8bit", action="store_true",
                    help="Load model in 8-bit via bitsandbytes (memory saver).")
    pt.set_defaults(func=cmd_train)

    # ---- predict -----------------------------------------------------
    pp = sub.add_parser("predict", help="Predict for one warped patch.")
    pp.add_argument("--image", type=str, required=True)
    pp.add_argument("--region-class", type=str, required=True,
                    choices=list(TASK_TOKENS))
    pp.add_argument("--ckpt", type=str,
                    default=str(DEFAULT_CKPT_DIR / "final"))
    pp.add_argument("--device", type=str, default=None)
    pp.add_argument("--out", type=str, default=None)
    pp.set_defaults(func=cmd_predict)

    # ---- batch -------------------------------------------------------
    pb = sub.add_parser(
        "batch",
        help="Run on a Stage 2 crop folder (Measure/GDT/Roughness subdirs).",
    )
    pb.add_argument("--input-dir", type=str, required=True)
    pb.add_argument("--out-dir", type=str, default=None)
    pb.add_argument("--ckpt", type=str,
                    default=str(DEFAULT_CKPT_DIR / "final"))
    pb.add_argument("--device", type=str, default=None)
    pb.set_defaults(func=cmd_batch)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

