"""
src/stage3_alphabetical.py

Stage 3-A — Alphabetical VLM (Donut, zero-shot)

Purpose
-------
Read **TitleBlock** and **Notes** crops produced by Stage 1 and emit
free-form structured JSON. **No fine-tuning** (D-001 / D-018):
zero-shot inference using the public Donut DocVQA / CORD-v2 checkpoints.

Language note
-------------
Donut public checkpoints were primarily trained on English documents.
For KO / JP / RU drawings (D-013), accuracy will be lower than EN.
The paper baseline (Khan 2025) reports zero-shot Alphabetical
overall F1 0.672 (TitleBlock 0.533 / Notes 0.810).

Two operating modes
-------------------
1. **DocVQA** (default, multi-question)
   - Asks a list of questions per TitleBlock to extract specific fields.
   - More controllable, slower (one forward pass per question).
   - Model: ``naver-clova-ix/donut-base-finetuned-docvqa``

2. **CORD-v2** (single-pass, structured)
   - Single forward pass, output uses CORD receipt schema.
   - Faster, but field names need re-mapping to drawing fields.
   - Model: ``naver-clova-ix/donut-base-finetuned-cord-v2``

CLI
---
::

    # 1) Predict for a single TitleBlock crop
    python src/stage3_alphabetical.py predict \
        --image outputs/crops/<drawing>/TitleBlock/<file>.jpg \
        --region titleblock --mode docvqa \
        --out outputs/<file>.alpha.json

    # 2) Predict for a single Notes crop
    python src/stage3_alphabetical.py predict \
        --image outputs/crops/<drawing>/Notes/<file>.jpg \
        --region notes --out outputs/<file>.alpha.json

    # 3) Batch over a folder of TitleBlock / Notes crops
    python src/stage3_alphabetical.py batch \
        --input-dir outputs/crops/<drawing> \
        --out-dir outputs/<drawing>/alphabetical

Output JSON schema (HANDOFF §5.3)
---------------------------------
TitleBlock::

    {
      "type": "TitleBlock",
      "source": "<image path>",
      "fields": {"drawing_no": "...", "title": "...", "material": "...", ...},
      "raw": {"<question>": "<answer>", ...},   # DocVQA mode only
      "model": "donut-base-finetuned-docvqa",
      "language_hint": "en"
    }

Notes::

    {
      "type": "Notes",
      "source": "<image path>",
      "items": ["1. UNLESS OTHERWISE SPECIFIED ...", "2. ..."],
      "raw": "<full answer>",
      "model": "donut-base-finetuned-docvqa",
      "language_hint": "en"
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Lazy import torch / transformers so the CLI help works even if not installed.
# Heavy modules are imported inside load_model().

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

DOCVQA_MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"
CORD_MODEL = "naver-clova-ix/donut-base-finetuned-cord-v2"

# Default DocVQA question set for TitleBlock.
# Keys map to fields in the unified JSON (HANDOFF §5.3).
DEFAULT_TITLEBLOCK_QUESTIONS: List[Tuple[str, str]] = [
    ("drawing_no",  "What is the drawing number?"),
    ("title",       "What is the title?"),
    ("material",    "What is the material?"),
    ("scale",       "What is the scale?"),
    ("revision",    "What is the revision?"),
    ("date",        "What is the date?"),
    ("drawn_by",    "Who drew this drawing?"),
    ("checked_by",  "Who checked this drawing?"),
    ("approved_by", "Who approved this drawing?"),
    ("part_no",     "What is the part number?"),
    ("sheet",       "What is the sheet number?"),
    ("project",     "What is the project name?"),
    ("weight",      "What is the weight?"),
    ("tolerance",   "What is the general tolerance?"),
]

DEFAULT_NOTES_QUESTION = "What do the notes say?"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage3_alphabetical")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str = DOCVQA_MODEL,
               device: Optional[str] = None) -> Tuple[Any, Any, str]:
    """Load Donut processor + model. Returns ``(processor, model, device)``.

    The model is moved to GPU when CUDA is available, FP16 for memory.
    """
    import torch  # noqa: PLC0415
    from transformers import DonutProcessor, VisionEncoderDecoderModel  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading Donut: %s (device=%s)", model_name, device)
    processor = DonutProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = model.to(device).half()  # FP16 to fit RTX 5080 16GB comfortably
    else:
        model = model.to(device)
    model.eval()
    return processor, model, device


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def _generate(processor, model, device: str,
              image: Image.Image, prompt: str,
              max_length: int = 768) -> str:
    """Run Donut ``generate`` for a given (image, prompt). Return decoded text."""
    import torch  # noqa: PLC0415

    pixel_values = processor(image, return_tensors="pt").pixel_values
    if device.startswith("cuda"):
        pixel_values = pixel_values.to(device).half()
    else:
        pixel_values = pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
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
    # Strip the initial task token (e.g. "<s_docvqa>") — Donut convention
    seq = re.sub(r"^<.*?>", "", seq, count=1).strip()
    return seq


def _safe_token2json(processor, seq: str) -> Any:
    """Try ``processor.token2json`` and fall back to raw string on failure."""
    try:
        return processor.token2json(seq)
    except Exception:  # noqa: BLE001
        return seq


# ---------------------------------------------------------------------------
# TitleBlock prediction
# ---------------------------------------------------------------------------
def predict_titleblock(image_path: Path,
                       processor, model, device: str,
                       questions: Optional[List[Tuple[str, str]]] = None,
                       mode: str = "docvqa",
                       language_hint: Optional[str] = None) -> Dict[str, Any]:
    """Zero-shot extraction of TitleBlock fields.

    Parameters
    ----------
    mode : "docvqa" | "cord"
        DocVQA = ask a list of questions (default).
        CORD = single-pass receipt-schema parse (faster, less accurate
        for engineering fields).
    """
    image = Image.open(image_path).convert("RGB")

    record: Dict[str, Any] = {
        "type": "TitleBlock",
        "source": str(image_path),
        "fields": {},
        "raw": {},
        "model": "",
        "language_hint": language_hint,
    }

    if mode == "cord":
        seq = _generate(processor, model, device, image, "<s_cord-v2>")
        parsed = _safe_token2json(processor, seq)
        record["fields"] = parsed if isinstance(parsed, dict) else {}
        record["raw"] = {"cord": seq}
        record["model"] = "donut-base-finetuned-cord-v2"
        return record

    # docvqa (default)
    qs = questions or DEFAULT_TITLEBLOCK_QUESTIONS
    record["model"] = "donut-base-finetuned-docvqa"
    for key, question in qs:
        prompt = (
            f"<s_docvqa><s_question>{question}</s_question><s_answer>"
        )
        seq = _generate(processor, model, device, image, prompt)
        parsed = _safe_token2json(processor, seq)
        if isinstance(parsed, dict) and "answer" in parsed:
            answer = (parsed.get("answer") or "").strip()
        else:
            answer = (seq or "").strip()
        record["raw"][question] = answer
        if answer and answer.lower() not in {"none", "n/a", "unknown", ""}:
            record["fields"][key] = answer
    return record


# ---------------------------------------------------------------------------
# Notes prediction
# ---------------------------------------------------------------------------
_NOTE_LINE_RE = re.compile(r"(?:^|\s)(\d+\s*[\.\)]\s*[^.\n]+)", re.UNICODE)


def _split_notes(text: str) -> List[str]:
    """Best-effort split of a notes blob into numbered items."""
    if not text:
        return []
    # First try numbered-list pattern (1. ..., 2) ...)
    matches = _NOTE_LINE_RE.findall(text)
    if len(matches) >= 2:
        return [m.strip() for m in matches]
    # Fallback: split on newlines or sentence terminators
    parts = [p.strip() for p in re.split(r"[\n\.;]+", text) if p.strip()]
    return parts


def predict_notes(image_path: Path,
                  processor, model, device: str,
                  question: str = DEFAULT_NOTES_QUESTION,
                  language_hint: Optional[str] = None) -> Dict[str, Any]:
    """Zero-shot extraction of free-form Notes content."""
    image = Image.open(image_path).convert("RGB")
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    seq = _generate(processor, model, device, image, prompt)
    parsed = _safe_token2json(processor, seq)
    if isinstance(parsed, dict) and "answer" in parsed:
        answer = (parsed.get("answer") or "").strip()
    else:
        answer = (seq or "").strip()
    items = _split_notes(answer)
    return {
        "type": "Notes",
        "source": str(image_path),
        "items": items,
        "raw": answer,
        "model": "donut-base-finetuned-docvqa",
        "language_hint": language_hint,
    }


# ---------------------------------------------------------------------------
# Public dispatcher (importable by pipeline.py)
# ---------------------------------------------------------------------------
def predict_one(image_path: Path,
                region_type: str = "titleblock",
                mode: str = "docvqa",
                processor=None, model=None, device: Optional[str] = None,
                questions: Optional[List[Tuple[str, str]]] = None,
                language_hint: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end zero-shot prediction for a single crop.

    Parameters
    ----------
    region_type : "titleblock" | "notes"
    mode        : "docvqa" | "cord"   (CORD only meaningful for titleblock)
    """
    if processor is None or model is None:
        model_name = DOCVQA_MODEL if mode == "docvqa" else CORD_MODEL
        processor, model, device = load_model(model_name, device)

    region_type = region_type.lower()
    if region_type in {"titleblock", "title", "title_block"}:
        return predict_titleblock(
            image_path, processor, model, device,
            questions=questions, mode=mode, language_hint=language_hint,
        )
    if region_type in {"notes", "note"}:
        return predict_notes(
            image_path, processor, model, device,
            language_hint=language_hint,
        )
    raise ValueError(f"Unknown region_type: {region_type!r}")


# ---------------------------------------------------------------------------
# CLI — predict (single image)
# ---------------------------------------------------------------------------
def cmd_predict(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        log.error("Image not found: %s", image_path)
        return 2

    rec = predict_one(
        image_path=image_path,
        region_type=args.region,
        mode=args.mode,
        device=args.device,
        language_hint=args.language,
    )

    out = Path(args.out) if args.out else (
        DEFAULT_OUTPUT_DIR / f"{image_path.stem}.alpha.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    log.info("Saved: %s", out)
    if rec["type"] == "TitleBlock":
        log.info("Extracted %d fields", len(rec.get("fields", {})))
        for k, v in rec.get("fields", {}).items():
            log.info("  %-12s : %s", k, v)
    else:
        log.info("Extracted %d notes items", len(rec.get("items", [])))
    return 0


# ---------------------------------------------------------------------------
# CLI — batch (directory)
# ---------------------------------------------------------------------------
def _iter_class_dirs(root: Path) -> List[Tuple[str, Path]]:
    """Discover ``<root>/{TitleBlock,Notes}/*.jpg`` produced by Stage 1.

    Returns a list of ``(region_type, image_path)`` tuples.
    """
    items: List[Tuple[str, Path]] = []
    mapping = {
        "TitleBlock": "titleblock",
        "Notes": "notes",
    }
    for cls_name, region_type in mapping.items():
        d = root / cls_name
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                items.append((region_type, p))
    return items


def cmd_batch(args: argparse.Namespace) -> int:
    root = Path(args.input_dir)
    if not root.exists():
        log.error("Input dir not found: %s", root)
        return 2

    items = _iter_class_dirs(root)
    if not items:
        log.warning("No TitleBlock / Notes images found under %s", root)
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else (root / "alphabetical")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once.
    model_name = DOCVQA_MODEL if args.mode == "docvqa" else CORD_MODEL
    processor, model, device = load_model(model_name, args.device)

    summary: List[Dict[str, Any]] = []
    for i, (region_type, img) in enumerate(items, 1):
        log.info("[%d/%d] %s (%s)", i, len(items), img.name, region_type)
        try:
            rec = predict_one(
                image_path=img,
                region_type=region_type,
                mode=args.mode,
                processor=processor, model=model, device=device,
                language_hint=args.language,
            )
        except Exception as e:  # noqa: BLE001
            log.error("Failed on %s: %s", img.name, e)
            rec = {"type": region_type, "source": str(img), "error": str(e)}

        out_path = out_dir / f"{img.stem}.alpha.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        summary.append({
            "image": str(img),
            "region": region_type,
            "json": str(out_path),
            "fields": len(rec.get("fields", {})) if region_type == "titleblock"
                      else len(rec.get("items", [])),
        })

    manifest = out_dir / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("Done. %d files. Manifest: %s", len(summary), manifest)
    return 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3-A — Donut Alphabetical VLM (zero-shot) "
                    "for TitleBlock and Notes crops.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- predict ---------------------------------------------------------
    pp = sub.add_parser("predict", help="Predict for a single crop image.")
    pp.add_argument("--image", type=str, required=True)
    pp.add_argument("--region", type=str, choices=["titleblock", "notes"],
                    default="titleblock")
    pp.add_argument("--mode", type=str, choices=["docvqa", "cord"],
                    default="docvqa",
                    help="docvqa = multi-question (recommended); "
                         "cord = single-pass receipt schema")
    pp.add_argument("--device", type=str, default=None,
                    help='e.g. "cuda", "cuda:0", "cpu" (default: auto)')
    pp.add_argument("--language", type=str, default=None,
                    help="Optional language hint stored in output (en/ko/ja/ru)")
    pp.add_argument("--out", type=str, default=None,
                    help="Output JSON path (default: outputs/<stem>.alpha.json)")
    pp.set_defaults(func=cmd_predict)

    # ---- batch -----------------------------------------------------------
    pb = sub.add_parser(
        "batch",
        help="Run on a Stage 1 crop dir: <input-dir>/{TitleBlock,Notes}/*.jpg"
    )
    pb.add_argument("--input-dir", type=str, required=True,
                    help="e.g. outputs/crops/<drawing_id>")
    pb.add_argument("--out-dir", type=str, default=None,
                    help="Default: <input-dir>/alphabetical")
    pb.add_argument("--mode", type=str, choices=["docvqa", "cord"],
                    default="docvqa")
    pb.add_argument("--device", type=str, default=None)
    pb.add_argument("--language", type=str, default=None)
    pb.set_defaults(func=cmd_batch)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
