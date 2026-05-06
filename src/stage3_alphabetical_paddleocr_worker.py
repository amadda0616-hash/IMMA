"""
src/stage3_alphabetical_paddleocr_worker.py

★ Long-running PaddleOCR-VL-1.5 worker — Phase 15c (D-039) backend.

Activated by `stage3_alphabetical_paddleocr.py` wrapper (subprocess Popen).
Reads JSON line from stdin → infers → writes JSON line to stdout.

★ Runs in `.venv-paddleocr` (transformers 5.0.0 + D-042 monkey-patch).
★ D-046 호출 방식 (task keyword + bf16 + apply_chat_template + decode 슬라이스).

Protocol
--------
Input (stdin, JSON line per request):
    {"image_path": "...", "region_type": "titleblock"|"notes",
     "language_hint": "en"|"ko"|... (optional)}
    또는 종료: {"action": "shutdown"}

Output (stdout, JSON line per response):
    {"title_block": {...}, "raw_text": "...", "model": "PaddleOCR-VL-1.5", ...}
    또는 {"notes": [...], "raw_text": "...", ...}
    또는 {"error": "...", "trace": "..."}

READY signal (모델 로드 완료):
    {"status": "ready"}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Project root bootstrap (D-049 패턴)
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))

# stderr 만 사용 (stdout 은 JSON 통신 전용)
logging.basicConfig(
    level=logging.INFO,
    format="[worker] %(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("paddleocr_worker")

# --- D-046 task keyword 매핑 ---
# titleblock → "Table Recognition:" (TitleBlock = 표 구조)
# notes      → "OCR:" (일반 OCR)
REGION_TASK_KEYWORDS = {
    "titleblock":   "Table Recognition:",
    "title_block":  "Table Recognition:",
    "title":        "Table Recognition:",
    "notes":        "OCR:",
    "note":         "OCR:",
}

DEFAULT_MAX_PIXELS = 1280 * 28 * 28  # 1,003,520 — README 권장


# ---------------------------------------------------------------------------
# Model loading (D-042 monkey-patch + bf16)
# ---------------------------------------------------------------------------
def load_model_and_processor(device: str = "cuda:0"):
    """PaddleOCR-VL-1.5 로드.

    D-042 monkey-patch: transformers 5.0.0 의 `AutoConfig` 가
    `text_config` attribute 를 누락 → `get_text_config()` 호출로 보강.
    """
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoConfig, AutoProcessor, AutoModelForImageTextToText,
    )

    mid = "PaddlePaddle/PaddleOCR-VL-1.5"
    log.info("Loading PaddleOCR-VL-1.5 ...")

    config = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
        config.text_config = config.get_text_config()
        log.info("★ D-042 monkey-patch applied")

    processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        mid, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info("★ Model loaded: %.2fB params  device=%s  dtype=bf16", n_params, device)
    return processor, model, device


# ---------------------------------------------------------------------------
# Inference (D-046 호출 방식)
# ---------------------------------------------------------------------------
def infer_one(processor, model, device: str,
              image_path: Path, region_type: str,
              language_hint: Optional[str] = None,
              max_pixels: int = DEFAULT_MAX_PIXELS) -> Dict[str, Any]:
    """D-046 호출 방식 — task keyword + apply_chat_template + decode 슬라이스."""
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    task = REGION_TASK_KEYWORDS.get(region_type.lower())
    if task is None:
        return {"error": f"Unknown region_type: {region_type!r}"}

    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    img = Image.open(image_path).convert("RGB")

    # ★ D-046 — messages 안에 image 직접 binding + task keyword
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": task},
        ]},
    ]
    # ★ Phase 15c (2026-05-06) fix — images_kwargs 제거.
    # PaddleOCR-VL 의 processor strict validation 이 max_pixels kwargs 거부:
    # "TypeError: merged_typed_dict.__init__() got an unexpected keyword 'max_pixels'"
    # → processor 자체 default (1280*28*28) 사용.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    seq = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return {
        "raw_text": seq,
        "task": task,
        "region_type": region_type,
        "language_hint": language_hint,
        "model": "PaddleOCR-VL-1.5",
    }


# ---------------------------------------------------------------------------
# Output formatting (pipeline.py 호환)
# ---------------------------------------------------------------------------
def parse_titleblock_raw(raw: str) -> Dict[str, Any]:
    """OTSL/markdown table → 1차 구조화. 정확한 schema 매핑은 후속 (D-044 23 필드)."""
    return {
        "raw": raw,
        "fields": {},  # placeholder — 후속 OTSL parser 연결
    }


def parse_notes_raw(raw: str) -> list:
    """Raw OCR text → 라인 분리 list."""
    if not raw:
        return []
    return [ln.strip() for ln in raw.split("\n") if ln.strip()]


def to_pipeline_record(infer_res: Dict[str, Any], region_type: str) -> Dict[str, Any]:
    """pipeline.py (line 393-405) 기대 형식과 호환되는 dict.

    ★ Phase 15c (2026-05-06) fix — 키 mismatch 해결:
    - TitleBlock: pipeline 이 ``rec["fields"]`` (dict) 기대 → "fields" 키로 반환
    - Notes:      pipeline 이 ``rec["items"]`` (list)  기대 → "items"  키로 반환
    """
    if "error" in infer_res:
        return {
            "error": infer_res["error"],
            "model": "PaddleOCR-VL-1.5",
            "language_hint": infer_res.get("language_hint"),
        }
    raw = infer_res.get("raw_text", "")
    rt = region_type.lower()

    if rt in {"titleblock", "title_block", "title"}:
        # OTSL parser 미구현 — 1차 baseline: raw text 를 _raw_text 필드로 보존
        # 후속 (D-047): OTSL → 23 필드 (D-044) parser 연결
        parsed = parse_titleblock_raw(raw)
        fields = parsed.get("fields", {}) or {}
        if raw and "_raw_text" not in fields:
            fields["_raw_text"] = raw
        return {
            "fields": fields,                     # ★ pipeline.py: rec["fields"] (dict)
            "raw": raw,
            "model": "PaddleOCR-VL-1.5",
            "language_hint": infer_res.get("language_hint"),
        }

    if rt in {"notes", "note"}:
        return {
            "items": parse_notes_raw(raw),        # ★ pipeline.py: rec["items"] (list of str)
            "raw": raw,
            "model": "PaddleOCR-VL-1.5",
            "language_hint": infer_res.get("language_hint"),
        }

    return {"raw": raw, "model": "PaddleOCR-VL-1.5"}


# ---------------------------------------------------------------------------
# Main loop (stdin JSON line protocol)
# ---------------------------------------------------------------------------
def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    args = parser.parse_args()

    # ---- Initialize ----
    try:
        processor, model, device = load_model_and_processor(args.device)
    except Exception as e:  # noqa: BLE001
        _emit({
            "status": "init_error",
            "error": str(e),
            "trace": traceback.format_exc(),
        })
        return 1

    _emit({"status": "ready"})
    log.info("★ READY — waiting for stdin requests")

    # ---- Request loop ----
    n_processed = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"error": f"json_decode: {e}", "raw": line[:200]})
            continue

        if req.get("action") == "shutdown":
            log.info("Shutdown requested (after %d requests)", n_processed)
            break
        try:
            image_path = Path(req["image_path"])
            region_type = req.get("region_type", "titleblock")
            language_hint = req.get("language_hint")
            infer_res = infer_one(
                processor, model, device,
                image_path, region_type, language_hint,
                max_pixels=args.max_pixels,
            )
            response = to_pipeline_record(infer_res, region_type)
        except Exception as e:  # noqa: BLE001
            response = {"error": str(e), "trace": traceback.format_exc()}

        _emit(response)
        n_processed += 1

    log.info("Worker exiting (processed %d requests)", n_processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
