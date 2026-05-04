"""
src/rescue_misclassified_notes.py

D-038 — Stage 1 false positive Notes Rescue.

배경
----
Stage 1 Version A 가 일반 주석 (Text 클래스) 영역을 PMI 로 잘못 검출하면,
Stage 2 에서 SKIP 처리되어 메타데이터 JSON 에서 정보가 누락됨.

본 도구는 SKIP `stage1_fp_notes` 로 마킹된 PMI crop 들을
Stage 3-A Donut zero-shot OCR 로 처리하여 텍스트를 추출하고,
최종 JSON 의 ``general_notes`` 필드에 병합 가능한 형태로 저장.

흐름
----
::

    CVAT 라벨링 (SKIP `stage1_fp_notes`)
         ↓
    extract_skip_list.py → outputs/skip_lists/stage1_fp_notes.txt
         ↓
    ★ rescue_misclassified_notes.py (이 파일)
         ↓
    Donut zero-shot OCR (각 crop)
         ↓
    outputs/rescued_notes.json
         ↓
    pipeline.py 또는 stage4 merger 에서
    최종 JSON 의 "general_notes" 필드로 병합

CLI
---
::

    # 기본 (단일 입력 파일)
    python src/rescue_misclassified_notes.py \
        --skip-list outputs/skip_lists/stage1_fp_notes.txt \
        --crops-dir outputs/cvat_stage2_input_v3_upscaled \
        --output outputs/rescued_notes.json

    # 다른 모델 또는 device 지정
    python src/rescue_misclassified_notes.py \
        --skip-list outputs/skip_lists/stage1_fp_notes.txt \
        --crops-dir outputs/cvat_stage2_input_v3_upscaled \
        --output outputs/rescued_notes.json \
        --device 0 \
        --question "What is written in this engineering note?"

    # 언어 힌트 (다국어 도면)
    python src/rescue_misclassified_notes.py \
        --skip-list outputs/skip_lists/stage1_fp_notes.txt \
        --crops-dir outputs/cvat_stage2_input_v3_upscaled \
        --output outputs/rescued_notes.json \
        --language ja

산출물 (JSON)
-------------
::

    {
      "metadata": {
        "source": "stage1_fp_notes_rescue",
        "decision": "D-038",
        "model": "donut-base-finetuned-docvqa",
        "n_input": 47,
        "n_success": 45,
        "n_failed": 2,
        "timestamp": "2026-05-01T12:00:00Z"
      },
      "rescued_notes": [
        {
          "crop_filename": "DwgFoo__PMI_023.jpg",
          "drawing_id": "DwgFoo",
          "pmi_idx": 23,
          "items": ["材料は鉄かSUS403"],
          "raw_text": "材料は鉄かSUS403",
          "language_hint": "ja"
        },
        {
          "crop_filename": "DwgFoo__PMI_045.jpg",
          "drawing_id": "DwgFoo",
          "pmi_idx": 45,
          "items": ["+0.1以下のものは機械加工のこと"],
          "raw_text": "+0.1以下のものは機械加工のこと",
          "language_hint": "ja"
        }
      ]
    }

병합 예시 (Stage 4 JSON merger)
-------------------------------
::

    {
      "drawing_id": "DwgFoo_001",
      "title_block": { ... },
      "measures": [ ... ],
      "gdt": [ ... ],
      "roughness": [ ... ],
      "general_notes": [           ← rescue 결과 병합
        {
          "source": "stage1_fp_notes_rescue",
          "content": "材料は鉄かSUS403",
          "crop_id": "DwgFoo_001__PMI_023.jpg"
        }
      ]
    }

관련 의사결정
-------------
- D-037 adaptive padding (v3 base)
- D-038 ★ Stage 1 false positive Notes rescue
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root 를 path 에 추가 (python src/xxx.py 직접 실행 시 src/ 패키지 import 가능하게)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rescue_notes")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
PMI_NAME_PATTERN = re.compile(r"^(?P<drawing>.+?)__PMI_(?P<idx>\d+)\.jpg$")


def parse_pmi_filename(filename: str) -> Dict[str, Any]:
    """
    PMI crop 파일명에서 metadata 추출.

    예: "DwgFoo_jpg.rf.abc__PMI_023.jpg" → {"drawing_id": "DwgFoo_jpg.rf.abc", "pmi_idx": 23}
    """
    m = PMI_NAME_PATTERN.match(filename)
    if not m:
        return {"drawing_id": filename, "pmi_idx": -1}
    return {
        "drawing_id": m.group("drawing"),
        "pmi_idx": int(m.group("idx")),
    }


# ---------------------------------------------------------------------------
# Skip list loader
# ---------------------------------------------------------------------------
def load_skip_list(path: Path) -> List[str]:
    """
    skip list 파일 (stage1_fp_notes.txt 등) 로드. 주석 (#) 제외.
    """
    if not path.exists():
        log.error("Skip list not found: %s", path)
        return []

    files: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            files.append(line)
    log.info("Loaded skip list: %d entries from %s", len(files), path)
    return files


# ---------------------------------------------------------------------------
# Donut rescue
# ---------------------------------------------------------------------------
def rescue_one(crop_path: Path,
               processor, model, device: str,
               question: str,
               language_hint: Optional[str]) -> Dict[str, Any]:
    """
    단일 crop 에 Donut zero-shot OCR 적용.

    Returns
    -------
    dict
        {
            "crop_filename": "...",
            "drawing_id": "...",
            "pmi_idx": ...,
            "items": [...],
            "raw_text": "...",
            "language_hint": ...,
            "error": (있으면) "..."
        }
    """
    from src.stage3_alphabetical import predict_notes  # noqa: PLC0415

    metadata = parse_pmi_filename(crop_path.name)
    result = {
        "crop_filename": crop_path.name,
        "drawing_id": metadata["drawing_id"],
        "pmi_idx": metadata["pmi_idx"],
        "language_hint": language_hint,
    }

    try:
        prediction = predict_notes(
            crop_path, processor, model, device,
            question=question,
            language_hint=language_hint,
        )
        result["items"] = prediction.get("items", [])
        result["raw_text"] = prediction.get("raw", "")
        result["model"] = prediction.get("model", "donut")
    except Exception as e:  # noqa: BLE001
        log.error("Failed to process %s: %s", crop_path.name, e)
        result["items"] = []
        result["raw_text"] = ""
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="D-038 Stage 1 false positive Notes Rescue (Donut zero-shot)",
    )
    p.add_argument("--skip-list", type=Path, required=True,
                   help="stage1_fp_notes.txt (extract_skip_list.py 산출물)")
    p.add_argument("--crops-dir", type=Path, required=True,
                   help="PMI crop 폴더 (e.g., outputs/cvat_stage2_input_v3_upscaled)")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/rescued_notes.json"),
                   help="결과 JSON (default: outputs/rescued_notes.json)")
    p.add_argument("--device", default=None,
                   help="GPU id (e.g. 0) 또는 cpu")
    p.add_argument("--question", default="What text is in this image?",
                   help="Donut DocVQA 질문 (default: 'What text is in this image?')")
    p.add_argument("--language", default=None,
                   help="언어 힌트 (en/ko/ja/zh/ru, optional)")
    p.add_argument("--model",
                   default="naver-clova-ix/donut-base-finetuned-docvqa",
                   help="Donut 모델 이름")
    p.add_argument("--limit", type=int, default=0,
                   help="처리 crop 수 제한 (0 = 전체)")
    args = p.parse_args()

    # --- Validate ----
    if not args.skip_list.exists():
        log.error("Skip list not found: %s", args.skip_list)
        log.error("→ 먼저 extract_skip_list.py 실행:")
        log.error("    python src/extract_skip_list.py --xml ... --output-dir outputs/skip_lists/")
        return 1

    if not args.crops_dir.exists():
        log.error("Crops dir not found: %s", args.crops_dir)
        return 2

    # --- Load skip list ----
    files = load_skip_list(args.skip_list)
    if not files:
        log.warning("Skip list 비어 있음. rescue 대상 없음.")
        return 0

    if args.limit > 0:
        files = files[: args.limit]
        log.info("Limited to first %d crops", len(files))

    # --- 존재하지 않는 파일 필터 + 경로 매핑 ----
    crop_paths: List[Path] = []
    n_missing = 0
    for fn in files:
        crop_path = args.crops_dir / fn
        if crop_path.exists():
            crop_paths.append(crop_path)
        else:
            log.warning("Crop file not found: %s", crop_path)
            n_missing += 1

    if not crop_paths:
        log.error("처리 가능한 crop 파일 없음. crops-dir 경로 확인 필요.")
        return 3

    log.info("Crops to rescue: %d (missing: %d)", len(crop_paths), n_missing)

    # --- Donut 모델 로드 (1회) ----
    log.info("Loading Donut model: %s", args.model)
    try:
        from src.stage3_alphabetical import load_model  # noqa: PLC0415
    except ImportError as e:
        log.error("stage3_alphabetical import 실패: %s", e)
        log.error("→ src/stage3_alphabetical.py 가 정상인지 확인")
        return 4

    try:
        processor, model_obj, device = load_model(args.model, args.device)
    except Exception as e:  # noqa: BLE001
        log.error("Donut load 실패: %s", e)
        log.error("→ transformers + torch 환경 점검 필요")
        return 5

    log.info("Model loaded. Device: %s", device)

    # --- Rescue 실행 ----
    log.info("=" * 60)
    log.info("Rescue 시작 (Donut zero-shot OCR)")
    log.info("=" * 60)

    try:
        from tqdm import tqdm  # noqa: PLC0415
        iter_paths = tqdm(crop_paths, desc="Rescue", unit="crop", dynamic_ncols=True)
    except ImportError:
        iter_paths = crop_paths

    rescued: List[Dict[str, Any]] = []
    n_success = 0
    n_failed = 0
    n_empty = 0

    for crop_path in iter_paths:
        result = rescue_one(crop_path, processor, model_obj, device,
                            args.question, args.language)
        if "error" in result:
            n_failed += 1
        elif not result.get("items") and not result.get("raw_text"):
            n_empty += 1
        else:
            n_success += 1
        rescued.append(result)

    # --- JSON 출력 ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "source": "stage1_fp_notes_rescue",
            "decision": "D-038",
            "model": args.model,
            "language_hint": args.language,
            "question": args.question,
            "n_input": len(files),
            "n_processed": len(crop_paths),
            "n_success": n_success,
            "n_empty": n_empty,
            "n_failed": n_failed,
            "n_missing_files": n_missing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "rescued_notes": rescued,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # --- Summary ----
    log.info("")
    log.info("=" * 60)
    log.info("Rescue 완료")
    log.info("  Input            : %d", len(files))
    log.info("  Processed        : %d", len(crop_paths))
    log.info("  Success (text)   : %d", n_success)
    log.info("  Empty result     : %d", n_empty)
    log.info("  Failed           : %d", n_failed)
    log.info("  Missing files    : %d", n_missing)
    log.info("=" * 60)
    log.info("Output: %s", args.output)
    log.info("")
    log.info("[다음 단계 — JSON 병합]")
    log.info("  pipeline.py 또는 stage4 merger 에서 'general_notes' 필드로 병합:")
    log.info("    final_json['general_notes'] = output_data['rescued_notes']")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
