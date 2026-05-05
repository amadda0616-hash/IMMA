"""
src/prepare_vlm_dataset.py

Step 4 — Stage 3 (VLM) 학습/평가용 image-text pair 자동 시드 생성

Bridge between
--------------
- Upstream : Stage 1 (yolo_det.pt) + Stage 2 (yolo_obb.pt) 학습 완료 모델
- Downstream: Stage 3-N (Step 6) 학습 데이터 + Stage 3-A (Step 5/V5) 평가 GT

Workflow
--------
::

    dataset/<drawing>.jpg
            │
            ▼
    Stage 1 predict → View / TitleBlock / Notes BBox
            │
            ├──► TitleBlock / Notes crop ──► data/vlm/alphabetical/<id>.jpg
            │                                + .json (template, 사람 검수)
            │
            └──► View crop ──► Stage 2 predict (OBB)
                                    │
                                    ▼  perspective-warp de-rotation
                            Measure / GDT / Roughness patch ──► data/vlm/numerical/
                            <id>.jpg + .json (template, 사람 검수)

Output (per patch)
------------------
- ``<id>.jpg`` — crop / warped patch
- ``<id>.json`` — schema template + ``_review`` metadata block
- ``manifest.csv`` — 모든 생성 패치의 인덱스 (group_key 포함, D-024)

The JSON templates contain ``null`` fields the user must fill. With
``--ocr-prefill`` flag the script also runs Pytesseract on each patch
and writes raw OCR text into the ``_review.ocr_hint`` field — this
shortens human review time without contaminating ground-truth fields.

CLI
---
::

    # 모두 (Stage 1 + Stage 2 학습 완료 후 권장)
    python src/prepare_vlm_dataset.py all \\
        --det-weights checkpoints/yolo_det.pt \\
        --obb-weights checkpoints/yolo_obb.pt \\
        --ocr-prefill --device 0

    # TitleBlock + Notes 만 (Stage 1 학습 직후 가능)
    python src/prepare_vlm_dataset.py alphabetical \\
        --det-weights checkpoints/yolo_det.pt --ocr-prefill

    # Measure + GDT + Roughness 만 (Stage 2 학습 직후, ★ Step 6 학습 데이터)
    python src/prepare_vlm_dataset.py numerical \\
        --det-weights checkpoints/yolo_det.pt \\
        --obb-weights checkpoints/yolo_obb.pt --ocr-prefill
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Project root bootstrap — allow `python src/prepare_vlm_dataset.py` direct
# execution. (Without this, lazy `from src.stage1_layout import ...` /
# `from src.stage2_obb_view import ...` calls fail with ModuleNotFoundError
# because src/ parent is not on sys.path.)
# Same pattern as src/pipeline.py (Task #92).
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_DET_WEIGHTS = PROJECT_ROOT / "checkpoints" / "yolo_det.pt"
DEFAULT_OBB_WEIGHTS = PROJECT_ROOT / "checkpoints" / "yolo_obb.pt"

VLM_NUMERICAL_DIR = PROJECT_ROOT / "data" / "vlm" / "numerical"
VLM_ALPHA_DIR     = PROJECT_ROOT / "data" / "vlm" / "alphabetical"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

OCR_LANGS = "kor+eng+rus+jpn"
WIN_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# ---------------------------------------------------------------------------
# JSON Schema Templates (HANDOFF §5.3, §5.4)
# ---------------------------------------------------------------------------
NUMERICAL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Measure": {
        "type": "Measure",
        "nominal": None,         # ← human fills (필수)
        "tolerance": None,       # ← {"upper": ..., "lower": ...} 또는 null
        "unit": "mm",            # default
    },
    "GDT": {
        "type": "GDT",
        "symbol": None,          # ← e.g. "⏤", "⌖" (필수)
        "tolerance": None,       # ← float (필수)
        "datum": None,           # ← list[str] e.g. ["A", "B"]
    },
    "Roughness": {
        "type": "Roughness",
        "Ra": None,              # ← human fills (필수)
        "unit": "μm",
    },
}

ALPHABETICAL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "TitleBlock": {
        "type": "TitleBlock",
        "fields": {
            "drawing_no":  None,
            "title":       None,
            "material":    None,
            "scale":       None,
            "revision":    None,
            "date":        None,
            "drawn_by":    None,
            "checked_by":  None,
            "approved_by": None,
            "part_no":     None,
            "weight":      None,
        },
    },
    "Notes": {
        "type": "Notes",
        "items": [],   # ← list[str] (각 항목)
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_vlm_dataset")


# ---------------------------------------------------------------------------
# IO helpers (Unicode-safe)
# ---------------------------------------------------------------------------
def imread_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:  # noqa: BLE001
        log.error("imread failed for %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Group key (D-024)
# ---------------------------------------------------------------------------
def extract_group_key(filename: str) -> str:
    """Extract group key from filename for group-aware split (D-024).

    Examples
    --------
    >>> extract_group_key("11_jpeg.rf.8b46c563d114.jpg")
    '11_jpeg'
    >>> extract_group_key("drawing__View_00__Measure_03.jpg")
    'drawing'
    """
    stem = Path(filename).stem
    # 1) Strip Roboflow augmentation hash
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    # 2) Strip Stage 1/2 patch suffix
    for marker in ("__View_", "__TitleBlock_", "__Notes_",
                   "__Measure_", "__GDT_", "__Roughness_"):
        if marker in stem:
            stem = stem.split(marker)[0]
            break
    return stem


# ---------------------------------------------------------------------------
# OCR pre-fill (optional)
# ---------------------------------------------------------------------------
def setup_tesseract(custom_path: Optional[str] = None) -> None:
    """Configure pytesseract path for Windows compatibility (Linux uses PATH)."""
    try:
        import pytesseract  # noqa: PLC0415
    except ImportError:
        return
    if custom_path:
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return
    if sys.platform.startswith("win"):
        default = Path(WIN_TESSERACT_DEFAULT)
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)


def ocr_image(image_path: Path, langs: str = OCR_LANGS) -> str:
    """Run Pytesseract on a patch image. Returns raw text or empty string."""
    try:
        import pytesseract  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return ""
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img, lang=langs, config="--psm 6").strip()
    except Exception as e:  # noqa: BLE001
        log.debug("OCR failed on %s: %s", image_path, e)
        return ""


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numeric_hint(ocr_text: str) -> Optional[float]:
    """Extract first numeric value from OCR text. Useful for Measure pre-fill."""
    if not ocr_text:
        return None
    m = _NUMERIC_RE.search(ocr_text)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------
def build_numerical_template(region_class: str,
                             source_image: Path,
                             ocr_text: str = "",
                             extra: Optional[Dict[str, Any]] = None,
                             ) -> Dict[str, Any]:
    """Create JSON template for a Numerical patch (Measure/GDT/Roughness)."""
    if region_class not in NUMERICAL_TEMPLATES:
        raise ValueError(f"Unknown numerical class: {region_class}")
    tpl = json.loads(json.dumps(NUMERICAL_TEMPLATES[region_class]))  # deep copy
    review = {
        "source_image": str(source_image),
        "completed": False,
    }
    if ocr_text:
        review["ocr_hint"] = ocr_text
        if region_class == "Measure":
            review["ocr_numeric"] = extract_numeric_hint(ocr_text)
    if extra:
        review.update(extra)
    tpl["_review"] = review
    return tpl


def build_alphabetical_template(region_class: str,
                                source_image: Path,
                                ocr_text: str = "",
                                extra: Optional[Dict[str, Any]] = None,
                                ) -> Dict[str, Any]:
    """Create JSON template for an Alphabetical patch (TitleBlock/Notes)."""
    if region_class not in ALPHABETICAL_TEMPLATES:
        raise ValueError(f"Unknown alphabetical class: {region_class}")
    tpl = json.loads(json.dumps(ALPHABETICAL_TEMPLATES[region_class]))
    review = {
        "source_image": str(source_image),
        "completed": False,
    }
    if ocr_text:
        review["ocr_text"] = ocr_text
    if extra:
        review.update(extra)
    tpl["_review"] = review
    return tpl


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Per-drawing pipelines
# ---------------------------------------------------------------------------
def process_drawing_alphabetical(image_path: Path,
                                 det_predict_fn,
                                 det_crop_fn,
                                 det_weights: Path,
                                 device: Optional[str],
                                 alpha_dir: Path,
                                 ocr_prefill: bool,
                                 imgsz_det: int = 1280,
                                 conf_thr: float = 0.25,
                                 ) -> List[Dict[str, Any]]:
    """Run Stage 1 → save TitleBlock + Notes patches + JSON templates.

    Returns manifest rows.
    """
    rows: List[Dict[str, Any]] = []
    try:
        rec = det_predict_fn(image_path, weights=det_weights,
                             conf_thr=conf_thr, imgsz=imgsz_det, device=device)
    except Exception as e:  # noqa: BLE001
        log.error("Stage 1 predict failed for %s: %s", image_path.name, e)
        return rows

    # Filter to TitleBlock + Notes only
    relevant = [r for r in rec.get("regions", [])
                if r["class"] in ("TitleBlock", "Notes")]
    if not relevant:
        return rows

    # Reuse Stage 1's crop_regions on relevant regions only
    tmp_out = alpha_dir / image_path.stem
    crops = det_crop_fn(image_path, relevant, tmp_out, padding=5)

    group_key = extract_group_key(image_path.name)

    # Move crops to flat alpha_dir/ and create JSON template per crop
    for c in crops:
        cls = c["class"]
        crop_path = Path(c["path"])
        # Move to flat dir with descriptive name
        dst_jpg = alpha_dir / crop_path.name
        if str(crop_path) != str(dst_jpg):
            try:
                crop_path.replace(dst_jpg)
            except OSError:
                # Fallback: copy
                import shutil  # noqa: PLC0415
                shutil.copy2(crop_path, dst_jpg)
                crop_path.unlink(missing_ok=True)

        ocr_text = ocr_image(dst_jpg) if ocr_prefill else ""
        tpl = build_alphabetical_template(cls, dst_jpg, ocr_text=ocr_text)
        json_path = dst_jpg.with_suffix(".json")
        write_json(json_path, tpl)

        rows.append({
            "filename": dst_jpg.name,
            "region_class": cls,
            "parent_drawing": image_path.name,
            "parent_view": "-",
            "group_key": group_key,
            "has_ocr_hint": bool(ocr_text),
            "json_path": str(json_path),
            "status": "pending_review",
        })

    # Cleanup empty subdir
    try:
        if tmp_out.exists() and not any(tmp_out.iterdir()):
            tmp_out.rmdir()
        for sub in tmp_out.glob("*"):
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
    except OSError:
        pass

    return rows


def process_drawing_numerical(image_path: Path,
                              det_predict_fn,
                              det_crop_fn,
                              obb_predict_fn,
                              obb_crop_fn,
                              det_weights: Path,
                              obb_weights: Path,
                              device: Optional[str],
                              num_dir: Path,
                              ocr_prefill: bool,
                              imgsz_det: int = 1280,
                              imgsz_obb: int = 1024,
                              conf_thr: float = 0.25,
                              ) -> List[Dict[str, Any]]:
    """Stage 1 → View crop → Stage 2 OBB → de-rotation patch + JSON template."""
    rows: List[Dict[str, Any]] = []

    # Stage 1
    try:
        det_rec = det_predict_fn(image_path, weights=det_weights,
                                 conf_thr=conf_thr, imgsz=imgsz_det, device=device)
    except Exception as e:  # noqa: BLE001
        log.error("Stage 1 failed for %s: %s", image_path.name, e)
        return rows

    views = [r for r in det_rec.get("regions", []) if r["class"] == "View"]
    if not views:
        return rows

    group_key = extract_group_key(image_path.name)
    tmp_dir = num_dir / image_path.stem
    view_crops = det_crop_fn(image_path, views, tmp_dir, padding=5)

    # Stage 2 on each View crop
    for vc in view_crops:
        view_path = Path(vc["path"])
        view_id = view_path.stem.split("__View_")[-1]   # e.g. "00"
        try:
            obb_rec = obb_predict_fn(
                view_path, weights=obb_weights,
                conf_thr=conf_thr, imgsz=imgsz_obb, device=device,
                parent_bbox=vc["bbox"],
            )
        except Exception as e:  # noqa: BLE001
            log.error("Stage 2 failed for %s: %s", view_path.name, e)
            continue

        annotations = obb_rec.get("annotations", [])
        if not annotations:
            continue

        # Save warped patches
        ann_out = tmp_dir / "annotations"
        crops = obb_crop_fn(view_path, annotations, ann_out)

        for c in crops:
            cls = c["class"]
            crop_path = Path(c["path"])
            # Flat name: <drawing>__<view_id>__<class>_<idx>.jpg
            new_name = (f"{image_path.stem}__View_{view_id}__"
                        f"{crop_path.name.split('__')[-1]}")
            dst_jpg = num_dir / new_name
            try:
                crop_path.replace(dst_jpg)
            except OSError:
                import shutil  # noqa: PLC0415
                shutil.copy2(crop_path, dst_jpg)
                crop_path.unlink(missing_ok=True)

            ocr_text = ocr_image(dst_jpg) if ocr_prefill else ""
            tpl = build_numerical_template(cls, dst_jpg, ocr_text=ocr_text,
                                           extra={"obb": c.get("obb")})
            json_path = dst_jpg.with_suffix(".json")
            write_json(json_path, tpl)

            rows.append({
                "filename": dst_jpg.name,
                "region_class": cls,
                "parent_drawing": image_path.name,
                "parent_view": f"View_{view_id}",
                "group_key": group_key,
                "has_ocr_hint": bool(ocr_text),
                "json_path": str(json_path),
                "status": "pending_review",
            })

    # Cleanup nested subdirs
    try:
        for p in sorted(tmp_dir.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except OSError:
        pass

    return rows


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        log.warning("No rows to write — manifest skipped: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "region_class", "parent_drawing", "parent_view",
        "group_key", "has_ocr_hint", "json_path", "status",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("Manifest written: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Main routines
# ---------------------------------------------------------------------------
def collect_drawings(dataset_dir: Path) -> List[Path]:
    return sorted(
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def run_alphabetical(args: argparse.Namespace) -> int:
    """alphabetical 모드 — Stage 1 학습 직후 가능 (TB + Notes)."""
    from src.stage1_layout import predict_one as det_predict
    from src.stage1_layout import crop_regions as det_crop

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        log.error("dataset dir not found: %s", dataset_dir)
        return 2

    if args.ocr_prefill:
        setup_tesseract(args.tesseract)
        log.info("OCR pre-fill ENABLED (%s)", OCR_LANGS)

    images = collect_drawings(dataset_dir)
    if args.limit > 0:
        images = images[:args.limit]
    log.info("Processing %d drawings → %s", len(images), VLM_ALPHA_DIR)
    VLM_ALPHA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for i, img in enumerate(images, 1):
        rows = process_drawing_alphabetical(
            image_path=img,
            det_predict_fn=det_predict,
            det_crop_fn=det_crop,
            det_weights=Path(args.det_weights),
            device=args.device,
            alpha_dir=VLM_ALPHA_DIR,
            ocr_prefill=args.ocr_prefill,
            imgsz_det=args.imgsz_det,
            conf_thr=args.conf,
        )
        all_rows.extend(rows)
        if i % 25 == 0 or i == len(images):
            log.info("[%d/%d] %s → +%d alpha pairs (total %d)",
                     i, len(images), img.name, len(rows), len(all_rows))

    manifest = VLM_ALPHA_DIR / "manifest.csv"
    write_manifest(all_rows, manifest)
    return 0


def run_numerical(args: argparse.Namespace) -> int:
    """numerical 모드 — Stage 2 학습 후. ★ Step 6 학습 데이터 시드."""
    from src.stage1_layout import predict_one as det_predict
    from src.stage1_layout import crop_regions as det_crop
    from src.stage2_annotation import predict_one as obb_predict
    from src.stage2_annotation import crop_obb_regions as obb_crop

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        log.error("dataset dir not found: %s", dataset_dir)
        return 2
    obb_w = Path(args.obb_weights)
    if not obb_w.exists():
        log.error("obb weights not found: %s "
                  "(Stage 2 학습 후 사용 가능)", obb_w)
        return 2

    if args.ocr_prefill:
        setup_tesseract(args.tesseract)
        log.info("OCR pre-fill ENABLED (%s)", OCR_LANGS)

    images = collect_drawings(dataset_dir)
    if args.limit > 0:
        images = images[:args.limit]
    log.info("Processing %d drawings → %s", len(images), VLM_NUMERICAL_DIR)
    VLM_NUMERICAL_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for i, img in enumerate(images, 1):
        rows = process_drawing_numerical(
            image_path=img,
            det_predict_fn=det_predict,
            det_crop_fn=det_crop,
            obb_predict_fn=obb_predict,
            obb_crop_fn=obb_crop,
            det_weights=Path(args.det_weights),
            obb_weights=obb_w,
            device=args.device,
            num_dir=VLM_NUMERICAL_DIR,
            ocr_prefill=args.ocr_prefill,
            imgsz_det=args.imgsz_det,
            imgsz_obb=args.imgsz_obb,
            conf_thr=args.conf,
        )
        all_rows.extend(rows)
        if i % 25 == 0 or i == len(images):
            log.info("[%d/%d] %s → +%d num pairs (total %d)",
                     i, len(images), img.name, len(rows), len(all_rows))

    manifest = VLM_NUMERICAL_DIR / "manifest.csv"
    write_manifest(all_rows, manifest)
    return 0


def run_all(args: argparse.Namespace) -> int:
    """all 모드 — alphabetical + numerical 동시."""
    rc1 = run_alphabetical(args)
    if rc1 != 0:
        log.warning("alphabetical 종료 코드 %d — numerical 계속 진행", rc1)
    rc2 = run_numerical(args)
    return max(rc1, rc2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_DIR))
    p.add_argument("--det-weights", type=str, default=str(DEFAULT_DET_WEIGHTS))
    p.add_argument("--obb-weights", type=str, default=str(DEFAULT_OBB_WEIGHTS))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--imgsz-det", type=int, default=1280)
    p.add_argument("--imgsz-obb", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--ocr-prefill", action="store_true",
                   help="Pytesseract 로 OCR 텍스트 추출해 _review.ocr_hint 에 저장 "
                        "(사람 검수 시간 단축)")
    p.add_argument("--tesseract", type=str, default=None,
                   help="(Windows) tesseract.exe 경로 override")
    p.add_argument("--limit", type=int, default=0,
                   help="처리할 도면 최대 개수 (0 = 전체). 디버깅/테스트용")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4 — VLM image-text pair 자동 시드 생성 "
                    "(Stage 3 학습/평가 데이터 준비)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("all", help="alphabetical + numerical 동시 (Stage 1+2 학습 후)")
    add_common_args(pa)
    pa.set_defaults(func=run_all)

    pal = sub.add_parser("alphabetical",
                         help="TitleBlock/Notes 만 (Stage 1 학습 직후 가능)")
    add_common_args(pal)
    pal.set_defaults(func=run_alphabetical)

    pn = sub.add_parser("numerical",
                        help="Measure/GDT/Roughness 만 (★ Step 6 학습 데이터, Stage 2 후)")
    add_common_args(pn)
    pn.set_defaults(func=run_numerical)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
