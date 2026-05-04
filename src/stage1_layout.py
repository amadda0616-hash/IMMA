"""
src/stage1_layout.py

Stage 1 — YOLOv11-det (Layout Segmentation)

Classes
-------
View / TitleBlock / Notes

This module exposes three CLI subcommands and three public functions for the
end-to-end pipeline (``src/pipeline.py``).

CLI
---
::

    # 1) Fine-tune YOLOv11m-det on the layout dataset
    python src/stage1_layout.py train \
        --data configs/yolo_det.yaml \
        --model yolo11m.pt --epochs 100 --imgsz 1280 --batch 8

    # 2) Predict regions for one image (writes JSON, schema = HANDOFF §5.1)
    python src/stage1_layout.py predict \
        --image data/raw/<file>.jpg \
        --weights checkpoints/yolo_det.pt \
        --out outputs/<file>.det.json

    # 3) Predict + crop each detected region into class-named subfolders
    python src/stage1_layout.py crop \
        --image data/raw/<file>.jpg \
        --weights checkpoints/yolo_det.pt \
        --out-dir outputs/crops/<file>

Output JSON schema (predict)
----------------------------
See PROJECT_HANDOFF.md §5.1::

    {
      "drawing_id":  "<stem>",
      "image_path":  "<abs path>",
      "image_size":  [W, H],
      "regions": [
        {"class": "View",       "bbox": [x1,y1,x2,y2], "conf": 0.97},
        {"class": "TitleBlock", "bbox": [...],          "conf": 0.99},
        {"class": "Notes",      "bbox": [...],          "conf": 0.98},
        {"class": "Isometric",  "bbox": [...],          "conf": 0.95},   # D-028
        {"class": "PMI",        "bbox": [...],          "conf": 0.96}    # D-028
      ]
    }

Class scheme (D-028 / D-029): Roboflow data.yaml uses
``[Isometric, PMI, Table, Text, View]``. Internal JSON uses semantic names —
``Table → TitleBlock``, ``Text → Notes`` (mapping in ``_result_to_schema``).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_CFG = PROJECT_ROOT / "configs" / "yolo_det.yaml"
DEFAULT_BASE_MODEL = "yolo11m.pt"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CKPT = CKPT_DIR / "yolo_det.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_RUN_DIR = CKPT_DIR / "yolo_det_runs"

# ---------------------------------------------------------------------------
# Class scheme (D-028, D-029)
# ---------------------------------------------------------------------------
# Roboflow data.yaml uses 5 classes (CLASS_NAMES_RF). Internal code uses
# semantic names (CLASS_NAMES) — TitleBlock / Notes preserved for clarity and
# to keep Stage 3-A Donut prompt tokens (`<s_titleblock>`, `<s_notes>`) stable.
#
# Mapping is applied ONCE in ``_result_to_schema`` so that all downstream
# modules (pipeline, prepare_vlm_dataset, stage3_alphabetical) continue to
# match against the internal names.

# Roboflow data.yaml order (must match configs/yolo_det.yaml)
CLASS_NAMES_RF: List[str] = ["Isometric", "PMI", "Table", "Text", "View"]

# Roboflow → internal mapping (D-029)
ROBOFLOW_TO_INTERNAL: Dict[str, str] = {
    "Isometric": "Isometric",   # NEW (D-028) — 3D isometric view
    "PMI":       "PMI",         # NEW (D-028) — Stage 2 OBB target region
    "Table":     "TitleBlock",  # rename (D-029) — semantic preservation
    "Text":      "Notes",       # rename (D-029) — semantic preservation
    "View":      "View",        # unchanged
}

# Internal canonical names (output JSON uses these)
CLASS_NAMES: List[str] = [ROBOFLOW_TO_INTERNAL[n] for n in CLASS_NAMES_RF]
# = ["Isometric", "PMI", "TitleBlock", "Notes", "View"]


def to_internal_class(rf_name: str) -> str:
    """Map a Roboflow class name to the internal canonical name (D-029).

    Unknown names are passed through unchanged (graceful fallback).
    """
    return ROBOFLOW_TO_INTERNAL.get(rf_name, rf_name)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage1_layout")


# ---------------------------------------------------------------------------
# Unicode-safe IO helpers (Windows + KO/JP/RU filenames)
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


def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    try:
        ext = path.suffix if path.suffix else ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception as e:  # noqa: BLE001
        log.error("imwrite failed for %s: %s", path, e)
        return False


def clamp_bbox(x1, y1, x2, y2, w, h):
    """Clamp BBox coordinates to image bounds and enforce x2>x1, y2>y1."""
    x1 = max(0, min(int(round(float(x1))), w - 1))
    y1 = max(0, min(int(round(float(y1))), h - 1))
    x2 = max(0, min(int(round(float(x2))), w))
    y2 = max(0, min(int(round(float(y2))), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    """Fine-tune YOLOv11-det on the layout dataset.

    Augmentation policy (HANDOFF §4.1): HSV/scale/translate ON, **flip OFF**.
    """
    data_cfg = Path(args.data).resolve()
    if not data_cfg.exists():
        log.error("Data config not found: %s", data_cfg)
        return 2

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model: %s", args.model)
    model = YOLO(args.model)

    log.info(
        "Training (data=%s, epochs=%d, imgsz=%d, batch=%d, device=%s)",
        data_cfg, args.epochs, args.imgsz, args.batch, args.device or "auto",
    )

    results = model.train(
        data=str(data_cfg),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(DEFAULT_RUN_DIR),
        name=args.name,
        exist_ok=True,
        # ---- Augmentation (per HANDOFF §4.1) ----
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        fliplr=0.0,   # ★ flip OFF (drawing orientation matters)
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        # ---- Misc ----
        save=True,
        plots=True,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", DEFAULT_RUN_DIR / args.name))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, DEFAULT_CKPT)
        log.info("Best weights copied to: %s", DEFAULT_CKPT)
    else:
        log.warning("best.pt not found at %s", best)
    log.info("Training run dir: %s", save_dir)
    return 0


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
def _result_to_schema(result, image_path: Path, conf_thr: float) -> Dict[str, Any]:
    """Convert a single ultralytics ``Results`` object to HANDOFF §5.1 dict."""
    img_h, img_w = result.orig_shape  # (h, w)
    schema: Dict[str, Any] = {
        "drawing_id": image_path.stem,
        "image_path": str(image_path),
        "image_size": [int(img_w), int(img_h)],
        "regions": [],
    }
    if result.boxes is None or len(result.boxes) == 0:
        return schema

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

    # name_map comes from the YOLO model (= Roboflow data.yaml order: CLASS_NAMES_RF)
    name_map = result.names if isinstance(result.names, dict) else \
        {i: n for i, n in enumerate(CLASS_NAMES_RF)}

    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        if p < conf_thr:
            continue
        ci = int(c)
        rf_name = name_map.get(ci, str(ci))
        # D-029: map Roboflow → internal canonical name once, here.
        name = to_internal_class(rf_name)
        x1c, y1c, x2c, y2c = clamp_bbox(x1, y1, x2, y2, img_w, img_h)
        schema["regions"].append({
            "class": name,
            "bbox": [x1c, y1c, x2c, y2c],
            "conf": float(round(float(p), 4)),
        })
    return schema


def predict_one(image_path: Path,
                weights: Path = DEFAULT_CKPT,
                conf_thr: float = 0.25,
                imgsz: int = 1280,
                device: Optional[str] = None) -> Dict[str, Any]:
    """Run YOLOv11-det on a single image and return the §5.1 schema dict.

    Importable by ``src/pipeline.py``.
    """
    weights = Path(weights)
    image_path = Path(image_path)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(weights))
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf_thr,
        device=device,
        verbose=False,
    )
    if not results:
        return {
            "drawing_id": image_path.stem,
            "image_path": str(image_path),
            "image_size": [0, 0],
            "regions": [],
        }
    return _result_to_schema(results[0], image_path, conf_thr)


def cmd_predict(args: argparse.Namespace) -> int:
    rec = predict_one(
        image_path=Path(args.image),
        weights=Path(args.weights),
        conf_thr=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    out = Path(args.out) if args.out else (
        DEFAULT_OUTPUT_DIR / f"{Path(args.image).stem}.det.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log.info("Detected %d regions. JSON: %s", len(rec["regions"]), out)
    for r in rec["regions"]:
        log.info("  - %-10s conf=%.3f bbox=%s", r["class"], r["conf"], r["bbox"])
    return 0


# ---------------------------------------------------------------------------
# CROP
# ---------------------------------------------------------------------------
def crop_regions(image_path: Path,
                 regions: List[Dict[str, Any]],
                 out_dir: Path,
                 padding: int = 0) -> List[Dict[str, Any]]:
    """Crop each region from the image and save to ``out_dir/<class>/<stem>__<class>_<idx>.jpg``.

    Returns a list of records: ``{class, bbox, conf, path}``.
    Importable by ``src/pipeline.py`` and ``src/prepare_vlm_dataset.py``.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    img = imread_unicode(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)

    counters: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    for r in regions:
        cls = r["class"]
        x1, y1, x2, y2 = r["bbox"]
        if padding:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            log.warning("Empty crop skipped: cls=%s bbox=[%d,%d,%d,%d]",
                        cls, x1, y1, x2, y2)
            continue
        idx = counters.get(cls, 0)
        counters[cls] = idx + 1
        save_path = out_dir / cls / f"{image_path.stem}__{cls}_{idx:02d}.jpg"
        if imwrite_unicode(save_path, crop):
            records.append({
                "class": cls,
                "bbox": [x1, y1, x2, y2],
                "conf": r.get("conf"),
                "path": str(save_path),
            })
    return records


def cmd_crop(args: argparse.Namespace) -> int:
    rec = predict_one(
        image_path=Path(args.image),
        weights=Path(args.weights),
        conf_thr=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        DEFAULT_OUTPUT_DIR / "crops" / Path(args.image).stem
    )
    crops = crop_regions(
        image_path=Path(args.image),
        regions=rec["regions"],
        out_dir=out_dir,
        padding=args.padding,
    )
    rec["crops"] = crops
    manifest = out_dir / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log.info("Saved %d crops under %s", len(crops), out_dir)
    log.info("Manifest: %s", manifest)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 — YOLOv11-det (Layout Segmentation)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train -----------------------------------------------------------
    pt = sub.add_parser("train", help="Fine-tune yolo11m on the layout dataset.")
    pt.add_argument("--data", type=str, default=str(DEFAULT_DATA_CFG),
                    help="Ultralytics data YAML (default: configs/yolo_det.yaml)")
    pt.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                    help="Base model weights (default: yolo11m.pt)")
    pt.add_argument("--epochs", type=int, default=100)
    pt.add_argument("--imgsz", type=int, default=1280)
    pt.add_argument("--batch", type=int, default=8,
                    help="Batch size (RTX 5080 16GB default: 8)")
    pt.add_argument("--patience", type=int, default=30)
    pt.add_argument("--device", type=str, default=None,
                    help='e.g. "0", "cpu", or omit for auto')
    pt.add_argument("--name", type=str, default="yolo_det",
                    help="Run name under checkpoints/yolo_det_runs/")
    pt.set_defaults(func=train)

    # ---- predict ---------------------------------------------------------
    pp = sub.add_parser("predict", help="Predict regions for a single image.")
    pp.add_argument("--image", type=str, required=True)
    pp.add_argument("--weights", type=str, default=str(DEFAULT_CKPT))
    pp.add_argument("--conf", type=float, default=0.25)
    pp.add_argument("--imgsz", type=int, default=1280)
    pp.add_argument("--device", type=str, default=None)
    pp.add_argument("--out", type=str, default=None,
                    help="Output JSON path (default: outputs/<stem>.det.json)")
    pp.set_defaults(func=cmd_predict)

    # ---- crop ------------------------------------------------------------
    pc = sub.add_parser("crop", help="Predict, then crop each region into class subfolders.")
    pc.add_argument("--image", type=str, required=True)
    pc.add_argument("--weights", type=str, default=str(DEFAULT_CKPT))
    pc.add_argument("--conf", type=float, default=0.25)
    pc.add_argument("--imgsz", type=int, default=1280)
    pc.add_argument("--device", type=str, default=None)
    pc.add_argument("--out-dir", type=str, default=None,
                    help="Output dir (default: outputs/crops/<stem>/)")
    pc.add_argument("--padding", type=int, default=0,
                    help="Pixels of padding around each crop (default: 0)")
    pc.set_defaults(func=cmd_crop)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
