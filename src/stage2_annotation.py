"""
src/stage2_annotation.py

Stage 2 — YOLOv11-obb (Annotation Localization)

Classes
-------
Measure / GDT / Roughness  (oriented bounding boxes)

Language note
-------------
This stage is **language-agnostic by design** (HANDOFF §11 D-009).
Engineering drawings in this project may be in KO / EN / JP / RU, but
YOLOv11-obb learns visual shape patterns (rotated rectangles, frame layouts,
roughness symbols), not text content. Language-aware processing only
matters at Stage 3 (Donut Alphabetical / Numerical VLMs).

Input
-----
Typically a **View crop** produced by ``src/stage1_layout.py`` ``crop``.
Running this on a full drawing also works but adds noise from TitleBlock /
Notes regions (which Stage 1 already isolates).

CLI
---
::

    # 1) Fine-tune YOLOv11m-obb on the annotation dataset
    python src/stage2_annotation.py train \
        --data configs/yolo_obb.yaml \
        --model yolo11m-obb.pt \
        --epochs 150 --imgsz 1024 --batch 8

    # 2) Predict OBBs for one View image (writes JSON, schema = HANDOFF §5.2)
    python src/stage2_annotation.py predict \
        --image outputs/crops/<drawing>/View/<file>.jpg \
        --weights checkpoints/yolo_obb.pt \
        --out outputs/<file>.obb.json

    # 3) Predict + crop each detected annotation as a de-rotated patch
    python src/stage2_annotation.py crop \
        --image outputs/crops/<drawing>/View/<file>.jpg \
        --weights checkpoints/yolo_obb.pt \
        --out-dir outputs/crops/<drawing>/annotations

Output JSON schema (predict)
----------------------------
See PROJECT_HANDOFF.md §5.2::

    {
      "view_id":      "<image stem>",
      "image_path":   "<abs path>",
      "image_size":   [W, H],
      "parent_bbox":  null,                    # set by pipeline.py if known
      "annotations": [
        {
          "class": "Measure",
          "obb":   [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],   # ordered TL,TR,BR,BL
          "angle": 12.5,                                # degrees, [-90, 90)
          "conf":  0.93
        }
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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
DEFAULT_DATA_CFG = PROJECT_ROOT / "configs" / "yolo_obb.yaml"
DEFAULT_BASE_MODEL = "yolo11m-obb.pt"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CKPT = CKPT_DIR / "yolo_obb.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_RUN_DIR = CKPT_DIR / "yolo_obb_runs"

# Class order MUST match configs/yolo_obb.yaml
CLASS_NAMES: List[str] = ["Measure", "GDT", "Roughness"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage2_annotation")


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


# ---------------------------------------------------------------------------
# OBB geometry helpers
# ---------------------------------------------------------------------------
def order_obb_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 OBB points as TL, TR, BR, BL by angle around centroid.

    Ultralytics already returns them ordered, but we re-order defensively so
    downstream perspective-warp produces an upright crop.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    # angle from centroid; upper-left has smallest angle in image coords
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    # rotate so the top-left (smallest x+y) starts the sequence
    s = pts.sum(axis=1)
    start = int(np.argmin(s))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def obb_angle_deg(pts: np.ndarray) -> float:
    """Compute OBB long-edge angle in degrees, normalized to [-90, 90)."""
    pts = order_obb_points(pts)
    edge = pts[1] - pts[0]            # TL -> TR
    a = math.degrees(math.atan2(edge[1], edge[0]))
    # normalize to [-90, 90)
    while a >= 90.0:
        a -= 180.0
    while a < -90.0:
        a += 180.0
    return float(round(a, 2))


def warp_obb_crop(img: np.ndarray, obb_pts: np.ndarray) -> np.ndarray:
    """Perspective-warp an OBB to an upright rectangle for downstream OCR/VLM."""
    pts = order_obb_points(obb_pts)
    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[3] - pts[0])
    h2 = np.linalg.norm(pts[2] - pts[1])
    width = max(1, int(round((float(w1) + float(w2)) / 2.0)))
    height = max(1, int(round((float(h1) + float(h2)) / 2.0)))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    """Fine-tune YOLOv11-obb on the annotation dataset.

    Augmentation policy (HANDOFF §4.2): rotation ON, flip OFF.
    Class imbalance: Roughness is rare; rely on dataset-level oversampling.
    """
    data_cfg = Path(args.data).resolve()
    if not data_cfg.exists():
        log.error("Data config not found: %s", data_cfg)
        return 2

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Resume 분기 (★ 2026-05-03 추가) ----
    resume_path: Optional[Path] = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            log.error("--resume-from 체크포인트 미발견: %s", resume_path)
            return 3
    elif args.resume:
        last_pt = DEFAULT_RUN_DIR / args.name / "weights" / "last.pt"
        if last_pt.exists():
            resume_path = last_pt
        else:
            log.warning(
                "--resume 지정됐지만 last.pt 미발견: %s\n"
                "  → fresh 학습으로 진행 (의도가 아니면 --name 확인 필요)",
                last_pt,
            )

    if resume_path is not None:
        log.info("★ Resume from checkpoint: %s", resume_path)
        model = YOLO(str(resume_path))
        train_kwargs = {"resume": True}
    else:
        log.info("Loading base model: %s", args.model)
        model = YOLO(args.model)
        train_kwargs = {}

    log.info(
        "Training OBB (data=%s, epochs=%d, imgsz=%d, batch=%d, device=%s, "
        "save_period=%d%s)",
        data_cfg, args.epochs, args.imgsz, args.batch, args.device or "auto",
        args.save_period,
        " [RESUME]" if resume_path is not None else "",
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
        # ---- Augmentation (Option C 강화, 2026-05-03) ----
        # 사유: GDT 88 / Roughness 106 데이터 부족 보완 (D-023 critical 임계값 위해)
        # - degrees 15→30: 회전 텍스트/심볼 다양성 (D-036 보완)
        # - scale 0.3→0.5: PMI 다양한 크기 대응
        # - mixup 0→0.15: ★ 부족 클래스 보완 효과 강함
        # - copy_paste 0.3: ★ Roughness/GDT 인스턴스 증강
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=30.0,    # ★ Option C: 회전 강화 (15→30)
        translate=0.1,
        scale=0.5,       # ★ Option C: 크기 변동 ↑ (0.3→0.5)
        shear=0.0,
        fliplr=0.0,      # D-001: flip OFF (도면 비대칭)
        flipud=0.0,      # D-001
        mosaic=1.0,
        mixup=0.15,      # ★ Option C: 부족 클래스 보완 (0→0.15)
        copy_paste=0.3,  # ★ Option C: Roughness/GDT 인스턴스 증강 (신규)
        # ---- Misc ----
        save=True,
        save_period=args.save_period,  # ★ 2026-05-03: N epoch마다 체크포인트 저장
        plots=True,
        verbose=True,
        **train_kwargs,                # ★ 2026-05-03: resume=True (resume 시만)
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
    """Convert a single ultralytics OBB ``Results`` to HANDOFF §5.2 dict."""
    img_h, img_w = result.orig_shape  # (h, w)
    schema: Dict[str, Any] = {
        "view_id": image_path.stem,
        "image_path": str(image_path),
        "image_size": [int(img_w), int(img_h)],
        "parent_bbox": None,   # filled by pipeline.py if View parent known
        "annotations": [],
    }
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return schema

    # Ultralytics OBB API:
    #   obb.xyxyxyxy : (N, 4, 2) float
    #   obb.cls      : (N,)
    #   obb.conf     : (N,)
    #   obb.xywhr    : (N, 5) -> cx, cy, w, h, rot(rad)  [optional but useful]
    pts_all = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, "cpu") else np.asarray(obb.xyxyxyxy)
    cls = obb.cls.cpu().numpy() if hasattr(obb.cls, "cpu") else np.asarray(obb.cls)
    conf = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else np.asarray(obb.conf)

    name_map = result.names if isinstance(result.names, dict) else \
        {i: n for i, n in enumerate(CLASS_NAMES)}

    for pts, c, p in zip(pts_all, cls, conf):
        if p < conf_thr:
            continue
        ci = int(c)
        name = name_map.get(ci, str(ci))
        ordered = order_obb_points(pts)
        # clamp to image bounds (defensive)
        ordered[:, 0] = np.clip(ordered[:, 0], 0, img_w - 1)
        ordered[:, 1] = np.clip(ordered[:, 1], 0, img_h - 1)
        angle = obb_angle_deg(ordered)
        schema["annotations"].append({
            "class": name,
            "obb": ordered.round(2).tolist(),
            "angle": angle,
            "conf": float(round(float(p), 4)),
        })
    return schema


def predict_one(image_path: Path,
                weights: Path = DEFAULT_CKPT,
                conf_thr: float = 0.25,
                imgsz: int = 1024,
                device: Optional[str] = None,
                parent_bbox: Optional[List[int]] = None) -> Dict[str, Any]:
    """Run YOLOv11-obb on a single image and return the §5.2 schema dict.

    Importable by ``src/pipeline.py``.

    Parameters
    ----------
    parent_bbox : Optional[List[int]]
        If this image is a View crop, supply the parent's BBox in the
        original drawing coordinate system. ``pipeline.py`` will use it to
        translate OBB points back to global coordinates.
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
            "view_id": image_path.stem,
            "image_path": str(image_path),
            "image_size": [0, 0],
            "parent_bbox": parent_bbox,
            "annotations": [],
        }
    rec = _result_to_schema(results[0], image_path, conf_thr)
    rec["parent_bbox"] = parent_bbox
    return rec


def cmd_predict(args: argparse.Namespace) -> int:
    rec = predict_one(
        image_path=Path(args.image),
        weights=Path(args.weights),
        conf_thr=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    out = Path(args.out) if args.out else (
        DEFAULT_OUTPUT_DIR / f"{Path(args.image).stem}.obb.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log.info("Detected %d annotations. JSON: %s",
             len(rec["annotations"]), out)
    by_cls: Dict[str, int] = {}
    for a in rec["annotations"]:
        by_cls[a["class"]] = by_cls.get(a["class"], 0) + 1
    for k, v in by_cls.items():
        log.info("  - %-9s %d", k, v)
    return 0


# ---------------------------------------------------------------------------
# CROP (de-rotated, perspective-warped patches for Stage 3 Numerical VLM)
# ---------------------------------------------------------------------------
def crop_obb_regions(image_path: Path,
                     annotations: List[Dict[str, Any]],
                     out_dir: Path) -> List[Dict[str, Any]]:
    """Warp each OBB to an upright rectangle and save under
    ``out_dir/<class>/<stem>__<class>_<idx>.jpg``.

    De-rotation matters for Donut VLM accuracy: the model is trained on
    upright text. Rotated callouts (e.g. vertical Measures) would otherwise
    degrade Stage 3-N performance.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    img = imread_unicode(image_path)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    counters: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    for a in annotations:
        cls = a["class"]
        pts = np.array(a["obb"], dtype=np.float32).reshape(4, 2)
        try:
            warped = warp_obb_crop(img, pts)
        except cv2.error as e:  # noqa: PERF203
            log.warning("Warp failed (%s) for cls=%s; skipping", e, cls)
            continue
        if warped.size == 0:
            log.warning("Empty warp skipped: cls=%s", cls)
            continue
        idx = counters.get(cls, 0)
        counters[cls] = idx + 1
        save_path = out_dir / cls / f"{image_path.stem}__{cls}_{idx:02d}.jpg"
        if imwrite_unicode(save_path, warped):
            records.append({
                "class": cls,
                "obb": a["obb"],
                "angle": a.get("angle"),
                "conf": a.get("conf"),
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
        DEFAULT_OUTPUT_DIR / "crops" / Path(args.image).stem / "annotations"
    )
    crops = crop_obb_regions(
        image_path=Path(args.image),
        annotations=rec["annotations"],
        out_dir=out_dir,
    )
    rec["crops"] = crops
    manifest = out_dir / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    log.info("Saved %d OBB crops under %s", len(crops), out_dir)
    log.info("Manifest: %s", manifest)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 — YOLOv11-obb (Annotation Localization). "
                    "Language-agnostic visual detector for Measure / GDT / Roughness."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- train -----------------------------------------------------------
    pt = sub.add_parser("train", help="Fine-tune yolo11m-obb on the annotation dataset.")
    pt.add_argument("--data", type=str, default=str(DEFAULT_DATA_CFG),
                    help="Ultralytics data YAML (default: configs/yolo_obb.yaml)")
    pt.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL,
                    help="Base model weights (default: yolo11m-obb.pt)")
    pt.add_argument("--epochs", type=int, default=150,
                    help="OBB tends to converge slower; default 150")
    pt.add_argument("--imgsz", type=int, default=1024,
                    help="1024 default; raise to 1280 for tiny GD&T frames "
                         "if VRAM permits")
    pt.add_argument("--batch", type=int, default=8,
                    help="Batch size (RTX 5080 16GB default: 8)")
    pt.add_argument("--patience", type=int, default=40)
    pt.add_argument("--device", type=str, default=None,
                    help='e.g. "0", "cpu", or omit for auto')
    pt.add_argument("--name", type=str, default="yolo_obb",
                    help="Run name under checkpoints/yolo_obb_runs/")
    # ---- Checkpoint / Resume (★ 2026-05-03 추가) ----
    pt.add_argument("--save-period", type=int, default=20,
                    help="N epoch마다 체크포인트 저장 (default: 20). "
                         "PC 중단 시 재개 가능. -1=비활성 (last+best만)")
    pt.add_argument("--resume", action="store_true",
                    help="중단된 학습 재개 — last.pt 자동 감지 "
                         "(checkpoints/yolo_obb_runs/<name>/weights/last.pt)")
    pt.add_argument("--resume-from", type=str, default=None,
                    help="특정 체크포인트에서 재개 (path 명시). "
                         "지정 시 --resume 옵션 무시.")
    pt.set_defaults(func=train)

    # ---- predict ---------------------------------------------------------
    pp = sub.add_parser("predict", help="Predict OBBs for a single image.")
    pp.add_argument("--image", type=str, required=True)
    pp.add_argument("--weights", type=str, default=str(DEFAULT_CKPT))
    pp.add_argument("--conf", type=float, default=0.25)
    pp.add_argument("--imgsz", type=int, default=1024)
    pp.add_argument("--device", type=str, default=None)
    pp.add_argument("--out", type=str, default=None,
                    help="Output JSON path (default: outputs/<stem>.obb.json)")
    pp.set_defaults(func=cmd_predict)

    # ---- crop ------------------------------------------------------------
    pc = sub.add_parser(
        "crop",
        help="Predict, then perspective-warp each OBB to an upright crop "
             "(Stage 3-N input)."
    )
    pc.add_argument("--image", type=str, required=True)
    pc.add_argument("--weights", type=str, default=str(DEFAULT_CKPT))
    pc.add_argument("--conf", type=float, default=0.25)
    pc.add_argument("--imgsz", type=int, default=1024)
    pc.add_argument("--device", type=str, default=None)
    pc.add_argument("--out-dir", type=str, default=None,
                    help="Output dir (default: outputs/crops/<stem>/annotations/)")
    pc.set_defaults(func=cmd_crop)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
