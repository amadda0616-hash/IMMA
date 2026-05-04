"""
src/ensemble_predict.py

Stage 2 — 5-Fold OBB Ensemble (D-023 회수율 강화).

Background
----------
- 단일 Fold 2 모델은 mAP@0.5 = 0.978 로 우수하지만,
  Measure missing_rate = 0.101 (D-023 임계 0.08 초과 → FAIL).
- 5개 fold 의 OBB 검출 결과를 concatenate 후 클래스별 NMS 로 합쳐
  recall 을 끌어올리는 ensemble 접근.

Pipeline
--------
1. ``load_fold_models()`` : checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_{0..4}/weights/best.pt 5개 로드
2. ``ensemble_predict()`` : 한 이미지 입력 → 5 모델 추론 → 클래스별 NMS rotated
3. evaluate mode  : val.txt 기반 GT 매칭 → per-class TP/FP/FN → P/R/missing_rate
4. predict mode   : 단일 이미지 → JSON 출력

CLI
---
::

    # D-023 재평가 (Fold 2 val set, 가장 어려운 split)
    python src/ensemble_predict.py evaluate \\
        --val-txt data/annotation_kfold/fold_2/val.txt \\
        --conf 0.25 --iou-nms 0.5 \\
        --output outputs/v3b_ensemble_eval.json

    # 단일 이미지 추론
    python src/ensemble_predict.py predict \\
        --image path/to/drawing.png \\
        --output predictions.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Measure", "GDT", "Roughness"]

D023_THRESHOLDS = {
    # missing_rate < threshold == PASS
    "Measure":   {"missing_rate": 0.08, "severity": "critical"},
    "GDT":       {"missing_rate": 0.05, "severity": "critical"},
    "Roughness": {"missing_rate": 0.30, "severity": "warning"},
}

DEFAULT_CKPT_ROOT = _PROJECT_ROOT / "checkpoints" / "yolo_obb_runs"
DEFAULT_FOLD_PATTERN = "yolo_obb_v3_kfold_{i}"
DEFAULT_N_FOLDS = 5

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    log = logging.getLogger("ensemble_predict")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log


log = setup_logger()


# ---------------------------------------------------------------------------
# Polygon IoU (shapely)
# ---------------------------------------------------------------------------
def polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """OBB polygon IoU using shapely. p1, p2: (4, 2) arrays."""
    try:
        from shapely.geometry import Polygon  # noqa: PLC0415
    except ImportError:
        log.warning("shapely missing — falling back to bbox IoU")
        return _bbox_iou_from_polys(p1, p2)

    poly1 = Polygon(p1).buffer(0)
    poly2 = Polygon(p2).buffer(0)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_iou_from_polys(p1: np.ndarray, p2: np.ndarray) -> float:
    a1 = np.array([p1[:, 0].min(), p1[:, 1].min(), p1[:, 0].max(), p1[:, 1].max()])
    a2 = np.array([p2[:, 0].min(), p2[:, 1].min(), p2[:, 0].max(), p2[:, 1].max()])
    xa = max(a1[0], a2[0]); ya = max(a1[1], a2[1])
    xb = min(a1[2], a2[2]); yb = min(a1[3], a2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = (a1[2] - a1[0]) * (a1[3] - a1[1])
    area2 = (a2[2] - a2[0]) * (a2[3] - a2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Rotated NMS resolver — handle ultralytics version drift
# ---------------------------------------------------------------------------
def _resolve_nms_rotated():
    """Try multiple ultralytics import paths for ``nms_rotated``.

    Returns a callable ``nms_rotated(boxes_xywhr, scores, iou_thr) -> keep_idx``.
    Falls back to a manual shapely-based greedy NMS when not found.
    """
    candidates = [
        ("ultralytics.utils.ops",    "nms_rotated"),
        ("ultralytics.utils.metrics","nms_rotated"),
        ("ultralytics.utils.tal",    "nms_rotated"),
        ("ultralytics.utils",        "nms_rotated"),
    ]
    for mod_path, attr in candidates:
        try:
            mod = __import__(mod_path, fromlist=[attr])
            fn = getattr(mod, attr, None)
            if callable(fn):
                log.info("Using rotated NMS from %s.%s", mod_path, attr)
                return fn
        except ImportError:
            continue
    log.info("ultralytics nms_rotated not found — using manual shapely-based greedy NMS")
    return manual_nms_rotated


def xywhr_to_corners(xywhr) -> np.ndarray:
    """Convert (cx, cy, w, h, r_radians) to (4, 2) corner points.

    Convention matches ultralytics OBB ``.xyxyxyxy`` ordering
    (clockwise starting from top-left of un-rotated box).
    """
    if hasattr(xywhr, "cpu"):
        xywhr = xywhr.cpu().numpy()
    xywhr = np.asarray(xywhr, dtype=float).reshape(-1)
    cx, cy, w, h, r = xywhr
    cos_r = float(np.cos(r))
    sin_r = float(np.sin(r))
    hw, hh = w / 2.0, h / 2.0
    # Local frame, clockwise from top-left
    local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=float)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=float)
    return local @ R.T + np.array([cx, cy], dtype=float)


def manual_nms_rotated(
    boxes_xywhr,        # (N, 5) tensor or array
    scores,             # (N,)
    iou_thr: float,
) -> torch.Tensor:
    """Greedy rotated NMS using shapely polygon IoU.

    Returns
    -------
    torch.Tensor (long)
        Indices of kept boxes, in score-descending order.
    """
    if hasattr(boxes_xywhr, "cpu"):
        boxes_np = boxes_xywhr.cpu().numpy()
    else:
        boxes_np = np.asarray(boxes_xywhr)
    if hasattr(scores, "cpu"):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    n = len(boxes_np)
    if n == 0:
        return torch.zeros(0, dtype=torch.long)

    # Pre-compute polygons
    polys = [xywhr_to_corners(b) for b in boxes_np]

    order = np.argsort(-scores_np).tolist()
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        rest: List[int] = []
        for j in order:
            iou = polygon_iou(polys[i], polys[j])
            if iou < iou_thr:
                rest.append(j)
        order = rest
    return torch.tensor(keep, dtype=torch.long)


# Resolve once at import time (None until first call avoids penalty if module unused)
_NMS_ROTATED_FN = None


def _get_nms_rotated():
    global _NMS_ROTATED_FN
    if _NMS_ROTATED_FN is None:
        _NMS_ROTATED_FN = _resolve_nms_rotated()
    return _NMS_ROTATED_FN


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_fold_models(
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_pattern: str = DEFAULT_FOLD_PATTERN,
    device: Optional[str] = None,
) -> List[Any]:
    """Load 5 YOLO models from ``checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_{i}/weights/best.pt``."""
    from ultralytics import YOLO  # noqa: PLC0415

    models = []
    for i in range(n_folds):
        run_dir = ckpt_root / fold_pattern.format(i=i)
        weights_path = run_dir / "weights" / "best.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Fold {i} weights missing: {weights_path}")
        log.info("Loading fold %d: %s", i, weights_path)
        m = YOLO(str(weights_path))
        if device:
            m.to(device)
        models.append(m)
    log.info("Loaded %d fold models", len(models))
    return models


# ---------------------------------------------------------------------------
# Ensemble inference (per image)
# ---------------------------------------------------------------------------
def ensemble_predict(
    models: List[Any],
    image: Any,
    conf: float = 0.25,
    iou_nms: float = 0.5,
    imgsz: int = 1024,
    device: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Run 5 models on one image, concatenate, class-wise rotated NMS.

    Parameters
    ----------
    models : list of YOLO
        5 fold models from ``load_fold_models``.
    image : str | Path | np.ndarray | PIL.Image
        Anything ultralytics ``model.predict`` accepts.
    conf : float
        Per-model confidence threshold (default 0.25).
    iou_nms : float
        Cross-model NMS IoU threshold (default 0.5).
    imgsz : int
        Inference image size.
    device : str, optional
        Device override (e.g., "cuda:0").

    Returns
    -------
    dict or None
        {
            "xyxyxyxy": (K, 4, 2) array,    # absolute pixel coords (4-corner)
            "xywhr":    (K, 5) array,       # rotated box format
            "conf":     (K,) array,
            "cls":      (K,) array (int),
            "img_w": int, "img_h": int,
        }
        ``None`` if no detections from any model.
    """
    nms_rotated = _get_nms_rotated()

    boxes_all: List[torch.Tensor] = []
    pts_all: List[torch.Tensor] = []
    scores_all: List[torch.Tensor] = []
    classes_all: List[torch.Tensor] = []
    img_w = img_h = None

    for m in models:
        kw = dict(conf=conf, imgsz=imgsz, verbose=False)
        if device:
            kw["device"] = device
        r = m.predict(image, **kw)[0]

        # Capture image size from first result
        if img_w is None and getattr(r, "orig_shape", None):
            img_h, img_w = r.orig_shape  # (h, w)

        obb = getattr(r, "obb", None)
        if obb is None or len(obb) == 0:
            continue

        boxes_all.append(obb.xywhr)         # (N, 5)  absolute (px)
        pts_all.append(obb.xyxyxyxy)        # (N, 4, 2) absolute (px)
        scores_all.append(obb.conf)
        classes_all.append(obb.cls)

    if not boxes_all:
        return None

    boxes = torch.cat(boxes_all, dim=0)
    pts = torch.cat(pts_all, dim=0)
    scores = torch.cat(scores_all, dim=0)
    classes = torch.cat(classes_all, dim=0)

    # Class-wise rotated NMS
    keep_idx_list: List[torch.Tensor] = []
    for c in classes.unique():
        mask = classes == c
        idx_all = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx_all.numel() == 0:
            continue
        keep_local = nms_rotated(boxes[mask], scores[mask], iou_nms)
        keep_idx_list.append(idx_all[keep_local])

    if not keep_idx_list:
        return None

    keep = torch.cat(keep_idx_list, dim=0)

    return {
        "xyxyxyxy": pts[keep].cpu().numpy().astype(float),
        "xywhr":    boxes[keep].cpu().numpy().astype(float),
        "conf":     scores[keep].cpu().numpy().astype(float),
        "cls":      classes[keep].cpu().numpy().astype(int),
        "img_w":    int(img_w) if img_w else 0,
        "img_h":    int(img_h) if img_h else 0,
    }


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------
def img_path_to_label_path(img_path: Path) -> Path:
    """YOLO convention: replace /images/ with /labels/, swap extension."""
    s = str(img_path)
    s = s.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    return Path(s).with_suffix(".txt")


def load_gt_obbs(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    """Parse YOLO OBB .txt → list of (cid, (4,2) normalized 0-1 points)."""
    if not label_path.exists():
        return []
    out: List[Tuple[int, np.ndarray]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) != 9:
                continue
            try:
                cid = int(parts[0])
                pts = np.array([float(x) for x in parts[1:]], dtype=float).reshape(4, 2)
            except ValueError:
                continue
            out.append((cid, pts))
    return out


def _wsl_to_native(p: str) -> str:
    """Convert /mnt/c/... path to native if running on Windows."""
    if sys.platform.startswith("win") and p.startswith("/mnt/"):
        # /mnt/c/Users/... → C:/Users/...
        parts = p.split("/", 3)
        if len(parts) >= 4 and len(parts[2]) == 1:
            drive = parts[2].upper() + ":"
            rest = parts[3]
            return f"{drive}/{rest}"
    return p


def load_val_image_list(val_txt: Path) -> List[Path]:
    paths: List[Path] = []
    with open(val_txt, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            s = _wsl_to_native(s)
            paths.append(Path(s))
    return paths


# ---------------------------------------------------------------------------
# GT vs Pred matching (greedy IoU, class-aware)
# ---------------------------------------------------------------------------
def match_gt_pred(
    gts: List[Tuple[int, np.ndarray]],          # normalized (0-1)
    pred_pts_norm: np.ndarray,                   # (K, 4, 2) normalized
    pred_classes: np.ndarray,                    # (K,)
    pred_scores: np.ndarray,                     # (K,)
    iou_thr: float = 0.5,
) -> Dict[int, Dict[str, int]]:
    """Greedy match. Class must agree.

    Returns per-class dict: {cid: {tp, fp, fn}}.
    """
    per_class: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    used_gt = set()

    # Sort preds by confidence desc
    order = np.argsort(-pred_scores)

    for k in order:
        cid_p = int(pred_classes[k])
        pts_p = pred_pts_norm[k]
        best_iou = 0.0
        best_idx = -1
        for gi, (cid_g, pts_g) in enumerate(gts):
            if gi in used_gt or cid_g != cid_p:
                continue
            iou = polygon_iou(pts_p, pts_g)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_thr and best_idx >= 0:
            used_gt.add(best_idx)
            per_class[cid_p]["tp"] += 1
        else:
            per_class[cid_p]["fp"] += 1

    for gi, (cid_g, _) in enumerate(gts):
        if gi not in used_gt:
            per_class[cid_g]["fn"] += 1

    return dict(per_class)


# ---------------------------------------------------------------------------
# evaluate mode
# ---------------------------------------------------------------------------
def evaluate_d023(
    val_txt: Path,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_pattern: str = DEFAULT_FOLD_PATTERN,
    conf: float = 0.25,
    iou_nms: float = 0.5,
    iou_match: float = 0.5,
    imgsz: int = 1024,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ensemble on val set, compute D-023 metrics."""
    image_paths = load_val_image_list(val_txt)
    log.info("Val set: %s (%d images)", val_txt, len(image_paths))

    models = load_fold_models(ckpt_root, n_folds, fold_pattern, device)

    totals: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    drawing_recalls: List[float] = []
    skipped = 0

    for i, img_path in enumerate(image_paths, 1):
        if not img_path.exists():
            log.warning("Missing image: %s", img_path)
            skipped += 1
            continue

        label_path = img_path_to_label_path(img_path)
        gts = load_gt_obbs(label_path)

        result = ensemble_predict(
            models, str(img_path),
            conf=conf, iou_nms=iou_nms,
            imgsz=imgsz, device=device,
        )

        if result is None:
            # No detections — all GTs become FN
            for cid_g, _ in gts:
                totals[cid_g]["fn"] += 1
            drawing_recalls.append(0.0 if gts else 1.0)
            if i % 20 == 0 or i == len(image_paths):
                log.info("[%d/%d] no detections", i, len(image_paths))
            continue

        # Normalize pred points to [0,1] using image size
        img_w = result["img_w"] or 1
        img_h = result["img_h"] or 1
        pts_norm = result["xyxyxyxy"].copy()
        pts_norm[..., 0] /= float(img_w)
        pts_norm[..., 1] /= float(img_h)

        per_class = match_gt_pred(
            gts, pts_norm, result["cls"], result["conf"], iou_thr=iou_match,
        )

        # Aggregate
        n_gt = len(gts)
        n_matched = 0
        for cid, st in per_class.items():
            totals[cid]["tp"] += st["tp"]
            totals[cid]["fp"] += st["fp"]
            totals[cid]["fn"] += st["fn"]
            n_matched += st["tp"]

        if n_gt > 0:
            drawing_recalls.append(n_matched / n_gt)
        else:
            drawing_recalls.append(1.0)

        if i % 20 == 0 or i == len(image_paths):
            log.info(
                "[%d/%d] %s — GT=%d Pred=%d",
                i, len(image_paths), img_path.name, n_gt, len(result["cls"]),
            )

    # ---- Final metrics ----
    per_class_results: Dict[str, Dict[str, Any]] = {}
    d023_pass = True
    d023_failures: List[str] = []

    for cid, name in enumerate(CLASS_NAMES):
        st = totals.get(cid, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        n_gt = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        missing = 1.0 - recall

        thr = D023_THRESHOLDS.get(name, {})
        miss_thr = thr.get("missing_rate", 1.0)
        passed = missing < miss_thr
        if not passed:
            d023_pass = False
            d023_failures.append(f"{name}: missing={missing:.4f} >= {miss_thr}")

        per_class_results[name] = {
            "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "missing_rate": round(missing, 4),
            "missing_threshold": miss_thr,
            "severity": thr.get("severity", "warning"),
            "pass": bool(passed),
        }

    drawing_recall_mean = float(np.mean(drawing_recalls)) if drawing_recalls else 0.0
    drawing_pass = drawing_recall_mean >= 0.85
    if not drawing_pass:
        d023_pass = False
        d023_failures.append(f"drawing_recall={drawing_recall_mean:.4f} < 0.85")

    summary = {
        "val_txt": str(val_txt),
        "n_images": len(image_paths),
        "n_skipped": skipped,
        "ensemble_n_folds": n_folds,
        "conf": conf,
        "iou_nms": iou_nms,
        "iou_match": iou_match,
        "imgsz": imgsz,
        "per_class": per_class_results,
        "drawing_recall_mean": round(drawing_recall_mean, 4),
        "drawing_recall_threshold": 0.85,
        "drawing_recall_pass": bool(drawing_pass),
        "d023_overall_pass": bool(d023_pass),
        "d023_failures": d023_failures,
    }
    return summary


# ---------------------------------------------------------------------------
# Pipeline schema adapter (HANDOFF §5.2 호환)
# ---------------------------------------------------------------------------
def predict_one_schema(
    models: List[Any],
    image_path: Path,
    conf: float = 0.25,
    iou_nms: float = 0.5,
    imgsz: int = 1024,
    device: Optional[str] = None,
    parent_bbox: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Pipeline-friendly ensemble inference returning HANDOFF §5.2 schema.

    Same shape as ``stage2_annotation.predict_one()`` so that
    ``src/pipeline.py`` can drop in this function transparently.

    Returns
    -------
    dict
        {
          "view_id": <stem>,
          "image_path": <str>,
          "image_size": [W, H],
          "parent_bbox": <list or None>,
          "annotations": [
              {"class": str, "obb": [[x,y]*4], "angle": float, "conf": float},
              ...
          ]
        }
    """
    # Reuse stage2 helpers for OBB ordering / angle (avoid drift).
    from src.stage2_annotation import order_obb_points, obb_angle_deg  # noqa: PLC0415

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = ensemble_predict(
        models, str(image_path),
        conf=conf, iou_nms=iou_nms,
        imgsz=imgsz, device=device,
    )

    if result is None:
        return {
            "view_id": image_path.stem,
            "image_path": str(image_path),
            "image_size": [0, 0],
            "parent_bbox": parent_bbox,
            "annotations": [],
        }

    img_w = int(result["img_w"])
    img_h = int(result["img_h"])

    annotations: List[Dict[str, Any]] = []
    for k in range(len(result["cls"])):
        cid = int(result["cls"][k])
        if 0 <= cid < len(CLASS_NAMES):
            name = CLASS_NAMES[cid]
        else:
            name = str(cid)
        pts = np.asarray(result["xyxyxyxy"][k], dtype=np.float32).reshape(4, 2)
        ordered = order_obb_points(pts)
        # clamp to image bounds (defensive — same as stage2_annotation)
        if img_w > 0 and img_h > 0:
            ordered[:, 0] = np.clip(ordered[:, 0], 0, img_w - 1)
            ordered[:, 1] = np.clip(ordered[:, 1], 0, img_h - 1)
        angle = obb_angle_deg(ordered)
        annotations.append({
            "class": name,
            "obb":   ordered.round(2).tolist(),
            "angle": angle,
            "conf":  float(round(float(result["conf"][k]), 4)),
        })

    return {
        "view_id":     image_path.stem,
        "image_path":  str(image_path),
        "image_size":  [img_w, img_h],
        "parent_bbox": parent_bbox,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# predict mode
# ---------------------------------------------------------------------------
def predict_single(
    image: Path,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    n_folds: int = DEFAULT_N_FOLDS,
    fold_pattern: str = DEFAULT_FOLD_PATTERN,
    conf: float = 0.25,
    iou_nms: float = 0.5,
    imgsz: int = 1024,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    models = load_fold_models(ckpt_root, n_folds, fold_pattern, device)
    result = ensemble_predict(
        models, str(image),
        conf=conf, iou_nms=iou_nms,
        imgsz=imgsz, device=device,
    )

    detections: List[Dict[str, Any]] = []
    if result is not None:
        for k in range(len(result["cls"])):
            cid = int(result["cls"][k])
            detections.append({
                "class_id": cid,
                "class_name": CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid),
                "confidence": round(float(result["conf"][k]), 4),
                "obb_xyxyxyxy": result["xyxyxyxy"][k].tolist(),
                "obb_xywhr":    result["xywhr"][k].tolist(),
            })

    return {
        "image":  str(image),
        "img_w":  result["img_w"] if result else 0,
        "img_h":  result["img_h"] if result else 0,
        "ensemble_n_folds": n_folds,
        "conf": conf,
        "iou_nms": iou_nms,
        "n_detections": len(detections),
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_evaluate_summary(s: Dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("  5-Fold Ensemble - Stage 2 OBB - D-023 Evaluation")
    print("=" * 72)
    print(f"  Val set       : {s['val_txt']}")
    print(f"  Images        : {s['n_images']}  (skipped: {s['n_skipped']})")
    print(f"  conf          : {s['conf']}")
    print(f"  iou_nms       : {s['iou_nms']}")
    print(f"  iou_match     : {s['iou_match']}")
    print()
    print(f"  {'Class':<10} {'P':>7} {'R':>7} {'miss':>7} {'thr':>6}  TP/FP/FN  GT  PASS")
    print("  " + "-" * 64)
    for name in CLASS_NAMES:
        c = s["per_class"][name]
        flag = "PASS" if c["pass"] else "FAIL"
        print(
            f"  {name:<10} {c['precision']:>7.4f} {c['recall']:>7.4f} "
            f"{c['missing_rate']:>7.4f} {c['missing_threshold']:>6.2f}  "
            f"{c['tp']}/{c['fp']}/{c['fn']}  {c['n_gt']}  {flag}"
        )
    print()
    print(
        f"  drawing_recall = {s['drawing_recall_mean']:.4f} "
        f"(threshold >= 0.85, {'PASS' if s['drawing_recall_pass'] else 'FAIL'})"
    )
    print()
    overall = "PASS" if s["d023_overall_pass"] else "FAIL"
    print(f"  D-023 overall : {overall}")
    if s["d023_failures"]:
        for f in s["d023_failures"]:
            print(f"    - {f}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 5-Fold OBB Ensemble - D-023 evaluator + predictor"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    common.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    common.add_argument("--fold-pattern", type=str, default=DEFAULT_FOLD_PATTERN)
    common.add_argument("--conf", type=float, default=0.25)
    common.add_argument("--iou-nms", type=float, default=0.5)
    common.add_argument("--imgsz", type=int, default=1024)
    common.add_argument("--device", type=str, default=None,
                        help="e.g., cuda:0  (default: ultralytics auto)")

    pe = sub.add_parser("evaluate", parents=[common],
                        help="D-023 ensemble evaluation on a val.txt list")
    pe.add_argument("--val-txt", type=Path, required=True)
    pe.add_argument("--iou-match", type=float, default=0.5,
                    help="IoU threshold for GT match (default 0.5)")
    pe.add_argument("--output", type=Path, default=None,
                    help="Write summary JSON to this path")

    pp = sub.add_parser("predict", parents=[common],
                        help="Single-image ensemble inference")
    pp.add_argument("--image", type=Path, required=True)
    pp.add_argument("--output", type=Path, default=None,
                    help="Write detections JSON to this path")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.mode == "evaluate":
        summary = evaluate_d023(
            val_txt=args.val_txt,
            ckpt_root=args.ckpt_root,
            n_folds=args.n_folds,
            fold_pattern=args.fold_pattern,
            conf=args.conf,
            iou_nms=args.iou_nms,
            iou_match=args.iou_match,
            imgsz=args.imgsz,
            device=args.device,
        )
        print_evaluate_summary(summary)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            log.info("Wrote summary JSON: %s", args.output)
        return 0 if summary["d023_overall_pass"] else 1

    if args.mode == "predict":
        result = predict_single(
            image=args.image,
            ckpt_root=args.ckpt_root,
            n_folds=args.n_folds,
            fold_pattern=args.fold_pattern,
            conf=args.conf,
            iou_nms=args.iou_nms,
            imgsz=args.imgsz,
            device=args.device,
        )
        out_text = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_text)
            log.info("Wrote detections JSON: %s", args.output)
        else:
            print(out_text)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
