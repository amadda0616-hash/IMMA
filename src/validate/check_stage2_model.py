"""
src/validate/check_stage2_model.py

V3-B — Stage 2 (YOLOv11-obb) 학습 모델 성능 검증.

★ 사용자 필수 임계값 (D-023):
- Measure 누락률 < 8%   (critical)
- GDT 누락률     < 5%   (critical)
- Roughness 누락률 < 30% (warning)
- 도면 단위 회수율 (drawing-level recall) ≥ 0.85 (critical)
- per-class accuracy: Measure 0.92 / GDT 0.95 / Roughness 0.50 (논문)
- mAP@0.5 ≥ 0.80

Inputs
------
- ``--weights`` : ``checkpoints/yolo_obb.pt``
- ``--data``    : ``configs/yolo_obb.yaml``
- ``--split``   : "val" (default)

Methodology
-----------
1. ultralytics ``model.val()`` 로 mAP / per-class P/R/F1 측정
2. ``model.predict()`` 를 val 이미지마다 돌려 per-image GT vs Pred 매칭
   (IoU ≥ 0.5, polygon IoU via shapely)
3. 클래스별 누락률 = FN / (TP + FN) = 1 - recall
4. 도면 단위 회수율 = mean(image_recall) where
   image_recall = #matched_GT / #total_GT in that image

CLI
---
::

    python -m src.validate.check_stage2_model \
        --weights checkpoints/yolo_obb.pt \
        --data configs/yolo_obb.yaml --device 0
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.validate.common import (
    DEFAULT_REPORTS_DIR,
    DEFAULT_THRESHOLDS_PATH,
    Severity,
    Status,
    ValidationReport,
    load_thresholds,
    make_bar_chart,
    make_confusion_matrix,
    setup_logging,
    threshold_lookup,
)

log = setup_logging("validate.stage2_model")

IOU_MATCH_THRESHOLD = 0.50    # standard for OBB detection eval
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Polygon IoU (shapely with fallback to bbox IoU)
# ---------------------------------------------------------------------------
def polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """OBB polygon IoU. Falls back to axis-aligned BBox IoU if shapely missing."""
    try:
        from shapely.geometry import Polygon  # noqa: PLC0415
        poly1 = Polygon(p1).buffer(0)
        poly2 = Polygon(p2).buffer(0)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return float(inter / union) if union > 0 else 0.0
    except ImportError:
        # Fallback: axis-aligned bbox IoU (over-estimates but avoids hard dep)
        return _bbox_iou_from_polys(p1, p2)


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
# GT loading
# ---------------------------------------------------------------------------
def load_data_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_class_names(cfg: Dict[str, Any]) -> List[str]:
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, list):
        return list(names)
    return ["Measure", "GDT", "Roughness"]


def find_split_dirs(cfg: Dict[str, Any], split: str) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) for the given split."""
    root = Path(cfg.get("path", ".")).resolve()
    rel = cfg.get(split, f"images/{split}")
    images_dir = (root / rel).resolve()
    # YOLO convention: labels are sibling of images with 'labels' instead of 'images'
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    return images_dir, labels_dir


def load_gt_obbs(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    """Parse one YOLO OBB .txt to list of (cid, 4x2 points)."""
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


def parse_pred_obbs(result, img_w: int, img_h: int) -> List[Tuple[int, float, np.ndarray]]:
    """Convert ultralytics OBB result to list of (cid, conf, 4x2 normalized points)."""
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return []
    pts_all = obb.xyxyxyxy
    cls = obb.cls
    conf = obb.conf
    if hasattr(pts_all, "cpu"):
        pts_all = pts_all.cpu().numpy()
        cls = cls.cpu().numpy()
        conf = conf.cpu().numpy()
    out: List[Tuple[int, float, np.ndarray]] = []
    for pts, c, p in zip(pts_all, cls, conf):
        # normalize to [0,1] for IoU comparison with GT
        pts_norm = pts.astype(float).copy()
        pts_norm[:, 0] /= float(img_w)
        pts_norm[:, 1] /= float(img_h)
        out.append((int(c), float(p), pts_norm))
    return out


# ---------------------------------------------------------------------------
# Greedy IoU matching
# ---------------------------------------------------------------------------
def match_gt_pred(
    gts: List[Tuple[int, np.ndarray]],
    preds: List[Tuple[int, float, np.ndarray]],
    iou_thr: float = IOU_MATCH_THRESHOLD,
) -> Tuple[int, int, int, Dict[int, Dict[str, int]]]:
    """Greedy match preds to GTs by IoU. Class must agree.

    Returns
    -------
    (tp_total, fp_total, fn_total, per_class_stats)
        per_class_stats[cid] = {'tp', 'fp', 'fn'}
    """
    per_class: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    used_gt = set()
    # Sort preds by confidence desc for stable greedy match
    preds_sorted = sorted(preds, key=lambda x: -x[1])

    for cid_p, _, pts_p in preds_sorted:
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

    # Anything unmatched in GT is FN, indexed by GT class
    for gi, (cid_g, _) in enumerate(gts):
        if gi not in used_gt:
            per_class[cid_g]["fn"] += 1

    tp = sum(v["tp"] for v in per_class.values())
    fp = sum(v["fp"] for v in per_class.values())
    fn = sum(v["fn"] for v in per_class.values())
    return tp, fp, fn, dict(per_class)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
def run(weights: Path,
        data_cfg: Path,
        thresholds_path: Path,
        split: str = "val",
        imgsz: int = 1024,
        batch: int = 8,
        device: Optional[str] = None,
        iou_thr: float = IOU_MATCH_THRESHOLD,
        conf_thr: float = 0.25) -> ValidationReport:
    from ultralytics import YOLO  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_cfg.exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg}")

    thr = load_thresholds(thresholds_path)
    cfg = load_data_yaml(data_cfg)
    class_names = get_class_names(cfg)
    n_classes = len(class_names)

    log.info("Loading model: %s", weights)
    model = YOLO(str(weights))

    # ---- 1) ultralytics aggregate metrics ---------------------------
    log.info("Running ultralytics val (aggregate)")
    metrics = model.val(
        data=str(data_cfg), split=split,
        imgsz=imgsz, batch=batch, device=device,
        verbose=False, plots=False,
    )

    report = ValidationReport(
        title="V3-B — Stage 2 YOLOv11-obb Model Validation (★ D-023)",
        step="stage2_model",
        metadata={
            "weights": str(weights), "data": str(data_cfg),
            "split": split, "imgsz": imgsz,
            "iou_threshold": iou_thr, "conf_threshold": conf_thr,
            "classes": class_names,
        },
    )

    box = getattr(metrics, "obb", None) or getattr(metrics, "box", None)
    map50 = float(getattr(box, "map50", 0.0) or 0.0) if box else 0.0
    map5095 = float(getattr(box, "map", 0.0) or 0.0) if box else 0.0

    try:
        node = threshold_lookup(thr, "stage2_model.map_at_50")
        m_thr, m_sev = node["threshold"], node.get("severity", "critical")
    except KeyError:
        m_thr, m_sev = 0.80, "critical"
    report.add_eval(
        "mAP@0.5 (overall)", value=map50, threshold=m_thr,
        direction="ge", severity=m_sev,
        message=f"mAP@0.5:0.95 = {map5095:.4f}",
    )

    # ---- 2) Per-image GT vs Pred matching ---------------------------
    images_dir, labels_dir = find_split_dirs(cfg, split)
    if not images_dir.exists() or not labels_dir.exists():
        log.warning("Split dirs missing: img=%s lbl=%s", images_dir, labels_dir)
        report.add_eval(
            "split_dirs_found", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message="Cannot compute missing rate / drawing recall",
        )
        return report

    image_files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    log.info("Per-image evaluation on %d images", len(image_files))

    drawing_recalls: List[float] = []
    per_class_totals: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    n_imgs_with_gt = 0

    for i, img_path in enumerate(image_files, 1):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gts = load_gt_obbs(lbl_path)
        if not gts:
            continue
        n_imgs_with_gt += 1

        # Read image dimensions
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Predict
        results = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf_thr,
            device=device, verbose=False,
        )
        preds = parse_pred_obbs(results[0], w, h) if results else []

        tp, fp, fn, per_cls = match_gt_pred(gts, preds, iou_thr=iou_thr)
        # Drawing-level recall
        gt_count = len(gts)
        img_recall = tp / gt_count if gt_count > 0 else 1.0
        drawing_recalls.append(img_recall)

        for cid, st in per_cls.items():
            per_class_totals[cid]["tp"] += st["tp"]
            per_class_totals[cid]["fp"] += st["fp"]
            per_class_totals[cid]["fn"] += st["fn"]

        if i % 25 == 0 or i == len(image_files):
            log.info("  [%d/%d] images processed", i, len(image_files))

    # ---- 3) ★ Missing rates per class (D-023) -----------------------
    try:
        miss_node = threshold_lookup(thr, "stage2_model.missing_rate_max")
    except KeyError:
        miss_node = {}

    miss_rows: List[Dict[str, str]] = []
    miss_values: List[float] = []
    miss_labels: List[str] = []

    for cid in range(n_classes):
        name = class_names[cid] if cid < len(class_names) else f"c{cid}"
        st = per_class_totals.get(cid, {"tp": 0, "fp": 0, "fn": 0})
        gt_total = st["tp"] + st["fn"]
        miss_rate = (st["fn"] / gt_total) if gt_total > 0 else 0.0
        recall = (st["tp"] / gt_total) if gt_total > 0 else 0.0
        precision = (st["tp"] / (st["tp"] + st["fp"])) \
            if (st["tp"] + st["fp"]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) \
            if (precision + recall) > 0 else 0.0

        miss_rows.append({
            "class": name,
            "tp": st["tp"], "fp": st["fp"], "fn": st["fn"],
            "GT_total": gt_total,
            "missing_rate": f"{miss_rate:.4f}",
            "recall": f"{recall:.4f}",
            "precision": f"{precision:.4f}",
            "f1": f"{f1:.4f}",
        })
        miss_values.append(miss_rate)
        miss_labels.append(name)

        # threshold check
        spec = miss_node.get(name) if isinstance(miss_node, dict) else None
        if spec:
            report.add_eval(
                f"missing_rate[{name}]",
                value=miss_rate, threshold=spec["threshold"],
                direction="le",
                severity=spec.get("severity", "warning"),
                message=f"FN={st['fn']} / GT={gt_total}",
            )
        else:
            report.add_eval(
                f"missing_rate[{name}]",
                value=miss_rate, threshold=None, direction="none",
                severity=Severity.INFO,
            )

    report.add_table(
        "Per-class TP / FP / FN / missing rate (★ D-023)",
        rows=miss_rows,
        columns=["class", "tp", "fp", "fn", "GT_total",
                 "missing_rate", "recall", "precision", "f1"],
        description=f"Greedy IoU≥{iou_thr} matching, conf≥{conf_thr}.",
    )
    report.add_plot(
        "Per-class missing rate (★)",
        make_bar_chart(
            labels=miss_labels, values=miss_values,
            title="Missing rate = FN / (TP + FN)",
            ylabel="missing rate", ylim=(0.0, max(0.5, max(miss_values + [0.0]))),
        ),
    )

    # ---- 4) ★ Drawing-level recall (D-023) --------------------------
    if drawing_recalls:
        dr_mean = float(np.mean(drawing_recalls))
        dr_std = float(np.std(drawing_recalls))
        try:
            dr_node = threshold_lookup(thr, "stage2_model.drawing_recall_min")
            dr_thr, dr_sev = dr_node["threshold"], dr_node.get("severity", "critical")
        except KeyError:
            dr_thr, dr_sev = 0.85, "critical"
        report.add_eval(
            "drawing_level_recall (★)",
            value=dr_mean, threshold=dr_thr,
            direction="ge", severity=dr_sev,
            message=f"mean over {n_imgs_with_gt} images "
                    f"(std={dr_std:.4f})",
        )

        # Histogram of per-image recall
        bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
        hist, _ = np.histogram(drawing_recalls, bins=bins)
        labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
        report.add_plot(
            "Drawing-level recall histogram",
            make_bar_chart(
                labels=labels, values=[int(v) for v in hist],
                title="Per-image recall distribution",
                ylabel="image count",
            ),
            description="Shows how many images fall into each recall bucket.",
        )
    else:
        report.add_eval(
            "drawing_level_recall (★)", value=0, threshold=0.85,
            direction="ge", severity=Severity.CRITICAL,
            message="No images with GT — cannot compute",
        )

    # ---- 5) Per-class accuracy (논문 confusion matrix style) ---------
    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is not None:
        try:
            cm_matrix = cm_obj.matrix
            if hasattr(cm_matrix, "cpu"):
                cm_matrix = cm_matrix.cpu().numpy()
            cm_matrix = np.asarray(cm_matrix, dtype=float)

            n_real = cm_matrix.shape[0] - 1
            try:
                pca_node = threshold_lookup(thr, "stage2_model.per_class_accuracy")
            except KeyError:
                pca_node = {}

            for cid in range(n_real):
                col_sum = cm_matrix[:, cid].sum()
                acc = float(cm_matrix[cid, cid] / col_sum) if col_sum > 0 else 0.0
                name = class_names[cid] if cid < len(class_names) else f"c{cid}"
                spec = pca_node.get(name) if isinstance(pca_node, dict) else None
                if spec:
                    report.add_eval(
                        f"per_class_accuracy[{name}]",
                        value=acc, threshold=spec["threshold"],
                        direction="ge",
                        severity=spec.get("severity", "warning"),
                    )
                else:
                    report.add_eval(
                        f"per_class_accuracy[{name}]",
                        value=acc, threshold=None, direction="none",
                        severity=Severity.INFO,
                    )

            labels = (class_names + ["background"])[:cm_matrix.shape[0]]
            report.add_plot(
                "Confusion matrix (val set)",
                make_confusion_matrix(
                    cm_matrix.astype(int).tolist(),
                    labels=labels,
                    title="Stage 2 YOLOv11-obb",
                ),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("CM extraction failed: %s", e)

    # ---- 6) Class confusion magnitude --------------------------------
    if cm_obj is not None and cm_matrix.size > 0:
        n_real = cm_matrix.shape[0] - 1
        diag = float(np.trace(cm_matrix[:n_real, :n_real]))
        off = float(np.sum(cm_matrix[:n_real, :n_real]) - diag)
        total = diag + off
        confusion_rate = (off / total) if total > 0 else 0.0
        try:
            node = threshold_lookup(thr, "stage2_model.class_confusion_max")
            cf_thr, cf_sev = node["threshold"], node.get("severity", "warning")
        except KeyError:
            cf_thr, cf_sev = 0.05, "warning"
        report.add_eval(
            "class_confusion_rate", value=confusion_rate,
            threshold=cf_thr, direction="le", severity=cf_sev,
            message=f"off-diagonal {int(off)} / total {int(total)}",
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V3-B — Stage 2 YOLOv11-obb model validation "
                    "(missing rate / drawing recall, ★ D-023)."
    )
    p.add_argument("--weights", type=Path,
                   default=Path("checkpoints/yolo_obb.pt"))
    p.add_argument("--data", type=Path,
                   default=Path("configs/yolo_obb.yaml"))
    p.add_argument("--split", type=str, default="val",
                   choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--iou", type=float, default=IOU_MATCH_THRESHOLD,
                   help="IoU threshold for GT-pred matching")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold for predictions")
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Weights : %s", args.weights)
    log.info("Data    : %s", args.data)
    log.info("Split   : %s   IoU=%.2f   conf=%.2f", args.split, args.iou, args.conf)

    try:
        report = run(
            weights=args.weights,
            data_cfg=args.data,
            thresholds_path=args.thresholds,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            iou_thr=args.iou,
            conf_thr=args.conf,
        )
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    except ImportError as e:
        log.error("Required dependency missing: %s", e)
        return 3

    paths = report.emit(
        reports_dir=args.reports_dir,
        use_color=not args.no_color,
    )
    log.info("HTML : %s", paths["html"])
    log.info("JSON : %s", paths["json"])

    return 0 if report.overall_status in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
