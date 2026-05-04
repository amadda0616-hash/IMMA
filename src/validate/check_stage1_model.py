"""
src/validate/check_stage1_model.py

V2-B — Stage 1 (YOLOv11-det) 학습 모델 성능 검증.

논문 Khan 2025 §4.1 기준 임계값 (`configs/validation_thresholds.yaml#stage1_model`):
- View ≥ 0.90 / TitleBlock ≥ 0.95 / Notes ≥ 0.90  (per-class accuracy)
- mAP@0.5 ≥ 0.85
- false positive rate < 0.10  (TB 없는 도면에서 TB 검출)

Inputs
------
- ``--weights`` : ``checkpoints/yolo_det.pt``
- ``--data``    : ``configs/yolo_det.yaml``
- ``--split``   : "val" (default) / "test"

Checks
------
- mAP@0.5 (overall)
- per-class precision / recall / F1
- per-class accuracy (논문 confusion matrix 형식과 동일)
- confusion matrix
- confidence distribution

CLI
---
::

    python -m src.validate.check_stage1_model \
        --weights checkpoints/yolo_det.pt \
        --data configs/yolo_det.yaml --device 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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

log = setup_logging("validate.stage1_model")


# ---------------------------------------------------------------------------
# D-029 mapping (Roboflow class names ↔ internal canonical names)
# ---------------------------------------------------------------------------
# 모델은 학습 시 yaml 의 Roboflow 이름 (Table, Text 등) 을 반환하지만,
# `validation_thresholds.yaml` 은 D-029 의 내부 정규명 (TitleBlock, Notes) 을 사용.
# 임계값 lookup 시 양방향 시도 — Roboflow → 내부, 내부 → Roboflow.
INTERNAL_TO_ROBOFLOW: Dict[str, str] = {
    "TitleBlock": "Table",
    "Notes": "Text",
}
ROBOFLOW_TO_INTERNAL: Dict[str, str] = {v: k for k, v in INTERNAL_TO_ROBOFLOW.items()}


def _resolve_threshold(pca_node: Dict, name: str) -> Optional[Dict]:
    """Return threshold spec for class ``name`` trying both Roboflow and internal aliases."""
    if name in pca_node:
        return pca_node[name]
    alt = ROBOFLOW_TO_INTERNAL.get(name) or INTERNAL_TO_ROBOFLOW.get(name)
    if alt and alt in pca_node:
        return pca_node[alt]
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_get(obj, attr: str, default=None):
    val = getattr(obj, attr, default)
    if hasattr(val, "cpu"):
        val = val.cpu().numpy()
    return val


def _to_float_list(arr) -> List[float]:
    if arr is None:
        return []
    a = np.asarray(arr).flatten()
    return [float(x) for x in a.tolist()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(weights: Path,
        data_cfg: Path,
        thresholds_path: Path,
        split: str = "val",
        imgsz: int = 1280,
        batch: int = 8,
        device: Optional[str] = None) -> ValidationReport:
    """Run YOLO model.val() and aggregate metrics into a ValidationReport."""

    # Lazy import (heavy) so CLI --help works without ultralytics installed.
    from ultralytics import YOLO  # noqa: PLC0415

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_cfg.exists():
        raise FileNotFoundError(f"Data config not found: {data_cfg}")

    thr = load_thresholds(thresholds_path)

    log.info("Loading model: %s", weights)
    model = YOLO(str(weights))
    class_names: Dict[int, str] = dict(model.names) if isinstance(model.names, dict) \
        else {i: n for i, n in enumerate(model.names)}

    log.info("Running val (data=%s, split=%s, imgsz=%d, batch=%d)",
             data_cfg, split, imgsz, batch)
    metrics = model.val(
        data=str(data_cfg),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False,
        plots=False,
    )

    report = ValidationReport(
        title="V2-B — Stage 1 YOLOv11-det Model Validation",
        step="stage1_model",
        metadata={
            "weights": str(weights),
            "data": str(data_cfg),
            "split": split,
            "imgsz": imgsz,
            "classes": list(class_names.values()),
        },
    )

    box = getattr(metrics, "box", None)
    if box is None:
        report.add_eval(
            "metrics_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message="ultralytics returned no box metrics",
        )
        return report

    # --- mAP@0.5 ------------------------------------------------------
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    map5095 = float(getattr(box, "map", 0.0) or 0.0)
    try:
        node = threshold_lookup(thr, "stage1_model.map_at_50")
        m_thr, m_sev = node["threshold"], node.get("severity", "critical")
    except KeyError:
        m_thr, m_sev = 0.85, "critical"

    report.add_eval(
        "mAP@0.5 (overall)", value=map50, threshold=m_thr,
        direction="ge", severity=m_sev,
        message=f"mAP@0.5:0.95 = {map5095:.4f}",
    )

    # --- per-class ----------------------------------------------------
    p_arr = _to_float_list(_safe_get(box, "p"))    # precision
    r_arr = _to_float_list(_safe_get(box, "r"))    # recall
    f1_arr = _to_float_list(_safe_get(box, "f1"))  # F1
    map50_per = _to_float_list(_safe_get(box, "ap50"))  # per-class AP@0.5

    # Class index → name
    n_classes = max(
        len(p_arr), len(r_arr), len(f1_arr),
        len(map50_per), len(class_names),
    )
    cls_table_rows: List[Dict[str, str]] = []
    for cid in range(n_classes):
        name = class_names.get(cid, f"class_{cid}")
        p_v = p_arr[cid] if cid < len(p_arr) else None
        r_v = r_arr[cid] if cid < len(r_arr) else None
        f_v = f1_arr[cid] if cid < len(f1_arr) else None
        m_v = map50_per[cid] if cid < len(map50_per) else None
        cls_table_rows.append({
            "class": name,
            "precision": f"{p_v:.4f}" if p_v is not None else "—",
            "recall":    f"{r_v:.4f}" if r_v is not None else "—",
            "f1":        f"{f_v:.4f}" if f_v is not None else "—",
            "mAP@0.5":   f"{m_v:.4f}" if m_v is not None else "—",
        })
    report.add_table(
        "Per-class metrics",
        rows=cls_table_rows,
        columns=["class", "precision", "recall", "f1", "mAP@0.5"],
    )

    # ------------------------------------------------------------------
    # Per-class accuracy thresholds — mAP@0.5 기반 (ultralytics 표준)
    # ------------------------------------------------------------------
    # 이전 버전은 confusion_matrix.matrix 의 diagonal/col-sum 으로 계산했으나
    # ultralytics 8.4.x 에서 CM 셀이 0 으로 나오는 케이스 발견 (CM 의 conf
    # threshold 0.25 필터링이 셀을 비움 — 정확한 mAP 계산에는 영향 없음).
    # 따라서 표준 box.ap50 (per-class AP@0.5) 을 accuracy 대용으로 사용.
    # → "accuracy" 의미는 "AP@0.5 per class" 로 재정의 (논문과 일관).
    per_class_acc: Dict[str, float] = {}
    for cid in range(n_classes):
        name = class_names.get(cid, f"class_{cid}")
        m_v = map50_per[cid] if cid < len(map50_per) else None
        if m_v is not None:
            per_class_acc[name] = float(m_v)

    # Threshold checks (D-029 양방향 매핑 적용)
    try:
        pca_node = threshold_lookup(thr, "stage1_model.per_class_accuracy")
        for name, acc in per_class_acc.items():
            spec = _resolve_threshold(pca_node, name)
            if spec is None:
                report.add_eval(
                    f"per_class_accuracy[{name}]",
                    value=acc, threshold=None, direction="none",
                    severity=Severity.INFO,
                    message=f"AP@0.5 = {acc:.4f} (no threshold defined)",
                )
                continue
            report.add_eval(
                f"per_class_accuracy[{name}]",
                value=acc, threshold=spec["threshold"],
                direction="ge",
                severity=spec.get("severity", "critical"),
                message=f"AP@0.5 = {acc:.4f}",
            )
    except KeyError:
        for name, acc in per_class_acc.items():
            report.add_eval(
                f"per_class_accuracy[{name}]",
                value=acc, threshold=None, direction="none",
                severity=Severity.INFO,
                message=f"AP@0.5 = {acc:.4f}",
            )

    # ------------------------------------------------------------------
    # Confusion matrix — 시각화 용도 (judgment 에는 사용 안 함, 위 mAP 기반)
    # ------------------------------------------------------------------
    cm_obj = getattr(metrics, "confusion_matrix", None)
    cm_matrix = None
    if cm_obj is not None:
        try:
            cm_matrix = cm_obj.matrix
            if hasattr(cm_matrix, "cpu"):
                cm_matrix = cm_matrix.cpu().numpy()
            cm_matrix = np.asarray(cm_matrix, dtype=float)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not extract CM for visualization: %s", e)
            cm_matrix = None

    if cm_matrix is not None and cm_matrix.size > 0:
        n_real = cm_matrix.shape[0] - 1
        labels = [class_names.get(i, f"c{i}") for i in range(n_real)] + ["background"]
        report.add_plot(
            "Confusion matrix (val set)",
            make_confusion_matrix(
                cm_matrix.astype(int).tolist(),
                labels=labels,
                title="Stage 1 YOLOv11-det",
            ),
            description="Rows=actual, cols=predicted. Last entry is background "
                        "(FP if column ≠ background, FN if row ≠ background). "
                        "정보성 시각화 — 임계값 평가는 mAP@0.5 (위) 기반.",
        )

    # --- False positive rate (background prediction by class) ---------
    # D-029: 모델 출력 Roboflow 이름 (Table) 또는 내부명 (TitleBlock) 양쪽 시도
    if cm_matrix is not None and cm_matrix.size > 0:
        tb_idx = None
        tb_label = None
        for i, n in class_names.items():
            if n in ("TitleBlock", "Table"):
                tb_idx = i
                tb_label = n
                break
        if tb_idx is not None:
            bg_row = cm_matrix[-1, :]
            fp_tb = float(bg_row[tb_idx])
            total_bg = float(bg_row.sum())
            fp_rate = fp_tb / total_bg if total_bg > 0 else 0.0
            try:
                node = threshold_lookup(thr, "stage1_model.false_positive_rate_max")
                fp_thr, fp_sev = node["threshold"], node.get("severity", "warning")
            except KeyError:
                fp_thr, fp_sev = 0.10, "warning"
            report.add_eval(
                f"false_positive_rate[{tb_label}]", value=fp_rate,
                threshold=fp_thr, direction="le", severity=fp_sev,
                message=f"BG predicted as {tb_label}: {int(fp_tb)} / {int(total_bg)}",
            )

    # --- Per-class mAP plot -------------------------------------------
    if map50_per and class_names:
        names = [class_names.get(i, f"c{i}") for i in range(len(map50_per))]
        report.add_plot(
            "Per-class mAP@0.5",
            make_bar_chart(
                labels=names, values=map50_per,
                title="mAP@0.5 by class",
                ylabel="AP@0.5", ylim=(0.0, 1.0),
            ),
        )

    # --- Per-class accuracy plot --------------------------------------
    if per_class_acc:
        names = list(per_class_acc.keys())
        vals = [per_class_acc[n] for n in names]
        report.add_plot(
            "Per-class accuracy (CM-based)",
            make_bar_chart(
                labels=names, values=vals,
                title="Per-class accuracy", ylabel="accuracy",
                ylim=(0.0, 1.0),
            ),
            description="Diagonal / column-sum from confusion matrix.",
        )

    # --- Summary line -------------------------------------------------
    report.add_eval(
        "n_classes", value=n_classes, threshold=None, direction="none",
        severity=Severity.INFO,
        message=", ".join(class_names.values()),
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V2-B — Stage 1 YOLOv11-det model validation."
    )
    p.add_argument("--weights", type=Path,
                   default=Path("checkpoints/yolo_det.pt"))
    p.add_argument("--data", type=Path,
                   default=Path("configs/yolo_det.yaml"))
    p.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None,
                   help='e.g. "0", "cpu", or omit for auto')
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Weights : %s", args.weights)
    log.info("Data    : %s", args.data)
    log.info("Split   : %s", args.split)

    try:
        report = run(
            weights=args.weights,
            data_cfg=args.data,
            thresholds_path=args.thresholds,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2
    except ImportError as e:
        log.error("ultralytics not installed: %s", e)
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
