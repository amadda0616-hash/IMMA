"""
src/validate/check_labels_yolo.py

V2-A — YOLO det 라벨 품질 검증 (Stage 1: View / TitleBlock / Notes).

Inputs
------
- ``--labels-dir`` : YOLO det ``.txt`` 라벨 폴더 (train 또는 val)
- ``--images-dir`` : 매칭 이미지 폴더 (옵션, 라벨-이미지 매칭 검사용)
- ``--cfg``       : configs/yolo_det.yaml (클래스 이름 + 개수)

YOLO det format
---------------
각 라벨 라인: ``<class_id> <x_center> <y_center> <width> <height>``
모두 정규화 [0, 1]. 빈 파일은 "객체 없음" (정상이지만 추적함).

Checks
------
- 빈 라벨 비율
- 라벨 라인 형식 (5개 필드, float, 좌표 범위)
- BBox 유효성 (면적>0, 정규화 범위)
- 클래스 ID 유효성
- 클래스 분포 (View / TitleBlock / Notes)
- 작은 BBox outlier (면적 임계값 미만)
- 라벨-이미지 매칭 (이름 일치 / 누락)

CLI
---
::

    python -m src.validate.check_labels_yolo \
        --labels-dir data/layout/labels/train \
        --images-dir data/layout/images/train \
        --cfg configs/yolo_det.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.validate.common import (
    DEFAULT_REPORTS_DIR,
    DEFAULT_THRESHOLDS_PATH,
    Severity,
    Status,
    ValidationReport,
    load_thresholds,
    make_bar_chart,
    setup_logging,
    threshold_lookup,
)

log = setup_logging("validate.labels_yolo")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SMALL_BBOX_AREA_RATIO = 1e-4   # ≈ 100 px² on 1000x1000 image


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_class_names(cfg_path: Path) -> List[str]:
    """Load class name list from ultralytics data YAML."""
    if not cfg_path.exists():
        return []
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    if isinstance(names, list):
        return list(names)
    return []


def parse_label_file(path: Path) -> Tuple[List[Tuple[int, float, float, float, float]],
                                          List[str]]:
    """Parse a YOLO det .txt file. Returns (valid_rows, error_messages)."""
    rows: List[Tuple[int, float, float, float, float]] = []
    errors: List[str] = []
    if not path.exists():
        return rows, [f"missing: {path.name}"]
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001
        return rows, [f"read_error: {e}"]

    for ln, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != 5:
            errors.append(f"L{ln}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
        except ValueError as e:  # noqa: PERF203
            errors.append(f"L{ln}: parse error ({e})")
            continue
        rows.append((cid, cx, cy, w, h))
    return rows, errors


def discover_pairs(labels_dir: Path,
                   images_dir: Optional[Path]) -> Tuple[List[Path], Dict[str, Path]]:
    """Return list of label files + a {stem: image_path} map."""
    labels = sorted(p for p in labels_dir.rglob("*.txt") if p.is_file())
    images: Dict[str, Path] = {}
    if images_dir is not None and images_dir.exists():
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                images[p.stem] = p
    return labels, images


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def validate_bbox(cid: int, cx: float, cy: float, w: float, h: float,
                  n_classes: int) -> List[str]:
    """Return list of issues for a single bbox row. Empty = valid."""
    issues: List[str] = []
    if cid < 0 or cid >= n_classes:
        issues.append(f"class_id={cid} out of [0, {n_classes - 1}]")
    for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not (0.0 <= val <= 1.0):
            issues.append(f"{name}={val:.4f} out of [0,1]")
    if w <= 0 or h <= 0:
        issues.append(f"non-positive size w={w:.4f} h={h:.4f}")
    if w >= 1.0 and h >= 1.0:
        issues.append(f"bbox covers full image (w*h={w * h:.3f})")
    return issues


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run(labels_dir: Path,
        images_dir: Optional[Path],
        cfg_path: Path,
        thresholds_path: Path) -> ValidationReport:
    thr = load_thresholds(thresholds_path)
    class_names = load_class_names(cfg_path)
    n_classes = len(class_names)
    if n_classes == 0:
        log.warning("Could not load class names from %s — using indices.", cfg_path)
        n_classes = 3   # safe default for stage 1
        class_names = [f"class_{i}" for i in range(n_classes)]

    label_files, image_map = discover_pairs(labels_dir, images_dir)

    report = ValidationReport(
        title="V2-A — YOLO det 라벨 품질 검증",
        step="stage1_labels",
        metadata={
            "labels_dir": str(labels_dir),
            "images_dir": str(images_dir) if images_dir else None,
            "n_label_files": len(label_files),
            "n_images": len(image_map),
            "classes": class_names,
        },
    )

    if not label_files:
        report.add_eval(
            "labels_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message=f"No .txt found under {labels_dir}",
        )
        return report

    # --- Iterate ------------------------------------------------------
    n_total = len(label_files)
    n_empty = 0
    n_parse_errors = 0
    n_invalid_bbox = 0
    n_small = 0
    parse_error_rows: List[Dict[str, str]] = []
    bbox_error_rows: List[Dict[str, str]] = []
    all_class_counts: Counter = Counter()
    aspect_ratios: List[float] = []
    label_stems: set = set()

    for lf in label_files:
        label_stems.add(lf.stem)
        rows, errors = parse_label_file(lf)
        if errors:
            n_parse_errors += 1
            for e in errors[:3]:
                parse_error_rows.append({"file": lf.name, "error": e})
        if not rows:
            n_empty += 1
            continue
        for cid, cx, cy, w, h in rows:
            issues = validate_bbox(cid, cx, cy, w, h, n_classes)
            if issues:
                n_invalid_bbox += 1
                bbox_error_rows.append({
                    "file": lf.name,
                    "row": f"{cid} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f}",
                    "issues": "; ".join(issues),
                })
                continue
            all_class_counts[cid] += 1
            if w * h < SMALL_BBOX_AREA_RATIO:
                n_small += 1
            if h > 0:
                aspect_ratios.append(w / h)

    n_total_bboxes = sum(all_class_counts.values())

    # --- Empty label rate ---------------------------------------------
    empty_rate = n_empty / n_total if n_total else 0.0
    try:
        node = threshold_lookup(thr, "stage1_labels.empty_label_rate_max")
        empty_thr, empty_sev = node["threshold"], node.get("severity", "critical")
    except KeyError:
        empty_thr, empty_sev = 0.05, "critical"
    report.add_eval(
        "empty_label_rate", value=empty_rate, threshold=empty_thr,
        direction="le", severity=empty_sev,
        message=f"{n_empty}/{n_total} files with no objects",
    )

    # --- Parse error rate ---------------------------------------------
    report.add_eval(
        "parse_error_rate", value=n_parse_errors / n_total if n_total else 0.0,
        threshold=0.0, direction="le", severity=Severity.CRITICAL,
        message=f"{n_parse_errors} files with malformed lines",
    )

    # --- BBox validity rate -------------------------------------------
    bbox_validity = (
        (n_total_bboxes / (n_total_bboxes + n_invalid_bbox))
        if (n_total_bboxes + n_invalid_bbox) > 0 else 1.0
    )
    try:
        node = threshold_lookup(thr, "stage1_labels.bbox_validity_rate")
        bv_thr, bv_sev = node["threshold"], node.get("severity", "critical")
    except KeyError:
        bv_thr, bv_sev = 1.0, "critical"
    report.add_eval(
        "bbox_validity_rate", value=bbox_validity, threshold=bv_thr,
        direction="ge", severity=bv_sev,
        message=f"{n_total_bboxes} valid / {n_invalid_bbox} invalid",
    )

    # --- Small bbox rate ----------------------------------------------
    small_rate = n_small / n_total_bboxes if n_total_bboxes else 0.0
    try:
        node = threshold_lookup(thr, "stage1_labels.small_bbox_rate_max")
        sm_thr, sm_sev = node["threshold"], node.get("severity", "warning")
    except KeyError:
        sm_thr, sm_sev = 0.05, "warning"
    report.add_eval(
        "small_bbox_rate", value=small_rate, threshold=sm_thr,
        direction="le", severity=sm_sev,
        message=f"{n_small} bboxes with area < {SMALL_BBOX_AREA_RATIO}",
    )

    # --- Class distribution -------------------------------------------
    cls_rows = []
    for cid, name in enumerate(class_names):
        cnt = all_class_counts.get(cid, 0)
        ratio = cnt / n_total_bboxes if n_total_bboxes else 0.0
        cls_rows.append({"class_id": cid, "name": name,
                         "count": cnt, "ratio": f"{ratio:.1%}"})
    report.add_table(
        "Class distribution",
        rows=cls_rows,
        columns=["class_id", "name", "count", "ratio"],
        description=f"Total {n_total_bboxes} valid bboxes across {n_total - n_empty} non-empty files.",
    )
    report.add_plot(
        "Class distribution chart",
        make_bar_chart(
            labels=class_names,
            values=[all_class_counts.get(i, 0) for i in range(n_classes)],
            title="Bbox count per class", ylabel="count",
        ),
    )

    # --- Class distribution thresholds (warning) ----------------------
    try:
        cd_node = threshold_lookup(thr, "stage1_labels.class_distribution")
        cd_sev = cd_node.get("severity", "warning")
        for cls_name, key in (("View", "View_min_ratio"),
                              ("TitleBlock", "TitleBlock_min_ratio"),
                              ("Notes", "Notes_min_ratio")):
            min_ratio = cd_node.get(key)
            if min_ratio is None or cls_name not in class_names:
                continue
            cid = class_names.index(cls_name)
            ratio = (all_class_counts.get(cid, 0) / n_total_bboxes) if n_total_bboxes else 0.0
            report.add_eval(
                f"class_ratio[{cls_name}]", value=ratio, threshold=min_ratio,
                direction="ge", severity=cd_sev,
                message=f"{all_class_counts.get(cid, 0)} bboxes",
            )
    except KeyError:
        pass

    # --- Label–image matching -----------------------------------------
    if image_map:
        labels_without_img = [s for s in label_stems if s not in image_map]
        images_without_lbl = [s for s in image_map.keys() if s not in label_stems]
        report.add_eval(
            "labels_without_images", value=len(labels_without_img),
            threshold=0, direction="le", severity=Severity.WARNING,
            message="Labels with no matching image",
        )
        report.add_eval(
            "images_without_labels", value=len(images_without_lbl),
            threshold=0, direction="le", severity=Severity.WARNING,
            message=("Images with no matching label "
                     "(may be intentional 'no objects' background)"),
        )
        if labels_without_img[:10]:
            report.add_table(
                "Sample: labels without images (first 10)",
                rows=[{"stem": s} for s in labels_without_img[:10]],
                columns=["stem"],
            )

    # --- Aspect ratio outliers ----------------------------------------
    if aspect_ratios:
        ar_extreme = sum(1 for r in aspect_ratios if r < 0.1 or r > 10)
        report.add_eval(
            "extreme_aspect_ratio_count", value=ar_extreme, threshold=0,
            direction="le", severity=Severity.WARNING,
            message=f"BBoxes with W/H < 0.1 or > 10 (count of {len(aspect_ratios)})",
        )

    # --- Error sample tables ------------------------------------------
    if parse_error_rows[:20]:
        report.add_table(
            "Parse errors (first 20)",
            rows=parse_error_rows[:20],
            columns=["file", "error"],
        )
    if bbox_error_rows[:20]:
        report.add_table(
            "BBox errors (first 20)",
            rows=bbox_error_rows[:20],
            columns=["file", "row", "issues"],
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V2-A — YOLO det 라벨 품질 검증."
    )
    p.add_argument("--labels-dir", type=Path, required=True,
                   help="e.g. data/layout/labels/train")
    p.add_argument("--images-dir", type=Path, default=None,
                   help="(optional) e.g. data/layout/images/train")
    p.add_argument("--cfg", type=Path,
                   default=Path("configs/yolo_det.yaml"),
                   help="ultralytics data YAML for class names")
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Labels dir : %s", args.labels_dir)
    log.info("Images dir : %s", args.images_dir or "(none)")
    log.info("Cfg        : %s", args.cfg)

    if not args.labels_dir.exists():
        log.error("labels-dir not found: %s", args.labels_dir)
        return 2

    report = run(args.labels_dir, args.images_dir, args.cfg, args.thresholds)
    paths = report.emit(
        reports_dir=args.reports_dir,
        stem=f"{args.labels_dir.parent.name}_{args.labels_dir.name}_yolo_labels",
        use_color=not args.no_color,
    )
    log.info("HTML : %s", paths["html"])
    log.info("JSON : %s", paths["json"])

    return 0 if report.overall_status in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
