"""
src/validate/check_labels_obb.py

V3-A — YOLO OBB 라벨 품질 검증 (Stage 2: Measure / GDT / Roughness).

YOLO OBB format
---------------
각 라벨 라인: ``<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>``
모두 정규화 [0, 1].

Inputs
------
- ``--labels-dir`` : YOLO obb ``.txt`` 라벨 폴더
- ``--images-dir`` : 매칭 이미지 폴더 (옵션)
- ``--cfg``       : configs/yolo_obb.yaml

Checks
------
- 빈 라벨 비율
- 라벨 라인 형식 (9개 필드)
- OBB 유효성 (좌표 범위, 자기교차 없음, 면적 > 0)
- 클래스 분포 (Measure / GDT / Roughness)
- ★ Roughness 최소 개수 (< 50 → synthetic_gen 검토, D-017)
- 각도 분포 (axis-aligned 외 각도 비율 ≥ 0.20)
- 작은 annotation outlier
- 라벨-이미지 매칭

CLI
---
::

    python -m src.validate.check_labels_obb \
        --labels-dir data/annotation/labels/train \
        --images-dir data/annotation/images/train \
        --cfg configs/yolo_obb.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    setup_logging,
    threshold_lookup,
)

log = setup_logging("validate.labels_obb")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SMALL_OBB_AREA_RATIO = 5e-5
AXIS_ALIGNED_TOL_DEG = 3.0   # ±3° from 0° / 90° → "axis-aligned"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_class_names(cfg_path: Path) -> List[str]:
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


def parse_obb_file(path: Path) -> Tuple[List[Tuple[int, np.ndarray]], List[str]]:
    """Parse OBB .txt → list of (class_id, 4x2 array). Errors as messages."""
    rows: List[Tuple[int, np.ndarray]] = []
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
        if len(parts) != 9:
            errors.append(f"L{ln}: expected 9 fields, got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]], dtype=float)
        except ValueError as e:  # noqa: PERF203
            errors.append(f"L{ln}: parse error ({e})")
            continue
        rows.append((cid, coords.reshape(4, 2)))
    return rows, errors


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def polygon_area(pts: np.ndarray) -> float:
    """Shoelace area for an Nx2 polygon."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_simple_quad(pts: np.ndarray) -> bool:
    """Check 4-point quad is non-self-intersecting and has consistent winding."""
    # Cross products of consecutive edge pairs should have same sign for convex.
    signs = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    signs = [s for s in signs if s != 0]
    if not signs:
        return False
    return all(s == signs[0] for s in signs)


def obb_long_edge_angle_deg(pts: np.ndarray) -> float:
    """Angle of the longest edge in degrees, normalized to [-90, 90)."""
    edges = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        d = b - a
        length = math.hypot(d[0], d[1])
        edges.append((length, math.degrees(math.atan2(d[1], d[0]))))
    _, ang = max(edges, key=lambda x: x[0])
    while ang >= 90.0:
        ang -= 180.0
    while ang < -90.0:
        ang += 180.0
    return float(ang)


def is_axis_aligned(angle_deg: float, tol: float = AXIS_ALIGNED_TOL_DEG) -> bool:
    """True if angle within ±tol of 0° or ±90°."""
    return abs(angle_deg) < tol or abs(abs(angle_deg) - 90.0) < tol


def validate_obb(cid: int, pts: np.ndarray, n_classes: int) -> List[str]:
    issues: List[str] = []
    if cid < 0 or cid >= n_classes:
        issues.append(f"class_id={cid} out of [0,{n_classes - 1}]")
    if pts.shape != (4, 2):
        issues.append(f"shape {pts.shape} ≠ (4,2)")
        return issues
    if np.any(pts < 0.0) or np.any(pts > 1.0):
        issues.append("coords outside [0,1]")
    area = polygon_area(pts)
    if area <= 0.0:
        issues.append(f"non-positive area={area:.6f}")
    if not is_simple_quad(pts):
        issues.append("self-intersecting / non-convex quad")
    return issues


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_pairs(labels_dir: Path,
                   images_dir: Optional[Path]) -> Tuple[List[Path], Dict[str, Path]]:
    labels = sorted(p for p in labels_dir.rglob("*.txt") if p.is_file())
    images: Dict[str, Path] = {}
    if images_dir is not None and images_dir.exists():
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                images[p.stem] = p
    return labels, images


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run(labels_dir: Path,
        images_dir: Optional[Path],
        cfg_path: Path,
        thresholds_path: Path) -> ValidationReport:
    thr = load_thresholds(thresholds_path)
    class_names = load_class_names(cfg_path)
    if not class_names:
        log.warning("No class names from %s — defaulting to 3 classes.", cfg_path)
        class_names = ["Measure", "GDT", "Roughness"]
    n_classes = len(class_names)

    label_files, image_map = discover_pairs(labels_dir, images_dir)

    report = ValidationReport(
        title="V3-A — YOLO OBB 라벨 품질 검증",
        step="stage2_labels",
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
    n_invalid_obb = 0
    n_small = 0
    n_axis_aligned = 0
    parse_error_rows: List[Dict[str, str]] = []
    obb_error_rows: List[Dict[str, str]] = []
    all_class_counts: Counter = Counter()
    angles: List[float] = []
    label_stems: set = set()

    for lf in label_files:
        label_stems.add(lf.stem)
        rows, errors = parse_obb_file(lf)
        if errors:
            n_parse_errors += 1
            for e in errors[:3]:
                parse_error_rows.append({"file": lf.name, "error": e})
        if not rows:
            n_empty += 1
            continue
        for cid, pts in rows:
            issues = validate_obb(cid, pts, n_classes)
            if issues:
                n_invalid_obb += 1
                obb_error_rows.append({
                    "file": lf.name,
                    "class_id": str(cid),
                    "issues": "; ".join(issues),
                })
                continue
            all_class_counts[cid] += 1
            area = polygon_area(pts)
            if area < SMALL_OBB_AREA_RATIO:
                n_small += 1
            ang = obb_long_edge_angle_deg(pts)
            angles.append(ang)
            if is_axis_aligned(ang):
                n_axis_aligned += 1

    n_total_obbs = sum(all_class_counts.values())

    # --- Empty rate ---------------------------------------------------
    empty_rate = n_empty / n_total if n_total else 0.0
    report.add_eval(
        "empty_label_rate", value=empty_rate, threshold=0.05, direction="le",
        severity=Severity.CRITICAL,
        message=f"{n_empty}/{n_total} files",
    )

    # --- Parse error -------------------------------------------------
    report.add_eval(
        "parse_error_rate", value=n_parse_errors / n_total if n_total else 0.0,
        threshold=0.0, direction="le", severity=Severity.CRITICAL,
        message=f"{n_parse_errors} files with malformed lines",
    )

    # --- OBB validity ------------------------------------------------
    obb_validity = (
        n_total_obbs / (n_total_obbs + n_invalid_obb)
        if (n_total_obbs + n_invalid_obb) > 0 else 1.0
    )
    try:
        node = threshold_lookup(thr, "stage2_labels.obb_validity_rate")
        bv_thr, bv_sev = node["threshold"], node.get("severity", "critical")
    except KeyError:
        bv_thr, bv_sev = 1.0, "critical"
    report.add_eval(
        "obb_validity_rate", value=obb_validity, threshold=bv_thr,
        direction="ge", severity=bv_sev,
        message=f"{n_total_obbs} valid / {n_invalid_obb} invalid",
    )

    # --- Class distribution ------------------------------------------
    cls_rows = []
    for cid, name in enumerate(class_names):
        cnt = all_class_counts.get(cid, 0)
        ratio = cnt / n_total_obbs if n_total_obbs else 0.0
        cls_rows.append({
            "class_id": cid, "name": name,
            "count": cnt, "ratio": f"{ratio:.1%}",
        })
    report.add_table(
        "Class distribution",
        rows=cls_rows,
        columns=["class_id", "name", "count", "ratio"],
        description=f"Total {n_total_obbs} valid OBBs across {n_total - n_empty} non-empty files.",
    )
    report.add_plot(
        "Class distribution chart",
        make_bar_chart(
            labels=class_names,
            values=[all_class_counts.get(i, 0) for i in range(n_classes)],
            title="OBB count per class", ylabel="count",
        ),
    )

    # --- ★ Roughness minimum count (D-017 trigger) -------------------
    try:
        cd_node = threshold_lookup(thr, "stage2_labels.class_distribution")
        rough_min = cd_node.get("Roughness_min_count")
        cd_sev = cd_node.get("severity", "warning")
    except KeyError:
        rough_min, cd_sev = 50, "warning"
    if rough_min is not None and "Roughness" in class_names:
        rough_id = class_names.index("Roughness")
        rough_count = all_class_counts.get(rough_id, 0)
        report.add_eval(
            "roughness_min_count", value=rough_count, threshold=rough_min,
            direction="ge", severity=cd_sev,
            message=("논문 152개 / 임계값 미달 시 D-017 synthetic_gen 검토"
                     if rough_count < rough_min else "OK"),
        )

    # --- Angle diversity ---------------------------------------------
    if angles:
        non_axis_ratio = 1.0 - (n_axis_aligned / len(angles))
        try:
            ang_node = threshold_lookup(thr, "stage2_labels.angle_diversity")
            ang_thr = ang_node.get("non_axis_aligned_min", 0.20)
            ang_sev = ang_node.get("severity", "warning")
        except KeyError:
            ang_thr, ang_sev = 0.20, "warning"
        report.add_eval(
            "non_axis_aligned_ratio", value=non_axis_ratio,
            threshold=ang_thr, direction="ge", severity=ang_sev,
            message=f"{len(angles) - n_axis_aligned}/{len(angles)} 회전 OBB "
                    f"(±{AXIS_ALIGNED_TOL_DEG}° 외)",
        )

        # Angle histogram
        bins = list(range(-90, 91, 15))
        hist, _ = np.histogram(angles, bins=bins)
        bin_labels = [f"{bins[i]}~{bins[i + 1]}" for i in range(len(bins) - 1)]
        report.add_plot(
            "OBB long-edge angle distribution",
            make_bar_chart(
                labels=bin_labels,
                values=[int(v) for v in hist],
                title="Angle histogram (degrees)",
                ylabel="count",
            ),
            description=f"Axis-aligned (±{AXIS_ALIGNED_TOL_DEG}°): {n_axis_aligned} / "
                        f"non-axis: {len(angles) - n_axis_aligned}",
        )

    # --- Small OBB ----------------------------------------------------
    small_rate = n_small / n_total_obbs if n_total_obbs else 0.0
    report.add_eval(
        "small_obb_rate", value=small_rate, threshold=0.05, direction="le",
        severity=Severity.WARNING,
        message=f"{n_small} OBBs with area < {SMALL_OBB_AREA_RATIO}",
    )

    # --- Label-image matching ----------------------------------------
    if image_map:
        labels_without_img = [s for s in label_stems if s not in image_map]
        images_without_lbl = [s for s in image_map if s not in label_stems]
        report.add_eval(
            "labels_without_images", value=len(labels_without_img),
            threshold=0, direction="le", severity=Severity.WARNING,
        )
        report.add_eval(
            "images_without_labels", value=len(images_without_lbl),
            threshold=0, direction="le", severity=Severity.WARNING,
            message="(may be intentional 'no objects' background)",
        )

    # --- Error tables -------------------------------------------------
    if parse_error_rows[:20]:
        report.add_table(
            "Parse errors (first 20)",
            rows=parse_error_rows[:20],
            columns=["file", "error"],
        )
    if obb_error_rows[:20]:
        report.add_table(
            "OBB errors (first 20)",
            rows=obb_error_rows[:20],
            columns=["file", "class_id", "issues"],
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V3-A — YOLO OBB 라벨 품질 검증.",
    )
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--cfg", type=Path,
                   default=Path("configs/yolo_obb.yaml"))
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Labels dir : %s", args.labels_dir)
    log.info("Images dir : %s", args.images_dir or "(none)")

    if not args.labels_dir.exists():
        log.error("labels-dir not found: %s", args.labels_dir)
        return 2

    report = run(args.labels_dir, args.images_dir, args.cfg, args.thresholds)
    paths = report.emit(
        reports_dir=args.reports_dir,
        stem=f"{args.labels_dir.parent.name}_{args.labels_dir.name}_obb_labels",
        use_color=not args.no_color,
    )
    log.info("HTML : %s", paths["html"])
    log.info("JSON : %s", paths["json"])
    return 0 if report.overall_status in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
