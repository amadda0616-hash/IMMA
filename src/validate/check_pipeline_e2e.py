"""
src/validate/check_pipeline_e2e.py

V7 — End-to-End Pipeline (Step 7) 사후 검증.

Inputs
------
- ``--predictions`` : ``pipeline.py batch`` 의 출력 폴더 (``<id>.json``)
- ``--gt``          : Ground truth 폴더 (stem 일치, HANDOFF §5.5 schema)
- ``--summary``     : 옵션, ``pipeline_summary.json`` (timing/error 집계)

Checks
------
- field_f1 (overall JSON, ≥ 0.75 critical)
- field_f1[title_block] / field_f1[notes]
- detection_metrics per class (★ D-023 누락률 재측정 — Measure < 0.08, GDT < 0.05)
- drawing-level recall ≥ 0.85
- numerical_content_f1 per class (Stage 3-N parsed JSON 매칭)
- per-drawing inference time (≤ 30s warning)
- per-stage timing breakdown (mean)
- failure_rate (from summary, < 0.01 critical)

CLI
---
::

    python -m src.validate.check_pipeline_e2e \\
        --predictions outputs/json/ \\
        --gt data/validation_gt/e2e/ \\
        --summary outputs/json/_pipeline_summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.metrics import (
    compare_titleblock, compare_notes,
    field_level_f1, hallucination_rate,
    detection_metrics, polygon_iou,
    compare_measure, compare_gdt, compare_roughness,
    pr_f1, safe_div,
)
from src.validate.common import (
    DEFAULT_REPORTS_DIR, DEFAULT_THRESHOLDS_PATH,
    Severity, Status, ValidationReport,
    load_thresholds, threshold_lookup,
    make_bar_chart, make_confusion_matrix,
    setup_logging,
)

log = setup_logging("validate.pipeline_e2e")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load %s: %s", path, e)
        return None


def discover(pred_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in pred_dir.glob("*.json"):
        if p.name.startswith("_") or p.name == "manifest.json":
            continue
        out[p.stem] = p
    return out


# ---------------------------------------------------------------------------
# Pair record
# ---------------------------------------------------------------------------
@dataclass
class E2EPair:
    image_stem: str
    pred: Dict[str, Any]
    gt: Dict[str, Any]

    overall_f1: Dict[str, Any] = dc_field(default_factory=dict)
    titleblock: Dict[str, Any] = dc_field(default_factory=dict)
    notes: Dict[str, Any] = dc_field(default_factory=dict)
    detection: Dict[str, Any] = dc_field(default_factory=dict)
    numerical_per_class: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)

    timing_total_s: Optional[float] = None
    timing_stages: Dict[str, float] = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------
def flatten_annotations(unified: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten ``views[].annotations`` to a single list with global coords."""
    out: List[Dict[str, Any]] = []
    for v in unified.get("views", []) or []:
        for a in v.get("annotations", []) or []:
            obb = a.get("obb_global") or a.get("obb")
            if not obb:
                continue
            out.append({
                "class": a.get("class"),
                "obb": obb,
                "conf": a.get("conf", 0.0),
                "parsed": a.get("parsed") or {},
            })
    return out


def evaluate_pair(pred: Dict[str, Any], gt: Dict[str, Any]) -> E2EPair:
    """Compute all per-pair metrics."""
    pair = E2EPair(image_stem=pred.get("drawing_id") or "?",
                   pred=pred, gt=gt)

    # Overall JSON F1 (excluding meta + per-annotation parsed details)
    pair.overall_f1 = field_level_f1(
        pred, gt, fuzzy_strings=True, abs_tol=0.01,
        skip_keys=("meta", "_review", "_meta", "type",
                   "raw_seq", "raw", "image_path", "source"),
    )

    # TitleBlock
    p_tb = pred.get("title_block") or {}
    g_tb = gt.get("title_block") or {}
    pair.titleblock = compare_titleblock(p_tb, g_tb, fuzzy=True)

    # Notes
    p_notes = pred.get("notes") or []
    g_notes = gt.get("notes") or []
    pair.notes = compare_notes(p_notes, g_notes, fuzzy=True, threshold=0.30)

    # Detection: flatten across views, match by IoU + class
    p_anns = flatten_annotations(pred)
    g_anns = flatten_annotations(gt)
    pair.detection = detection_metrics(
        p_anns, g_anns, iou_thr=0.5, iou_fn=polygon_iou,
    )

    # Numerical content F1: among matched annotations, compare parsed
    pair.numerical_per_class = _eval_numerical_content(p_anns, g_anns)

    # Timing
    timing = (pred.get("meta") or {}).get("timing_seconds") or {}
    pair.timing_stages = {
        k: float(v) for k, v in timing.items() if isinstance(v, (int, float))
    }
    pair.timing_total_s = pair.timing_stages.get("total")

    return pair


def _eval_numerical_content(p_anns: List[Dict[str, Any]],
                            g_anns: List[Dict[str, Any]]
                            ) -> Dict[str, Dict[str, Any]]:
    """For each (pred, gt) annotation match (greedy IoU + class), compare
    their ``parsed`` content per class.
    """
    # Re-do greedy matching here so we can capture matched pairs for parsed comparison
    used_g = set()
    matched: List[tuple] = []
    p_sorted = sorted(range(len(p_anns)), key=lambda i: -p_anns[i].get("conf", 0.0))
    for pi in p_sorted:
        p = p_anns[pi]
        best_iou, best_g = 0.0, -1
        for gi, g in enumerate(g_anns):
            if gi in used_g:
                continue
            if g.get("class") != p.get("class"):
                continue
            iou = polygon_iou(p["obb"], g["obb"])
            if iou > best_iou:
                best_iou, best_g = iou, gi
        if best_iou >= 0.5 and best_g >= 0:
            matched.append((pi, best_g, p["class"]))
            used_g.add(best_g)

    # Compare parsed content per class
    by_class: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp_field": 0, "fp_field": 0, "fn_field": 0, "n_pairs": 0,
                  "nominal_correct": 0, "tolerance_correct": 0,
                  "symbol_correct": 0, "datum_correct": 0,
                  "Ra_correct": 0},
    )

    for pi, gi, cls in matched:
        p_parsed = p_anns[pi]["parsed"]
        g_parsed = g_anns[gi]["parsed"]
        if not p_parsed and not g_parsed:
            continue
        if cls == "Measure":
            r = compare_measure(p_parsed, g_parsed)
            if r.get("nominal_correct"):
                by_class[cls]["nominal_correct"] += 1
            if r.get("tolerance_correct"):
                by_class[cls]["tolerance_correct"] += 1
        elif cls == "GDT":
            r = compare_gdt(p_parsed, g_parsed)
            if r.get("symbol_correct"):
                by_class[cls]["symbol_correct"] += 1
            if r.get("datum_correct"):
                by_class[cls]["datum_correct"] += 1
        elif cls == "Roughness":
            r = compare_roughness(p_parsed, g_parsed)
            if r.get("Ra_correct"):
                by_class[cls]["Ra_correct"] += 1
        else:
            r = field_level_f1(p_parsed, g_parsed)

        by_class[cls]["tp_field"] += r.get("tp", 0)
        by_class[cls]["fp_field"] += r.get("fp", 0)
        by_class[cls]["fn_field"] += r.get("fn", 0)
        by_class[cls]["n_pairs"] += 1

    # Compute F1 per class
    out: Dict[str, Dict[str, Any]] = {}
    for cls, c in by_class.items():
        f = pr_f1(c["tp_field"], c["fp_field"], c["fn_field"])
        out[cls] = {
            "n_pairs": c["n_pairs"],
            "f1": f["f1"], "precision": f["precision"], "recall": f["recall"],
        }
        if cls == "Measure" and c["n_pairs"] > 0:
            out[cls]["nominal_accuracy"] = round(
                c["nominal_correct"] / c["n_pairs"], 4)
            out[cls]["tolerance_match"] = round(
                c["tolerance_correct"] / c["n_pairs"], 4)
        elif cls == "GDT" and c["n_pairs"] > 0:
            out[cls]["symbol_accuracy"] = round(
                c["symbol_correct"] / c["n_pairs"], 4)
            out[cls]["datum_accuracy"] = round(
                c["datum_correct"] / c["n_pairs"], 4)
        elif cls == "Roughness" and c["n_pairs"] > 0:
            out[cls]["Ra_accuracy"] = round(
                c["Ra_correct"] / c["n_pairs"], 4)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(pairs: List[E2EPair]) -> Dict[str, Any]:
    if not pairs:
        return {}

    # Overall JSON F1 (mean of per-pair F1)
    overall_f1s = [p.overall_f1.get("f1", 0.0) for p in pairs]
    tb_f1s     = [p.titleblock.get("f1", 0.0)  for p in pairs]
    notes_f1s  = [p.notes.get("f1", 0.0)       for p in pairs]

    # Detection: aggregate by summing TP/FP/FN per class
    det_by_class: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0},
    )
    drawing_recalls: List[float] = []
    for p in pairs:
        det = p.detection.get("per_class", {})
        for cls, m in det.items():
            det_by_class[cls]["tp"] += m.get("tp", 0)
            det_by_class[cls]["fp"] += m.get("fp", 0)
            det_by_class[cls]["fn"] += m.get("fn", 0)
        ov = p.detection.get("overall", {})
        gt_total = ov.get("tp", 0) + ov.get("fn", 0)
        if gt_total > 0:
            drawing_recalls.append(ov.get("tp", 0) / gt_total)

    det_summary: Dict[str, Dict[str, Any]] = {}
    total_det = {"tp": 0, "fp": 0, "fn": 0}
    for cls, c in det_by_class.items():
        f = pr_f1(c["tp"], c["fp"], c["fn"])
        f["missing_rate"] = round(safe_div(c["fn"], c["tp"] + c["fn"]), 4)
        det_summary[cls] = f
        for k in total_det:
            total_det[k] += c[k]

    # Numerical content F1: aggregate by class
    num_by_class: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"f1_sum": 0.0, "n": 0,
                 "nominal_correct": 0, "tolerance_correct": 0,
                 "symbol_correct": 0, "datum_correct": 0,
                 "Ra_correct": 0,
                 "n_measure": 0, "n_gdt": 0, "n_rough": 0},
    )
    for p in pairs:
        for cls, m in p.numerical_per_class.items():
            num_by_class[cls]["f1_sum"] += m.get("f1", 0.0)
            num_by_class[cls]["n"] += 1
            n_pairs = m.get("n_pairs", 0)
            if cls == "Measure" and n_pairs > 0:
                num_by_class[cls]["nominal_correct"] += int(
                    m.get("nominal_accuracy", 0.0) * n_pairs)
                num_by_class[cls]["tolerance_correct"] += int(
                    m.get("tolerance_match", 0.0) * n_pairs)
                num_by_class[cls]["n_measure"] += n_pairs
            elif cls == "GDT" and n_pairs > 0:
                num_by_class[cls]["symbol_correct"] += int(
                    m.get("symbol_accuracy", 0.0) * n_pairs)
                num_by_class[cls]["datum_correct"] += int(
                    m.get("datum_accuracy", 0.0) * n_pairs)
                num_by_class[cls]["n_gdt"] += n_pairs
            elif cls == "Roughness" and n_pairs > 0:
                num_by_class[cls]["Ra_correct"] += int(
                    m.get("Ra_accuracy", 0.0) * n_pairs)
                num_by_class[cls]["n_rough"] += n_pairs

    # Timing
    totals = [p.timing_total_s for p in pairs if p.timing_total_s is not None]
    timing_summary: Dict[str, Any] = {}
    if totals:
        timing_summary["total"] = {
            "mean": round(statistics.mean(totals), 3),
            "median": round(statistics.median(totals), 3),
            "p95": round(_percentile(totals, 95), 3),
            "max": round(max(totals), 3),
            "n": len(totals),
        }
    # Per-stage means
    stage_keys = set()
    for p in pairs:
        stage_keys.update(p.timing_stages.keys())
    stage_keys.discard("total")
    for key in sorted(stage_keys):
        vals = [p.timing_stages[key] for p in pairs if key in p.timing_stages]
        if vals:
            timing_summary[key] = {
                "mean": round(statistics.mean(vals), 3),
                "max": round(max(vals), 3),
                "n": len(vals),
            }

    return {
        "n_pairs": len(pairs),
        "overall_f1": {
            "mean":   round(statistics.mean(overall_f1s), 4) if overall_f1s else 0.0,
            "median": round(statistics.median(overall_f1s), 4) if overall_f1s else 0.0,
        },
        "titleblock_f1_mean": round(statistics.mean(tb_f1s), 4) if tb_f1s else 0.0,
        "notes_f1_mean":      round(statistics.mean(notes_f1s), 4) if notes_f1s else 0.0,
        "detection_per_class": det_summary,
        "detection_overall":   pr_f1(total_det["tp"], total_det["fp"], total_det["fn"]),
        "drawing_level_recall": round(
            statistics.mean(drawing_recalls), 4) if drawing_recalls else 0.0,
        "numerical_per_class": dict(num_by_class),
        "timing": timing_summary,
    }


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    d = sorted(data)
    k = (len(d) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(d) - 1)
    return d[f] + (d[c] - d[f]) * (k - f)


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run(predictions_dir: Path,
        gt_dir: Path,
        thresholds_path: Path,
        summary_path: Optional[Path] = None) -> ValidationReport:
    thr = load_thresholds(thresholds_path)
    pred_map = discover(predictions_dir)
    gt_map   = discover(gt_dir)

    summary = load_json_safe(summary_path) if summary_path else None

    report = ValidationReport(
        title="V7 — End-to-End Pipeline (Step 7) Validation",
        step="pipeline_e2e",
        metadata={
            "predictions_dir": str(predictions_dir),
            "gt_dir": str(gt_dir),
            "n_pred": len(pred_map),
            "n_gt": len(gt_map),
            "summary_used": bool(summary),
        },
    )

    if not pred_map or not gt_map:
        report.add_eval(
            "data_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message=f"pred={len(pred_map)} gt={len(gt_map)}",
        )
        return report

    # ---- Per-pair eval ----------------------------------------------
    pairs: List[E2EPair] = []
    unmatched: List[str] = []
    parse_errors: List[str] = []
    for stem, ppath in pred_map.items():
        if stem not in gt_map:
            unmatched.append(stem)
            continue
        pred = load_json_safe(ppath)
        gt   = load_json_safe(gt_map[stem])
        if pred is None or gt is None:
            parse_errors.append(stem)
            continue
        pair = evaluate_pair(pred, gt)
        pair.image_stem = stem
        pairs.append(pair)

    log.info("Evaluated %d pairs (unmatched=%d errors=%d)",
             len(pairs), len(unmatched), len(parse_errors))

    if not pairs:
        report.add_eval(
            "pairs_evaluated", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message="No matching (pred, gt) pairs",
        )
        return report

    agg = aggregate(pairs)

    # ---- field_f1 (overall) -----------------------------------------
    try:
        f_node = threshold_lookup(thr, "pipeline_e2e.field_f1_min")
        f_thr, f_sev = f_node["threshold"], f_node.get("severity", "critical")
    except KeyError:
        f_thr, f_sev = 0.75, "critical"
    report.add_eval(
        "field_f1 (overall, mean)",
        value=agg["overall_f1"]["mean"], threshold=f_thr,
        direction="ge", severity=f_sev,
        message=f"n={agg['n_pairs']} | median={agg['overall_f1']['median']}",
    )
    report.add_eval(
        "field_f1[title_block] (mean)",
        value=agg["titleblock_f1_mean"], threshold=None, direction="none",
        severity=Severity.INFO,
    )
    report.add_eval(
        "field_f1[notes] (mean)",
        value=agg["notes_f1_mean"], threshold=None, direction="none",
        severity=Severity.INFO,
    )

    # ---- ★ Detection per-class (D-023 재검증) -----------------------
    try:
        miss_node = threshold_lookup(thr, "stage2_model.missing_rate_max")
    except KeyError:
        miss_node = {}
    det_per_class = agg["detection_per_class"]
    for cls in ("Measure", "GDT", "Roughness"):
        if cls not in det_per_class:
            continue
        m = det_per_class[cls]
        spec = miss_node.get(cls) if isinstance(miss_node, dict) else None
        if spec:
            report.add_eval(
                f"missing_rate[{cls}] (★ e2e)",
                value=m["missing_rate"],
                threshold=spec["threshold"],
                direction="le",
                severity=spec.get("severity", "warning"),
                message=f"FN={m['fn']} TP+FN={m['tp'] + m['fn']}",
            )
        else:
            report.add_eval(
                f"missing_rate[{cls}]", value=m["missing_rate"],
                threshold=None, direction="none", severity=Severity.INFO,
            )
        report.add_eval(
            f"detection_f1[{cls}]", value=m["f1"],
            threshold=None, direction="none", severity=Severity.INFO,
        )

    # ---- Drawing-level recall ---------------------------------------
    try:
        dr_node = threshold_lookup(thr, "stage2_model.drawing_recall_min")
        dr_thr, dr_sev = dr_node["threshold"], dr_node.get("severity", "critical")
    except KeyError:
        dr_thr, dr_sev = 0.85, "critical"
    report.add_eval(
        "drawing_level_recall (★ e2e)",
        value=agg["drawing_level_recall"], threshold=dr_thr,
        direction="ge", severity=dr_sev,
        message="mean per-image recall (e2e)",
    )

    # ---- Numerical content per class --------------------------------
    num_pc = agg["numerical_per_class"]
    for cls, c in num_pc.items():
        if c["n"] == 0:
            continue
        f1 = round(c["f1_sum"] / c["n"], 4)
        report.add_eval(
            f"numerical_content_f1[{cls}]",
            value=f1, threshold=None, direction="none", severity=Severity.INFO,
            message=f"n_pairs across {c['n']} drawings",
        )
        if cls == "Measure" and c["n_measure"]:
            report.add_eval(
                "numerical_accuracy (e2e)",
                value=round(c["nominal_correct"] / c["n_measure"], 4),
                threshold=0.95, direction="ge", severity=Severity.WARNING,
                message=f"matched {c['n_measure']} measures",
            )

    # ---- Timing ----------------------------------------------------
    timing = agg.get("timing", {})
    if timing.get("total"):
        try:
            t_node = threshold_lookup(thr, "pipeline_e2e.inference_time_per_drawing_max_s")
            t_thr, t_sev = t_node["threshold"], t_node.get("severity", "warning")
        except KeyError:
            t_thr, t_sev = 30, "warning"
        report.add_eval(
            "inference_time_per_drawing (mean)",
            value=timing["total"]["mean"], threshold=t_thr,
            direction="le", severity=t_sev,
            message=f"median={timing['total']['median']} "
                    f"p95={timing['total']['p95']} max={timing['total']['max']}",
        )

        # Per-stage timing table
        stage_rows = []
        for stage_key, stats_v in timing.items():
            if stage_key == "total":
                continue
            stage_rows.append({
                "stage": stage_key,
                "mean_s": stats_v["mean"],
                "max_s": stats_v["max"],
                "n": stats_v["n"],
            })
        if stage_rows:
            report.add_table(
                "Per-stage timing (mean / max seconds)",
                rows=stage_rows,
                columns=["stage", "mean_s", "max_s", "n"],
            )
            # Bar chart
            report.add_plot(
                "Mean time per stage (seconds)",
                make_bar_chart(
                    labels=[r["stage"] for r in stage_rows],
                    values=[r["mean_s"] for r in stage_rows],
                    title="Pipeline stage timing (mean)",
                    ylabel="seconds",
                ),
            )

    # ---- Failure rate (from summary, optional) ----------------------
    if summary:
        n_total = summary.get("n_total", 0)
        n_err   = summary.get("n_err", 0)
        rate = round(safe_div(n_err, max(1, n_total)), 4)
        try:
            fr_node = threshold_lookup(thr, "pipeline_e2e.failure_rate_max")
            fr_thr, fr_sev = fr_node["threshold"], fr_node.get("severity", "critical")
        except KeyError:
            fr_thr, fr_sev = 0.01, "critical"
        report.add_eval(
            "failure_rate", value=rate,
            threshold=fr_thr, direction="le", severity=fr_sev,
            message=f"errors={n_err} / total={n_total}",
        )

    # ---- Per-class detection summary table --------------------------
    det_rows = []
    for cls, m in det_per_class.items():
        det_rows.append({
            "class": cls,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "missing_rate": m["missing_rate"],
        })
    if det_rows:
        report.add_table(
            "Detection metrics per class (★ e2e)",
            rows=det_rows,
            columns=["class", "tp", "fp", "fn", "f1",
                     "precision", "recall", "missing_rate"],
            description="Aggregated across all drawings (sum TP/FP/FN, recompute F1).",
        )
        report.add_plot(
            "Detection F1 per class (e2e)",
            make_bar_chart(
                labels=[r["class"] for r in det_rows],
                values=[r["f1"] for r in det_rows],
                title="End-to-end detection F1",
                ylabel="F1", ylim=(0.0, 1.0),
            ),
        )

    # ---- Per-pair table (worst 10 by overall F1) --------------------
    worst = sorted(pairs, key=lambda p: p.overall_f1.get("f1", 0.0))[:10]
    if worst:
        report.add_table(
            "Worst 10 drawings (by overall F1)",
            rows=[
                {"image_stem": p.image_stem,
                 "overall_f1": p.overall_f1.get("f1"),
                 "tb_f1":      p.titleblock.get("f1"),
                 "notes_f1":   p.notes.get("f1"),
                 "total_s":    p.timing_total_s}
                for p in worst
            ],
            columns=["image_stem", "overall_f1", "tb_f1",
                     "notes_f1", "total_s"],
        )

    # ---- Summary diagnostics ----------------------------------------
    if unmatched:
        report.add_eval(
            "unmatched_predictions", value=len(unmatched),
            threshold=0, direction="le", severity=Severity.WARNING,
        )
    if parse_errors:
        report.add_eval(
            "parse_errors", value=len(parse_errors),
            threshold=0, direction="le", severity=Severity.WARNING,
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V7 — End-to-end pipeline (Step 7) validation."
    )
    p.add_argument("--predictions", type=Path, required=True,
                   help="pipeline.py batch 출력 폴더")
    p.add_argument("--gt", type=Path, required=True,
                   help="GT 폴더 (HANDOFF §5.5 schema)")
    p.add_argument("--summary", type=Path, default=None,
                   help="옵션, _pipeline_summary.json")
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Predictions : %s", args.predictions)
    log.info("Ground truth: %s", args.gt)
    log.info("Summary     : %s", args.summary or "(none)")

    if not args.predictions.exists():
        log.error("predictions dir not found: %s", args.predictions)
        return 2
    if not args.gt.exists():
        log.error("gt dir not found: %s", args.gt)
        return 2

    report = run(args.predictions, args.gt, args.thresholds, args.summary)
    paths = report.emit(reports_dir=args.reports_dir,
                        use_color=not args.no_color)
    log.info("HTML : %s", paths["html"])
    log.info("JSON : %s", paths["json"])
    return 0 if report.overall_status in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
