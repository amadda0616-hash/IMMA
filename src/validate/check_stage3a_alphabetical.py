"""
src/validate/check_stage3a_alphabetical.py

V5 — Stage 3-A (Donut Alphabetical, zero-shot) 사후 검증

논문 베이스라인 (Khan 2025 §4.3):
- TitleBlock F1   : 0.533
- Notes F1        : 0.810   ← critical
- Overall F1      : 0.672
- Hallucination   : 0.40 ~ 0.48 (TB 0.478, Notes 0.319)

Inputs
------
- ``--predictions`` : Stage 3-A 출력 폴더 (``*.alpha.json`` 또는 ``*.json``)
- ``--gt``          : Ground truth 폴더 (stem 일치, type/fields/items 포함)

GT format (stem 일치하는 ``<stem>.json``)
-----------------------------------------
TitleBlock::

    {"type": "TitleBlock",
     "fields": {"drawing_no": "DWG-001-A", "material": "SS400", ...}}

Notes::

    {"type": "Notes", "items": ["1. ...", "2. ..."]}

Checks (validation_thresholds.yaml#stage3a)
-------------------------------------------
- field_f1_min[titleblock] ≥ 0.50  warning
- field_f1_min[notes]      ≥ 0.75  critical (논문 0.810)
- field_f1_min[overall]    ≥ 0.50  warning
- hallucination_rate_max   < 0.50  warning
- empty_response_rate_max  < 0.10  warning
- per_language_f1_gap_max  ≤ 0.30  info
- edit_distance_avg_max    ≤ 5     info

CLI
---
::

    python -m src.validate.check_stage3a_alphabetical \\
        --predictions outputs/sample/alphabetical/ \\
        --gt data/validation_gt/stage3a/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.metrics import (
    compare_titleblock, compare_notes,
    hallucination_rate, normalized_edit_distance, normalize_text,
    safe_div, pr_f1,
)
from src.validate.common import (
    DEFAULT_REPORTS_DIR, DEFAULT_THRESHOLDS_PATH,
    Severity, Status, ValidationReport,
    load_thresholds, threshold_lookup,
    make_bar_chart, make_confusion_matrix,
    setup_logging,
)

log = setup_logging("validate.stage3a")

# ---------------------------------------------------------------------------
# Filename language heuristic (V1 와 같은 룰)
# ---------------------------------------------------------------------------
def detect_language(filename: str) -> str:
    has_hangul = any(0xAC00 <= ord(c) <= 0xD7A3 for c in filename)
    has_kana   = any(0x3040 <= ord(c) <= 0x30FF for c in filename) \
              or any(0x4E00 <= ord(c) <= 0x9FFF for c in filename)
    has_cyril  = any(0x0400 <= ord(c) <= 0x04FF for c in filename)
    if has_hangul: return "ko"
    if has_kana:   return "ja"
    if has_cyril:  return "ru"
    return "en"


# ---------------------------------------------------------------------------
# Pair record
# ---------------------------------------------------------------------------
@dataclass
class PairResult:
    image_stem: str
    region_type: str          # "titleblock" / "notes"
    language: str             # language_hint or filename heuristic
    pred: Dict[str, Any]
    gt: Dict[str, Any]
    metrics: Dict[str, Any]   # field_level_f1 / compare_notes 결과
    hallucination: Dict[str, Any]
    edit_dist_avg: float
    is_empty_pred: bool


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


def discover_predictions(pred_dir: Path) -> Dict[str, Path]:
    """Return ``{stem: path}`` for ``*.alpha.json`` and ``*.json`` (excluding manifest)."""
    out: Dict[str, Path] = {}
    for ext in ("*.alpha.json", "*.json"):
        for p in pred_dir.rglob(ext):
            if p.name in ("manifest.json", "manifest.csv"):
                continue
            stem = p.stem
            if stem.endswith(".alpha"):
                stem = stem[:-len(".alpha")]
            if stem not in out:
                out[stem] = p
    return out


def discover_gt(gt_dir: Path) -> Dict[str, Path]:
    """Return ``{stem: path}`` for GT JSONs."""
    return {p.stem: p for p in gt_dir.rglob("*.json")
            if p.name not in ("manifest.json",)}


# ---------------------------------------------------------------------------
# Per-pair evaluation
# ---------------------------------------------------------------------------
def evaluate_titleblock(pred: Dict[str, Any], gt: Dict[str, Any]
                        ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Evaluate TitleBlock prediction. Returns (metrics, hallucination, avg_edit)."""
    pred_fields = pred.get("fields") or {}
    gt_fields   = gt.get("fields") or {}

    metrics = compare_titleblock(pred_fields, gt_fields, fuzzy=True)

    allowed = set(gt_fields.keys()) if gt_fields else None
    hall = hallucination_rate(pred_fields, gt_fields,
                              allowed_keys=allowed)

    # Per-field edit distance avg
    edits: List[float] = []
    for k, gv in gt_fields.items():
        pv = pred_fields.get(k)
        if pv is None or gv is None:
            continue
        edits.append(normalized_edit_distance(str(pv), str(gv)))
    avg_edit = round(sum(edits) / len(edits), 4) if edits else 0.0

    return metrics, hall, avg_edit


def evaluate_notes(pred: Dict[str, Any], gt: Dict[str, Any]
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Evaluate Notes prediction."""
    pred_items = pred.get("items") or []
    gt_items   = gt.get("items") or []

    metrics = compare_notes(pred_items, gt_items, fuzzy=True, threshold=0.30)

    # Hallucination: items in pred not matched to any gt item
    hall_count = max(0, len(pred_items) - metrics["tp"])
    hall = {
        "rate": round(safe_div(hall_count, max(1, len(pred_items))), 4),
        "n_pred_fields": len(pred_items),
        "n_hallucinations": hall_count,
        "schema_violations": [],
        "extra_fields": [],
        "value_mismatches": [],
    }

    # Average edit distance over greedy matches
    edits: List[float] = []
    used_gt: set = set()
    for p in pred_items:
        best_d, best_j = float("inf"), -1
        for j, g in enumerate(gt_items):
            if j in used_gt:
                continue
            d = normalized_edit_distance(p, g)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            edits.append(best_d)
            used_gt.add(best_j)
    avg_edit = round(sum(edits) / len(edits), 4) if edits else 0.0

    return metrics, hall, avg_edit


def is_empty_pred(pred: Dict[str, Any], region_type: str) -> bool:
    if region_type == "titleblock":
        fields = pred.get("fields") or {}
        return not any(v for v in fields.values())
    if region_type == "notes":
        return not (pred.get("items") or [])
    return True


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_overall(pairs: List[PairResult]) -> Dict[str, Any]:
    """Aggregate by region_type + overall (weighted by tp+fn)."""
    by_region: Dict[str, List[PairResult]] = defaultdict(list)
    for p in pairs:
        by_region[p.region_type].append(p)

    region_summaries: Dict[str, Dict[str, float]] = {}
    total_tp = total_fp = total_fn = 0
    total_hall_count = 0
    total_pred_fields = 0
    total_empty = 0
    edit_sum = 0.0
    edit_n = 0

    for region, ps in by_region.items():
        tp = sum(p.metrics.get("tp", 0) for p in ps)
        fp = sum(p.metrics.get("fp", 0) for p in ps)
        fn = sum(p.metrics.get("fn", 0) for p in ps)
        f = pr_f1(tp, fp, fn)
        h_count = sum(p.hallucination["n_hallucinations"] for p in ps)
        h_total = sum(p.hallucination["n_pred_fields"]   for p in ps)
        h_rate = round(safe_div(h_count, max(1, h_total)), 4)
        empty = sum(1 for p in ps if p.is_empty_pred)
        empty_rate = round(safe_div(empty, len(ps)), 4)
        avg_e = round(
            safe_div(sum(p.edit_dist_avg for p in ps), len(ps)), 4,
        )
        region_summaries[region] = {
            "n_pairs": len(ps),
            "f1": f["f1"],
            "precision": f["precision"],
            "recall": f["recall"],
            "tp": tp, "fp": fp, "fn": fn,
            "hallucination_rate": h_rate,
            "empty_response_rate": empty_rate,
            "edit_distance_avg": avg_e,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_hall_count += h_count
        total_pred_fields += h_total
        total_empty += empty
        edit_sum += sum(p.edit_dist_avg for p in ps)
        edit_n += len(ps)

    overall = pr_f1(total_tp, total_fp, total_fn)
    overall["hallucination_rate"] = round(
        safe_div(total_hall_count, max(1, total_pred_fields)), 4,
    )
    overall["empty_response_rate"] = round(
        safe_div(total_empty, max(1, len(pairs))), 4,
    )
    overall["edit_distance_avg"] = round(safe_div(edit_sum, edit_n), 4)
    overall["n_pairs"] = len(pairs)

    return {"by_region": region_summaries, "overall": overall}


def aggregate_by_language(pairs: List[PairResult]) -> Dict[str, Dict[str, float]]:
    by_lang: Dict[str, List[PairResult]] = defaultdict(list)
    for p in pairs:
        by_lang[p.language].append(p)
    out: Dict[str, Dict[str, float]] = {}
    for lang, ps in sorted(by_lang.items()):
        tp = sum(p.metrics.get("tp", 0) for p in ps)
        fp = sum(p.metrics.get("fp", 0) for p in ps)
        fn = sum(p.metrics.get("fn", 0) for p in ps)
        f = pr_f1(tp, fp, fn)
        out[lang] = {"n_pairs": len(ps), **f}
    return out


def field_level_breakdown(pairs: List[PairResult]) -> List[Dict[str, Any]]:
    """For TB pairs only — per-field hit rate."""
    counter: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"hits": 0, "total": 0, "missed": 0, "extra": 0},
    )
    for p in pairs:
        if p.region_type != "titleblock":
            continue
        gt_fields   = p.gt.get("fields")   or {}
        pred_fields = p.pred.get("fields") or {}
        for k, gv in gt_fields.items():
            if not gv:
                continue
            counter[k]["total"] += 1
            pv = pred_fields.get(k)
            if pv and normalize_text(str(pv)) == normalize_text(str(gv)):
                counter[k]["hits"] += 1
            elif not pv:
                counter[k]["missed"] += 1
        for k in pred_fields:
            if k not in gt_fields and pred_fields[k]:
                counter[k]["extra"] += 1

    rows: List[Dict[str, Any]] = []
    for k, c in sorted(counter.items(),
                        key=lambda x: -x[1]["total"]):
        hit_rate = safe_div(c["hits"], c["total"])
        rows.append({
            "field": k,
            "total_gt": c["total"],
            "hits": c["hits"],
            "missed": c["missed"],
            "extra_pred": c["extra"],
            "hit_rate": f"{hit_rate:.4f}",
        })
    return rows


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run(predictions_dir: Path,
        gt_dir: Path,
        thresholds_path: Path) -> ValidationReport:
    thr = load_thresholds(thresholds_path)

    pred_map = discover_predictions(predictions_dir)
    gt_map   = discover_gt(gt_dir)

    report = ValidationReport(
        title="V5 — Stage 3-A (Donut Alphabetical zero-shot) Validation",
        step="stage3a",
        metadata={
            "predictions_dir": str(predictions_dir),
            "gt_dir": str(gt_dir),
            "n_pred": len(pred_map),
            "n_gt":   len(gt_map),
        },
    )

    if not pred_map:
        report.add_eval(
            "predictions_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message=f"No prediction files under {predictions_dir}",
        )
        return report
    if not gt_map:
        report.add_eval(
            "gt_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message=f"No GT files under {gt_dir}",
        )
        return report

    # ---- Per-pair eval ----------------------------------------------
    pairs: List[PairResult] = []
    unmatched_pred: List[str] = []
    parse_errors: List[str] = []

    for stem, ppath in pred_map.items():
        if stem not in gt_map:
            unmatched_pred.append(stem)
            continue
        pred = load_json_safe(ppath)
        gt   = load_json_safe(gt_map[stem])
        if pred is None or gt is None:
            parse_errors.append(stem)
            continue

        region = (gt.get("type") or pred.get("type") or "").lower()
        region = "titleblock" if region == "titleblock" else (
            "notes" if region == "notes" else region
        )
        if region not in ("titleblock", "notes"):
            parse_errors.append(f"{stem} (unknown type {region!r})")
            continue

        if region == "titleblock":
            metrics, hall, avg_e = evaluate_titleblock(pred, gt)
        else:
            metrics, hall, avg_e = evaluate_notes(pred, gt)

        lang = (pred.get("language_hint")
                or detect_language(stem))
        pairs.append(PairResult(
            image_stem=stem, region_type=region, language=lang,
            pred=pred, gt=gt,
            metrics=metrics, hallucination=hall,
            edit_dist_avg=avg_e,
            is_empty_pred=is_empty_pred(pred, region),
        ))

    log.info("Evaluated %d pairs (unmatched preds: %d, parse errors: %d)",
             len(pairs), len(unmatched_pred), len(parse_errors))

    if not pairs:
        report.add_eval(
            "pairs_evaluated", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message="No matching (pred, gt) pairs",
        )
        return report

    agg = aggregate_overall(pairs)
    by_lang = aggregate_by_language(pairs)
    field_rows = field_level_breakdown(pairs)

    # ---- Field F1 thresholds ----------------------------------------
    try:
        ff_node = threshold_lookup(thr, "stage3a.field_f1_min")
    except KeyError:
        ff_node = {}

    overall = agg["overall"]
    by_region = agg["by_region"]

    # Notes critical (논문 0.810 → 0.75)
    if "notes" in by_region:
        spec = ff_node.get("notes", {})
        report.add_eval(
            "field_f1[notes]", value=by_region["notes"]["f1"],
            threshold=spec.get("threshold", 0.75),
            direction="ge",
            severity=spec.get("severity", "critical"),
            message=f"n={by_region['notes']['n_pairs']} | 논문 0.810",
        )
    # TitleBlock warning
    if "titleblock" in by_region:
        spec = ff_node.get("titleblock", {})
        report.add_eval(
            "field_f1[titleblock]", value=by_region["titleblock"]["f1"],
            threshold=spec.get("threshold", 0.50),
            direction="ge",
            severity=spec.get("severity", "warning"),
            message=f"n={by_region['titleblock']['n_pairs']} | 논문 0.533",
        )
    # Overall
    spec = ff_node.get("overall", {})
    report.add_eval(
        "field_f1[overall]", value=overall["f1"],
        threshold=spec.get("threshold", 0.50),
        direction="ge",
        severity=spec.get("severity", "warning"),
        message=f"n={overall['n_pairs']} | 논문 0.672",
    )

    # ---- Hallucination ----------------------------------------------
    try:
        h_node = threshold_lookup(thr, "stage3a.hallucination_rate_max")
        h_thr, h_sev = h_node["threshold"], h_node.get("severity", "warning")
    except KeyError:
        h_thr, h_sev = 0.50, "warning"
    report.add_eval(
        "hallucination_rate (overall)",
        value=overall["hallucination_rate"],
        threshold=h_thr, direction="le", severity=h_sev,
        message="논문 TB=0.478, Notes=0.319",
    )
    if "titleblock" in by_region:
        report.add_eval(
            "hallucination_rate[titleblock]",
            value=by_region["titleblock"]["hallucination_rate"],
            threshold=None, direction="none",
            severity=Severity.INFO,
        )
    if "notes" in by_region:
        report.add_eval(
            "hallucination_rate[notes]",
            value=by_region["notes"]["hallucination_rate"],
            threshold=None, direction="none",
            severity=Severity.INFO,
        )

    # ---- Empty response ---------------------------------------------
    try:
        e_node = threshold_lookup(thr, "stage3a.empty_response_rate_max")
        e_thr, e_sev = e_node["threshold"], e_node.get("severity", "warning")
    except KeyError:
        e_thr, e_sev = 0.10, "warning"
    report.add_eval(
        "empty_response_rate", value=overall["empty_response_rate"],
        threshold=e_thr, direction="le", severity=e_sev,
        message="응답 자체가 비어있는 경우",
    )

    # ---- Edit distance avg ------------------------------------------
    try:
        ed_node = threshold_lookup(thr, "stage3a.edit_distance_avg_max")
        ed_thr  = ed_node["threshold"]
    except KeyError:
        ed_thr = 5
    report.add_eval(
        "edit_distance_avg", value=overall["edit_distance_avg"],
        threshold=ed_thr, direction="le", severity=Severity.INFO,
        message="정규화 edit distance (0=동일, 1=완전히 다름)",
    )

    # ---- Per-language gap -------------------------------------------
    if by_lang:
        lang_f1s = {l: v["f1"] for l, v in by_lang.items()}
        max_f1 = max(lang_f1s.values())
        min_f1 = min(lang_f1s.values())
        gap = round(max_f1 - min_f1, 4)
        try:
            g_node = threshold_lookup(thr, "stage3a.per_language_f1_gap_max")
            g_thr  = g_node["threshold"]
        except KeyError:
            g_thr = 0.30
        report.add_eval(
            "per_language_f1_gap", value=gap,
            threshold=g_thr, direction="le", severity=Severity.INFO,
            message=f"max={max_f1:.4f} - min={min_f1:.4f}",
        )

        report.add_table(
            "Per-language F1",
            rows=[
                {"language": l, "n": v["n_pairs"],
                 "f1": v["f1"], "precision": v["precision"],
                 "recall": v["recall"]}
                for l, v in by_lang.items()
            ],
            columns=["language", "n", "f1", "precision", "recall"],
            description="Language detected from prediction's language_hint "
                        "or filename heuristic.",
        )
        report.add_plot(
            "F1 per language",
            make_bar_chart(
                labels=list(by_lang.keys()),
                values=[v["f1"] for v in by_lang.values()],
                title="Stage 3-A F1 by language",
                ylabel="F1",
                ylim=(0.0, 1.0),
            ),
        )

    # ---- Per-region table -------------------------------------------
    region_rows = [
        {"region": k,
         "n": v["n_pairs"],
         "f1": v["f1"],
         "precision": v["precision"],
         "recall": v["recall"],
         "hall_rate": v["hallucination_rate"],
         "empty_rate": v["empty_response_rate"],
         "edit_avg": v["edit_distance_avg"]}
        for k, v in by_region.items()
    ]
    report.add_table(
        "Per-region summary",
        rows=region_rows,
        columns=["region", "n", "f1", "precision", "recall",
                 "hall_rate", "empty_rate", "edit_avg"],
    )

    # ---- TB field-level breakdown -----------------------------------
    if field_rows:
        report.add_table(
            "TitleBlock field-level breakdown (top 15)",
            rows=field_rows[:15],
            columns=["field", "total_gt", "hits", "missed",
                     "extra_pred", "hit_rate"],
            description="hit_rate = exact match (case+whitespace 정규화) / total_gt",
        )
        # Hit rate plot
        report.add_plot(
            "Field-level hit rate (top 10 by GT count)",
            make_bar_chart(
                labels=[r["field"] for r in field_rows[:10]],
                values=[float(r["hit_rate"]) for r in field_rows[:10]],
                title="Per-field exact hit rate",
                ylabel="hit rate", ylim=(0.0, 1.0),
                horizontal=True,
            ),
        )

    # ---- Unmatched / parse errors -----------------------------------
    if unmatched_pred:
        report.add_eval(
            "unmatched_predictions", value=len(unmatched_pred),
            threshold=0, direction="le", severity=Severity.WARNING,
            message=f"GT 없는 예측 {len(unmatched_pred)}개 (sample 검수만 했을 수 있음)",
        )
    if parse_errors:
        report.add_eval(
            "parse_errors", value=len(parse_errors),
            threshold=0, direction="le", severity=Severity.WARNING,
            message="JSON 파싱 / type 인식 실패",
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V5 — Stage 3-A (Donut Alphabetical zero-shot) validation."
    )
    p.add_argument("--predictions", type=Path, required=True,
                   help="Stage 3-A 출력 폴더 (*.alpha.json)")
    p.add_argument("--gt", type=Path, required=True,
                   help="Ground truth 폴더 (*.json with type/fields/items)")
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Predictions : %s", args.predictions)
    log.info("Ground truth: %s", args.gt)

    if not args.predictions.exists():
        log.error("predictions dir not found: %s", args.predictions)
        return 2
    if not args.gt.exists():
        log.error("gt dir not found: %s", args.gt)
        return 2

    report = run(args.predictions, args.gt, args.thresholds)
    paths = report.emit(reports_dir=args.reports_dir,
                        use_color=not args.no_color)
    log.info("HTML : %s", paths["html"])
    log.info("JSON : %s", paths["json"])

    return 0 if report.overall_status in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
