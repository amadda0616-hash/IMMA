"""
src/validate/check_enrichment.py

V9 — Step 9 (Metadata Enrichment) 사후 검증.

Inputs
------
- ``--enriched`` : Step 9 출력 폴더 (``<stem>.enriched.json``)
- ``--expert``   : (옵션) 도메인 전문가 검수 GT — material recommendation accuracy 측정

Checks
------
- provenance_completeness   = 1.0 (critical)   모든 필드에 method/source/rationale 존재
- llm_method_rate_max       < 0.40 (warning)   비용 통제
- hitl_flag_rate_max        < 0.25 (warning)
- empty_suggestion_rate_max < 0.10 (warning)   suggested 가 None
- material_recommendation_accuracy ≥ 0.70 (warning, GT 있을 때)
- cost_per_drawing_max_usd  ≤ $0.005 (warning, gemini 추정)
- per-category 분포 표
- per-method 분포 차트
- per-provider 분포 (mock vs gemini vs qwen)

CLI
---
::

    python -m src.validate.check_enrichment \\
        --enriched outputs/enriched/ \\
        --expert data/validation_gt/enrichment_expert.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.metrics import safe_div, pr_f1
from src.validate.common import (
    DEFAULT_REPORTS_DIR, DEFAULT_THRESHOLDS_PATH,
    Severity, Status, ValidationReport,
    load_thresholds, threshold_lookup,
    make_bar_chart,
    setup_logging,
)

log = setup_logging("validate.enrichment")

# Cost estimates (USD per ~5K input + ~1K output tokens, drawing-level)
COST_PER_DRAWING_USD = {
    "mock":   0.0,
    "gemini": 0.0008,
    "qwen":   0.0,        # local, no API cost
    "claude": 0.003,
}

# Required provenance fields (D-022)
REQUIRED_PROVENANCE_FIELDS = ["method", "source", "rationale"]


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


def discover_enriched(d: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in ("*.enriched.json", "*.json"):
        for p in d.rglob(ext):
            if p.name.startswith("_") or p.name == "manifest.json":
                continue
            stem = p.stem
            if stem.endswith(".enriched"):
                stem = stem[:-len(".enriched")]
            if stem not in out:
                out[stem] = p
    return out


# ---------------------------------------------------------------------------
# Per-record evaluation
# ---------------------------------------------------------------------------
@dataclass
class EnrichedSummary:
    image_stem: str
    provider: str
    n_fields_total: int
    method_counts: Dict[str, int]   # {deterministic, heuristic, llm, hitl}
    hitl_flagged: int
    empty_suggestions: int
    provenance_violations: List[str]
    fields_by_category: Dict[str, Dict[str, Any]]   # for per-category breakdown


def evaluate_one(stem: str, enriched_json: Dict[str, Any]) -> Optional[EnrichedSummary]:
    enr = enriched_json.get("enrichment")
    if not enr:
        return None

    fields = enr.get("fields") or {}
    provider = enr.get("provider", "unknown")

    method_counts: Counter = Counter()
    hitl_flagged = 0
    empty_count = 0
    provenance_violations: List[str] = []

    for cat, info in fields.items():
        if not isinstance(info, dict):
            continue
        method = info.get("method", "unknown")
        method_counts[method] += 1
        if info.get("flagged_for_review"):
            hitl_flagged += 1
        if not info.get("suggested"):
            empty_count += 1
        # Provenance completeness
        for field_name in REQUIRED_PROVENANCE_FIELDS:
            if not info.get(field_name):
                provenance_violations.append(f"{cat}.{field_name}")

    return EnrichedSummary(
        image_stem=stem,
        provider=provider,
        n_fields_total=len(fields),
        method_counts=dict(method_counts),
        hitl_flagged=hitl_flagged,
        empty_suggestions=empty_count,
        provenance_violations=provenance_violations,
        fields_by_category=fields,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(records: List[EnrichedSummary]) -> Dict[str, Any]:
    if not records:
        return {}

    n_records = len(records)
    total_fields = sum(r.n_fields_total for r in records)
    total_hitl   = sum(r.hitl_flagged for r in records)
    total_empty  = sum(r.empty_suggestions for r in records)
    total_violations = sum(len(r.provenance_violations) for r in records)

    # Method distribution
    method_totals: Counter = Counter()
    for r in records:
        method_totals.update(r.method_counts)

    # Provider distribution
    provider_counts = Counter(r.provider for r in records)

    # Per-category breakdown
    by_category: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "method_counts": Counter(),
                 "hitl_flagged": 0, "empty": 0,
                 "avg_confidence": 0.0, "_conf_sum": 0.0},
    )
    for r in records:
        for cat, info in r.fields_by_category.items():
            if not isinstance(info, dict):
                continue
            c = by_category[cat]
            c["n"] += 1
            c["method_counts"][info.get("method", "unknown")] += 1
            if info.get("flagged_for_review"):
                c["hitl_flagged"] += 1
            if not info.get("suggested"):
                c["empty"] += 1
            c["_conf_sum"] += float(info.get("confidence") or 0.0)

    for cat, c in by_category.items():
        c["avg_confidence"] = round(c["_conf_sum"] / max(1, c["n"]), 4)
        del c["_conf_sum"]

    # Provenance completeness rate
    # = 1.0 - (violations / required_total)
    required_total = total_fields * len(REQUIRED_PROVENANCE_FIELDS)
    prov_rate = 1.0 - safe_div(total_violations, max(1, required_total))

    # Cost estimate
    estimated_cost = 0.0
    for prov, count in provider_counts.items():
        per = COST_PER_DRAWING_USD.get(prov, 0.001)   # default fallback
        estimated_cost += per * count

    return {
        "n_records": n_records,
        "n_fields_total": total_fields,
        "method_distribution": {
            m: round(safe_div(c, max(1, total_fields)), 4)
            for m, c in method_totals.items()
        },
        "method_counts": dict(method_totals),
        "hitl_flag_rate": round(safe_div(total_hitl, max(1, total_fields)), 4),
        "hitl_flagged_total": total_hitl,
        "empty_suggestion_rate": round(safe_div(total_empty, max(1, total_fields)), 4),
        "provenance_completeness": round(prov_rate, 4),
        "provenance_violations_total": total_violations,
        "provider_counts": dict(provider_counts),
        "estimated_cost_usd": round(estimated_cost, 4),
        "cost_per_drawing_usd": round(safe_div(estimated_cost, n_records), 6),
        "by_category": dict(by_category),
    }


def evaluate_material_accuracy(records: List[EnrichedSummary],
                                expert_gt: Dict[str, Any]
                                ) -> Optional[Dict[str, Any]]:
    """If expert GT provided, compute material recommendation accuracy.

    Expert GT format::

        {
          "drawing_001": {
            "material": {"expert_correct": true, "expert_value": "SUS304 No.2D"},
            ...
          }
        }
    """
    if not expert_gt:
        return None

    n_evaluated = 0
    n_correct = 0
    examples: List[Dict[str, Any]] = []
    for r in records:
        gt = expert_gt.get(r.image_stem)
        if not gt:
            continue
        for cat, gt_info in gt.items():
            if cat not in r.fields_by_category:
                continue
            pred_info = r.fields_by_category[cat]
            if not isinstance(pred_info, dict) or not isinstance(gt_info, dict):
                continue
            n_evaluated += 1
            is_correct = bool(gt_info.get("expert_correct", False))
            if is_correct:
                n_correct += 1
            else:
                examples.append({
                    "image_stem": r.image_stem,
                    "category": cat,
                    "predicted": pred_info.get("suggested"),
                    "expert_value": gt_info.get("expert_value"),
                })

    if n_evaluated == 0:
        return None
    return {
        "n_evaluated": n_evaluated,
        "n_correct":   n_correct,
        "accuracy":    round(n_correct / n_evaluated, 4),
        "wrong_examples": examples[:10],
    }


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run(enriched_dir: Path,
        expert_path: Optional[Path],
        thresholds_path: Path) -> ValidationReport:
    thr = load_thresholds(thresholds_path)
    files = discover_enriched(enriched_dir)

    expert_gt: Dict[str, Any] = {}
    if expert_path and expert_path.exists():
        expert_gt = load_json_safe(expert_path) or {}

    report = ValidationReport(
        title="V9 — Metadata Enrichment (Step 9) Validation",
        step="enrichment",
        metadata={
            "enriched_dir": str(enriched_dir),
            "expert_gt": str(expert_path) if expert_path else None,
            "n_files": len(files),
            "n_expert_records": len(expert_gt),
        },
    )

    if not files:
        report.add_eval(
            "files_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message=f"No enriched JSONs under {enriched_dir}",
        )
        return report

    # ---- Per-record summary ----------------------------------------
    records: List[EnrichedSummary] = []
    parse_errors: List[str] = []
    no_enrichment: List[str] = []
    for stem, p in files.items():
        d = load_json_safe(p)
        if d is None:
            parse_errors.append(stem)
            continue
        s = evaluate_one(stem, d)
        if s is None:
            no_enrichment.append(stem)
            continue
        records.append(s)

    log.info("Loaded %d records (parse_errors=%d no_enrichment=%d)",
             len(records), len(parse_errors), len(no_enrichment))

    if not records:
        report.add_eval(
            "records_evaluated", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message="No records with enrichment block",
        )
        return report

    agg = aggregate(records)

    # ---- ★ Provenance completeness (critical) ----------------------
    try:
        node = threshold_lookup(thr, "enrichment.provenance_completeness")
        p_thr, p_sev = node["threshold"], node.get("severity", "critical")
    except KeyError:
        p_thr, p_sev = 1.0, "critical"
    report.add_eval(
        "provenance_completeness",
        value=agg["provenance_completeness"], threshold=p_thr,
        direction="ge", severity=p_sev,
        message=f"violations={agg['provenance_violations_total']} / "
                f"required={agg['n_fields_total'] * len(REQUIRED_PROVENANCE_FIELDS)}",
    )

    # ---- LLM method rate -------------------------------------------
    llm_rate = agg["method_distribution"].get("llm", 0.0)
    try:
        node = threshold_lookup(thr, "enrichment.llm_method_rate_max")
        l_thr, l_sev = node["threshold"], node.get("severity", "warning")
    except KeyError:
        l_thr, l_sev = 0.40, "warning"
    report.add_eval(
        "llm_method_rate", value=llm_rate, threshold=l_thr,
        direction="le", severity=l_sev,
        message=f"비용 통제 — LLM 호출 비율 (count={agg['method_counts'].get('llm', 0)})",
    )

    # ---- HITL flag rate --------------------------------------------
    try:
        node = threshold_lookup(thr, "enrichment.hitl_flag_rate_max")
        h_thr, h_sev = node["threshold"], node.get("severity", "warning")
    except KeyError:
        h_thr, h_sev = 0.25, "warning"
    report.add_eval(
        "hitl_flag_rate", value=agg["hitl_flag_rate"],
        threshold=h_thr, direction="le", severity=h_sev,
        message=f"flagged={agg['hitl_flagged_total']} / total_fields={agg['n_fields_total']}",
    )

    # ---- Empty suggestion rate -------------------------------------
    report.add_eval(
        "empty_suggestion_rate", value=agg["empty_suggestion_rate"],
        threshold=0.10, direction="le", severity=Severity.WARNING,
        message="suggested 필드가 None 인 비율",
    )

    # ---- Cost per drawing ------------------------------------------
    try:
        node = threshold_lookup(thr, "enrichment.cost_per_drawing_max_usd")
        c_thr, c_sev = node["threshold"], node.get("severity", "warning")
    except KeyError:
        c_thr, c_sev = 0.005, "warning"
    report.add_eval(
        "cost_per_drawing_usd",
        value=agg["cost_per_drawing_usd"], threshold=c_thr,
        direction="le", severity=c_sev,
        message=f"providers={dict(agg['provider_counts'])}, "
                f"total=${agg['estimated_cost_usd']}",
    )

    # ---- Material accuracy (optional, with GT) ----------------------
    mat_acc = evaluate_material_accuracy(records, expert_gt)
    if mat_acc:
        try:
            node = threshold_lookup(thr, "enrichment.material_recommendation_accuracy_min")
            m_thr, m_sev = node["threshold"], node.get("severity", "warning")
        except KeyError:
            m_thr, m_sev = 0.70, "warning"
        report.add_eval(
            "material_recommendation_accuracy",
            value=mat_acc["accuracy"], threshold=m_thr,
            direction="ge", severity=m_sev,
            message=f"{mat_acc['n_correct']} / {mat_acc['n_evaluated']} "
                    f"(domain expert validations)",
        )
        if mat_acc.get("wrong_examples"):
            report.add_table(
                "Wrong material recommendations (first 10)",
                rows=mat_acc["wrong_examples"],
                columns=["image_stem", "category", "predicted", "expert_value"],
            )

    # ---- Method distribution chart ---------------------------------
    if agg["method_distribution"]:
        report.add_plot(
            "Method distribution (%)",
            make_bar_chart(
                labels=list(agg["method_distribution"].keys()),
                values=[v * 100 for v in agg["method_distribution"].values()],
                title="Enrichment method distribution",
                ylabel="% of fields",
            ),
            description="Cascade tier distribution: deterministic / heuristic / llm / hitl",
        )

    # ---- Per-category table ----------------------------------------
    cat_rows = []
    for cat, c in agg["by_category"].items():
        det_n = c["method_counts"].get("deterministic", 0)
        heu_n = c["method_counts"].get("heuristic", 0)
        llm_n = c["method_counts"].get("llm", 0)
        hitl_n = c["method_counts"].get("hitl", 0)
        cat_rows.append({
            "category": cat,
            "n": c["n"],
            "det": det_n, "heur": heu_n, "llm": llm_n, "hitl_method": hitl_n,
            "hitl_flagged": c["hitl_flagged"],
            "empty": c["empty"],
            "avg_conf": c["avg_confidence"],
        })
    if cat_rows:
        report.add_table(
            "Per-category enrichment summary",
            rows=cat_rows,
            columns=["category", "n", "det", "heur", "llm", "hitl_method",
                     "hitl_flagged", "empty", "avg_conf"],
            description=("det/heur/llm/hitl_method = cascade tier 분포 / "
                         "hitl_flagged = confidence 임계값 미달 / "
                         "empty = suggested 가 None"),
        )

    # ---- Per-provider table ----------------------------------------
    if len(agg["provider_counts"]) > 1:
        prov_rows = [{"provider": p, "n_drawings": c,
                      "estimated_cost_usd":
                          round(c * COST_PER_DRAWING_USD.get(p, 0.001), 4)}
                     for p, c in agg["provider_counts"].items()]
        report.add_table(
            "Per-provider distribution",
            rows=prov_rows,
            columns=["provider", "n_drawings", "estimated_cost_usd"],
        )

    # ---- Diagnostics ------------------------------------------------
    if parse_errors:
        report.add_eval(
            "parse_errors", value=len(parse_errors),
            threshold=0, direction="le", severity=Severity.WARNING,
        )
    if no_enrichment:
        report.add_eval(
            "files_without_enrichment_block", value=len(no_enrichment),
            threshold=0, direction="le", severity=Severity.WARNING,
            message="enrichment 블록 없는 JSON (Step 9 미실행 가능)",
        )

    # Summary log
    log.info(
        "Summary: provenance=%.4f  hitl_rate=%.4f  llm_rate=%.4f  cost/drawing=$%.4f",
        agg["provenance_completeness"], agg["hitl_flag_rate"],
        llm_rate, agg["cost_per_drawing_usd"],
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V9 — Metadata Enrichment (Step 9) validation."
    )
    p.add_argument("--enriched", type=Path, required=True,
                   help="Step 9 출력 폴더 (*.enriched.json)")
    p.add_argument("--expert", type=Path, default=None,
                   help="(옵션) 도메인 전문가 검수 GT JSON")
    p.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Enriched dir: %s", args.enriched)
    log.info("Expert GT   : %s", args.expert or "(none)")

    if not args.enriched.exists():
        log.error("enriched dir not found: %s", args.enriched)
        return 2

    report = run(args.enriched, args.expert, args.thresholds)
    paths = report.emit(reports_dir=args.reports_dir,
                        use_color=not args.no_color)
    log.info("HTML : %s", paths["html"])
    log.info("JSON : %s", paths["json"])
    return 0 if report.overall_status in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
