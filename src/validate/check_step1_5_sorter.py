"""
src/validate/check_step1_5_sorter.py

V1 — sort_by_titleblock.py 결과 검증.

Inputs
------
1. ``outputs/sort_titleblock_manifest.csv``  (sorter output, required)
2. ``data/validation_gt/step1_5_titleblock_gt.csv`` (optional)
   columns: ``filename, has_titleblock_actual``  (1 / 0)

Without GT, only descriptive stats (no accuracy).
With GT, also compute accuracy / precision / recall / per-language.

CLI
---
::

    python -m src.validate.check_step1_5_sorter \
        --manifest outputs/sort_titleblock_manifest.csv \
        --gt data/validation_gt/step1_5_titleblock_gt.csv

Outputs
-------
- console: PASS/FAIL summary
- ``reports/<date>_step1_5_sorter.html``
- ``reports/<date>_step1_5_sorter.json``
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

log = setup_logging("validate.step1_5")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> List[Dict[str, str]]:
    """Load sort_by_titleblock.py manifest CSV. utf-8-sig friendly."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def load_gt(path: Optional[Path]) -> Dict[str, int]:
    """Load ground-truth CSV. Returns {filename: 1/0}."""
    if path is None or not path.exists():
        return {}
    gt: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = (r.get("filename") or "").strip()
            val = (r.get("has_titleblock_actual") or "").strip().lower()
            if not fn:
                continue
            gt[fn] = 1 if val in {"1", "true", "y", "yes", "tb"} else 0
    return gt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def detect_language(filename: str) -> str:
    """Crude language heuristic from filename (Hangul / Kana / Cyrillic / Latin).

    Real language tagging happens at Stage 3; here it's only for breakdown.
    """
    has_hangul = any(0xAC00 <= ord(c) <= 0xD7A3 for c in filename)
    has_kana   = any(0x3040 <= ord(c) <= 0x30FF for c in filename) \
              or any(0x4E00 <= ord(c) <= 0x9FFF for c in filename)
    has_cyril  = any(0x0400 <= ord(c) <= 0x04FF for c in filename)
    if has_hangul:  return "ko"
    if has_kana:    return "ja"
    if has_cyril:   return "ru"
    return "en"


def decision_to_pred(decision: str) -> Optional[int]:
    """Map sorter decision to has_titleblock prediction (1=yes, 0=no, None=ambiguous)."""
    if decision == "stage1_titleblock":
        return 1
    if decision == "stage2_no_titleblock":
        return 0
    return None  # manual_review / error


def confusion(rows_with_gt: List[Tuple[int, int]]) -> Dict[str, int]:
    tp = sum(1 for p, a in rows_with_gt if p == 1 and a == 1)
    tn = sum(1 for p, a in rows_with_gt if p == 0 and a == 0)
    fp = sum(1 for p, a in rows_with_gt if p == 1 and a == 0)
    fn = sum(1 for p, a in rows_with_gt if p == 0 and a == 1)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b > 0 else default


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------
def run(manifest_path: Path,
        gt_path: Optional[Path],
        thresholds_path: Path) -> ValidationReport:
    rows = load_manifest(manifest_path)
    gt = load_gt(gt_path)
    thr = load_thresholds(thresholds_path)

    report = ValidationReport(
        title="Step 1.5 — TitleBlock Sorter Validation",
        step="step1.5_sorter",
        metadata={
            "manifest": str(manifest_path),
            "gt": str(gt_path) if gt_path else None,
            "n_rows": len(rows),
            "n_gt": len(gt),
        },
    )

    if not rows:
        report.add_eval(
            "manifest_loaded", value=0, threshold=1, direction="ge",
            severity=Severity.CRITICAL,
            message="Manifest is empty",
        )
        return report

    # --- Decision distribution ----------------------------------------
    decisions = Counter(r.get("decision", "?") for r in rows)
    n_total = len(rows)
    n_stage1 = decisions.get("stage1_titleblock", 0)
    n_stage2 = decisions.get("stage2_no_titleblock", 0)
    n_manual = decisions.get("manual_review", 0)
    n_error = decisions.get("error", 0)

    manual_rate = safe_div(n_manual, n_total)
    error_rate = safe_div(n_error, n_total)

    try:
        manual_thr = threshold_lookup(thr, "step1_5.manual_review_rate_max")
        manual_thr_val = manual_thr["threshold"]
        manual_sev = manual_thr.get("severity", "warning")
    except KeyError:
        manual_thr_val = 0.20
        manual_sev = "warning"

    report.add_eval(
        "manual_review_rate", value=manual_rate,
        threshold=manual_thr_val, direction="le",
        severity=manual_sev,
        message=f"{n_manual}/{n_total} → {manual_rate:.1%}",
    )

    report.add_eval(
        "error_rate", value=error_rate,
        threshold=0.01, direction="le",
        severity=Severity.WARNING,
        message=f"{n_error} imread or OCR failures",
    )

    # --- Language breakdown -------------------------------------------
    lang_decisions: Dict[str, Counter] = {}
    for r in rows:
        lang = detect_language(r.get("filename", ""))
        lang_decisions.setdefault(lang, Counter())[r.get("decision", "?")] += 1
    lang_rows = []
    for lang, c in sorted(lang_decisions.items()):
        n = sum(c.values())
        lang_rows.append({
            "language": lang,
            "total": n,
            "stage1": c.get("stage1_titleblock", 0),
            "stage2": c.get("stage2_no_titleblock", 0),
            "manual": c.get("manual_review", 0),
            "manual_rate": f"{safe_div(c.get('manual_review', 0), n):.1%}",
        })
    report.add_table(
        "Decision distribution by language",
        rows=lang_rows,
        columns=["language", "total", "stage1", "stage2", "manual", "manual_rate"],
        description="Filename-based language heuristic (4-script detection).",
    )

    # --- Decision distribution chart ----------------------------------
    report.add_plot(
        "Decision distribution",
        make_bar_chart(
            labels=["stage1_titleblock", "stage2_no_titleblock",
                    "manual_review", "error"],
            values=[n_stage1, n_stage2, n_manual, n_error],
            title="Sorter decisions", ylabel="count",
        ),
    )

    # --- GT-based accuracy (if GT provided) ---------------------------
    if gt:
        # Build (pred, actual) pairs only for rows present in GT
        joined: List[Tuple[int, int, str, str]] = []
        for r in rows:
            fn = r.get("filename", "")
            if fn not in gt:
                continue
            pred = decision_to_pred(r.get("decision", ""))
            actual = gt[fn]
            if pred is None:
                # manual_review counted as 'unknown' — exclude from accuracy
                continue
            joined.append((pred, actual, fn, detect_language(fn)))

        if not joined:
            report.add_eval(
                "gt_join", value=0, threshold=1, direction="ge",
                severity=Severity.CRITICAL,
                message="GT provided but no overlap with manifest filenames",
            )
        else:
            n_evaluated = len(joined)
            cm = confusion([(p, a) for p, a, _, _ in joined])
            accuracy = safe_div(cm["tp"] + cm["tn"], n_evaluated)
            precision = safe_div(cm["tp"], cm["tp"] + cm["fp"])
            recall    = safe_div(cm["tp"], cm["tp"] + cm["fn"])
            f1        = safe_div(2 * precision * recall, precision + recall)

            try:
                acc_thr = threshold_lookup(thr, "step1_5.classifier_accuracy")
                acc_val, acc_sev = acc_thr["threshold"], acc_thr.get("severity", "critical")
            except KeyError:
                acc_val, acc_sev = 0.85, "critical"

            report.add_eval(
                "classifier_accuracy", value=accuracy,
                threshold=acc_val, direction="ge",
                severity=acc_sev,
                message=f"on {n_evaluated} GT-evaluated rows "
                        f"(manual_review excluded)",
            )
            report.add_eval(
                "precision_TB_present", value=precision,
                threshold=None, direction="none", severity=Severity.INFO,
                message=f"TP={cm['tp']} FP={cm['fp']}",
            )
            report.add_eval(
                "recall_TB_present", value=recall,
                threshold=None, direction="none", severity=Severity.INFO,
                message=f"TP={cm['tp']} FN={cm['fn']}",
            )
            report.add_eval(
                "f1_TB_present", value=f1,
                threshold=None, direction="none", severity=Severity.INFO,
            )

            # Confusion matrix plot
            cm_matrix = [[cm["tn"], cm["fp"]],
                         [cm["fn"], cm["tp"]]]
            report.add_plot(
                "Confusion matrix (TB-present)",
                make_confusion_matrix(
                    cm_matrix, labels=["No-TB", "Has-TB"],
                    title="Sorter vs human GT",
                ),
                description="Rows = actual, cols = predicted. "
                            "manual_review rows excluded.",
            )

            # Per-language accuracy
            try:
                lang_thr = threshold_lookup(thr, "step1_5.per_language_min")
                lang_thr_val, lang_sev = lang_thr["threshold"], lang_thr.get("severity", "warning")
            except KeyError:
                lang_thr_val, lang_sev = 0.80, "warning"

            per_lang_rows = []
            for lang in sorted({l for _, _, _, l in joined}):
                subset = [(p, a) for p, a, _, l in joined if l == lang]
                acc_l = safe_div(
                    sum(1 for p, a in subset if p == a), len(subset)
                )
                per_lang_rows.append({
                    "language": lang,
                    "n": len(subset),
                    "accuracy": f"{acc_l:.4f}",
                })
                report.add_eval(
                    f"per_language_accuracy[{lang}]", value=acc_l,
                    threshold=lang_thr_val, direction="ge",
                    severity=lang_sev,
                    message=f"n={len(subset)}",
                )
            report.add_table(
                "Per-language accuracy",
                rows=per_lang_rows,
                columns=["language", "n", "accuracy"],
            )

    else:
        report.add_eval(
            "gt_provided", value=0, threshold=None, direction="none",
            severity=Severity.INFO,
            message="No GT supplied — skipping accuracy. Pass --gt to enable.",
        )

    # --- Keyword hit distribution (descriptive) -----------------------
    hits = []
    for r in rows:
        try:
            hits.append(int(r.get("keyword_hits", "0") or 0))
        except ValueError:
            hits.append(0)
    if hits:
        hist = Counter(hits)
        labels = sorted(hist.keys())
        report.add_plot(
            "Keyword hits distribution",
            make_bar_chart(
                labels=[str(k) for k in labels],
                values=[hist[k] for k in labels],
                title="Multilingual TitleBlock keyword hits per drawing",
                ylabel="count of drawings",
            ),
            description="Higher hits → more confident TB-present classification.",
        )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V1 — Validate Step 1.5 sort_by_titleblock.py output."
    )
    p.add_argument(
        "--manifest", type=Path,
        default=Path("outputs/sort_titleblock_manifest.csv"),
        help="Sorter manifest CSV.",
    )
    p.add_argument(
        "--gt", type=Path, default=None,
        help="Optional ground-truth CSV (filename, has_titleblock_actual).",
    )
    p.add_argument(
        "--thresholds", type=Path, default=DEFAULT_THRESHOLDS_PATH,
        help="Validation thresholds YAML.",
    )
    p.add_argument(
        "--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
        help="Output directory for HTML + JSON.",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color in console output.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("Manifest : %s", args.manifest)
    log.info("GT       : %s", args.gt or "(none — descriptive only)")
    log.info("Thresholds: %s", args.thresholds)

    try:
        report = run(args.manifest, args.gt, args.thresholds)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 2

    paths = report.emit(
        reports_dir=args.reports_dir,
        use_color=not args.no_color,
    )
    log.info("HTML  : %s", paths["html"])
    log.info("JSON  : %s", paths["json"])

    overall = report.overall_status
    return 0 if overall in (Status.PASS, Status.WARN, Status.INFO) else 1


if __name__ == "__main__":
    sys.exit(main())
