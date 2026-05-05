"""
src/auto_fill_numerical_gt.py

Phase 16a/b — Numerical GT 자동 채움 (1차 baseline 학습용)

Phase 16a (`prepare_vlm_dataset.py numerical`) 가 생성한 JSON 템플릿의
GT field (`nominal`, `tolerance`, `symbol`, `Ra` 등) 는 모두 ``null``
이므로 그대로 Donut fine-tune 에 투입할 수 없다. 본 스크립트는
``_review.ocr_hint`` (Pytesseract OCR 결과) 를 정규식으로 파싱하여
GT field 를 ★ 1차 자동 채움 한다.

목적
----
사람 검수 없이 Phase 16b 학습을 진행하기 위한 ``noisy GT`` 생성.
Phase 17 e2e 평가에서 Stage 3-N 자리를 채우고, 후속 검수 / 재학습의
기준점을 확보한다.

지원 클래스
-----------
- **Measure**  : ``nominal`` (numeric), ``tolerance`` (±X / +Y/-Z)
- **GDT**      : ``symbol`` (⌖/⏤/⊥/∥/⌭/...), ``tolerance``, ``datum`` (A,B,C)
- **Roughness**: ``Ra`` (numeric)

매핑 실패 region 은 ``status="auto_filled_failed"`` 로 표시되어
학습 데이터에서 제외된다 (Donut Lightning DataModule 이 ``completed=True``
만 사용).

CLI
---
::

    # 검증 (실제 파일 수정 X)
    python src/auto_fill_numerical_gt.py --dry-run

    # 실제 채움 + 통계 리포트
    python src/auto_fill_numerical_gt.py --report outputs/auto_fill_report.md

    # 다른 디렉토리 처리
    python src/auto_fill_numerical_gt.py \\
        --input-dir data/vlm/numerical/ \\
        --report outputs/auto_fill_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root bootstrap — same pattern as src/pipeline.py (Task #92)
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "vlm" / "numerical"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "auto_fill_numerical_report.md"

# ---------------------------------------------------------------------------
# Regex patterns (★ 핵심)
# ---------------------------------------------------------------------------
# 일반 numeric (소수점 포함, 음수 허용)
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")

# Tolerance: ±X
_TOL_SYMMETRIC_RE = re.compile(r"[±+\-]\s*(\d+(?:\.\d+)?)\s*(?![/\d])")
_TOL_PM_RE = re.compile(r"±\s*(\d+(?:\.\d+)?)")

# Tolerance: +X/-Y or +X / -Y
_TOL_ASYMMETRIC_RE = re.compile(
    r"\+\s*(\d+(?:\.\d+)?)\s*/\s*-\s*(\d+(?:\.\d+)?)"
)

# Tolerance: H7, h6, g6 등 ISO fit class
_TOL_ISO_RE = re.compile(r"\b([A-Za-z][A-Za-z]?\d{1,2})\b")

# Diameter prefix: ∅, Ø, %, Phi
_DIAMETER_RE = re.compile(r"[⌀Ø∅Φ]")  # ∅ Ø ∅ Φ

# Roughness: Ra X.X 또는 Ra/X.X 형식 (다양한 인식)
_ROUGHNESS_RA_RE = re.compile(
    r"R\s*a\s*[:=]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# GDT symbols (Unicode + ASCII fallback)
GDT_SYMBOL_PATTERNS: List[Tuple[str, str]] = [
    # (symbol, regex pattern in OCR text)
    ("⌖",  r"[⌖⌖]|TRUE\s*POSITION|Position|\bPOS\b"),
    ("⏤",  r"[⏤⏤]|STRAIGHT|Straightness|\bSTR\b"),
    ("⊥",  r"[⊥⊥]|PERPEND|Perpendicular|\bPERP\b"),
    ("∥",  r"[∥∥]|PARALLEL|Parallelism|\bPARA\b"),
    ("⌭",  r"[⌭⌭]|CYLIND|Cylindricity|\bCYL\b"),
    ("○",  r"\bROUND|Roundness|Circular(?:ity)?|\bCIRC\b"),
    ("⌒",  r"[⌒⌒]|Profile.*Line"),
    ("⌓",  r"[⌓⌓]|Profile.*Surface"),
    ("∠",  r"[∠∠]|ANGUL|Angularity|\bANG\b"),
    ("⌯",  r"[⌯⌯]|Symmetry|\bSYM\b"),
    ("◎",  r"[◎◎]|Concentricity|\bCON\b|\bCONC\b"),
    ("⌰",  r"[⌰⌰]|Total.*Run"),
    ("↗",  r"[↗↗]|Run.*[Oo]ut|\bRO\b"),
    ("⌗",  r"[⌗⌗]|Flatness|\bFLAT\b"),
]

# Datum letters (A, B, C, D, A-B, etc.)
_DATUM_RE = re.compile(r"\b([A-D](?:\s*[-‐]\s*[A-D])?)\b")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_fill_numerical_gt")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
class FillResult:
    __slots__ = ("path", "region_class", "filled", "fields", "reason")

    def __init__(self, path: Path, region_class: str):
        self.path = path
        self.region_class = region_class
        self.filled = False
        self.fields: List[str] = []
        self.reason: str = ""

# ---------------------------------------------------------------------------
# Per-class auto fill
# ---------------------------------------------------------------------------
def fill_measure(record: Dict[str, Any], ocr: str,
                 ocr_numeric: Optional[float]) -> Tuple[bool, List[str], str]:
    """Measure: nominal + tolerance.

    Returns (filled, fields_filled, fail_reason).
    """
    fields: List[str] = []

    # nominal
    nominal = ocr_numeric
    if nominal is None:
        m = _NUMERIC_RE.search(ocr)
        if m:
            try:
                nominal = float(m.group())
            except ValueError:
                nominal = None
    if nominal is None:
        return False, fields, "no_numeric_in_ocr"

    record["nominal"] = nominal
    fields.append("nominal")

    # tolerance — symmetric ±
    tol: Optional[Dict[str, float]] = None
    m = _TOL_PM_RE.search(ocr)
    if m:
        try:
            v = float(m.group(1))
            tol = {"upper": v, "lower": v}
        except ValueError:
            pass

    # tolerance — asymmetric +X/-Y
    if tol is None:
        m = _TOL_ASYMMETRIC_RE.search(ocr)
        if m:
            try:
                tol = {"upper": float(m.group(1)),
                       "lower": float(m.group(2))}
            except ValueError:
                pass

    if tol is not None:
        record["tolerance"] = tol
        fields.append("tolerance")
    # tolerance 없어도 nominal 만 있으면 통과 (일반 dimension 흔함)

    return True, fields, ""


def fill_gdt(record: Dict[str, Any], ocr: str
             ) -> Tuple[bool, List[str], str]:
    """GDT: symbol + tolerance + datum."""
    fields: List[str] = []

    # symbol
    symbol = None
    for sym, pattern in GDT_SYMBOL_PATTERNS:
        if re.search(pattern, ocr, re.IGNORECASE | re.UNICODE):
            symbol = sym
            break
    if symbol is None:
        return False, fields, "no_gdt_symbol_match"

    record["symbol"] = symbol
    fields.append("symbol")

    # tolerance — first numeric in ocr
    m = _NUMERIC_RE.search(ocr)
    if m:
        try:
            record["tolerance"] = float(m.group())
            fields.append("tolerance")
        except ValueError:
            pass

    # datum — capital letters not part of tolerance
    datums: List[str] = []
    # Strip numeric portions first to reduce false positives
    ocr_stripped = _NUMERIC_RE.sub(" ", ocr)
    for m in _DATUM_RE.finditer(ocr_stripped):
        d = m.group(1).strip().replace(" ", "")
        if d not in datums and len(d) <= 3:
            datums.append(d)
    if datums:
        record["datum"] = datums[:3]  # max 3 datum
        fields.append("datum")

    return True, fields, ""


def fill_roughness(record: Dict[str, Any], ocr: str
                   ) -> Tuple[bool, List[str], str]:
    """Roughness: Ra value."""
    fields: List[str] = []

    # Try Ra-specific pattern first
    m = _ROUGHNESS_RA_RE.search(ocr)
    if m:
        try:
            record["Ra"] = float(m.group(1))
            fields.append("Ra")
            return True, fields, ""
        except ValueError:
            pass

    # Fallback: first numeric (less reliable)
    m = _NUMERIC_RE.search(ocr)
    if m:
        try:
            record["Ra"] = float(m.group())
            fields.append("Ra")
            return True, fields, "fallback_first_numeric"
        except ValueError:
            pass

    return False, fields, "no_numeric_in_ocr"


# ---------------------------------------------------------------------------
# Main process
# ---------------------------------------------------------------------------
FILLERS = {
    "Measure":   fill_measure,
    "GDT":       fill_gdt,
    "Roughness": fill_roughness,
}


def process_one_json(path: Path, dry_run: bool) -> FillResult:
    try:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        r = FillResult(path, "Unknown")
        r.reason = f"json_load_error: {e}"
        return r

    region_class = record.get("type", "Unknown")
    result = FillResult(path, region_class)

    if region_class not in FILLERS:
        result.reason = "unknown_class"
        return result

    review = record.get("_review", {})
    ocr_hint: str = review.get("ocr_hint", "") or ""
    ocr_numeric = review.get("ocr_numeric")

    if not ocr_hint:
        result.reason = "no_ocr_hint"
        return result

    # Apply filler
    filler = FILLERS[region_class]
    if region_class == "Measure":
        ok, fields, reason = filler(record, ocr_hint, ocr_numeric)
    else:
        ok, fields, reason = filler(record, ocr_hint)

    if ok:
        result.filled = True
        result.fields = fields
        # Update _review metadata
        review["completed"] = True
        review["auto_filled"] = True
        review["fill_method"] = f"auto_{region_class.lower()}"
        review["fill_fields"] = fields
        if reason:
            review["fill_warning"] = reason
        record["_review"] = review

        if not dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
    else:
        result.reason = reason

    return result


def write_report(results: List[FillResult], report_path: Path) -> None:
    """Write Markdown report with per-class statistics."""
    by_class: Dict[str, List[FillResult]] = defaultdict(list)
    for r in results:
        by_class[r.region_class].append(r)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Auto-fill Numerical GT Report\n\n")
        f.write(f"Total processed: **{len(results)}**\n\n")

        # Per-class summary
        f.write("## Per-class statistics\n\n")
        f.write("| Class | Total | Filled | Rate | Top fail reason |\n")
        f.write("|---|---|---|---|---|\n")
        for cls in ["Measure", "GDT", "Roughness", "Unknown"]:
            items = by_class.get(cls, [])
            if not items:
                continue
            total = len(items)
            filled = sum(1 for r in items if r.filled)
            rate = filled / total * 100 if total else 0
            reasons = Counter(r.reason for r in items if not r.filled)
            top_reason = reasons.most_common(1)[0][0] if reasons else "-"
            f.write(f"| {cls} | {total} | {filled} | {rate:.1f}% | {top_reason} |\n")

        # Overall
        total = len(results)
        filled = sum(1 for r in results if r.filled)
        rate = filled / total * 100 if total else 0
        f.write(f"\n**Overall fill rate**: {filled}/{total} = **{rate:.1f}%**\n\n")

        # Field-level statistics for Measure
        f.write("## Measure field coverage\n\n")
        measure_filled = [r for r in by_class.get("Measure", []) if r.filled]
        if measure_filled:
            field_counts = Counter()
            for r in measure_filled:
                for fld in r.fields:
                    field_counts[fld] += 1
            f.write("| Field | Count | Rate |\n|---|---|---|\n")
            for fld, cnt in field_counts.most_common():
                rate = cnt / len(measure_filled) * 100
                f.write(f"| {fld} | {cnt} | {rate:.1f}% |\n")

        # Field-level statistics for GDT
        f.write("\n## GDT field coverage\n\n")
        gdt_filled = [r for r in by_class.get("GDT", []) if r.filled]
        if gdt_filled:
            field_counts = Counter()
            for r in gdt_filled:
                for fld in r.fields:
                    field_counts[fld] += 1
            f.write("| Field | Count | Rate |\n|---|---|---|\n")
            for fld, cnt in field_counts.most_common():
                rate = cnt / len(gdt_filled) * 100
                f.write(f"| {fld} | {cnt} | {rate:.1f}% |\n")

        # Failure breakdown
        f.write("\n## Failure reasons\n\n")
        all_reasons = Counter(r.reason for r in results if not r.filled)
        if all_reasons:
            f.write("| Reason | Count |\n|---|---|\n")
            for reason, cnt in all_reasons.most_common():
                f.write(f"| {reason} | {cnt} |\n")
        else:
            f.write("(none)\n")

    log.info("Report written: %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 16a/b — Numerical GT 자동 채움 (OCR hint → GT field)",
    )
    p.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR),
                   help="JSON 템플릿이 있는 디렉토리 (default: data/vlm/numerical/)")
    p.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH),
                   help="통계 리포트 출력 경로 (.md)")
    p.add_argument("--dry-run", action="store_true",
                   help="실제 파일 수정 없이 검증만 수행")
    p.add_argument("--limit", type=int, default=0,
                   help="처리할 JSON 최대 개수 (0 = 전체). 디버깅용")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        log.error("Input dir not found: %s", input_dir)
        return 2

    json_paths = sorted(input_dir.glob("*.json"))
    if args.limit > 0:
        json_paths = json_paths[: args.limit]

    if not json_paths:
        log.warning("No *.json found in %s", input_dir)
        return 0

    log.info("Processing %d JSON files (dry_run=%s)", len(json_paths), args.dry_run)

    results: List[FillResult] = []
    for i, p in enumerate(json_paths, 1):
        result = process_one_json(p, dry_run=args.dry_run)
        results.append(result)
        if i % 1000 == 0 or i == len(json_paths):
            filled_so_far = sum(1 for r in results if r.filled)
            log.info("[%d/%d] filled %d (%.1f%%)",
                     i, len(json_paths), filled_so_far,
                     filled_so_far / i * 100)

    # Final report
    write_report(results, Path(args.report))

    total = len(results)
    filled = sum(1 for r in results if r.filled)
    rate = filled / total * 100 if total else 0
    log.info("Done. %d/%d filled (%.1f%%). Report: %s",
             filled, total, rate, args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
