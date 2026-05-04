"""
src/utils/metrics.py

Step 8 — Evaluation metrics for the multi-stage hybrid pipeline.

Library module (no CLI). Used by every downstream validator:

- V5 (`check_stage3a_alphabetical`)  : field-level F1, hallucination, edit dist
- V6 (`check_stage3n_numerical`)     : schema F1, numerical accuracy, hallucination
- V7 (`check_pipeline_e2e`)          : end-to-end JSON field F1, timing, failure rate
- V9 (`check_enrichment`)            : method distribution, provenance, HITL rate

Also reused by `check_stage1_model` / `check_stage2_model` for shared helpers
(IoU, set-based P/R/F1, confusion matrix construction).

Decision references
-------------------
- D-022  Provenance필수 (provenance_completeness 측정)
- D-023  사용자 필수 임계값 측정에 사용되는 지표
- D-024  Group-aware split 후 평가 (이 모듈은 그룹과 무관, 결과만 평가)

Sections
--------
1. Core P / R / F1
2. Set-based comparison
3. String metrics (edit distance, fuzzy match)
4. Numerical comparison (with tolerance)
5. JSON flattening + field-level F1
6. Hallucination rate
7. Schema-aware comparators (Measure / GDT / Roughness)
8. TitleBlock / Notes comparators
9. Object detection (BBox / OBB IoU + matching)
10. Aggregators (per-class, confusion matrix)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# =====================================================================
# 1. Core P / R / F1
# =====================================================================
def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom > 0 else default


def pr_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Standard precision / recall / F1 from raw counts."""
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f1, 4),
    }


# =====================================================================
# 2. Set-based comparison
# =====================================================================
def set_pr_f1(pred: Iterable[Any],
              gt: Iterable[Any]) -> Dict[str, float]:
    """P/R/F1 treating both sides as sets (order-independent, dedup)."""
    p_set, g_set = set(pred), set(gt)
    tp = len(p_set & g_set)
    fp = len(p_set - g_set)
    fn = len(g_set - p_set)
    return pr_f1(tp, fp, fn)


# =====================================================================
# 3. String metrics
# =====================================================================
def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance. Uses ``editdistance`` lib if available, else
    a pure-Python fallback."""
    if s1 == s2:
        return 0
    try:
        import editdistance  # noqa: PLC0415
        return int(editdistance.eval(s1, s2))
    except ImportError:
        return _pure_python_edit_distance(s1, s2)


def _pure_python_edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _pure_python_edit_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            ins  = prev[j + 1] + 1
            dele = curr[j] + 1
            sub  = prev[j] + (c1 != c2)
            curr[j + 1] = min(ins, dele, sub)
        prev = curr
    return prev[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Edit distance / max(len). Returns 0.0 on identical, 1.0 on disjoint."""
    s1, s2 = (s1 or ""), (s2 or "")
    n = max(len(s1), len(s2))
    if n == 0:
        return 0.0
    return edit_distance(s1, s2) / n


def fuzzy_match(s1: str, s2: str, threshold: float = 0.20) -> bool:
    """True if normalized edit distance ≤ threshold (default 20%)."""
    return normalized_edit_distance(s1, s2) <= threshold


def normalize_text(s: str) -> str:
    """Uppercase + collapse whitespace (for case/space-insensitive match)."""
    return " ".join((s or "").split()).upper()


# =====================================================================
# 4. Numerical comparison (with tolerance)
# =====================================================================
def coerce_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+(?:\.\d+)?", x)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def numerical_match(pred: Any, gt: Any,
                    abs_tol: float = 0.01,
                    rel_tol: float = 0.001) -> bool:
    """True if pred ≈ gt within abs OR rel tolerance.

    Useful for engineering tolerances (e.g. nominal 25.0 ±0.01 mm).
    """
    p = coerce_number(pred)
    g = coerce_number(gt)
    if p is None or g is None:
        return p == g
    if p == g:
        return True
    if abs(p - g) <= abs_tol:
        return True
    return abs(p - g) <= rel_tol * max(abs(p), abs(g))


# =====================================================================
# 5. JSON flattening + field-level F1
# =====================================================================
def flatten_json(d: Any, prefix: str = "", sep: str = ".",
                 skip_keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Flatten nested dict/list to ``{dotted_key: scalar}``.

    Lists become ``key[0]``, ``key[1]``, ... unless homogeneous strings
    (kept as a tuple for set-based comparison).
    """
    skip_keys = set(skip_keys or [])
    out: Dict[str, Any] = {}

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in skip_keys:
                    continue
                key = f"{path}{sep}{k}" if path else str(k)
                _walk(v, key)
        elif isinstance(node, list):
            # Treat list of scalars as ordered sequence
            if all(not isinstance(x, (dict, list)) for x in node):
                out[path] = tuple(node)
            else:
                for i, x in enumerate(node):
                    _walk(x, f"{path}[{i}]")
        else:
            out[path] = node

    _walk(d, prefix)
    return out


def field_level_f1(pred: Any,
                   gt: Any,
                   *,
                   fuzzy_strings: bool = False,
                   abs_tol: float = 0.01,
                   skip_keys: Iterable[str] = ("_review", "_meta", "type"),
                   ) -> Dict[str, Any]:
    """Field-level F1 between two JSON-like dicts.

    Match rule per leaf field:
        - both scalars equal (exact) OR within numeric tolerance
        - both lists equal as tuple (or set if same length 1)
        - both strings fuzzy-match (only if fuzzy_strings=True)

    Returns
    -------
    {
        "tp", "fp", "fn",
        "precision", "recall", "f1",
        "matched_fields": [...],
        "missed_fields":  [...],   # in gt but not predicted
        "extra_fields":   [...],   # in pred but not in gt (potential hallucination)
    }
    """
    p_flat = flatten_json(pred or {}, skip_keys=skip_keys)
    g_flat = flatten_json(gt or {}, skip_keys=skip_keys)

    pred_keys = {k for k, v in p_flat.items() if v not in (None, "", [], {})}
    gt_keys   = {k for k, v in g_flat.items() if v not in (None, "", [], {})}

    matched: List[str] = []
    missed: List[str] = []
    extra: List[str] = []

    # Matched: both have the key + values agree
    for k in sorted(pred_keys & gt_keys):
        p_val, g_val = p_flat[k], g_flat[k]
        if _values_match(p_val, g_val, fuzzy_strings, abs_tol):
            matched.append(k)
        else:
            missed.append(k)   # key matches but value disagrees → counted as miss
            extra.append(k)    # also counts as wrong prediction

    missed.extend(sorted(gt_keys - pred_keys))
    extra.extend(sorted(pred_keys - gt_keys))

    tp = len(matched)
    fn = len([k for k in missed if k in gt_keys or k in (gt_keys - pred_keys)])
    # Simpler: tp / total_gt = recall; tp / total_pred = precision
    fp = len(pred_keys) - tp
    fn = len(gt_keys) - tp

    base = pr_f1(tp, fp, fn)
    base.update({
        "matched_fields": matched,
        "missed_fields":  sorted(set(missed)),
        "extra_fields":   sorted(set(extra)),
    })
    return base


def _values_match(p: Any, g: Any,
                  fuzzy_strings: bool,
                  abs_tol: float) -> bool:
    if p == g:
        return True
    if isinstance(p, (int, float)) or isinstance(g, (int, float)):
        return numerical_match(p, g, abs_tol=abs_tol)
    if isinstance(p, str) and isinstance(g, str):
        if normalize_text(p) == normalize_text(g):
            return True
        if fuzzy_strings:
            return fuzzy_match(p, g)
        return False
    if isinstance(p, tuple) and isinstance(g, tuple):
        # ordered list comparison
        if len(p) != len(g):
            return False
        return all(_values_match(x, y, fuzzy_strings, abs_tol)
                   for x, y in zip(p, g))
    return False


# =====================================================================
# 6. Hallucination rate
# =====================================================================
def hallucination_rate(pred: Any,
                       gt: Any,
                       allowed_keys: Optional[Iterable[str]] = None,
                       skip_keys: Iterable[str] = ("_review", "_meta", "type"),
                       ) -> Dict[str, Any]:
    """Hallucination = predicted fields/values not justified by GT.

    Three categories:
        - schema_violations  : keys outside ``allowed_keys`` (if provided)
        - extra_fields       : keys present in pred but not in gt
        - value_mismatches   : keys present in both but values disagree

    Hallucination rate = (#schema_violations + #extra_fields + #value_mismatches)
                       / max(1, #pred_fields)
    """
    p_flat = flatten_json(pred or {}, skip_keys=skip_keys)
    g_flat = flatten_json(gt or {}, skip_keys=skip_keys)
    pred_keys = {k for k, v in p_flat.items() if v not in (None, "", [], {})}
    gt_keys   = {k for k, v in g_flat.items() if v not in (None, "", [], {})}

    schema_violations: List[str] = []
    if allowed_keys is not None:
        allowed = set(allowed_keys)
        # Use top-level segment for schema match (e.g. "tolerance.upper" → "tolerance")
        for k in pred_keys:
            top = k.split(".")[0].split("[")[0]
            if top not in allowed:
                schema_violations.append(k)

    extra_fields = sorted(pred_keys - gt_keys)
    value_mismatches: List[str] = []
    for k in pred_keys & gt_keys:
        if not _values_match(p_flat[k], g_flat[k], fuzzy_strings=False, abs_tol=0.01):
            value_mismatches.append(k)

    n_hall = len(set(schema_violations) | set(extra_fields) | set(value_mismatches))
    rate = safe_div(n_hall, max(1, len(pred_keys)))
    return {
        "rate": round(rate, 4),
        "n_pred_fields": len(pred_keys),
        "n_hallucinations": n_hall,
        "schema_violations": sorted(set(schema_violations)),
        "extra_fields": extra_fields,
        "value_mismatches": sorted(set(value_mismatches)),
    }


# =====================================================================
# 7. Schema-aware comparators (Numerical: Measure / GDT / Roughness)
# =====================================================================
NUMERICAL_SCHEMAS: Dict[str, Set[str]] = {
    "Measure":   {"type", "nominal", "tolerance", "unit",
                  "diameter", "radius", "thread", "depth"},
    "GDT":       {"type", "symbol", "tolerance", "datum", "modifier"},
    "Roughness": {"type", "Ra", "Rz", "Rmax", "unit"},
}


def compare_measure(pred: Dict[str, Any],
                    gt: Dict[str, Any],
                    abs_tol: float = 0.01) -> Dict[str, Any]:
    """Per-Measure schema match. Critical fields: nominal, tolerance bounds, unit."""
    out = field_level_f1(pred, gt, abs_tol=abs_tol,
                         skip_keys=("_review", "_meta", "type"))
    # Add specific flags
    out["nominal_correct"] = numerical_match(
        pred.get("nominal"), gt.get("nominal"), abs_tol=abs_tol,
    )
    if pred.get("tolerance") and gt.get("tolerance"):
        out["tolerance_correct"] = (
            numerical_match(
                (pred.get("tolerance") or {}).get("upper"),
                (gt.get("tolerance") or {}).get("upper"), abs_tol=abs_tol,
            )
            and numerical_match(
                (pred.get("tolerance") or {}).get("lower"),
                (gt.get("tolerance") or {}).get("lower"), abs_tol=abs_tol,
            )
        )
    else:
        out["tolerance_correct"] = (pred.get("tolerance") == gt.get("tolerance"))
    return out


def compare_gdt(pred: Dict[str, Any],
                gt: Dict[str, Any]) -> Dict[str, Any]:
    """Per-GDT schema match. Critical: symbol (Unicode exact), datum (ordered list)."""
    out = field_level_f1(pred, gt, fuzzy_strings=False,
                         skip_keys=("_review", "_meta", "type"))
    out["symbol_correct"] = pred.get("symbol") == gt.get("symbol")
    p_datum = pred.get("datum") or []
    g_datum = gt.get("datum") or []
    out["datum_correct"] = list(p_datum) == list(g_datum)
    return out


def compare_roughness(pred: Dict[str, Any],
                      gt: Dict[str, Any],
                      abs_tol: float = 0.05) -> Dict[str, Any]:
    """Per-Roughness schema match. Critical: Ra value within tolerance."""
    out = field_level_f1(pred, gt, abs_tol=abs_tol,
                         skip_keys=("_review", "_meta", "type"))
    out["Ra_correct"] = numerical_match(
        pred.get("Ra"), gt.get("Ra"), abs_tol=abs_tol,
    )
    return out


def compare_numerical_pair(pred: Dict[str, Any],
                           gt: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the per-class comparator based on ``type`` field."""
    t = (pred.get("type") or gt.get("type") or "").strip()
    if t == "Measure":
        return compare_measure(pred, gt)
    if t == "GDT":
        return compare_gdt(pred, gt)
    if t == "Roughness":
        return compare_roughness(pred, gt)
    return field_level_f1(pred, gt)


# =====================================================================
# 8. TitleBlock / Notes (Stage 3-A) comparators
# =====================================================================
def compare_titleblock(pred_fields: Dict[str, Any],
                       gt_fields: Dict[str, Any],
                       *,
                       fuzzy: bool = True) -> Dict[str, Any]:
    """Compare TitleBlock field dict (post-Stage 3-A).

    Both inputs are the inner ``fields`` dict (drawing_no / material / scale ...).
    """
    return field_level_f1(pred_fields, gt_fields,
                          fuzzy_strings=fuzzy,
                          skip_keys=("_review", "_meta", "type", "fields"))


def compare_notes(pred_items: List[str],
                  gt_items: List[str],
                  *,
                  fuzzy: bool = True,
                  threshold: float = 0.20) -> Dict[str, Any]:
    """Compare two lists of note items. Greedy fuzzy matching."""
    p_items = list(pred_items or [])
    g_items = list(gt_items or [])
    matched_g_idx: Set[int] = set()
    matched_p_idx: Set[int] = set()

    if fuzzy:
        # Greedy: for each pred, match best unused gt within threshold
        for i, p in enumerate(p_items):
            best_j, best_dist = -1, float("inf")
            for j, g in enumerate(g_items):
                if j in matched_g_idx:
                    continue
                d = normalized_edit_distance(p, g)
                if d < best_dist:
                    best_dist, best_j = d, j
            if best_j >= 0 and best_dist <= threshold:
                matched_p_idx.add(i)
                matched_g_idx.add(best_j)
    else:
        norm_p = {normalize_text(x) for x in p_items}
        norm_g = {normalize_text(x) for x in g_items}
        # We can't keep order indices easily here; do set-based
        common = norm_p & norm_g
        return pr_f1(len(common),
                     len(norm_p - norm_g),
                     len(norm_g - norm_p))

    tp = len(matched_p_idx)
    fp = len(p_items) - tp
    fn = len(g_items) - tp
    return pr_f1(tp, fp, fn)


# =====================================================================
# 9. Object detection (BBox / OBB)
# =====================================================================
def bbox_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Axis-aligned BBox IoU. Both as ``[x1, y1, x2, y2]``."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return safe_div(inter, union)


def polygon_iou(p1: Sequence[Sequence[float]],
                p2: Sequence[Sequence[float]]) -> float:
    """OBB polygon IoU using shapely (with axis-aligned fallback)."""
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
        # Fallback: bbox of each polygon
        import numpy as np  # noqa: PLC0415
        a = np.asarray(p1)
        b = np.asarray(p2)
        return bbox_iou(
            [a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()],
            [b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()],
        )


def match_predictions(preds: List[Dict[str, Any]],
                      gts: List[Dict[str, Any]],
                      *,
                      iou_thr: float = 0.5,
                      same_class: bool = True,
                      iou_fn = polygon_iou,
                      pred_box_key: str = "obb",
                      gt_box_key: str = "obb",
                      ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy IoU matching between pred / gt detections.

    Returns
    -------
    (matched, unmatched_pred_idx, unmatched_gt_idx)
        matched : list of (pred_idx, gt_idx)
    """
    used_gt: Set[int] = set()
    matched: List[Tuple[int, int]] = []
    unmatched_pred: List[int] = []

    # Sort preds by conf desc for stable greedy
    order = sorted(range(len(preds)),
                   key=lambda i: -preds[i].get("conf", 0.0))

    for pi in order:
        p = preds[pi]
        best_iou, best_gj = 0.0, -1
        for gj, g in enumerate(gts):
            if gj in used_gt:
                continue
            if same_class and p.get("class") != g.get("class"):
                continue
            iou = iou_fn(p[pred_box_key], g[gt_box_key])
            if iou > best_iou:
                best_iou, best_gj = iou, gj
        if best_iou >= iou_thr and best_gj >= 0:
            matched.append((pi, best_gj))
            used_gt.add(best_gj)
        else:
            unmatched_pred.append(pi)

    unmatched_gt = [i for i in range(len(gts)) if i not in used_gt]
    return matched, unmatched_pred, unmatched_gt


def detection_metrics(preds: List[Dict[str, Any]],
                      gts: List[Dict[str, Any]],
                      *,
                      iou_thr: float = 0.5,
                      classes: Optional[List[str]] = None,
                      iou_fn = polygon_iou,
                      ) -> Dict[str, Any]:
    """Compute per-class TP/FP/FN/P/R/F1. ``classes=None`` → infer from data."""
    if classes is None:
        classes = sorted({d.get("class") for d in (preds + gts) if d.get("class")})

    matched, unm_p, unm_g = match_predictions(
        preds, gts, iou_thr=iou_thr, same_class=True, iou_fn=iou_fn,
    )

    per_class: Dict[str, Dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0} for c in classes
    }
    for pi, gj in matched:
        c = preds[pi]["class"]
        per_class.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
    for pi in unm_p:
        c = preds[pi]["class"]
        per_class.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
    for gj in unm_g:
        c = gts[gj]["class"]
        per_class.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    out: Dict[str, Any] = {"per_class": {}, "overall": None}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for c, st in per_class.items():
        out["per_class"][c] = pr_f1(st["tp"], st["fp"], st["fn"])
        out["per_class"][c]["missing_rate"] = round(
            safe_div(st["fn"], st["tp"] + st["fn"]), 4,
        )
        for k in total:
            total[k] += st[k]
    out["overall"] = pr_f1(total["tp"], total["fp"], total["fn"])
    return out


# =====================================================================
# 10. Aggregators
# =====================================================================
def aggregate_per_class(records: List[Dict[str, Any]],
                        key: str = "class") -> Dict[str, Dict[str, int]]:
    """Group ``[{class, tp, fp, fn}, ...]`` into class-keyed totals."""
    out: Dict[str, Dict[str, int]] = {}
    for r in records:
        c = r.get(key, "_unknown")
        bucket = out.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})
        for k in ("tp", "fp", "fn"):
            bucket[k] += int(r.get(k, 0))
    return out


def confusion_matrix(predictions: Sequence[Tuple[int, int]],
                     n_classes: int) -> List[List[int]]:
    """Build n×n CM from list of (predicted_class, actual_class) pairs."""
    cm = [[0] * n_classes for _ in range(n_classes)]
    for p, a in predictions:
        if 0 <= p < n_classes and 0 <= a < n_classes:
            cm[a][p] += 1
    return cm


def numerical_accuracy(preds: List[Any],
                       gts: List[Any],
                       *,
                       abs_tol: float = 0.01) -> Dict[str, Any]:
    """For a list of (pred, gt) numeric pairs, compute hit rate."""
    n = min(len(preds), len(gts))
    if n == 0:
        return {"n": 0, "hits": 0, "accuracy": 0.0}
    hits = sum(
        1 for p, g in zip(preds, gts)
        if numerical_match(p, g, abs_tol=abs_tol)
    )
    return {"n": n, "hits": hits,
            "accuracy": round(hits / n, 4)}


# =====================================================================
# Self-test (sanity check)
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" src/utils/metrics.py — sanity tests")
    print("=" * 60)

    # 1. Core P/R/F1
    r = pr_f1(tp=8, fp=2, fn=3)
    assert r["precision"] == 0.8 and r["recall"] == 0.7273, r
    print("✓ pr_f1(8,2,3) →", r)

    # 2. Set-based
    r = set_pr_f1(["a", "b", "c"], ["b", "c", "d"])
    assert r["tp"] == 2, r
    print("✓ set_pr_f1 →", r)

    # 3. Edit distance
    assert edit_distance("kitten", "sitting") == 3
    assert normalized_edit_distance("hello", "hello") == 0.0
    assert fuzzy_match("DWG-001-A", "DWG-001-B", threshold=0.2)
    print("✓ edit_distance / fuzzy_match")

    # 4. Numerical match
    assert numerical_match(25.0, 25.005, abs_tol=0.01)
    assert not numerical_match(25.0, 25.5, abs_tol=0.01)
    assert numerical_match("Ø25.4", 25.4)   # string with prefix
    print("✓ numerical_match (incl. string coercion)")

    # 5. Field-level F1 — Measure
    pred = {"type": "Measure", "nominal": 25.0,
            "tolerance": {"upper": 0.05, "lower": -0.05}, "unit": "mm"}
    gt   = {"type": "Measure", "nominal": 25.0,
            "tolerance": {"upper": 0.05, "lower": -0.05}, "unit": "mm"}
    r = field_level_f1(pred, gt)
    assert r["f1"] == 1.0, r
    print("✓ field_level_f1 perfect →", {"f1": r["f1"], "tp": r["tp"]})

    # 6. Field-level F1 — partial
    pred2 = {"type": "Measure", "nominal": 25.0, "unit": "mm"}
    gt2   = {"type": "Measure", "nominal": 25.0,
             "tolerance": {"upper": 0.05, "lower": -0.05}, "unit": "mm"}
    r = field_level_f1(pred2, gt2)
    print("✓ field_level_f1 partial →",
          {"f1": r["f1"], "missed": r["missed_fields"]})
    assert "tolerance.upper" in r["missed_fields"]

    # 7. Hallucination
    pred3 = {"nominal": 25.0, "unit": "mm",
             "imaginary_field": 999, "tolerance": {"upper": 0.05, "lower": -0.05}}
    gt3   = {"nominal": 25.0, "unit": "mm"}
    h = hallucination_rate(pred3, gt3, allowed_keys={"nominal", "unit"})
    print("✓ hallucination_rate →", h["rate"],
          "schema_viol:", h["schema_violations"],
          "extra:", h["extra_fields"])

    # 8. Compare Measure
    r = compare_measure(pred, gt)
    assert r["nominal_correct"] and r["tolerance_correct"]
    print("✓ compare_measure full match")

    # 9. Compare GDT
    g_pred = {"type": "GDT", "symbol": "⏤", "tolerance": 0.02,
              "datum": ["A", "B"]}
    g_gt   = {"type": "GDT", "symbol": "⏤", "tolerance": 0.02,
              "datum": ["A", "B"]}
    r = compare_gdt(g_pred, g_gt)
    assert r["symbol_correct"] and r["datum_correct"]
    print("✓ compare_gdt full match")

    # 10. Compare Roughness
    r_pred = {"type": "Roughness", "Ra": 1.6, "unit": "μm"}
    r_gt   = {"type": "Roughness", "Ra": 1.605, "unit": "μm"}
    r = compare_roughness(r_pred, r_gt)
    assert r["Ra_correct"]
    print("✓ compare_roughness within tol")

    # 11. TitleBlock
    tb_p = {"drawing_no": "DWG-001-A", "material": "SS400", "scale": "1:2"}
    tb_g = {"drawing_no": "DWG-001-A", "material": "SS400",
            "scale": "1:2", "revision": "B"}
    r = compare_titleblock(tb_p, tb_g)
    print("✓ compare_titleblock →",
          {"f1": r["f1"], "missed": r["missed_fields"][:3]})

    # 12. Notes (fuzzy, partial match)
    notes_p = ["1. UNLESS OTHERWISE SPECIFIED",
               "2. ALL DIMS IN MM",
               "3. break sharp edges"]
    notes_g = ["1. UNLESS OTHERWISE SPECIFIED",
               "2. ALL DIMENSIONS IN MM",
               "3. BREAK ALL SHARP EDGES"]
    r = compare_notes(notes_p, notes_g, fuzzy=True, threshold=0.30)
    print("✓ compare_notes fuzzy →", r)

    # 13. BBox IoU
    iou = bbox_iou([10, 10, 50, 50], [30, 30, 70, 70])
    print(f"✓ bbox_iou (expected ~0.143) → {iou:.4f}")

    # 14. Polygon IoU
    p1 = [[0, 0], [10, 0], [10, 10], [0, 10]]
    p2 = [[5, 5], [15, 5], [15, 15], [5, 15]]
    iou = polygon_iou(p1, p2)
    print(f"✓ polygon_iou (expected ~0.143) → {iou:.4f}")

    # 15. Detection metrics
    preds_d = [
        {"class": "Measure", "obb": [[0, 0], [10, 0], [10, 10], [0, 10]], "conf": 0.9},
        {"class": "GDT",     "obb": [[20, 20], [30, 20], [30, 30], [20, 30]], "conf": 0.8},
        {"class": "Measure", "obb": [[100, 100], [110, 100], [110, 110], [100, 110]], "conf": 0.7},
    ]
    gts_d = [
        {"class": "Measure", "obb": [[1, 1], [11, 1], [11, 11], [1, 11]]},
        {"class": "GDT",     "obb": [[20, 20], [30, 20], [30, 30], [20, 30]]},
        {"class": "Roughness", "obb": [[50, 50], [60, 50], [60, 60], [50, 60]]},
    ]
    m = detection_metrics(preds_d, gts_d, iou_thr=0.5)
    print("✓ detection_metrics →")
    print("  overall:", m["overall"])
    for c, v in m["per_class"].items():
        print(f"  {c:10s}: {v}")

    print()
    print("All sanity tests passed ✓")
