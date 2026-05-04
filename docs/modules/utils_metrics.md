# `src/utils/metrics.py`

> **Step 8** — 평가 지표 라이브러리. 모든 검증기(V5~V9) 가 공통 사용.

## 1. 구현 요약

순수 라이브러리 모듈 (CLI 없음). 10개 섹션으로 구성된 ~600 lines.

| § | 섹션 | 핵심 함수 |
|---|---|---|
| 1 | Core P/R/F1 | `pr_f1()`, `safe_div()` |
| 2 | Set-based 비교 | `set_pr_f1()` |
| 3 | String 지표 | `edit_distance()`, `normalized_edit_distance()`, `fuzzy_match()`, `normalize_text()` |
| 4 | Numerical (tolerance) | `coerce_number()`, `numerical_match(abs_tol, rel_tol)` |
| 5 | JSON flatten + field F1 | `flatten_json()`, `field_level_f1()` |
| 6 | Hallucination | `hallucination_rate(allowed_keys=)` |
| 7 | Schema-aware (Donut) | `compare_measure()`, `compare_gdt()`, `compare_roughness()`, `compare_numerical_pair()` |
| 8 | TitleBlock / Notes | `compare_titleblock()`, `compare_notes()` |
| 9 | Detection (BBox/OBB) | `bbox_iou()`, `polygon_iou()`, `match_predictions()`, `detection_metrics()` |
| 10 | Aggregators | `aggregate_per_class()`, `confusion_matrix()`, `numerical_accuracy()` |

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 라이브러리 vs CLI | **라이브러리만** (CLI 없음) | 검증기에서 import 해 사용 |
| Edit distance | `editdistance` 라이브러리 + 순수 Python fallback | 의존성 graceful |
| Polygon IoU | `shapely` + axis-aligned bbox fallback | check_stage2_model 와 동일 정책 |
| **Numerical tolerance** | `abs_tol` (e.g. ±0.01) 또는 `rel_tol` (e.g. ±0.1%) **OR** | 엔지니어링 도면 표준 (작은 값은 abs, 큰 값은 rel) |
| `coerce_number("Ø25.4")` | 정규식으로 첫 숫자 추출 → 25.4 | 엔지니어링 표기 호환 |
| Fuzzy string | normalized edit distance ≤ 20% (default) | 도면 텍스트 OCR 노이즈 |
| Field-level F1 | flatten_json() 으로 nested → dotted-key 평탄화 | 일관된 비교 |
| Skip keys 기본값 | `_review`, `_meta`, `type` | provenance 메타는 평가 제외 |
| Schema 비교 | per-class 함수 분리 (Measure/GDT/Roughness) | 클래스별 critical field 다름 |
| Notes 비교 | greedy fuzzy 매칭 (item 단위) | 순서 변동 + 텍스트 변형 허용 |
| Detection 매칭 | greedy IoU + same-class | 표준 평가 방식 |
| 자체 sanity test | `__main__` 블록에 15개 | 회귀 방지 |

## 3. 사용법

### 3.1 검증기에서 import

```python
from src.utils.metrics import (
    pr_f1, set_pr_f1,
    edit_distance, normalized_edit_distance, fuzzy_match,
    numerical_match, coerce_number,
    flatten_json, field_level_f1, hallucination_rate,
    compare_measure, compare_gdt, compare_roughness, compare_numerical_pair,
    compare_titleblock, compare_notes,
    bbox_iou, polygon_iou, match_predictions, detection_metrics,
    aggregate_per_class, confusion_matrix, numerical_accuracy,
)
```

### 3.2 V5 (Stage 3-A) 에서 활용 예

```python
# TitleBlock F1
tb_metrics = compare_titleblock(pred_fields, gt_fields, fuzzy=True)
# → {tp, fp, fn, precision, recall, f1, matched_fields, missed_fields, extra_fields}

# Notes 항목 단위 F1
notes_metrics = compare_notes(pred_items, gt_items, fuzzy=True, threshold=0.30)

# Hallucination
h = hallucination_rate(pred, gt, allowed_keys=ALPHABETICAL_FIELDS)
# → {rate, n_hallucinations, schema_violations, extra_fields, value_mismatches}
```

### 3.3 V6 (Stage 3-N) 에서 활용 예

```python
# 클래스별 비교
for cls, pred, gt in pairs:
    if cls == "Measure":
        m = compare_measure(pred, gt, abs_tol=0.01)
        # m["nominal_correct"], m["tolerance_correct"]
    elif cls == "GDT":
        m = compare_gdt(pred, gt)
        # m["symbol_correct"], m["datum_correct"]
    elif cls == "Roughness":
        m = compare_roughness(pred, gt, abs_tol=0.05)

# 또는 dispatcher
m = compare_numerical_pair(pred, gt)
```

### 3.4 V7 (end-to-end) 에서 활용 예

```python
# 전체 통합 JSON 비교
e2e = field_level_f1(unified_pred, unified_gt,
                     fuzzy_strings=True, abs_tol=0.01)

# Detection 매칭 (Stage 1+2 합)
det = detection_metrics(all_preds, all_gts, iou_thr=0.5)
# det["per_class"]["Measure"]["missing_rate"]
```

## 4. 검증 결과

### 4.1 15개 Sanity Test 모두 PASS

```
✓ pr_f1(8,2,3) → P=0.8, R=0.7273, F1=0.7619
✓ set_pr_f1(['a','b','c'], ['b','c','d']) → tp=2 P=R=F1=0.667
✓ edit_distance("kitten", "sitting") = 3
✓ fuzzy_match("DWG-001-A", "DWG-001-B", 0.2) = True
✓ numerical_match(25.0, 25.005, 0.01) = True
✓ numerical_match("Ø25.4", 25.4) = True   (string 압축)
✓ field_level_f1 perfect → F1=1.0
✓ field_level_f1 partial → F1=0.667 (tolerance 누락 정확 검출)
✓ hallucination_rate → 0.6 (3 가짜 필드 / 5 예측 필드)
✓ compare_measure full match → nominal_correct=True, tolerance_correct=True
✓ compare_gdt full match → symbol_correct=True, datum_correct=True
✓ compare_roughness 0.05 tol → Ra_correct=True (1.6 vs 1.605)
✓ compare_titleblock → F1=0.857 (1 누락)
✓ compare_notes fuzzy → tp=2 fp=1 fn=1 (3개 중 2개 매칭)
✓ bbox_iou([10,10,50,50], [30,30,70,70]) → 0.1429
✓ polygon_iou (정사각형 겹침) → 0.1429 (shapely)
✓ detection_metrics → 클래스별 P/R/F1/missing_rate 정확 산출
```

### 4.2 BBox vs Polygon IoU 일치성

axis-aligned 정사각형 두 개에 대해 `bbox_iou` 와 `polygon_iou` 가 **정확히 같은 값** (0.1429) 산출 → 두 함수 일관성 확인.

### 4.3 Field-level F1 partial 케이스

```
pred:  {nominal: 25.0, unit: "mm"}
gt:    {nominal: 25.0, tolerance: {upper: 0.05, lower: -0.05}, unit: "mm"}
→ F1 = 0.667
  missed_fields: ['tolerance.lower', 'tolerance.upper']
```

flatten 으로 nested 키도 정확히 추적.

## 5. 주요 함수 상세

### 5.1 `numerical_match(pred, gt, abs_tol, rel_tol)`

엔지니어링 공차 매칭. 다음 중 하나면 True:
- `pred == gt` (exact)
- `|pred - gt| ≤ abs_tol` (절대 공차, 작은 값 대응)
- `|pred - gt| ≤ rel_tol × max(|pred|, |gt|)` (상대 공차, 큰 값 대응)

문자열 입력은 `coerce_number()` 로 첫 숫자 추출:
- `"Ø25.4"` → 25.4
- `"M8 thread"` → 8.0
- `"Ra 1.6 μm"` → 1.6

### 5.2 `field_level_f1(pred, gt, fuzzy_strings, abs_tol)`

JSON dict 두 개 비교. 처리 단계:
1. `flatten_json()` 으로 nested → dotted-key (`tolerance.upper`)
2. 비어있지 않은 키 집합 비교 (`tp = pred ∩ gt with values matching`)
3. 값 매칭 룰:
   - 숫자 → `numerical_match(abs_tol)`
   - 문자열 → 정규화 동일 OR fuzzy (옵션)
   - 리스트(tuple) → 순서대로 재귀
4. 반환: `tp/fp/fn/precision/recall/f1` + `matched/missed/extra_fields`

### 5.3 `hallucination_rate(pred, gt, allowed_keys)`

3가지 hallucination 카테고리 합산:
1. **schema_violations** — `allowed_keys` 외의 필드 (예: 정의 외 영역 추가)
2. **extra_fields** — pred 에 있지만 gt 에 없는 필드
3. **value_mismatches** — 같은 키지만 값 다름

`rate = n_hallucinations / max(1, n_pred_fields)` (D-021 임계값과 비교 가능)

### 5.4 `detection_metrics(preds, gts, iou_thr, classes)`

YOLO 검출 결과 평가. 흐름:
1. `match_predictions()` — greedy IoU + same-class 매칭
2. 클래스별 TP/FP/FN 누적
3. `pr_f1()` 으로 P/R/F1 산출
4. **`missing_rate = FN / (TP + FN)`** ← D-023 사용자 필수 임계값

## 6. 의존성

```
편의 (옵션):
- editdistance>=0.8.1   # Levenshtein, 순수 Python fallback 있음
- shapely>=2.0.0        # OBB polygon IoU, axis-aligned bbox fallback 있음
- numpy                 # confusion_matrix 등
```

핵심 함수는 stdlib 만으로 동작 (graceful degradation).

## 7. 관련 의사결정

- **D-021** Severity 분류 — 검증기는 이 모듈 결과를 임계값과 비교해 PASS/FAIL/WARN 판정
- **D-022** Provenance — `_review` / `_meta` 자동 skip
- **D-023** 사용자 필수 임계값 — `detection_metrics` 의 `missing_rate` 가 직접 측정
- **D-024** Group-aware split (이 모듈은 평가만, 그룹 분리는 학습 시점)

## 8. 검증 대상 모듈 (이 모듈을 사용)

- [`check_stage2_model.py`](./check_stage2_model.md) — `polygon_iou`, `match_predictions` 활용 가능
- [`check_stage1_model.py`](./check_stage1_model.md) — `bbox_iou`, `pr_f1` 활용 가능
- **V5** `check_stage3a_alphabetical.py` — `compare_titleblock`, `compare_notes`, `hallucination_rate`
- **V6** `check_stage3n_numerical.py` — `compare_measure`, `compare_gdt`, `compare_roughness`, `numerical_accuracy`
- **V7** `check_pipeline_e2e.py` — `field_level_f1`, `detection_metrics`
- **V9** `check_enrichment.py` — `set_pr_f1`, `aggregate_per_class`

## 9. 자체 테스트 실행

```bash
python -m src.utils.metrics
```

15개 sanity test 가 즉시 실행됨. CI 또는 작업 시작 시 활용.

## 10. 확장 가능성

추가 가능한 지표 (필요 시):
- **mAP** (average precision over IoU thresholds) — 현재는 mAP@0.5 만 지원, ultralytics 가 IoU sweep 지원
- **CER / WER** — 텍스트 인식 평가 (Stage 3-A Notes 에 도움)
- **BLEU / ROUGE** — Notes 자유 텍스트 평가
- **Cohen's kappa** — inter-annotator agreement
- **AUC / PR-curve** — confidence threshold sweep
