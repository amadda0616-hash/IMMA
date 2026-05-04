# `src/validate/check_pipeline_e2e.py`

> **V7** — End-to-End Pipeline (Step 7) 사후 검증 — D-023 재검증 포함

## 1. 구현 요약

`pipeline.py` 의 통합 출력 JSON 을 GT 와 비교해 **field-level F1, detection 누락률, numerical content 정확도, per-stage timing, failure rate** 을 전부 측정.

**검증 항목 (13개)**

| 항목 | 임계값 | Severity |
|---|---|---|
| `field_f1 (overall, mean)` | ≥ 0.75 | **critical** |
| `field_f1[title_block]` | — | info |
| `field_f1[notes]` | — | info |
| `missing_rate[Measure]` (★ e2e) | < 0.08 | **critical** (D-023 재측정) |
| `missing_rate[GDT]` (★ e2e) | < 0.05 | **critical** (D-023 재측정) |
| `missing_rate[Roughness]` | < 0.30 | warning |
| `drawing_level_recall` (★ e2e) | ≥ 0.85 | **critical** |
| `detection_f1[Measure/GDT/Roughness]` | — | info |
| `numerical_content_f1[Measure/GDT/Roughness]` | — | info |
| `numerical_accuracy (e2e)` | ≥ 0.95 | warning |
| `inference_time_per_drawing` (mean) | ≤ 30s | warning |
| `failure_rate` (from summary) | < 0.01 | **critical** |
| Per-stage timing 표 + 막대 차트 | — | — |
| 최악 10건 표 (overall F1 낮은 순) | — | — |

★ = D-023 사용자 필수 임계값을 e2e 시점에서 재측정 (Stage 2 단독 측정과는 별개).

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| Annotation 매칭 | 모든 view 의 annotation 평탄화 → global IoU 매칭 | view 단위 매칭은 복잡, 누적 오차 누락 |
| IoU 함수 | `polygon_iou` (shapely + bbox fallback) | check_stage2_model 와 일관 |
| 매칭 후 numerical 비교 | 같은 (pred, gt) 쌍의 `parsed` 만 비교 | 매칭 안 된 건 detection 측정에 반영 |
| Per-class 집계 | TP/FP/FN 합산 → recompute F1 | 평균이 아닌 micro-F1 |
| Drawing-level recall | 도면별 TP/(TP+FN) 평균 | D-023 정의 |
| Timing 분석 | mean / median / p95 / max | outlier 추적 |
| Per-stage breakdown | mean per stage_key | 병목 식별 |
| Failure rate | summary JSON 의 n_err/n_total | pipeline.py 가 생성한 매니페스트 활용 |
| 최악 10건 | overall F1 정렬 후 첫 10 | 디버깅 시 우선순위 |
| Skip keys | meta / _review / raw_seq / image_path | 평가 무관 메타데이터 |

## 3. 사용법

```bash
# 기본
python -m src.validate.check_pipeline_e2e \
    --predictions outputs/json/ \
    --gt data/validation_gt/e2e/ \
    --summary outputs/json/_pipeline_summary.json

# summary 없이 (timing 만 pred 안에서)
python -m src.validate.check_pipeline_e2e \
    --predictions outputs/json/ \
    --gt data/validation_gt/e2e/

# 임계값 커스터마이즈
python -m src.validate.check_pipeline_e2e \
    --predictions outputs/json/ --gt data/validation_gt/e2e/ \
    --thresholds configs/validation_thresholds.yaml
```

### GT 형식

`data/validation_gt/e2e/<stem>.json` — `pipeline.py` 출력과 같은 schema (HANDOFF §5.5).

```json
{
  "drawing_id": "drawing_001",
  "title_block": {"drawing_no": "DWG-001", "material": "SUS304", ...},
  "notes": ["1. ...", "2. ..."],
  "views": [{
    "view_id": "view_0",
    "bbox": [...],
    "annotations": [
      {"class": "Measure", "obb_global": [...], "parsed": {"nominal": 25.0, ...}},
      {"class": "GDT",     "obb_global": [...], "parsed": {"symbol": "⏤", ...}}
    ]
  }]
}
```

## 4. 검증 결과

### 4.1 2건 더미 데이터 — 모든 케이스 정확 검출

**입력**:
- sample_001: 거의 완벽 매칭 (Notes 1개 단어 차이만)
- sample_002: 부분 매칭 (TB material/scale 누락 + Notes 1개 누락 + Measure 1개 누락 + GDT 1개 누락)
- summary: 2건 모두 OK, total 56s

**결과 (FAIL 4건 — 모두 의도된 fail)**:

```
[ 1/13] field_f1 (overall, mean)         0.7099  ✗ FAIL  ≥ 0.7500   critical
[ 4/13] missing_rate[Measure] (★ e2e)    0.3333  ✗ FAIL  ≤ 0.0800   critical (FN=1/3)
[ 6/13] missing_rate[GDT] (★ e2e)        0.5000  ✗ FAIL  ≤ 0.0500   critical (FN=1/2)
[ 8/13] drawing_level_recall (★ e2e)     0.6667  ✗ FAIL  ≥ 0.8500   critical
[ 5/13] detection_f1[Measure]            0.8000  · INFO
[ 9/13] numerical_content_f1[Measure]    1.0000  · INFO  (매칭된 건은 모두 정확)
[10/13] numerical_accuracy (e2e)         1.0000  ✓ PASS  ≥ 0.9500
[12/13] inference_time_per_drawing       28.000  ✓ PASS  ≤ 30
[13/13] failure_rate                     0.0000  ✓ PASS  ≤ 0.0100

Overall: FAIL   PASS=3 WARN=0 FAIL=4 INFO=6
```

**해석**:
- sample_002 가 GT 의 3개 annotation 중 1개만 검출 → 누락률 정확 측정
- 매칭된 annotation 들의 numerical content 는 모두 정확 → numerical_accuracy 1.0
- 도면 평균 회수율 = (1.0 + 0.33) / 2 = 0.67 → drawing_level_recall FAIL

## 5. 출력 형식

### 5.1 콘솔 (위 §4.1)

### 5.2 HTML 첨부

- **Per-stage timing** 표 (stage / mean_s / max_s / n)
- **Mean time per stage** 막대 차트 (병목 시각화)
- **Detection metrics per class (★ e2e)** 표 (TP/FP/FN/F1/missing_rate)
- **Detection F1 per class** 막대 차트
- **Worst 10 drawings** 표 (image_stem / overall_f1 / tb_f1 / notes_f1 / total_s)

### 5.3 JSON

```json
{
  "step": "pipeline_e2e",
  "overall_status": "FAIL",
  "metadata": {
    "predictions_dir": "...",
    "gt_dir": "...",
    "n_pred": 2, "n_gt": 2,
    "summary_used": true
  },
  "checks": [...13 items...],
  "artifacts": [
    {"kind": "table", "title": "Per-stage timing"},
    {"kind": "plot",  "title": "Mean time per stage"},
    {"kind": "table", "title": "Detection metrics per class"},
    {"kind": "plot",  "title": "Detection F1 per class"},
    {"kind": "table", "title": "Worst 10 drawings"}
  ]
}
```

## 6. 의존성

```
PyYAML, numpy
matplotlib, jinja2  (via common.py)
src.utils.metrics   (detection_metrics, compare_*, polygon_iou, field_level_f1)
src.validate.common
shapely  (옵션, polygon IoU 정확도 향상)
```

## 7. 관련 의사결정

- **D-020** 검증 의무화
- **D-021** Severity (4개 critical: field_f1 / missing_rate × 2 / failure_rate)
- **D-022** Provenance / timing 측정 의무
- **D-023** ★ 누락률 + 회수율 e2e 시점 재검증 (Stage 2 단독과 별개로)
- **D-024** Group-aware split (이 모듈은 평가만)

## 8. 검증 대상 모듈

[`pipeline.md`](./pipeline.md) — Step 7

## 9. Plan A → Plan B 분기점 재확인

V7 의 critical FAIL 발생 시:

| FAIL | 진단 | 1차 / 2차 조치 |
|---|---|---|
| `field_f1 < 0.75` | Stage 3 출력 부족 | 1) Donut fine-tune 더 / 2) Step 9 enrichment 강화 |
| `missing_rate[Measure] > 8%` | Stage 2 누락 | 1) check_stage2_model 결과 재확인 / 2) Plan B (D-017 symbol_postcorrect) |
| `drawing_level_recall < 0.85` | 도면 단위 일관성 부족 | 1) 라벨 재검수 / 2) padding 조정 |
| `inference_time > 30s` | DocVQA 14회 호출이 비싼 부분 | 1) batch 옵션 / 2) `--skip-alphabetical` |
| `failure_rate > 0.01` | pipeline.py 예외 | summary JSON 의 error 메시지 확인 |

## 10. 사용 흐름

```
Step 7 (pipeline.py batch)
       ↓
outputs/json/<id>.json + _pipeline_summary.json
       ↓
사용자가 검수 완료한 GT — data/validation_gt/e2e/<id>.json
       ↓
★ V7 실행
       ↓
PASS → production 배포 / Step 9 enrichment 진행
FAIL → 단계별 V2/V3/V5/V6 결과 재확인 → 약점 단계 보강
```

## 11. 한계

- GT 작성이 매우 노동집약적 (도면 한 장에 모든 영역+annotation 라벨링)
- 50~100건 sampling 으로도 의미 있음 (전수 불필요)
- View 단위 매칭 안 함 — 같은 클래스 + IoU 만으로 매칭하므로, GT 와 view 분할이 다르면 결과 영향 가능

## 12. 향후 개선

- View-aware 매칭 (parent_bbox 기반 view 매칭 → annotation 매칭)
- GPU memory peak 측정 (현재 timing 만)
- Confidence-vs-correctness curve (PR curve 별도 산출)
- Per-language e2e F1 (현재 V5 만 분석)
