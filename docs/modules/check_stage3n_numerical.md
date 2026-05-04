# `src/validate/check_stage3n_numerical.py`

> **V6** ★ — Stage 3-N (Donut Numerical, fine-tuned) 사후 검증 — **D-023 핵심**

## 1. 구현 요약

Donut Numerical fine-tuned 모델의 예측 JSON 과 사람 검수 GT 를 비교해 **사용자 필수 임계값 (D-023)** 을 측정하는 가장 중요한 검증기.

**검증 항목 (16개)**

| 항목 | 임계값 | Severity | 출처 |
|---|---|---|---|
| `field_f1[Measure]` | ≥ 0.90 | **critical** | 논문 0.923 (★) |
| `field_f1[GDT]` | ≥ 0.95 | **critical** | 논문 0.965 (★) |
| `field_f1[Roughness]` | ≥ 0.95 | warning | 논문 1.0 |
| `field_f1[overall]` | — | info | 논문 0.963 |
| `numerical_accuracy` | ≥ 0.95 | **critical** | nominal ±0.01 mm 매칭 (★) |
| `tolerance_match` | ≥ 0.90 | **critical** | upper+lower 동시 (★) |
| `symbol_accuracy[GDT]` | ≥ 0.95 | warning | 14 ISO 1101 심볼 |
| `datum_accuracy[GDT]` | ≥ 0.90 | warning | 순서 일치 |
| `rare_symbol_accuracy` | ≥ 0.70 | warning | 빈도 하위 50% 심볼 |
| `Ra_accuracy[Roughness]` | ≥ 0.90 | warning | ±0.05 μm |
| `hallucination_rate (overall)` | < 0.10 | **critical** | 논문 0.067 (★) |
| `hallucination_rate[Measure/GDT/Roughness]` | — | info | per-class 분석 |
| `empty_rate` | < 0.05 | warning | parsed 비어있는 비율 |
| `unmatched_predictions` | — | info | GT 없는 예측 |
| `skipped_incomplete_gt` | — | info | `_review.completed=false` 제외 |
| `parse_errors` | = 0 | warning | JSON 파싱 실패 |

★ = 사용자 필수 임계값 (D-023). 미달 시 Plan B 발동 검토 (eDOCr2 차용).

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 매칭 방식 | 파일 stem 일치 (`<stem>.num.json` ↔ `<stem>.json`) | V5 와 동일 패턴 |
| Pred 구조 처리 | `pred.parsed` 가 inner schema | stage3_numerical.py 출력 형식 |
| GT review skip | `_review.completed != true` 면 평가에서 제외 | prepare_vlm_dataset.py 호환 |
| Per-class 평가 | `compare_measure / compare_gdt / compare_roughness` from metrics.py | 클래스별 critical field 다름 |
| ★ Numerical accuracy | Measure 의 nominal 만 별도 측정 (±0.01 tol) | D-023 사용자 필수 |
| ★ Tolerance match | Measure 의 upper+lower 동시 매칭 | D-023 |
| Symbol accuracy | Unicode exact match | ISO 1101 표준 |
| Datum accuracy | list ordered comparison | 순서 의미 있음 |
| Rare symbol | 빈도 하위 50% (info but separately tracked) | 희귀 심볼 별도 추적 |
| Hallucination | NUMERICAL_SCHEMAS 의 키만 허용 | metrics.py 의 schema |
| Per-symbol breakdown | GDT 만 (14개 심볼별 hit rate 표 + 차트) | 약점 시각화 |
| Empty 정의 | parsed 의 모든 값이 falsy | recall=0 추적 |

## 3. 사용법

```bash
# 기본
python -m src.validate.check_stage3n_numerical \
    --predictions outputs/sample/numerical/ \
    --gt data/vlm/numerical/

# 임계값 커스터마이즈
python -m src.validate.check_stage3n_numerical \
    --predictions outputs/sample/numerical/ \
    --gt data/vlm/numerical/ \
    --thresholds configs/validation_thresholds.yaml
```

### GT 형식 호환성

V6 는 두 가지 GT 형식을 모두 지원:

**형식 A** — 순수 GT (학습 데이터로 직접 사용 가능)
```json
{
  "type": "Measure",
  "nominal": 25.0,
  "tolerance": {"upper": 0.05, "lower": -0.05},
  "unit": "mm"
}
```

**형식 B** — `prepare_vlm_dataset.py` 출력 (검수 진행 추적)
```json
{
  "type": "Measure",
  "nominal": 25.0,
  ...,
  "_review": {
    "completed": true,        ← false 면 평가 skip
    "ocr_hint": "25.0",
    "source_image": "..."
  }
}
```

`_review.completed=false` 인 GT 는 자동으로 `skipped_incomplete_gt` 카운트.

## 4. 검증 결과

### 4.1 7건 더미 데이터 검증 (모든 케이스 정확 검출)

| 케이스 | 입력 | 기대 결과 |
|---|---|---|
| 1 | Measure 완벽 매칭 | F1↑, nominal_correct=True |
| 2 | Measure nominal 틀림 (25 vs 26.5) | numerical_accuracy ↓ |
| 3 | GDT 완벽 매칭 | symbol+datum_correct=True |
| 4 | GDT 심볼+datum 틀림 (희귀) | symbol_accuracy ↓ |
| 5 | Roughness 완벽 | Ra_correct=True |
| 6 | Roughness incomplete (`completed: false`) | **자동 skip** |
| 7 | Measure + 가짜 필드 | hallucination ↑ |

**최종 콘솔**

```
Evaluated 6 pairs (unmatched=0 incomplete=1 errors=0)

[ 1/16] field_f1[Measure]            0.7059  ✗ FAIL  ≥ 0.9000   ★ critical
[ 2/16] field_f1[GDT]                0.5000  ✗ FAIL  ≥ 0.9500   ★ critical
[ 3/16] field_f1[Roughness]          1.0000  ✓ PASS  ≥ 0.9500
[ 5/16] numerical_accuracy (★)       0.6667  ✗ FAIL  ≥ 0.9500   ★ critical
[ 6/16] tolerance_match (★)          1.0000  ✓ PASS  ≥ 0.9000
[ 7/16] symbol_accuracy[GDT]         0.5000  ! WARN  ≥ 0.9500
[ 8/16] datum_accuracy[GDT]          0.5000  ! WARN  ≥ 0.9000
[10/16] Ra_accuracy[Roughness]       1.0000  ✓ PASS  ≥ 0.9000
[11/16] hallucination_rate (★)       0.3529  ✗ FAIL  ≤ 0.1000   ★ critical
[16/16] skipped_incomplete_gt              1  · INFO  (sample_006)

Overall: FAIL   PASS=5  WARN=2  FAIL=4  INFO=5
```

4개 critical FAIL 정확 → 학습 후 실제 데이터에서도 명확한 PASS/FAIL 판정 가능.

### 4.2 Plan B 발동 결정점

V6 의 critical 임계값 미달 시 **Plan B (eDOCr2 차용) 검토**:

| FAIL | 1차 조치 | 2차 (Plan B) |
|---|---|---|
| Measure F1 < 0.90 | epochs ↑ / batch ↑ / 학습 데이터 ↑ | `utils/symbol_postcorrect.py` (∅ 템플릿 매칭) |
| GDT F1 < 0.95 | rare 심볼 추가 라벨링 | `utils/fcf_split.py` (D-015) |
| Roughness F1 < 0.95 | data scarcity 가능 | `utils/synthetic_gen.py` (D-017) |
| numerical_accuracy < 0.95 | nominal 추출 강화 | Pytesseract pre-filter (D-017) |
| Hallucination > 0.10 | task token 강화, schema 제약 | LLM verification (D-017) |

## 5. 출력 형식

### 5.1 콘솔 (위 §4.1)

### 5.2 HTML 첨부

- **Per-class summary** 표 (class / n / f1 / precision / recall / hall / empty / nominal_acc / tol_match / sym_acc / datum_acc / Ra_acc)
- **Per-symbol hit rate (GDT)** 표 + 막대 차트 (ISO 1101 심볼별)
- **F1 per class** 막대 차트

### 5.3 JSON

```json
{
  "step": "stage3n",
  "overall_status": "FAIL",
  "metadata": {
    "predictions_dir": "...",
    "gt_dir": "...",
    "n_pred": 7, "n_gt": 7
  },
  "checks": [...16 items...],
  "artifacts": [
    {"kind": "table", "title": "Per-class summary", ...},
    {"kind": "table", "title": "Per-symbol hit rate (GDT)", ...},
    {"kind": "plot",  "title": "F1 per class"},
    {"kind": "plot",  "title": "Per-symbol hit rate (GDT)"}
  ]
}
```

## 6. 의존성

```
PyYAML, numpy
matplotlib, jinja2  (via common.py)
src.utils.metrics   (compare_measure/gdt/roughness, hallucination_rate)
src.validate.common
```

## 7. 관련 의사결정

- **D-001** 아키텍처 = Donut fine-tune (D-023 임계값 재현 목표)
- **D-005** 학습 하이퍼파라미터 (epoch 30 / AdamW / cosine 1e-6 / batch 4 / FP16)
- **D-013** 4개 언어 (Numerical 은 숫자/심볼 위주라 영향 적음)
- **D-015** FCF 분리 미수행 — V6 GDT F1 < 0.85 시 fcf_split.py 검토
- **D-016** eDOCr2 다단계 차용 안 함 — V6 결과로 재논의 가능
- **D-017** 백업 모듈 조건부 추가 — V6 가 발동 결정점
- **D-020** 검증 의무화
- **D-021** Severity (★ critical 4개)
- **D-022** Provenance (raw_seq + source 보존)
- **D-023** ★ 사용자 필수 임계값 — 본 검증기가 직접 측정
- **D-024** Group-aware split (Step 6 학습 시 적용, V6 는 평가만)

## 8. 검증 대상 모듈

[`stage3_numerical.md`](./stage3_numerical.md) — Step 6

## 9. 사용 흐름

```
Step 6 (Donut Numerical fine-tune)
       ↓
checkpoints/donut_numerical/final/
       ↓
사용자가 test split 에 대해 stage3_numerical.py predict 또는 batch 실행
       ↓
outputs/<id>/numerical/*.num.json
       ↓
prepare_vlm_dataset.py 의 검수 완료된 GT (data/vlm/numerical/) 와 매칭
       ↓
★ V6 실행 → reports/<date>_stage3n.html
       ↓
PASS → Step 7 / Step 8 진행
FAIL → Plan B 발동 검토 (D-017)
```

## 10. 흔한 FAIL + 진단

### Measure F1 < 0.90

진단 흐름:
1. `numerical_accuracy` 도 낮은가? → nominal 추출 자체 문제 (OCR/숫자 인식)
2. `tolerance_match` 만 낮은가? → tolerance 분리 인식 문제
3. `hallucination_rate[Measure]` 도 높은가? → Donut 이 가짜 필드 생성
4. Per-class summary 의 fp 가 큰가? → 잘못된 매칭

### GDT F1 < 0.95

진단:
1. `symbol_accuracy` 낮음 → 14 심볼 중 어떤 게 약한지 per-symbol breakdown 표 확인
2. `rare_symbol_accuracy` 낮음 → 합성 데이터로 rare 심볼 보강 (D-017)
3. `datum_accuracy` 낮음 → datum list 순서 / 길이 문제
4. Per-symbol 표에서 hit_rate 0.0 인 심볼 있음 → 학습 데이터에 없거나 부족

### Hallucination > 0.10

진단:
1. `extra_fields` 가 많음 → schema 외 필드 자주 생성 (스키마 제약 강화 필요)
2. `value_mismatches` 가 많음 → 값 자체가 부정확 (numerical_accuracy 와 연동)

## 11. 한계

- 사람 검수 GT 의 품질에 직접 의존 — GT 자체가 부정확하면 V6 결과도 부정확
- Per-symbol breakdown 은 GT 빈도가 충분해야 의미 있음 (희귀 심볼 5건 미만은 통계 부족)
- Datum accuracy 는 순서 엄격 비교 — 일부 표준에서 datum 순서 무관할 수 있어 false negative 가능
