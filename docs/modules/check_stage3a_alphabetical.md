# `src/validate/check_stage3a_alphabetical.py`

> **V5** — Stage 3-A (Donut Alphabetical, zero-shot) 사후 검증 — 논문 베이스라인 0.672

## 1. 구현 요약

Stage 3-A (Donut zero-shot) 의 예측 JSON 과 사람 검수 GT JSON 을 비교해 **field-level F1, hallucination, edit distance, 언어별 격차** 측정.

**입력 매칭 규약**

| 입력 | 형식 |
|---|---|
| Predictions | `<dir>/<stem>.alpha.json` 또는 `<stem>.json` (Stage 3-A 출력) |
| Ground Truth | `<gt_dir>/<stem>.json` — pred 와 stem 일치 |

**검증 항목 (10개)**

| 항목 | 임계값 (논문 0.672) | Severity |
|---|---|---|
| `field_f1[notes]` | ≥ 0.75 (논문 0.810) | **critical** |
| `field_f1[titleblock]` | ≥ 0.50 (논문 0.533) | warning |
| `field_f1[overall]` | ≥ 0.50 (논문 0.672) | warning |
| `hallucination_rate (overall)` | < 0.50 (논문 0.40~0.48) | warning |
| `hallucination_rate[titleblock]` | — | info |
| `hallucination_rate[notes]` | — | info |
| `empty_response_rate` | < 0.10 | warning |
| `edit_distance_avg` | ≤ 5 (정규화) | info |
| `per_language_f1_gap` (max-min) | ≤ 0.30 | info |
| `unmatched_predictions` | = 0 | warning |

**HTML 첨부**
- Per-region summary 표 (TB / Notes 분리)
- Per-language F1 표 + 막대 차트
- TitleBlock field-level breakdown (top 15) + horizontal 막대
- 모든 plot 은 base64 PNG 임베디드

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 매칭 방식 | 파일 stem 일치 | 단순·재현성 |
| TB 비교 | `compare_titleblock` (fuzzy) | OCR 노이즈 허용 |
| Notes 비교 | `compare_notes` (greedy fuzzy, threshold 0.30) | 순서 변동 허용 |
| Hallucination — TB | metrics.hallucination_rate (allowed_keys = GT keys) | 도메인 정의 |
| Hallucination — Notes | `len(pred) - tp` / `len(pred)` | 매칭 안 된 pred 항목 = 가짜 |
| 언어 감지 | pred.language_hint 우선 → filename heuristic fallback | V1 와 동일 룰 |
| Per-field breakdown | normalize_text (case+space 무시) exact match | 통계 명확성 |
| Empty response | TB fields 모두 falsy / Notes items 비어있음 | recall=0 케이스 추적 |
| Per-language gap | max F1 - min F1 (info only) | 4개 언어 균형 모니터링 |
| 임계값 | `validation_thresholds.yaml#stage3a` 동적 로드 | D-021 |
| Region 자동 감지 | gt.type 우선, pred.type fallback | 일관성 |

## 3. 사용법

```bash
# Stage 3-A 출력 + GT 비교
python -m src.validate.check_stage3a_alphabetical \
    --predictions outputs/sample/alphabetical/ \
    --gt data/validation_gt/stage3a/

# 임계값 커스터마이즈
python -m src.validate.check_stage3a_alphabetical \
    --predictions outputs/sample/alphabetical/ \
    --gt data/validation_gt/stage3a/ \
    --thresholds configs/validation_thresholds.yaml
```

### GT JSON 형식

`data/validation_gt/stage3a/<stem>.json`:

**TitleBlock**
```json
{
  "type": "TitleBlock",
  "fields": {
    "drawing_no": "DWG-001-A",
    "material": "SS400",
    "scale": "1:2",
    "revision": "B"
  }
}
```

**Notes**
```json
{
  "type": "Notes",
  "items": [
    "1. UNLESS OTHERWISE SPECIFIED",
    "2. ALL DIMENSIONS IN MM"
  ]
}
```

### Stage 4 의 prepare_vlm_dataset.py 와 연동

`prepare_vlm_dataset.py alphabetical --ocr-prefill` 실행 → 사람이 `_review.completed=true` 로 마킹한 JSON 들을 `data/validation_gt/stage3a/` 로 복사하면 GT 로 사용 가능.

## 4. 검증 결과

### 4.1 5건 더미 데이터 PASS (모든 케이스 정확 검출)

| 케이스 | 입력 | 기대 결과 | 실제 |
|---|---|---|---|
| 1 | TitleBlock 완벽 매칭 (3 필드 모두 일치) | F1=1.0 | ✓ TB region tp=3 |
| 2 | TitleBlock 부분 매칭 + 가짜 필드 2개 | hallucination ↑, F1 부분 | ✓ TB hall=0.571 |
| 3 | Notes fuzzy 매칭 (3개 중 2개) | F1=0.67 | ✓ Notes F1=0.5 (전체) |
| 4 | Notes 빈 응답 | empty_rate ↑ | ✓ empty_rate=0.25 |
| 5 | GT 없는 예측 | unmatched_predictions=1 | ✓ WARN 발생 |

**최종 콘솔**

```
[ 1/10] field_f1[notes]              0.5000  ✗ FAIL  ≥ 0.7500   (critical)
[ 2/10] field_f1[titleblock]         0.5714  ✓ PASS  ≥ 0.5000
[ 3/10] field_f1[overall]            0.5455  ✓ PASS  ≥ 0.5000
[ 4/10] hallucination_rate (overall) 0.5000  ✓ PASS  ≤ 0.5000
[ 7/10] empty_response_rate          0.2500  ! WARN  ≤ 0.1000
[ 9/10] per_language_f1_gap          0.8333  · INFO  ≤ 0.3000   (en 1.0 vs ja 0.0)
[10/10] unmatched_predictions             1  ! WARN  ≤ 0

Overall: FAIL   PASS=4 WARN=2 FAIL=1 INFO=3
```

Notes critical 임계값 미달 (0.5 < 0.75) 정확히 FAIL 판정.

### 4.2 산출물

- `reports/<date>_stage3a.html` — 55KB, plot 2개 (per-language F1, field hit rate) + 표 3개
- `reports/<date>_stage3a.json` — 5,977 bytes 정량 메트릭

## 5. 출력 형식

### 5.1 콘솔

위 §4.1 참조.

### 5.2 HTML 첨부 (4개)

1. **Per-region summary** 표 (region / n / f1 / precision / recall / hall_rate / empty_rate / edit_avg)
2. **Per-language F1** 표 (en / ko / ja / ru)
3. **F1 per language** 막대 차트
4. **TitleBlock field-level breakdown** (top 15) + horizontal 막대
   - drawing_no / material / scale / revision / date / drawn_by / ... 별 hit_rate

### 5.3 JSON

```json
{
  "step": "stage3a",
  "overall_status": "FAIL",
  "metadata": {
    "predictions_dir": "...",
    "gt_dir": "...",
    "n_pred": 5,
    "n_gt": 4
  },
  "checks": [
    {"name": "field_f1[notes]", "value": 0.5, "threshold": 0.75,
     "direction": "ge", "severity": "critical", "status": "FAIL", ...},
    ...
  ],
  "artifacts": [
    {"kind": "table", "title": "Per-region summary", ...},
    {"kind": "plot",  "title": "F1 per language"},
    ...
  ]
}
```

## 6. 의존성

```
PyYAML, numpy
matplotlib, jinja2  (via common.py)
src.utils.metrics
src.validate.common
```

전부 stdlib + metrics.py 의 함수 활용.

## 7. 관련 의사결정

- **D-013** 4개 언어 — per-language F1 측정
- **D-018** Stage 3-A = Donut zero-shot — fine-tune 안 함
- **D-020** 검증 의무화
- **D-021** Severity (Notes critical, TB warning)
- **D-022** 콘솔 + HTML + JSON 3종 출력
- **D-023** 사용자 필수 임계값:
  - Notes F1 ≥ 0.75 (critical, 논문 0.810)
  - Hallucination < 0.50 (warning)

## 8. 검증 대상 모듈

[`stage3_alphabetical.md`](./stage3_alphabetical.md) — Step 5

## 9. 흔한 FAIL + 해결

| FAIL 항목 | 원인 | 조치 |
|---|---|---|
| `field_f1[notes] < 0.75` | Notes 항목 분리 실패 / OCR 약함 | 1) Donut DocVQA 단일 질문 → 다중 질문 분리 2) Step 9 enrichment 로 보정 |
| `hallucination_rate > 0.5` | DocVQA 가 모르는 답을 지어냄 | 1) "no answer" 명시 프롬프트 강화 2) Confidence 임계값 높이기 |
| `per_language_f1_gap > 0.30` | EN 강함, 비-EN 약함 (논문에서도 동일) | 1) 비-EN 도면을 영어 prompt 로 (mismatch) 2) Qwen2.5-VL 검토 (D-018 재검토) |
| `empty_response_rate > 0.1` | TitleBlock crop 가 너무 작거나 모호 | 1) Stage 1 padding ↑ 2) crop conf threshold ↓ |
| 특정 field hit_rate 낮음 | DocVQA 질문 부정확 | 14개 기본 질문 중 해당 필드만 reformulate |

## 10. 사람 검수 부담 감소 팁

V5 의 ground truth 는 prepare_vlm_dataset.py 의 `--ocr-prefill` 출력에서 `_review.ocr_text` 를 참고해 채우면 빠르게 작성 가능.

**전체 검수 불필요** — 50~100건 sampling 만으로 V5 결과는 의미 있음 (특히 4개 언어 균형).
