# `src/validate/check_enrichment.py`

> **V9** — Step 9 (Metadata Enrichment) 사후 검증 — 마지막 검증기

## 1. 구현 요약

`stage5_enrichment.py` 출력의 품질·비용·provenance 를 측정하는 마지막 검증기.

**검증 항목 (5 + 옵션 1)**

| 항목 | 임계값 | Severity |
|---|---|---|
| `provenance_completeness` | = 1.0 | **critical** (D-022) |
| `llm_method_rate` | ≤ 0.40 | warning (비용 통제) |
| `hitl_flag_rate` | ≤ 0.25 | warning |
| `empty_suggestion_rate` | ≤ 0.10 | warning |
| `cost_per_drawing_usd` | ≤ $0.005 | warning |
| `material_recommendation_accuracy` | ≥ 0.70 | warning (expert GT 옵션) |

**HTML 첨부**
- Method 분포 표 + 막대 차트 (deterministic / heuristic / llm / hitl)
- Per-category 분포 표 (각 카테고리별 method 분포 + confidence)
- Mean confidence per category 막대 차트
- Provider 분포 (mock / gemini / qwen / claude)
- Provenance 위반 샘플 표 (첫 20건)
- HITL flag 많은 도면 top 10

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 입력 | `*.enriched.json` (Step 9 출력) | stage5_enrichment.py 출력 형식 |
| Expert GT 옵션 | 단일 index JSON `{stem: {material: ...}}` 또는 폴더 | 30~50건 sample 검수면 충분 |
| Provenance 룰 | `method/source/rationale/confidence` 모두 있어야 | D-022 의무 |
| HITL fallback 예외 | `method='hitl'` 인 경우 source/rationale 누락 허용 | cascade 실패 케이스 정상 |
| Material 매칭 | fuzzy (≤ 20% edit dist) | "SUS304 No.2D" vs "SUS304" 등 변형 허용 |
| Cost 산정 | provider 별 per-LLM-call 표 | mock=0 / qwen=0 / gemini-flash=$0.0008 / gemini-pro=$0.005 |
| Empty suggestion | suggested 가 None / "" | recall=0 추적 |
| LLM method rate | 비용 통제용 (deterministic+heuristic 위주여야) | KB/룰이 충분히 정교한지 검증 |
| HITL flag rate | 사람 검수 부담 추적 | UI 부담 측정 |
| Per-category | 5 카테고리 각각의 cascade 분포 | 약점 카테고리 식별 |

## 3. 사용법

```bash
# 기본 (GT 없이)
python -m src.validate.check_enrichment \
    --enriched outputs/enriched/

# GT 와 함께 (material recommendation 정확도 측정)
python -m src.validate.check_enrichment \
    --enriched outputs/enriched/ \
    --expert data/validation_gt/enrichment_expert.json

# 임계값 커스터마이즈
python -m src.validate.check_enrichment \
    --enriched outputs/enriched/ \
    --thresholds configs/validation_thresholds.yaml
```

### Expert GT 형식

**Format A — 단일 index JSON**
```json
{
  "drawing_001": {"material": "SUS304 No.2D"},
  "drawing_002": {"material": "AL6061-T6"},
  "drawing_003": {"material": "SM490A"}
}
```

**Format B — 폴더 (per-stem 파일)**
```
data/validation_gt/enrichment/
├── drawing_001.json    {"material": "SUS304 No.2D"}
├── drawing_002.json    {"material": "AL6061-T6"}
└── drawing_003.json    {"material": "SM490A"}
```

## 4. 검증 결과

### 4.1 3건 더미 데이터 — 모든 케이스 정확 검출

**입력**:
- sample_001 (Mock provider): 5 카테고리, det 2 + heur 2 + llm 1, HITL 1
- sample_002 (Gemini provider): 4 카테고리, det 1 + llm 3, HITL 0
- sample_003 (Provenance 누락 케이스): 1 카테고리, source/rationale 빈 문자열

**결과**:

```
[ 1/5] provenance_completeness         0.9333  ✗ FAIL  ≥ 1.0000   critical
       · violations=2 / required=30  (sample_003 의 source/rationale 빈 문자열)
[ 2/5] llm_method_rate                 0.5000  ! WARN  ≤ 0.4000
       · 비용 통제 — LLM 호출 비율 (count=5)
       · sample_002 (Gemini 3건) + sample_001 (Mock 1건) + sample_003 (LLM 1건)
[ 3/5] hitl_flag_rate                  0.2000  ✓ PASS  ≤ 0.2500
       · flagged=2 / total_fields=10
[ 4/5] empty_suggestion_rate           0.0000  ✓ PASS  ≤ 0.1000
[ 5/5] cost_per_drawing_usd            0.0003  ✓ PASS  ≤ 0.0050
       · providers={'gemini-2.0-flash': 1, 'mock': 2}, total=$0.001

Overall: FAIL   PASS=3 WARN=1 FAIL=1 INFO=0
```

**핵심 검증 포인트**:
- Provenance: sample_003 의 빈 source / rationale 정확 검출 (2 위반 / 30 필요)
- LLM rate: 5개 LLM 호출 / 10개 필드 = 50% → WARN (비용 한계 초과)
- HITL: 2 / 10 = 20% → PASS (한계 25% 이내)
- Cost: Gemini 3 LLM 호출 × $0.0008 = $0.0024 / 3 도면 ≈ $0.0003 → PASS

## 5. 출력 형식

### 5.1 콘솔 (위 §4.1)

### 5.2 HTML 첨부

- **Method 분포 표** (deterministic/heuristic/llm/hitl 카운트 + 비율)
- **Method 분포 막대 차트** (cascade tier 시각화)
- **Per-category 분포 표** (5 카테고리 × 4 method)
- **Mean confidence per category** horizontal 막대
- **Provider 분포 표** (mock / gemini / qwen / claude)
- **Provenance violations** 샘플 표 (첫 20)
- **Top 10 HITL drawings** 표

### 5.3 JSON

```json
{
  "step": "enrichment",
  "overall_status": "FAIL",
  "metadata": {
    "predictions_dir": "outputs/enriched/",
    "expert_gt_path": "data/validation_gt/enrichment_expert.json",
    "n_pred": 3,
    "n_gt": 3
  },
  "checks": [...],
  "artifacts": [...],
}
```

## 6. 의존성

```
PyYAML, numpy
matplotlib, jinja2  (via common.py)
src.utils.metrics   (fuzzy_match, normalize_text, safe_div)
src.validate.common
```

## 7. 관련 의사결정

- **D-019** Step 9 = 논문 외 확장 — V9 도 확장 검증
- **D-020** 검증 의무화
- **D-021** Severity (provenance critical, 나머지 warning)
- **D-022** Provenance 필수 — V9 가 직접 측정
- **D-024** Provider 비용 추적 (Mock/Gemini/Qwen/Claude)

## 8. 검증 대상 모듈

[`stage5_enrichment.md`](./stage5_enrichment.md) — Step 9

## 9. 흔한 FAIL + 조치

| FAIL | 원인 | 조치 |
|---|---|---|
| `provenance_completeness < 1.0` | LLM 응답이 source / rationale 누락 | stage5_enrichment.py 의 LLM 프롬프트 강화 |
| `llm_method_rate > 0.40` | KB / 휴리스틱 부족 → LLM 호출 과다 | KB JSON 확장 (D-021 Q2) |
| `hitl_flag_rate > 0.25` | LLM confidence 낮음 | 1) 프롬프트 개선 2) confidence 임계값 조정 |
| `cost_per_drawing > $0.005` | LLM 호출 너무 많음 | 1) deterministic/heuristic 비중 ↑ 2) Mock fallback |
| `material_accuracy < 0.70` | LLM material 추천 부정확 | 1) KB material_catalog.json 확장 2) RAG retrieval 강화 |

## 10. 사용 흐름

```
Step 9 (stage5_enrichment.py batch)
       ↓
outputs/enriched/<id>.enriched.json
       ↓
(옵션) 도메인 전문가가 30~50건 sampling 검수
       ↓
data/validation_gt/enrichment_expert.json
       ↓
★ V9 실행
       ↓
PASS → production 배포
FAIL → KB 강화 / 프롬프트 튜닝 / cost 통제 후 재실행
```

## 11. 한계

- Expert GT 가 없으면 material recommendation accuracy 측정 불가
- Cost 추정은 provider 별 평균값 기준 — 실제 token 수 무관
- Hallucination 측정은 V5 / V6 와 별개 (V9 는 구조적 검증 위주)
- Per-language 분석 미포함 (V5 와 다름)

## 12. 향후 개선

- Real-time cost tracker (provider API 응답 token usage 활용)
- Confidence calibration plot (confidence vs accuracy)
- Cascade efficiency (deterministic 비율을 더 높이는 KB 확장 가이드)
- HITL queue prioritization (어떤 도면을 사람이 우선 검수해야 하는지)
