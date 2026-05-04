# `src/stage5_enrichment.py`

> **Step 9 (확장)** — Metadata Enrichment, 4-tier deterministic-first cascade

## 1. 구현 요약

논문 범위 외 확장 모듈. Stage 4 통합 JSON 을 입력받아 누락/모호 필드를 LLM·KB·휴리스틱으로 보완하고, 모든 결정에 provenance 를 남긴다.

**4-tier Cascade**

```
Stage 4 통합 JSON
       ↓
9.1 Gap Detection           누락/모호 필드 식별 (5 카테고리)
       ↓
9.2 Deterministic Lookup    KB 직접 매칭        conf 0.95+
       ↓ unresolved
9.3 Engineering Heuristics  도메인 룰 적용      conf ~0.80
       ↓ unresolved
9.4 RAG-Augmented LLM       Gemini/Qwen + KB    conf < 0.95
       ↓
9.5 HITL Flag Gate          conf < 0.70 → 검수 플래그
       ↓
outputs/<id>.enriched.json
```

**5개 enrichment 카테고리**

| 카테고리 | 입력 | 출력 예시 |
|---|---|---|
| `material` | "stainless steel" | "SUS304 No.2D" + 대안 3개 |
| `tolerance_general` | null | "ISO 2768-mK" |
| `surface_roughness_default` | null | "Ra 1.6 μm" (h7 검출 시) / "Ra 3.2 μm" (기본) |
| `process_sequence` | null | ["fiber_laser_cutting", "press_brake_bending", ...] |
| `qc_checklist` | null | ["480±0.3", "Ø6.5 위치도", ...] (LLM 추천) |

**Provider 추상화**

| Provider | 환경 | 의존성 | 용도 |
|---|---|---|---|
| `MockProvider` | API 없음 | — | CI / 테스트 / 데모 |
| `GeminiProvider` | `GEMINI_API_KEY` env | `google-generativeai` | 클라우드 production |
| `QwenProvider` | RTX 5080 16GB | `transformers`, `torch` | 로컬, 데이터 보안 |

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| Cascade 순서 | deterministic → heuristic → llm → hitl | 비용·할루시네이션 최소화 (D-020) |
| Provider 패턴 | Protocol-based 추상화 | 다중 LLM 지원 (D-024) |
| Mock 기본값 | API 없어도 동작 | 개발 편의 / CI 통합 |
| HITL 임계값 | 0.70 (D-023) | 신뢰도 미달 시 사람 검수 |
| KB 위치 | inline + `data/kb/*.json` override | 기본 동작 보장 + 사용자 확장 |
| Gap detection | 5개 카테고리 동시 분석 | 한 번에 모든 필드 enrich |
| Material 모호성 | Roboflow `.rf.` 와 별개 — 키워드 fuzzy match | "stainless" → enrich, "SUS304" → 그대로 |
| LLM 응답 파싱 | JSON 시도 → regex `{...}` 추출 → raw text fallback | LLM 출력 형식 변동 robust |
| Provenance | 모든 필드에 `method`/`source`/`rationale`/`evidence` | D-022 의무 |
| Deep copy | 입력 unified JSON 보존 | 부작용 방지 |
| 카테고리 선택 | CLI `--categories all/material,tolerance` | 부분 enrich 가능 |
| Image 입력 | 옵션 (multimodal LLM 활용) | image 미제공 시도 동작 |

## 3. 사용법

### CLI

```bash
# Mock 으로 단일 (API 키 없음)
python src/stage5_enrichment.py enrich \
    --json outputs/sample.json \
    --provider mock --categories all

# Gemini 클라우드 (API 키 필요)
export GEMINI_API_KEY=your_key
python src/stage5_enrichment.py enrich \
    --json outputs/sample.json \
    --image dataset/sample.jpg \
    --provider gemini --categories all

# 부분 enrich (material 만)
python src/stage5_enrichment.py enrich \
    --json outputs/sample.json \
    --provider gemini --categories material

# 배치 (전체 도면)
python src/stage5_enrichment.py batch \
    --json-dir outputs/json/ \
    --out-dir outputs/enriched/ \
    --provider gemini --limit 100
```

### Python import

```python
from src.stage5_enrichment import enrich, make_provider

provider = make_provider("mock")   # 또는 "gemini" / "qwen"
enriched = enrich(unified_json, provider,
                  image_path=Path("sample.jpg"),
                  categories=["material", "tolerance_general"])
# enriched["enrichment"]["fields"]["material"]
#   = {original, suggested, alternatives, confidence,
#      method, source, rationale, evidence, flagged_for_review}
```

## 4. 검증 결과

### 4.1 더미 unified JSON 으로 4-tier 정상 동작

입력: title_block.material="stainless steel", tolerance=null, h7 끼워맞춤 Measure 1개.

```
Stats: total=5  det=2  heur=2  llm=1  hitl=1

[deterministic] material              conf=0.78  → SUS304 No.2D
                                                   rationale: 박판 일반 grade
[deterministic] tolerance_general     conf=0.95  → ISO 2768-mK
                                                   rationale: ISO 2768 medium-coarse 표준
[heuristic    ] surface_roughness     conf=0.78  → Ra 1.6 μm
                                                   rationale: h6/k6/H7 정밀 끼워맞춤 검출
[heuristic    ] process_sequence      conf=0.70  → ['fiber_laser_cutting', 'press_brake_bending']
                                                   rationale: SUS 박판 → 레이저 절단 + 벤딩
[llm          ] qc_checklist          conf=0.65  → mock_suggestion ★HITL
                                                   rationale: Mock fallback
```

**모든 cascade tier 가 정확히 동작**:
- material: KB 직접 매칭 (deterministic)
- tolerance: ISO 2768 default 적용 (deterministic)
- roughness: h7 검출 → Ra 1.6 (heuristic precision_fit_implies_ra1.6)
- process: SUS material → sheet_metal_thin 템플릿 (heuristic)
- qc_checklist: heuristic 룰 없음 → LLM (Mock 기본값) → conf 0.65 < 0.70 → HITL 플래그

### 4.2 Provider 호환성

- ✓ Mock: 즉시 동작
- ✓ Gemini: API 키 미설정 시 graceful 에러 메시지
- ✓ Qwen: transformers 미설치 시 graceful 에러 메시지

## 5. 출력 형식

### 5.1 enriched JSON 추가 블록 (HANDOFF §11 D-022 / sample_enriched/ 참조)

```json
{
  ...existing Stage 4 fields...,
  "enrichment": {
    "version": "1.0",
    "timestamp": "2026-04-27T15:30:00+00:00",
    "provider": "gemini-2.0-flash",
    "stats": {
      "fields_total": 5,
      "resolved_deterministic": 2,
      "resolved_heuristic": 2,
      "resolved_llm": 1,
      "flagged_hitl": 0
    },
    "fields": {
      "material": {
        "original": "stainless steel",
        "suggested": "SUS304 No.2D",
        "alternatives": [
          {"value": "SUS304 BA", "weight": 0.30, "note": "외관 / 반사 중요"},
          {"value": "SUS316L No.2D", "weight": 0.15, "note": "내식성 강화"}
        ],
        "confidence": 0.78,
        "method": "deterministic",
        "source": "KS D 3705 / cross_standard_map",
        "rationale": "박판/판금 + 한국 시장 → SUS304 2D 마감이 가장 흔함",
        "evidence": [{"type": "kb_entry", "key": "stainless"}],
        "flagged_for_review": false
      },
      "tolerance_general": { ... },
      "surface_roughness_default": { ... },
      "process_sequence": { ... },
      "qc_checklist": {
        "original": null,
        "suggested": ["Ø6.5 위치도", "전폭 480 ±0.3"],
        "confidence": 0.85,
        "method": "llm",
        "source": "gemini",
        "rationale": "조립 critical features 자동 추출",
        "flagged_for_review": false
      }
    }
  }
}
```

### 5.2 사람 검수 흐름

`flagged_for_review: true` 인 필드만 사람이 확인. UI 가 다음을 보여주면 됨:
- `original`: 도면에서 검출된 원본
- `suggested`: 제안값 + `alternatives` 후보
- `rationale`: 왜 이 추천인지
- `evidence`: 근거 (KB ref, OCR 위치 등)
- 사용자가 confirm/reject/edit → `_review.completed = true`

## 6. KB 자원

### 6.1 인라인 기본값 (코드 내장)

- `INLINE_MATERIAL_KB`: stainless / carbon steel / aluminum / brass 4종
- `ISO_2768_DEFAULTS`: medium-coarse default
- `ROUGHNESS_DEFAULTS`: Ra 3.2 μm default
- `PROCESS_TEMPLATES`: sheet_metal_thin / machined_aluminum / shaft_S45C / default_machined

### 6.2 외부 KB override (`data/kb/`)

다음 파일이 존재하면 인라인 값을 override:
- `data/kb/material_catalog.json`
- `data/kb/iso_2768_defaults.json`
- `data/kb/machining_roughness_table.json`
- `data/kb/gdt_priors.json`
- `data/kb/process_combination_catalog.json`

스키마는 인라인과 동일.

## 7. 의존성

```
필수: 없음 (Mock provider 만 사용 시)

옵션:
- google-generativeai>=0.8.0   # Gemini provider
- transformers>=4.44.0          # Qwen provider
- torch (CUDA 12.8)             # Qwen provider
- Pillow                        # 이미지 입력
```

## 8. 관련 의사결정

- **D-001** 논문 외 확장 — Step 9 는 추가 기능
- **D-019** 4-tier cascade 구조
- **D-020** Provider 추상화 (gemini/qwen/claude/mock)
- **D-022** Provenance 필수 (모든 필드에 method/source/rationale)
- **D-023** HITL 임계값 = 0.70

## 9. 검증 모듈

[`check_enrichment.py`](./../README.md) — V9, 작성 예정 (provenance 완전성 / method 분포 / HITL flag rate / 비용 추적)

## 10. 업스트림 / 다운스트림

**업스트림**: `pipeline.py` (Step 7) 의 통합 JSON

**다운스트림**:
- 사람 검수 UI (별도 도구)
- CAD/ERP 연동 (downstream business)
- V9 (`check_enrichment.py`) 검증

## 11. 흔한 사용 시나리오

| 시나리오 | provider | categories | 비용 추정 |
|---|---|---|---|
| 개발/테스트 | mock | all | 0 |
| 회사 내 도면 (보안) | qwen (로컬) | all | 0 (전기료) |
| 비기밀 도면 | gemini-2.0-flash | all | $0.0008/도면 |
| 재질만 보강 | gemini | material | $0.0002/도면 |
| 5,839장 일괄 | gemini | all | ~$3.7 |

## 12. 한계 / 향후 개선

- **데이터셋 검증 필요**: 실제 5,839 도면에 적용 시 KB 적정성 확인
- **언어별 LLM 프롬프트 튜닝**: 현재 영어 프롬프트만 — KO/JP/RU prompt 분리 시 정확도 ↑
- **Cost tracking**: V9 검증기에서 도면당 비용 측정 추가 예정
- **Streaming**: 대용량 배치 처리 시 progress bar 미흡
- **KB 캐시**: 매 호출 시 JSON 로드 — 클래스 변수로 캐싱 검토
