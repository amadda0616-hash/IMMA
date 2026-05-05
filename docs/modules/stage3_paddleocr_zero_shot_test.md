# `src/stage3_paddleocr_zero_shot_test.py`

> **Phase 15b** — PaddleOCR-VL-1.5 다국어 zero-shot 평가 (D-039 / D-043 검증)

## 1. 구현 요약

Phase 15a (`stage3_paddleocr_install_check.py`) 환경 검증 PASS 후, 실제 다국어 도면 5장 (한/영/일/중/러) 으로 zero-shot 정성 평가. 사용자 확정 sample 7장 중 언어별 1장씩 선별.

**3 prompt 흐름** (도면 1장당):
1. `titleblock` — ★ ISO 7200 + KS A 0005 + 첨부 이미지 통합 **23 필드** 추출 (JSON)
2. `notes` — 다국어 General Notes / 주서 / 注記 등 list 추출
3. `full_text` — 전체 visible text transcribe (정성 baseline)

**핵심 추가 (★ 2026-05-04)**:
- `TITLEBLOCK_STANDARD_SCHEMA` 상수 — 23 필드 표준 schema (identification 5 / descriptive 10 / administrative 8)
- 다국어 keyword hints (中文/한국어/Deutsch/Русский)
- D-042 monkey-patch 자동 적용 (`config.text_config = config.get_text_config()`)

## 2. TitleBlock 표준 Schema (23 필드)

### Identification (5)
| 필드 | 의미 | 표준 출처 |
|---|---|---|
| `drawing_no` | 도면번호 | ISO 7200 mandatory / KS A 0005 |
| `project_id` | 프로젝트/조립체 ID | ★ 첨부 이미지 |
| `title` | 도면 제목 | ISO 7200 mandatory |
| `sheet` | 시트번호 (예: 1 OF 1) | ISO 7200 mandatory |
| `revision` | 개정번호 | ISO 7200 optional |

### Descriptive (10)
| 필드 | 의미 | 표준 출처 |
|---|---|---|
| `part_name` | 부품명 | KS A 0005 |
| `material` | 재질 | KS A 0005 |
| `mass` | 질량 | ★ 첨부 이미지 |
| `scale` | 척도 (예: 1:1) | ISO 7200 동적 |
| `projection` | 투상법 (1각/3각) | KS A 0005 |
| `paper_size` | 용지 크기 (A3 등) | ISO 7200 optional |
| `quantity` | 수량 | KS A 0005 |
| `surface_treatment` | 표면 처리 | — |
| `heat_treatment` | 열처리 | — |
| `general_tolerance` | 일반 공차 | ISO 7200 동적 |

### Administrative (8)
| 필드 | 의미 | 표준 출처 |
|---|---|---|
| `company` | 회사/법인 | ISO 7200 mandatory (legal_owner) |
| `department` | 책임 부서 | ISO 7200 optional |
| `drawn_by` | 작성자 | ISO 7200 mandatory (creator) |
| `designed_by` | 설계자 | ★ 첨부 이미지 (drawn_by 와 분리) |
| `checked_by` | 검도자 | KS A 0005 |
| `approved_by` | 승인자 | ISO 7200 mandatory |
| `date` | 발행일 | ISO 7200 mandatory |
| `state` | 도면 상태 (Released/Draft) | ★ 첨부 이미지, ISO 7200 optional |

## 3. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 모델 | `PaddlePaddle/PaddleOCR-VL-1.5` (D-039) | 다국어 SOTA, OmniDocBench 94.5% |
| Schema | 23 필드 통합 (★ 2026-05-04) | ISO 7200 + KS A 0005 + 첨부 이미지 종합 |
| 다국어 hint | 6개 언어 키워드 prompt 명시 | 한/영/일/중/러/독 키워드 자동 인식 |
| max_new_tokens | 1024 | TitleBlock + Notes 충분 |
| dtype | float16 | RTX 5080 16GB 충분 + 추론 속도 ↑ |
| chat template | `apply_chat_template` 사용 | PaddleOCR-VL 권장 방식 |
| 후처리 | JSON parse (3 fallback) + Notes list parse | TitleBlock JSON / Notes 항목별 분리 |
| 언어 감지 | 파일명 prefix (`en_` / `ja_` / `ko_` / `zh_` / `ru_` / `de_`) | 자동화 + 보고서 정렬 |
| 출력 | JSON + MD 보고서 동시 | 정량 (JSON) + 정성 (MD) 검토 |

## 4. 사용법

### 4.1 표준 5장 평가

```bash
cd /mnt/c/Users/user/github/Drawing
source .venv-paddleocr/bin/activate

# 기본 실행 (3 prompt 모두)
python src/stage3_paddleocr_zero_shot_test.py
```

**산출물**:
- `outputs/stage3a_zero_shot_eval.json` — 구조화 JSON
- `outputs/stage3a_zero_shot_eval.md` — 정성 검토 보고서

### 4.2 옵션

```bash
# 빠른 모드 (TitleBlock + Notes 만)
python src/stage3_paddleocr_zero_shot_test.py --prompts titleblock,notes

# 처음 1장만 (디버깅)
python src/stage3_paddleocr_zero_shot_test.py --limit 1

# 다른 디렉토리 (예: 독일어 ~10장 후속)
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir data/stage3a_eval_samples_de/ \
    --output-json outputs/stage3a_zero_shot_eval_de.json \
    --output-md   outputs/stage3a_zero_shot_eval_de.md

# stage1_fp_notes 23개 OCR (D-038 Notes Rescue 미리 검증)
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir <stage1_fp_notes 디렉토리> \
    --prompts notes,full_text \
    --output-json outputs/stage1_fp_notes_paddleocr_eval.json
```

### 4.3 CLI 인자

```
--samples-dir SAMPLES_DIR    평가 도면 디렉토리 (default: data/stage3a_eval_samples)
--output-json OUTPUT_JSON    결과 JSON (default: outputs/stage3a_zero_shot_eval.json)
--output-md   OUTPUT_MD      MD 보고서 (default: outputs/stage3a_zero_shot_eval.md)
--prompts     PROMPTS        실행할 prompt (콤마 구분, default: titleblock,notes,full_text)
--max-new-tokens MAX         max_new_tokens (default: 1024)
--device      DEVICE         (default: cuda:0)
--limit       LIMIT          최대 도면 개수 (0 = 전체)
```

## 5. 출력 형식

### 5.1 JSON 구조

```json
{
  "phase": "15b",
  "model_id": "PaddlePaddle/PaddleOCR-VL-1.5",
  "samples_dir": "data/stage3a_eval_samples",
  "n_samples": 5,
  "prompts": ["titleblock", "notes", "full_text"],
  "max_new_tokens": 1024,
  "device": "cuda:0",
  "total_inference_time_s": 45.32,
  "languages_processed": ["Chinese", "English", "Japanese", "Korean", "Russian"],
  "titleblock_schema": { ... },
  "titleblock_field_count": 23,
  "results": [
    {
      "filename": "en_drawing.jpg",
      "language": "English",
      "image_size": [W, H],
      "titleblock": {
        "raw_output": "...",
        "answer": "...",
        "parsed_json": {
          "drawing_no": "810-101-112",
          "title": "MOTOR MTG. PLATE",
          "material": "MS",
          ...
        },
        "json_parsed_ok": true,
        "inference_time_s": 2.45,
        "output_chars": 142
      },
      "notes": {
        "items": ["...", "..."],
        "n_items": 3,
        ...
      },
      "full_text": { ... }
    },
    ...
  ]
}
```

### 5.2 MD 보고서 (정성 검토용)

- 요약 표 (모든 도면 한눈에)
- 도면별 상세 (TitleBlock JSON / Notes 항목 / Full text)
- 사용자 정성 평가 가이드 (5개 항목, 1~5점)

## 6. D-013 임계값 (V5)

| 지표 | 임계값 | PASS 조건 |
|---|---|---|
| 평균 char accuracy | ≥ 0.85 | 5개 언어 평균 |
| field-level F1 | ≥ 0.80 | TitleBlock 필드별 |
| hallucination rate | ≤ 0.05 | 빈 영역 텍스트 생성 |
| 언어별 gap (best-worst) | ≤ 0.20 | 다국어 균형 |

**D-043 가중치** (sample 부족 보정):
- 영어 1장 → low confidence
- 한국어 1장 → "학습용 한정"
- 일본어/중국어 → high confidence
- 러시아 1장 → mid
- 독일어 ~10장 → mid (15b 후속)

## 7. 의존성

- `.venv-paddleocr` 환경 (Phase 15a 별도 venv)
- Python 3.10+, torch 2.11+cu128, transformers 5.0.0
- accelerate / sentencepiece / protobuf / einops / pillow
- HuggingFace Hub 접속 (모델 cache 활용)
- GPU: ~5 GB VRAM 권장

## 8. 트러블슈팅

### Q1. JSON parse 실패
→ `parsed_json` 이 null. raw output 확인 → prompt 조정 또는 후처리 강화.

### Q2. Notes 항목 파싱 실패
→ `parse_notes_list()` 가 numbered/bullet 패턴만 인식. 라벨이 "1." 등 시작이어야 함.

### Q3. CUDA OOM
→ `--max-new-tokens 512` 또는 `--device cuda:0` 명시 + 다른 프로세스 정리.

### Q4. chat template 불일치
→ PaddleOCR-VL `apply_chat_template` 실패 시 `prompt_text = prompt` fallback. 출력 품질 일부 저하 가능.

## 9. 관련 파일

- `src/stage3_paddleocr_zero_shot_test.py` — 본 스크립트 (786 lines)
- `src/stage3_paddleocr_install_check.py` — 환경 검증 (Phase 15a)
- `data/stage3a_eval_samples/` — 5장 평가 도면 (사용자 저장)
- `outputs/stage3a_zero_shot_eval.{json,md}` — 결과
- `docs/PHASE15_CHECKLIST.md` — 15b 체크리스트
- `history.md §A.12.3` — 작성 박제

## 10. 차후 검토

| 조건 | 액션 |
|---|---|
| TitleBlock 필드 추출 < 70% | prompt 재조정 (필드별 individual prompt) |
| 다국어 char accuracy < 0.80 | fine-tune 검토 또는 폴백 (Qwen3-VL / DeepSeek-OCR-3) |
| Notes parse 정확도 부족 | LLM 기반 후처리 (Step 9 enrichment 활용) |
| 독일어 정확도 측정 | 별도 batch (15b 후속) |
| stage1_fp_notes 23개 통합 | Phase 15d 본격 실행 |

---

**Last updated**: 2026-05-04 (Phase 15b 작성, 평가 실행은 다음 세션)
