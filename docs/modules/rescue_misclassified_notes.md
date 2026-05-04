# `src/rescue_misclassified_notes.py`

> **Step 5.9 — D-038 Stage 1 False Positive Notes Rescue** — SKIP `stage1_fp_notes` 를 Donut zero-shot OCR 로 복구하여 메타데이터 JSON 에 병합

## 1. 구현 요약

Stage 2 라벨링에서 SKIP `stage1_fp_notes` 로 마킹된 PMI crop 들을 Stage 3-A Donut zero-shot DocVQA 모델로 처리하여 일반 주석 텍스트를 추출. 추출 결과를 JSON 구조로 저장하여 pipeline.py / stage4 merger 에서 최종 도면 JSON 의 `general_notes` 필드로 병합. 

**배경 (왜 필요한가)**

```
Stage 1 Version A 오검출:
  일반 주석 (재질/가공/공차) 영역 → PMI 로 잘못 분류
           ↓
Stage 2 라벨링:
  Measure/GDT/Roughness 아님 → SKIP 처리 (reason=stage1_fp_notes)
           ↓
Stage 3-A:
  이미 SKIP 되었으므로 Notes 검출 X (메타데이터 누락 위험)
           ↓
해결:
  ★ 본 도구가 SKIP 된 crop 들을 Donut 로 재분석 → JSON 저장
     → Stage 4 에서 'general_notes' 필드로 병합 → 메타데이터 완성
```

**워크플로 (D-038 구현 흐름)**

```
extract_skip_list.py → outputs/skip_lists/stage1_fp_notes.txt
        ↓
★ rescue_misclassified_notes.py (이 파일)
        ↓
Donut zero-shot DocVQA (단일 모델로 전수 처리)
        ↓
outputs/rescued_notes.json
        ↓
pipeline.py / stage4 merger
        ↓
최종 JSON 의 "general_notes" 필드에 병합
```

**핵심 컴포넌트** (~380 lines)

| 함수 | 역할 |
|---|---|
| `load_skip_list(path)` | stage1_fp_notes.txt 로드 (주석 제외) |
| `parse_pmi_filename(filename)` | crop 파일명 → metadata 추출 (drawing_id, pmi_idx) |
| `rescue_one(crop_path, processor, model, device, ...)` | 단일 crop → Donut predict → result dict |
| `main()` | argparse + 모델 로드 + tqdm progress + JSON 저장 |

**1개 CLI 서브커맨드 (단일 실행)**

```bash
python src/rescue_misclassified_notes.py \
    --skip-list <stage1_fp_notes.txt> \
    --crops-dir <crops_directory> \
    --output <rescued_notes.json>
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 모델 선택 | **Donut zero-shot DocVQA** | Stage 3-A 와 동일 → 재사용 가능, 다국어 지원, fine-tuned 보다 빠름 |
| 입력 | stage1_fp_notes.txt (line-by-line) | extract_skip_list.py 산출 형식 일치 |
| 샘플링 | **전수 처리** (crop 수 제한 옵션 있음) | rescue 의 특성상 누락되는 주석이 최소화되어야 함 |
| OCR 질문 | `"What text is in this image?"` (기본, 커스텀 가능) | 다국어 일반 주석 추출용 (특정 스키마 강제 X) |
| 다국어 지원 | `--language ja/ko/en/zh/ru` flag | 도면의 기본 언어 힌트 (optional) |
| 오류 처리 | per-crop try/except + error field 기록 | 단일 crop 실패가 전체 중단 X, 최종 JSON 에 failure log 포함 |
| 진행바 | tqdm (optional import, fallback) | ~50개 crop × 5초/개 = ~250초 작업 가시화 |
| 메타데이터 | crop_filename, drawing_id, pmi_idx, raw_text, items, error | 병합 시 최종 JSON 맵핑 용이 |
| 출력 형식 | JSON (UTF-8, 줄바꿈 2칸 indent, ensure_ascii=False) | pipeline.py 와 호환성 |

## 3. 사용법

### CLI

```bash
# 기본
python src/rescue_misclassified_notes.py \
    --skip-list outputs/skip_lists/stage1_fp_notes.txt \
    --crops-dir outputs/cvat_stage2_input_v3_upscaled \
    --output outputs/rescued_notes.json

# 언어 힌트 추가 (다국어 도면)
python src/rescue_misclassified_notes.py \
    --skip-list outputs/skip_lists/stage1_fp_notes.txt \
    --crops-dir outputs/cvat_stage2_input_v3_upscaled \
    --output outputs/rescued_notes.json \
    --language ja

# 커스텀 질문 + 다른 device
python src/rescue_misclassified_notes.py \
    --skip-list outputs/skip_lists/stage1_fp_notes.txt \
    --crops-dir outputs/cvat_stage2_input_v3_upscaled \
    --output outputs/rescued_notes.json \
    --device 0 \
    --question "What text is written in this engineering note?" \
    --language ja

# 디버깅 — 처음 10개만 처리
python src/rescue_misclassified_notes.py \
    --skip-list outputs/skip_lists/stage1_fp_notes.txt \
    --crops-dir outputs/cvat_stage2_input_v3_upscaled \
    --output outputs/rescued_notes.json \
    --limit 10
```

### CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--skip-list` | (필수) | stage1_fp_notes.txt (extract_skip_list.py 산출) |
| `--crops-dir` | (필수) | PMI crop 폴더 (e.g., outputs/cvat_stage2_input_v3_upscaled) |
| `--output` | `outputs/rescued_notes.json` | 결과 JSON 경로 |
| `--device` | None (auto) | GPU id (0, 1, ...) 또는 "cpu" |
| `--question` | `"What text is in this image?"` | Donut DocVQA 질문 (커스텀 가능) |
| `--language` | None | 언어 힌트 (en/ko/ja/zh/ru, optional) |
| `--model` | `naver-clova-ix/donut-base-finetuned-docvqa` | Donut 모델명 (커스텀 가능) |
| `--limit` | 0 (전체) | 처리 crop 수 제한 (디버깅용, 0 = 무제한) |

### Python import (선택)

```python
from pathlib import Path
from src.rescue_misclassified_notes import load_skip_list, parse_pmi_filename

# skip list 로드
files = load_skip_list(Path("outputs/skip_lists/stage1_fp_notes.txt"))
print(f"Rescue 대상: {len(files)} 개")

# 파일명 메타데이터 추출
meta = parse_pmi_filename("DwgFoo__PMI_023.jpg")
print(meta)  # {"drawing_id": "DwgFoo", "pmi_idx": 23}
```

## 4. 산출 형식 (JSON)

### 전체 구조

```json
{
  "metadata": {
    "source": "stage1_fp_notes_rescue",
    "decision": "D-038",
    "model": "naver-clova-ix/donut-base-finetuned-docvqa",
    "language_hint": "ja",
    "question": "What text is in this image?",
    "n_input": 47,
    "n_processed": 45,
    "n_success": 42,
    "n_empty": 2,
    "n_failed": 1,
    "n_missing_files": 2,
    "timestamp": "2026-05-01T12:00:00Z"
  },
  "rescued_notes": [
    {
      "crop_filename": "DwgFoo__PMI_023.jpg",
      "drawing_id": "DwgFoo",
      "pmi_idx": 23,
      "items": ["材料は鉄かSUS403"],
      "raw_text": "材料は鉄かSUS403",
      "language_hint": "ja"
    },
    {
      "crop_filename": "DwgFoo__PMI_045.jpg",
      "drawing_id": "DwgFoo",
      "pmi_idx": 45,
      "items": ["+0.1以下のものは機械加工のこと"],
      "raw_text": "+0.1以下のものは機械加工のこと",
      "language_hint": "ja"
    },
    {
      "crop_filename": "Bad__PMI_999.jpg",
      "drawing_id": "Bad",
      "pmi_idx": 999,
      "items": [],
      "raw_text": "",
      "language_hint": "ja",
      "error": "Image file corrupted"
    }
  ]
}
```

### 필드 설명

**metadata**
- `source`: 항상 "stage1_fp_notes_rescue"
- `decision`: 항상 "D-038"
- `model`: 사용된 Donut 모델명
- `language_hint`: 전달된 언어 힌트 (없으면 null)
- `question`: Donut 에 전달한 질문
- `n_input`: skip list 에 있던 파일 수
- `n_processed`: 실제로 찾아서 처리한 파일 수
- `n_success`: 텍스트 추출 성공 (items 또는 raw_text 비어있지 않음)
- `n_empty`: 추출 실패했지만 error 없음 (모델이 텍스트 못 찾음)
- `n_failed`: 처리 중 exception 발생
- `n_missing_files`: crops_dir 에 없는 파일
- `timestamp`: ISO 8601 형식 UTC 시간

**rescued_notes[*]**
- `crop_filename`: 원본 crop 파일명
- `drawing_id`: 파일명에서 추출한 도면 ID
- `pmi_idx`: 파일명에서 추출한 PMI index
- `items`: 추출된 텍스트 리스트 (보통 1개 항목)
- `raw_text`: Donut 가 반환한 raw 텍스트 (items 기반)
- `language_hint`: 언어 힌트 (metadata 와 동일하게 복사)
- `error`: (있으면) 처리 중 발생한 에러 메시지 (성공 시 필드 없음)

## 5. Stage 4 병합 예시

### pipeline.py 또는 stage4 merger 에서의 사용

```python
import json

# rescued_notes.json 로드
with open("outputs/rescued_notes.json", "r") as f:
    rescue_data = json.load(f)

# 기존 도면 JSON 에 병합
final_json = {
    "drawing_id": "DwgFoo_001",
    "title_block": { ... },
    "measures": [ ... ],
    "gdt": [ ... ],
    "roughness": [ ... ],
    "general_notes": []    # ← rescue 결과 추가
}

for rescued in rescue_data["rescued_notes"]:
    if rescued["items"]:  # 성공한 항목만
        final_json["general_notes"].append({
            "source": "stage1_fp_notes_rescue",
            "content": rescued["raw_text"],
            "crop_id": rescued["crop_filename"],
            "drawing_match": rescued["drawing_id"]
        })

# 또는 간단히
final_json["general_notes"] = [
    {
        "source": "stage1_fp_notes_rescue",
        "content": r["raw_text"],
        "crop_id": r["crop_filename"]
    }
    for r in rescue_data["rescued_notes"]
    if r["items"] or r["raw_text"]
]
```

## 6. 의존성

| 라이브러리 | 버전 | 역할 |
|---|---|---|
| `pathlib` | stdlib | 파일 경로 처리 |
| `json` | stdlib | JSON I/O |
| `logging` | stdlib | 로깅 |
| `datetime` | stdlib | timestamp 생성 |
| `tqdm` | 4.65+ | progress bar (optional, fallback 있음) |
| `torch` | 2.0+ | GPU 처리 |
| `transformers` | 4.30+ | Donut 모델 로드 |
| `src.stage3_alphabetical` | 자체 | Donut model load + predict_notes 함수 |

## 7. 관련 의사결정

- **D-037** — adaptive padding v3 (PMI crop 품질 기반)
- **D-038** — ★ Stage 1 false positive Notes Rescue (본 모듈의 핵심)
  - 흐름: extract_skip_list.py → stage1_fp_notes.txt → **rescue_misclassified_notes.py** → rescued_notes.json → pipeline.py 병합
- **차후 검토**:
  - OCR 품질 검증 (다국어, 특수 기호, 손글씨 케이스) → rescue 결과 성능 통계
  - `general_notes` 필드의 JSON schema 확정 (현재: items list + raw_text)
  - Stage 1 Version B 학습 시 Text 클래스 보강 → false positive 해소 (rescue 의존 최소화)
  - Stage 3 fine-tune 데이터에 rescue 결과 포함 여부 결정

## 8. ★ Donut DocVQA 한계 + 차후 대안 (2026-05-03 박제)

### 8.1 1차 시도 결과 — 실패 (Day 2, 2026-05-03)

**환경**:
- 모델: `naver-clova-ix/donut-base-finetuned-docvqa`
- 입력: 23개 stage1_fp_notes crops (다국어 — 일본어 다수)
- 모델 다운로드: 1.6GB / ~75초
- OCR 처리: 5.5초 / 4.17 crops/sec (RTX 5080 cu128)

**표면 통계**: 23/23 success (에러 없음)

**실질 품질**: **4% (1/23 만 의미 있는 결과)** ★

| 결과 유형 | 개수 | 비율 | 예시 |
|---|---|---|---|
| 단일 문자 | 11 | 48% | `r`, `m`, `x`, `2`, `6` |
| 환각 답변 | 5 | 22% | `let yourself` (DocVQA generic 응답) |
| 부분 추출 | 5 | 22% | `gpi`, `iii`, `to ict` |
| **의미 있는 결과** | **1** | **4%** | `d'sus403` (← 일본어 `鉄かSUS403`) |

### 8.2 실패 원인 분석

| 원인 | 설명 |
|---|---|
| **다국어 미스매칭** ★ | Donut DocVQA 학습 = 영어 문서. 우리 노트 = 일본어 다수 (`材料は鉄かSUS403`, `機械加工のこと`) → 일본어 인식 불가 |
| **모델 부적합** | DocVQA = 문서 질문응답 (`What is the title?`). 단순 OCR 작업 X |
| **Crop 컨텍스트 부족** | DocVQA = 전체 문서 처리용. 작은 PMI crop fragment 에는 컨텍스트 정보 부족 |
| **환각 패턴** | DocVQA 모델이 답을 모를 때 generic 응답 (`let yourself`) 생성 |

### 8.3 결정

**rescue 결과 폐기** — JSON 메타데이터 병합 안 함:
- 환각 텍스트 ("let yourself") 가 `general_notes` 에 들어가면 메타데이터 오염
- **빈 필드 < 잘못된 정보**

### 8.4 ★ D-039 결정: PaddleOCR-VL-1.5 채택 (2026-05-03)

26년 4월 SOTA 모델 비교 후 **PaddleOCR-VL-1.5** 채택:

| 모델 | 출시 | 크기 | OmniDocBench | 채택 |
|---|---|---|---|---|
| **PaddleOCR-VL-1.5** | **2026-01-29** | **0.9B** | **94.50%** | ✅ **★ 채택** |
| DeepSeek-OCR-2 | 2026-01-27 | 3B | 91.09% | ❌ (VRAM ↑) |
| Qwen3-VL | 2025-11 | 8B / 30B | 별도 | △ (폴백) |
| Donut DocVQA | 2022 | 200M | (벤치 없음, 영어만) | ❌ (다국어 ✗) |

**채택 사유 8가지**:
1. OmniDocBench 94.50% (DeepSeek 91.09% 대비 +3.41%)
2. **0.9B 모델** → RTX 5080 16GB 에서 Stage 2 동시 로드 가능
3. Table TEDS 92.76% 명시 → Title Block 표 처리
4. Formula CDM 94.21% 명시 → Notes 수식/공차 정확
5. **Seal Recognition (1.5 신규)** → 도장/검도 도장 처리
6. CJK industry-leading → 일/한/중 도면
7. JSON cell 좌표 제공 → Stage 4 merge 위치 정보
8. 2026-03-06 update (llama.cpp 추론) → 활발한 개발

### 8.5 코드 확장 계획 (Day 3 진입 시 작성)

`rescue_misclassified_notes.py` 에 백엔드 옵션 추가:

```python
p.add_argument("--backend",
               choices=["donut-docvqa", "paddleocr-vl"],
               default="paddleocr-vl",  # ★ default 변경 (D-039)
               help="OCR 백엔드 (default: paddleocr-vl — D-039 채택)")
```

**예상 결과** (Day 3 사전 검증 통과 시):
- 23개 stage1_fp_notes crops 다국어 OCR
- 한/일/중 도면 처리 가능
- TitleBlock cell 좌표 + Notes 자유 텍스트 모두 처리
- 환각률 < 10% 기대 (Donut DocVQA 96% → 대폭 개선)

### 8.6 폴백 트리 (PaddleOCR-VL-1.5 부족 시)

```
PaddleOCR-VL-1.5 (★ D-039 채택)
   ↓ V6 검증 등에서 미흡 시
Qwen3-VL-8B (다국어 + 더 큰 모델)
   ↓
DeepSeek-OCR-2 (수식 + 효율, VRAM 16GB+ 필요)
```

### 8.7 관련 박제

- **D-039**: `PROJECT_HANDOFF.md §11.39` (Stage 3-A 모델 채택, 8가지 사유)
- **§A.11.9**: `history.md` (모델 선정 + 2026 SOTA 비교 + 하이브리드 아키텍처)

### 8.5 코드 확장 계획

`rescue_misclassified_notes.py` 에 OCR 백엔드 옵션 추가:

```bash
# 현재 (실패): Donut DocVQA
python src/rescue_misclassified_notes.py --backend donut-docvqa

# 신규 (계획): easyOCR
python src/rescue_misclassified_notes.py --backend easyocr --languages ja,ko,en

# 신규 (계획): PaddleOCR
python src/rescue_misclassified_notes.py --backend paddleocr --language japan
```

→ Day 3 진입 시 코드 작성 + 재실행 예정.
