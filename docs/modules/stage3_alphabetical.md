# `src/stage3_alphabetical.py`

> **Step 5** — Donut Alphabetical VLM (zero-shot, no fine-tuning)

## 1. 구현 요약

Stage 1 이 잘라낸 **TitleBlock + Notes crop** 에 대해 Donut 사전학습 모델을 zero-shot 모드로 호출, 자유 형식 JSON 추출.

**2개 동작 모드**

| 모드 | 모델 | 동작 |
|---|---|---|
| `docvqa` (기본) | `naver-clova-ix/donut-base-finetuned-docvqa` | 14개 질문 멀티 패스 — 정확도 ↑, 속도 ↓ |
| `cord` | `naver-clova-ix/donut-base-finetuned-cord-v2` | 단일 패스, CORD 영수증 스키마 → 필드명 매핑 필요 |

**14개 기본 질문 (TitleBlock, DocVQA mode)**

drawing_no / title / material / scale / revision / date / drawn_by / checked_by / approved_by / part_no / sheet / project / weight / tolerance

**Notes 처리**: 단일 질문 `"What do the notes say?"` → 정규식으로 numbered list 분리.

**공개 함수**

```python
load_model(model_name, device) -> (processor, model, device)
predict_titleblock(image_path, processor, model, device, ...) -> dict
predict_notes(image_path, processor, model, device, ...) -> dict
predict_one(image_path, region_type, mode, ...) -> dict   # dispatcher
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| Fine-tuning | **없음 (zero-shot)** | D-001 / D-018 / 논문 §4.3 (TitleBlock schema 변동성 큼) |
| 기본 모드 | DocVQA 멀티 질문 | 엔지니어링 필드 직접 매핑, 정확도 우선 |
| 대안 모드 | CORD-v2 단일 패스 | 빠르지만 receipt 스키마 매핑 필요 |
| 메모리 최적화 | **FP16** (`model.half()`) | RTX 5080 16GB / 700MB → 350MB |
| 모델 캐싱 | 배치 처리 시 1회 로드 | DocVQA 14회 호출 시 중요 |
| 언어 힌트 | 옵션 파라미터 `language_hint` 출력에 보존 | 후속 enrichment 활용 |
| Notes 분리 | 정규식 (numbered list 우선, fallback 줄바꿈) | 자유 텍스트 구조화 |
| 빈/실패 응답 처리 | "none" / "n/a" / "unknown" 필터링 | hallucination 감소 |
| Token2json 안전 처리 | try/except + raw fallback | Donut 출력 형식 깨짐 대응 |

## 3. 사용법

### CLI

```bash
# TitleBlock (DocVQA mode, 14 fields)
python src/stage3_alphabetical.py predict \
    --image outputs/crops/sample/TitleBlock/sample__TitleBlock_00.jpg \
    --region titleblock --mode docvqa --language en

# Notes (free-form)
python src/stage3_alphabetical.py predict \
    --image outputs/crops/sample/Notes/sample__Notes_00.jpg \
    --region notes

# 배치 (Stage 1 crop 폴더 통째로)
python src/stage3_alphabetical.py batch \
    --input-dir outputs/crops/sample \
    --out-dir outputs/sample/alphabetical

# CORD 모드 (단일 패스, 빠름)
python src/stage3_alphabetical.py predict \
    --image titleblock.jpg --region titleblock --mode cord
```

### 공개 함수

```python
from src.stage3_alphabetical import load_model, predict_one

processor, model, device = load_model()  # 1회 로드
for image_path in images:
    rec = predict_one(image_path, region_type="titleblock",
                      processor=processor, model=model, device=device)
```

## 4. 검증 결과

### 4.1 정량 검증 (V5, 작성 예정)

논문 zero-shot 성능 베이스라인:

| 항목 | 논문 값 | 임계값 |
|---|---|---|
| TitleBlock F1 | 0.533 | warning ≥ 0.50 |
| **Notes F1** | **0.810** | **critical ≥ 0.75** |
| Hallucination rate | 0.40 ~ 0.48 | warning < 0.50 |

(V5 검증기 작성 후 실제 측정)

### 4.2 단위 검증

- CLI `--help` 정상
- `transformers` 미설치 시 ImportError 처리
- 빈 응답 / 잘못된 JSON 출력 시 raw fallback

## 5. 출력 형식

### 5.1 TitleBlock 출력 (HANDOFF §5.3)

```json
{
  "type": "TitleBlock",
  "source": "outputs/crops/sample/TitleBlock/sample__TitleBlock_00.jpg",
  "fields": {
    "drawing_no": "DWG-001-A",
    "material": "SS400",
    "scale": "1:2",
    "revision": "B",
    "drawn_by": "Kim"
  },
  "raw": {
    "What is the drawing number?": "DWG-001-A",
    "What is the material?": "SS400",
    ...
  },
  "model": "donut-base-finetuned-docvqa",
  "language_hint": "en"
}
```

`fields` 는 답변이 있는 14개 질문만 정제된 결과, `raw` 는 전체 질문→답변 원본.

### 5.2 Notes 출력

```json
{
  "type": "Notes",
  "source": "outputs/crops/.../Notes/sample__Notes_00.jpg",
  "items": [
    "1. UNLESS OTHERWISE SPECIFIED",
    "2. ALL DIMENSIONS IN MM",
    "3. BREAK ALL SHARP EDGES"
  ],
  "raw": "1. UNLESS OTHERWISE SPECIFIED 2. ALL ...",
  "model": "donut-base-finetuned-docvqa",
  "language_hint": "en"
}
```

### 5.3 배치 manifest

```
outputs/sample/alphabetical/
├── sample__TitleBlock_00.alpha.json
├── sample__Notes_00.alpha.json
└── manifest.json   ← {image, region, json, fields/items count}
```

## 6. 의존성

```
torch (CUDA 12.8)
transformers>=4.44.0
sentencepiece>=0.2.0
Pillow>=10.3.0
```

## 7. 관련 의사결정

- **D-001** 아키텍처 = Donut
- **D-013** 4개 언어 (Donut 영어 강함, 비-EN 정확도 낮음 — 논문 0.672 baseline)
- **D-018** Stage 3 모델 = Donut 유지 (성능 미달 시 Step 7 평가 후 재논의)

## 8. 검증 모듈

[`check_stage3a_alphabetical.py`](./../README.md) — V5, 작성 예정

## 9. 다운스트림

Stage 3-A 의 출력은:

- **`pipeline.py`** (Step 7) 에서 통합 JSON 으로 합쳐짐
- **`stage5_enrichment.py`** (Step 9) 에서 LLM 보정 (Hallucination / 누락 필드 보강)

## 10. 성능 한계

- Donut 영어 중심 사전학습 → KO/JP/RU 정확도 차이 있음
- TitleBlock F1 0.533 (논문) — Step 9 enrichment 로 보정 필요
- Hallucination rate 측정 → V5 검증기에서 자동 산출
