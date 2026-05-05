# 엔지니어링 도면 자동 해석 파이프라인 구현 계획

> 출처 논문: *A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model* (Khan et al.)

---

## 0. 프로젝트 조건 (확정)

| 항목 | 값 |
|---|---|
| 아키텍처 | 논문 재현 (YOLOv11-det + YOLOv11-obb + Donut Alphabetical/Numerical) |
| 데이터셋 | **자체 도면** (직접 수집 + 라벨링 필요) |
| 입력 포맷 | **JPG only** (PDF 전처리 모듈 제외) |
| GPU | RTX 5080 (16GB VRAM) |
| CPU/RAM | i9-13900K / 128GB |

### 하드웨어 기반 학습 설정 조정
- **YOLOv11**: `yolo11m` 권장, 입력 1280, batch 8–12
- **Donut fine-tune (Numerical VLM)**:
  - 16GB VRAM 한계 → 논문 batch=4 그대로 / FP16 / gradient checkpointing 활성
  - 입력 해상도: 960×1280 (Donut 기본) 유지 가능
- **Donut Alphabetical**: zero-shot이라 추론만 → 메모리 여유

---

## 0-1. 자체 데이터셋 수집·라벨링 워크플로우 (신규)

논문 데이터셋이 없으므로 직접 구축 필요.

### A. 도면 수집
- 형식: **JPG** 통일 (회사 도면 / 오픈 데이터 / GrabCAD 등)
- 권장 수량 (논문 비례 → 자체 환경 최소치):
  - Stage 1 (Layout): 최소 200–300장 (논문 1,000)
  - Stage 2 (Annotation): 최소 300–500장 (논문 1,406)
  - Stage 3 (VLM image–text 쌍): Numerical 최소 2,000–3,000쌍
- 해상도: 단변 ≥ 1500px 권장 (작은 GD&T 가독성 확보)

### B. 라벨링 도구 (제안)
- **CVAT** (오픈소스, 로컬 docker 가능, OBB 지원) ★권장
- 또는 Roboflow (클라우드, OBB·YOLO export 즉시)
- 또는 labelImg (det만, OBB 미지원이라 Stage 2엔 부적합)

### C. 클래스 정의 (D-028 갱신)
- Stage 1: **`Isometric`, `PMI`, `Table`, `Text`, `View`** (5클래스, Roboflow data.yaml 기준).
  코드 내부 매핑 (D-029): `Table → TitleBlock`, `Text → Notes` (Stage 3-A 호환). `Isometric/PMI` 는 신규.
- Stage 2: `Measure`, `GDT`, `Roughness` (논문 그대로 유지). 입력 영역은 Stage 1 의 `PMI` crop.
- 도면 언어: **EN / KO / JP / RU / CN / DE** (6개, D-025 — DE 2026-05-04 추가, ~10장 보유).
- 도면 종류: 가공도면 + 조립도면 혼재 (D-026, `src/sort_by_drawing_type.py` 로 자동 분류).
- TB 핵심 필드 (material/quantity) 누락 흔함 (D-027) — Step 9 enrichment 가 보강.

### D. Stage 3 Image–Text Pair 생성 절차
1. Stage 1/2 학습 완료된 YOLO로 자체 도면을 **자동 crop**
2. 각 패치에 대해 사람이 ground-truth JSON을 작성/검수
3. 결과를 `data/vlm/{alphabetical, numerical}/` 에 `image.jpg + label.json` 쌍으로 저장

---

## 0-2. 전체 파이프라인 한눈에 보기

```
[원본 도면 (PNG / PDF)]
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: YOLOv11-det (Layout, 5 클래스 D-028)                 │
│  → Isometric / PMI / Table=TB / Text=Notes / View            │  axis-aligned BBox
└──────────────────────────────────────────────────────────────┘
        │   (각 영역 crop, D-029 매핑 적용)
        ├────────────────────────────────────────┐
        │                                        │
        ▼ PMI crop (★ Stage 2 입력)              ▼ TitleBlock / Notes crop
┌──────────────────────────────────────┐   ┌──────────────────────────────┐
│ Stage 2: YOLOv11-obb (Annotation)    │   │ Stage 3-A: PaddleOCR-VL-1.5  │
│  ★ 5-Fold Ensemble (D-040)           │   │   (★ D-039 채택, zero-shot)  │
│  → Measure / GDT / Roughness         │   │   ★ Task: Table Recognition  │
│     (oriented BBox, D-023 PASS)      │   │     OCR / Spotting / Seal    │
└──────────────────────────────────────┘   │   ★ D-046 monkey-patch 필수   │
                                            │   → 텍스트/카테고리 JSON       │
                                            └──────────────────────────────┘
        │   (각 annotation crop)
        ▼
┌──────────────────────────────────────┐
│ Stage 3-N: Numerical VLM             │
│   (Donut, fine-tuned)                │
│   → 수치/심볼 JSON                    │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│ Stage 4: JSON 통합 / 후처리           │
│   → 최종 구조화 출력 (.json)          │
└──────────────────────────────────────┘
```

---

## 1. 환경 및 디렉터리 구조

### 1.1 의존성
- Python 3.10+
- CUDA 12.x + PyTorch (RTX 5080 호환 빌드)
- `ultralytics` (YOLOv11-det, YOLOv11-obb)
- `transformers`, `torch`, `accelerate`, `sentencepiece` (Donut VLM)
- `Pillow`, `opencv-python`, `numpy`
- (선택) `wandb` 학습 모니터링
- ~~`pdf2image`~~ — JPG only이므로 제외

### 1.2 디렉터리 구조 (제안)
```
Drawing/
├── PIPELINE.md                  # 본 문서
├── requirements.txt
├── configs/
│   ├── yolo_det.yaml            # 1단계 데이터셋 설정 (Views/Title/Notes)
│   ├── yolo_obb.yaml            # 2단계 데이터셋 설정 (Measures/GD&T/Roughness)
│   └── donut_numerical.yaml     # Numerical VLM 학습 설정
├── data/
│   ├── raw/                     # 자체 수집 원본 JPG (라벨 전)
│   ├── layout/                  # Stage 1 자체 데이터셋
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}   # YOLO det 형식
│   ├── annotation/              # Stage 2 자체 데이터셋
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}   # YOLO obb 형식
│   └── vlm/                     # Stage 3 image-text pair (자체 생성)
│       ├── alphabetical/        # title block + notes 패치
│       └── numerical/           # 수치/심볼 패치
├── src/
│   ├── stage1_layout.py         # YOLOv11-det 학습/추론
│   ├── stage2_annotation.py     # YOLOv11-obb 학습/추론
│   ├── stage3_alphabetical.py   # Donut zero-shot 추론
│   ├── stage3_numerical.py      # Donut fine-tuning + 추론
│   ├── pipeline.py              # end-to-end 실행기
│   ├── prepare_vlm_dataset.py   # YOLO 결과로 VLM pair 자동 crop
│   └── utils/
│       ├── crop.py              # BBox/OBB crop 유틸
│       ├── json_merge.py        # 통합 JSON 빌더
│       └── metrics.py           # P/R/F1, hallucination rate
├── checkpoints/
│   ├── yolo_det.pt
│   ├── yolo_obb.pt
│   └── donut_numerical/
└── outputs/
    └── <drawing_id>.json        # 최종 결과
```

---

## 2. Stage 1 — YOLOv11-det (Layout Segmentation)

### 2.1 목적
도면 한 장에서 **Views / Title Block / Notes** 3개 클래스의 axis-aligned bounding box 검출.

### 2.2 데이터
- 1,000장, 라벨 수: Views 3,498 / Title Block 458 / Notes 1,127
- 분할: Train 80% / Test 20%
- 형식: YOLO det 표준 (`class cx cy w h`, normalized)

### 2.3 학습
- 모델: `yolo11{n,s,m}.pt` 중 도면 해상도/속도 trade-off 보고 선택 (기본 `yolo11m`)
- 입력 해상도: 1280 권장 (도면 디테일 반영)
- 데이터 증강: HSV/scale/translate만, **flip 비활성** (도면 방향성 보존)
- 출력 체크포인트: `checkpoints/yolo_det.pt`

### 2.4 추론 출력
```json
{
  "drawing_id": "...",
  "regions": [
    {"class": "View",       "bbox": [x1,y1,x2,y2], "conf": 0.97},
    {"class": "TitleBlock", "bbox": [...],         "conf": 0.99},
    {"class": "Notes",      "bbox": [...],         "conf": 0.98}
  ]
}
```

---

## 3. Stage 2 — YOLOv11-obb (Annotation Localization)

### 3.0 입력 준비: PMI crop 추출 (D-034, D-037)

> **신규 (2026-04-30)**: Stage 1 의 PMI 영역 자동 crop 추출. **D-034 hierarchical** 구조에 따라 Stage 1 의 axis-aligned PMI bbox 를 crop 해서 Stage 2 의 입력으로 사용.

**배경 (D-034)**:
- Stage 1: PMI 영역 axis-aligned 검출 (거친 영역 후보)
- Stage 2: PMI crop 내부에서 Measure / GDT / Roughness OBB 정밀 검출
- Hierarchical 구조로 노이즈 감소 + 후속 Stage 3-N 성능 향상

**PMI crop 추출 (D-037 padding 진화)**:

1. **v2 (per-axis adaptive)**:
   ```bash
   python src/extract_pmi_crops.py
   # → outputs/cvat_stage2_input_v2/  (844 PMI crops)
   ```
   - 수식: `pad_x = clamp(bbox_w × 0.4, [30, 80])`, 동일하게 pad_y
   - 통계 (20도면): pad_x mean=33.2 / pad_y mean=30.6 / max=44 px
   - 만족도: 비회전 90% / 회전 80%

2. **v3 (aspect-aware, 권장)**:
   ```bash
   python src/extract_pmi_crops_v3.py
   # → outputs/cvat_stage2_input_v3/  (844 PMI crops)
   ```
   - aspect = max(w,h) / min(w,h) 로 정사각형 판정 (threshold 1.5)
   - 정사각형 (aspect<1.5): uniform pad = long_side × 0.6 → 45° 회전 화살표 보강
   - 비정사각형: per-axis (v2 동일) → 인접 치수 침입 최소화
   - 예상 만족도: 약 90% (회전 텍스트 개선)

**manifest 파일**:
- v2: `pad_x`, `pad_y`, `padding_mode` 컬럼 기록
- v3: 추가로 `aspect_ratio`, `padding_strategy` 컬럼
- 모두 Group key (D-024) 보존으로 원본 도면 추적 가능

**차후 단계**: PMI crop 들을 CVAT 로 업로드 → Stage 2 OBB 라벨링.

상세: [`docs/modules/extract_pmi_crops.md`](./docs/modules/extract_pmi_crops.md) / [`extract_pmi_crops_v3.md`](./docs/modules/extract_pmi_crops_v3.md) / [`MANUAL.md` §4.7](./MANUAL.md)

### 3.1 목적
**PMI crop 이미지** 안에서 **Measures / GD&T / Surface Roughness**를 OBB(회전 BBox)로 검출 (D-034 hierarchical 구조).

### 3.2 데이터
- 입력: Stage 1 의 PMI crop (자체 데이터셋 844장 ← 20도면 × ~42 PMI/도면)
- 라벨 수: 자체 라벨링 중 (목표 500~1,000장 seed 예상, 논문 1,406 비대비)
- 분할: Train 80% / Test 20%
- 형식: YOLO obb (`class x1 y1 x2 y2 x3 y3 x4 y4`)
- **불균형 처리**: Roughness oversampling 또는 class weighting 필요 (seed 데이터 부족)

### 3.3 학습 (★ K-fold CV + Ensemble, D-040)
- 모델: `yolo11l-obb.pt` (★ K-fold CV 채택)
- 입력 해상도: 1024
- 회전 augmentation 강화 (Option C — degrees 30, mixup 0.15, copy_paste 0.3)
- **5-Fold Cross-Validation** (D-024 group-aware):
  - 9.0h overnight 학습
  - mean mAP@0.5 = 0.932 ± 0.062
  - Best fold = 2 (mAP 0.978)
- 출력: `checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_{0..4}/weights/best.pt`

### 3.4 ★ Ensemble 추론 (D-040, 2026-05-04)

**5-Fold Ensemble** (`src/ensemble_predict.py`):
- 5개 fold detection concatenate → class-wise rotated NMS (`iou_nms=0.5`)
- D-023 PASS (Measure/GDT/Roughness missing = 0.000, drawing_recall = 1.000)
- Trade-off: Recall +0.101 (Measure) / Precision -0.266 — D-023 만족 우선

**Pipeline 통합** (`src/pipeline.py`):
- `use_ensemble=True` (default, D-040)
- 5 fold lazy-load + class-wise NMS
- model_versions: `"5fold_ensemble (kfold_0..4, iou_nms=0.5, conf=0.25)"`

**ultralytics 호환성** (★ NMS resolver):
- ultralytics 8.3+ 에서 `nms_rotated` 위치 변경 → 다중 import 경로 + manual shapely fallback
- `_resolve_nms_rotated()` → 자동 감지

### 3.5 추론 출력
```json
{
  "pmi_crop_id": "crop_0",
  "annotations": [
    {"class": "Measure",   "obb": [[x,y]*4], "angle": 12.5, "conf": 0.93},
    {"class": "GDT",       "obb": [...],     "angle": 0.0,  "conf": 0.96},
    {"class": "Roughness", "obb": [...],     "angle": 90.0, "conf": 0.71}
  ]
}
```

---

## 4. Stage 3-A — PaddleOCR-VL-1.5 (★ D-039, 2026-05-03 채택 — Donut 폐기)

### 4.1 목적
Title Block / Notes 영역 이미지를 입력받아 **다국어 OCR + structured 텍스트** JSON 추출.

### 4.2 모델 변경 이력 (★ 박제)

| 버전 | 모델 | 결과 | 박제 |
|---|---|---|---|
| 1차 | Donut DocVQA zero-shot | 4% 성공 (실질 실패) | D-018 폐기 |
| **★ 2차** | **`PaddlePaddle/PaddleOCR-VL-1.5`** | OmniDocBench 94.5% (SOTA) | D-039 채택 |

**선정 사유 (8가지)**:
1. OmniDocBench 94.50% (DeepSeek-OCR-2 91.09% 대비 +3.41%)
2. 0.9B params → RTX 5080 16GB 에서 Stage 2 동시 로드 가능 (3.29 GB)
3. Table TEDS 92.76% — Title Block 표 처리
4. Formula CDM 94.21% — Notes 수식 정확
5. Seal Recognition 1.5 신규 — 도장/검도 도장 처리
6. CJK industry-leading — 일/한/중 도면 처리
7. 100+ 다국어 — DE/RU 포함 6개 언어 모두 지원 (D-025)
8. 라이선스 Apache 2.0

### 4.3 ★ Critical Workaround (D-042 / D-046)

**D-042 — `text_config` monkey-patch** (transformers 5.x 호환성):
```python
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
import torch

config = AutoConfig.from_pretrained("PaddlePaddle/PaddleOCR-VL-1.5", trust_remote_code=True)
if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
    config.text_config = config.get_text_config()    # ★ 1줄 patch 필수

model = AutoModelForImageTextToText.from_pretrained(
    "PaddlePaddle/PaddleOCR-VL-1.5",
    config=config,
    torch_dtype=torch.bfloat16,                       # ★ D-046: bfloat16 (NOT float16)
).to("cuda").eval()
```

**D-046 — README 권장 호출 방식** (NOT 자연어 prompt):
```python
# ★ Task keyword (자연어 prompt → 모델 confuse)
TASKS = {
    "table":    "Table Recognition:",   # ★ TitleBlock
    "ocr":      "OCR:",                 # Notes / 전체 텍스트
    "spotting": "Spotting:",            # 텍스트 + bbox
    "formula":  "Formula Recognition:", # 수식
    "seal":     "Seal Recognition:",    # 도장 (D-038 stage1_fp_table)
}

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},   # ★ image 직접 binding
        {"type": "text", "text": TASKS["table"]},
    ]
}]

# ★ apply_chat_template 통합 호출 (NOT processor(images=, text=) 분리)
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True,
    tokenize=True, return_dict=True, return_tensors="pt",
    images_kwargs={
        "size": {
            "shortest_edge": processor.image_processor.min_pixels,
            "longest_edge":  1280 * 28 * 28,    # ~1M pixels
        }
    },
).to(model.device)

# ★ Pure generate (D-045 의 repetition_penalty 등 모두 폐기)
outputs = model.generate(**inputs, max_new_tokens=512)

# ★ Input 부분 슬라이스 + processor.decode
result = processor.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
```

### 4.4 출력 형식

| Task | 출력 형식 | 후처리 |
|---|---|---|
| `Table Recognition:` | **OTSL token** (`<fcel>`/`<lcel>`/`<nl>`) — D-047 | markdown / JSON 변환 필요 |
| `OCR:` | plain text (다국어) | numbered list parse (Notes) |
| `Spotting:` | text + bbox | bbox 좌표 활용 |

### 4.5 환경 — 별도 venv (★ D-042)

Phase 14 ultralytics venv 와 분리:
```bash
uv venv --python 3.10 .venv-paddleocr
source .venv-paddleocr/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==5.0.0" accelerate sentencepiece protobuf einops pillow
```

### 4.6 Phase 15b 평가 결과 (★ 박제)

| 회차 | 결과 | 박제 |
|---|---|---|
| 1차 (자연어 prompt) | degenerate 무한 반복 (95%) | D-045 |
| 2차 (D-045 fix) | layout token 누출 + emoji hallucination | — |
| 3차 (D-046 fix) | en/ru 부분 성공 (avg char acc ~0.50) | D-047 |
| 4차 (Real-ESRGAN, 진행 중) | TBD | TBD |

**임계값 (D-013)**: char acc ≥ 0.85, F1 ≥ 0.80, hallucination ≤ 0.05.

### 4.7 차후 검토

- 4차 (Real-ESRGAN) 결과 PASS 시 → Stage 3-A 백엔드 교체 (Phase 15c)
- FAIL 시 → fine-tune 또는 폴백 (Qwen3-VL / DeepSeek-OCR-3)
- vLLM 도입 (Phase 17 batch 시) — PaddleOCR-VL 공식 지원
- Real-ESRGAN 외 OTSL → markdown 후처리 추가 (Phase 15c)

---

## 4-old. (구) Stage 3-A — Donut Alphabetical (★ D-039 폐기)

> Donut DocVQA zero-shot 4% 성공 (실질 실패) → D-039 로 PaddleOCR-VL-1.5 채택. 본 절 참고용.

- 베이스: `naver-clova-ix/donut-base-finetuned-cord-v2`
- F1 0.672 (Title Block 0.533, Notes 0.810)
- 다국어 (CJK / DE / RU) 처리 한계 → 폐기

---

## 5. Stage 3-N — Numerical VLM (Donut, Fine-tuned)

### 5.1 목적
Stage 2에서 OBB로 crop된 패치(Measure / GD&T / Roughness)를 **schema 기반 수치 JSON**으로 변환.

### 5.2 학습 데이터 준비 흐름 (★ Phase 16a/b, 2026-05-06)

```
Phase 16a — prepare_vlm_dataset.py numerical
  ├─ Stage 1 inference (yolo_det.pt) → View / TitleBlock / Notes crop
  ├─ Stage 2 inference (yolo_obb.pt 단일 fold) → Measure / GDT / Roughness OBB
  ├─ perspective-warp de-rotation
  ├─ Pytesseract OCR (--ocr-prefill) → _review.ocr_hint / ocr_numeric
  └─ JSON 템플릿 + manifest.csv (group_key 포함, D-024)

Phase 16a 결과 (500 도면, 2026-05-06):
  - 11,470 region 생성 (도면당 평균 22.94)
  - 처리 시간: 24분 04초

★ auto_fill_numerical_gt.py (D-049 sys.path bootstrap 적용)
  ├─ Measure: ocr_numeric → nominal 매핑, tolerance regex (±X / +X/-Y)
  ├─ GDT: 14개 symbol pattern + datum letter 추출
  ├─ Roughness: Ra regex + first numeric fallback
  └─ _review.completed = True / auto_filled / fill_method 박제

Auto-fill 결과 (Phase 16a 직후):
  - Total: 11,470 → Filled 5,784 (50.4%)
  - Measure: 8,750 → 5,381 (61.5%) ★ baseline 가능
  - GDT:        531 →     1 (0.2%)  ★ 학습 사실상 불가 (D-051)
  - Roughness: 2,189 →   402 (18.4%)

Phase 16b — stage3_numerical.py train
  └─ configs/donut_numerical.yaml 기반 학습 시작
```

### 5.3 학습 설정 (논문 재현 + cfg 파일 통합)

**CLI**:
```bash
python src/stage3_numerical.py train \
    --cfg configs/donut_numerical.yaml \
    --device 0
# 모든 hyperparameter 는 cfg 파일에 정의
```

**configs/donut_numerical.yaml 핵심**:
- 데이터: `data/vlm/numerical/` (Phase 16a 출력)
- 학습 가능 sample (completed=True): 5,784
- 분할: 70 / 20 / 10 (train ~4,049 / val ~1,156 / test ~579)
- 에포크: 30
- Optimizer: AdamW
- LR scheduler: cosine decay, init `1e-6`, no warm-up
- Batch size: 4 (RTX 5080 16GB 대응)
- Precision: FP16 mixed
- gradient_checkpointing: true
- 출력: `checkpoints/donut_numerical/`

### 5.4 ★ 1차 baseline 정의 (D-051)

- **★ Phase 16b 1차 baseline = Measure nominal extraction only**
- GDT 학습 사실상 불가 (auto-fill 0%, sample 1/531)
- Roughness 제한적 (auto-fill 18.4%)
- Phase 17 e2e 평가에서 Stage 3-N 자리만 채움 → 후속 개선 우선순위 정량화
- 후속 (Phase 18+): 검수 도구 + 사람 검수 + GDT crop 추가 라벨링 → 재학습

### 5.3 출력 스키마 예시
```json
{
  "type": "Measure",
  "nominal": 25.0,
  "tolerance": {"upper": 0.05, "lower": -0.05},
  "unit": "mm"
}
```
```json
{
  "type": "GDT",
  "symbol": "⏤",          // Unicode 인코딩
  "tolerance": 0.02,
  "datum": ["A", "B"]
}
```
```json
{"type": "Roughness", "Ra": 1.6, "unit": "μm"}
```

---

## 6. Stage 4 — 통합 JSON & 후처리

### 6.1 통합 구조
```json
{
  "drawing_id": "drawing_001",
  "title_block": { ...alphabetical VLM... },
  "notes":       [ ...alphabetical VLM... ],
  "views": [
    {
      "view_id": "view_0",
      "bbox": [...],
      "annotations": [
        { ...numerical VLM JSON..., "obb": [...], "conf": 0.93 }
      ]
    }
  ]
}
```

### 6.2 후처리 체크
- 빈/오류 JSON 검출 → null 채우기
- Hallucination 필터: 신뢰도 임계값 + schema validator
- 단위 정규화 (mm/inch, μm/Ra, …)

---

## 7. 평가 지표

| 단계 | 지표 |
|---|---|
| Stage 1 / 2 | mAP@0.5, per-class accuracy (논문 confusion matrix 재현) |
| Stage 3-A | Precision / Recall / F1 / Hallucination rate (Title, Notes) |
| Stage 3-N | Precision / Recall / F1 / Hallucination rate (Measure, GD&T, Roughness) |
| 전체 | end-to-end JSON field-level F1 |

---

## 8. 구현 순서 (모두 완료 ✅)

| # | 단계 | 산출물 | 상태 |
|---|---|---|---|
| 1 | 환경 세팅 & 디렉터리 골격 | `requirements.txt`, 폴더 트리, `configs/*.yaml` 템플릿 | ✅ |
| 1.5 | TitleBlock 분류기 (선택) | `src/sort_by_titleblock.py` | ✅ |
| 2 | Stage 1 학습/추론 스크립트 | `src/stage1_layout.py` | ✅ |
| 3 | Stage 2 학습/추론 스크립트 | `src/stage2_annotation.py` | ✅ |
| 4 | VLM 학습 데이터 자동 생성 | `src/prepare_vlm_dataset.py` (라벨
---

## 5. Stage 3-N — Numerical VLM (Donut, Fine-tuned, ★ Phase 16a/b)

### 5.1 목적
Stage 2에서 OBB로 crop된 patch (Measure / GD&T / Roughness)를 **schema 기반 수치 JSON**으로 변환.

### 5.2 학습 데이터 준비 흐름 (Phase 16a/b, 2026-05-06)

```
Phase 16a — prepare_vlm_dataset.py numerical
  ├─ Stage 1 inference (yolo_det.pt) → View / TitleBlock / Notes crop
  ├─ Stage 2 inference (yolo_obb.pt 단일 fold) → Measure / GDT / Roughness OBB
  ├─ perspective-warp de-rotation
  ├─ Pytesseract OCR (--ocr-prefill) → _review.ocr_hint / ocr_numeric
  └─ JSON 템플릿 + manifest.csv (group_key 포함, D-024)

Phase 16a 실측 결과 (2026-05-06, 500 도면):
  - 처리 시간: 24분 04초
  - 산출 region: 11,470 (도면당 평균 22.94)

★ auto_fill_numerical_gt.py (D-049 sys.path bootstrap 적용)
  ├─ Measure: ocr_numeric → nominal 매핑, tolerance regex (±X / +X/-Y)
  ├─ GDT: 14개 symbol pattern + datum letter 추출
  ├─ Roughness: Ra regex + first numeric fallback
  └─ _review.completed = True / auto_filled / fill_method 박제

Auto-fill 실측 결과:
  - Total: 11,470 → Filled 5,784 (50.4%)
  - Measure: 8,750 → 5,381 (61.5%) ★ baseline 가능
  - GDT:        531 →     1 ( 0.2%) ★ 학습 사실상 불가 (D-051)
  - Roughness: 2,189 →   402 (18.4%)

Phase 16b — stage3_numerical.py train
  └─ configs/donut_numerical.yaml 기반 학습 시작 (overnight ~6h)
```

### 5.3 학습 설정 (cfg 파일 통합)

**CLI**:
```bash
python src/stage3_numerical.py train \
    --cfg configs/donut_numerical.yaml \
    --device 0
# 모든 hyperparameter 는 cfg 파일에 정의
```

**configs/donut_numerical.yaml 핵심**:
- 데이터: `data/vlm/numerical/` (Phase 16a 출력)
- 학습 가능 sample (completed=True): 5,784
- 분할: 70 / 20 / 10 (train ~4,049 / val ~1,156 / test ~579)
- 에포크: 30 / Batch 4 / LR 1e-6 cosine decay / no warm-up
- Precision: FP16 mixed + gradient_checkpointing (RTX 5080 16GB 대응)
- 출력: `checkpoints/donut_numerical/`

### 5.4 ★ 1차 baseline 정의 (D-051)

- **★ Phase 16b 1차 baseline = Measure nominal extraction only**
- GDT 학습 사실상 불가 (auto-fill 0%, sample 1/531) — Stage 2 라벨 부족 (KNOWN_LIMITATIONS §2.1) + Tesseract OCR 한계 (D-050) 의 결합
- Roughness 제한적 (auto-fill 18.4%)
- Phase 17 e2e 평가에서 Stage 3-N 자리만 채움 → 후속 개선 우선순위 정량화
- 후속 (Phase 18+): 검수 도구 + 사람 검수 + GDT crop 추가 라벨링 → 재학습

### 5.5 출력 스키마 예시

```json
{"type": "Measure", "nominal": 25.0, "tolerance": {"upper": 0.05, "lower": -0.05}, "unit": "mm"}
{"type": "GDT", "symbol": "⏤", "tolerance": 0.02, "datum": ["A", "B"]}
{"type": "Roughness", "Ra": 1.6, "unit": "μm"}
```

### 5.6 ★ 한계 박제 (docs/KNOWN_LIMITATIONS.md §4 참조)

| ID | 한계 | 영향 |
|---|---|---|
| D-049 | sys.path bootstrap (해결) | prepare_vlm_dataset / auto_fill 직접 실행 가능 |
| D-050 | Tesseract OCR 본질적 한계 (Critical) | tolerance/GDT/Ra 인식 0% — regex 보강 효과 ≈ 0 |
| D-051 | 1차 baseline = Measure-only | Phase 17 e2e 자리 채움 + 후속 정량화 |

---

## 6. Pipeline 통합 (Phase 17 e2e — 2026-05-06 이후)

`pipeline.py` 가 Stage 1 → Stage 2 → Stage 3-A → Stage 3-N → Stage 4 (JSON Merger) → Step 9 Enrichment 통합 실행.

자세한 흐름은 `outputs/workflow_diagram_v4.png` 참조.
