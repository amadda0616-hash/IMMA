# 프로젝트 인수인계 문서 (PROJECT HANDOFF)
## Multi-Stage Hybrid Framework for Engineering Drawings — 자체 구현

> **이 문서의 목적**
> 본 문서는 어떤 AI 어시스턴트(Claude / GPT / Gemini 등)에게 이 프로젝트를 인계하더라도 **즉시 이어서 작업**할 수 있도록 작성된 자기완결(self-contained) 사양서다.
> **작업 시작 전 반드시 본 문서 §0 (체크리스트)부터 읽고, §10 (작업 진행 현황)을 갱신하면서 진행할 것.**

---

## 0. 새 세션 시작 시 체크리스트 (READ FIRST)

새로운 AI 세션이 이 프로젝트를 이어받을 때 **반드시 다음 순서로 확인**한다.

1. [ ] 본 문서 §1 (프로젝트 개요) 와 §2 (제약 조건) 를 읽는다.
2. [ ] §3 (전체 아키텍처) 의 다이어그램을 머릿속에 그린다.
3. [ ] §10 (작업 진행 현황) 에서 **마지막으로 끝난 단계**를 확인한다.
4. [ ] §11 (의사결정 로그) 의 항목들을 모두 존중한다 (재논의 금지).
5. [ ] §12 (사용자 선호) 를 준수한다.
6. [ ] 워크스페이스 폴더(`C:\Users\user\github\Drawing`)의 실제 파일 구조와 §6 의 디렉터리 구조를 비교해 어디까지 구현됐는지 검증한다.
7. [ ] **§12 의 "코드 작성 전 확인 룰"을 준수**: 코드 작성 직전에 항상 한국어로 “…를 작성해드릴까요?” 확인을 먼저 받는다.

---

## 1. 프로젝트 개요

### 1.1 목표
2D 엔지니어링 도면(JPG)을 입력받아 구조화된 JSON으로 변환하는 **3-stage hybrid pipeline** 을 자체 구축한다. 아키텍처는 아래 논문을 100% 재현하되, **데이터셋은 사용자가 자체 수집·라벨링** 한다.

### 1.2 출처 논문
*A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model*
Khan, Yong, Chen, Feng, Tan, Moon (Singapore) — 논문 PDF는 `uploads/Multi-Stage Hybrid Framework for Engineering Drawings.pdf`

### 1.3 논문에서 보고된 성능 (재현 목표)
| Stage | 모델 | 핵심 지표 |
|---|---|---|
| 1 | YOLOv11-det | acc: View 0.96 / TitleBlock 0.99 / Notes 0.98 |
| 2 | YOLOv11-obb | acc: Measure 0.95 / GD&T 0.97 / Roughness 0.54 |
| 3-A | Donut (zero-shot) | F1: TitleBlock 0.533 / Notes 0.810 / overall 0.672 |
| 3-N | Donut (fine-tuned) | F1: Measure 0.923 / GD&T 0.965 / Roughness 1.0 / overall 0.963 |

---

## 2. 제약 조건 (CONFIRMED)

| 항목 | 값 | 비고 |
|---|---|---|
| 아키텍처 | 논문 그대로 재현 | YOLOv11 + Donut |
| 데이터셋 | 자체 수집·라벨링 | 양은 가용 범위에서 |
| 입력 포맷 | **JPG only** | PDF 전처리 모듈 제외 |
| GPU | RTX 5080 (16GB VRAM, Blackwell, **sm_120**) | **CUDA 12.8+** (D-030: cu124 빌드는 sm_90 까지만 지원하여 비호환) |
| CPU | Intel i9-13900K | |
| RAM | 128GB | |
| OS | **Ubuntu 22.04 LTS on WSL2** (Windows 11 host) | 2026-04-25 변경 |
| IDE | **Antigravity** (VS Code 기반, WSL Remote) | 2026-04-25 추가 |
| Workspace 경로 | `/mnt/c/Users/user/github/Drawing` (WSL2에서 접근) / Windows 측 `C:\Users\user\github\Drawing` 동일 위치 | |
| 도면 언어 | **EN / KO / JP / RU / CN** (도면 1장당 단일 언어) | D-010 / D-013 / D-025 |
| 사용자 응답 언어 | 한국어 / 코드 주석은 영어 권장 | |

---

## 3. 전체 아키텍처

### 3.1 파이프라인 다이어그램
```
[입력: 엔지니어링 도면 JPG]
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1 — YOLOv11-det  (Layout Segmentation)            │
│  Input : 원본 JPG                                        │
│  Output: axis-aligned BBox                              │
│          classes = {View, TitleBlock, Notes}            │
└─────────────────────────────────────────────────────────┘
        │
        ├─────────────────────────────────────────┐
        │                                         │
        ▼ View crop들                             ▼ TitleBlock / Notes crop들
┌──────────────────────────────────────┐   ┌─────────────────────────────────┐
│ Stage 2 — YOLOv11-obb (Annotation)   │   │ Stage 3-A — Alphabetical VLM    │
│  Input : View crop                   │   │  Model : Donut (zero-shot)      │
│  Output: oriented BBox               │   │  Input : Title/Notes crop       │
│   classes = {Measure, GDT, Roughness}│   │  Output: free-form JSON         │
└──────────────────────────────────────┘   └─────────────────────────────────┘
        │
        ▼ annotation crop들
┌──────────────────────────────────────┐
│ Stage 3-N — Numerical VLM            │
│  Model : Donut (fine-tuned)          │
│  Input : measure/gdt/roughness crop  │
│  Output: schema-defined JSON         │
└──────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4 — JSON Merger / Post-processing                 │
│  Output: <drawing_id>.json (unified structured output)  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 데이터 흐름과 이미지 좌표계
- Stage 1 BBox 좌표 = **원본 도면 좌표계**
- Stage 2 OBB 좌표 = **Stage 1 View crop 좌표계** → 통합 시 부모 BBox offset 더해 원본 좌표로 환산
- Stage 3 입력 = crop된 패치 (좌표 정보 미보유)

---

## 4. 단계별 상세 사양

### 4.1 Stage 1 — YOLOv11-det
- **베이스 모델**: `yolo11m.pt` (Ultralytics)
- **클래스**: `["View", "TitleBlock", "Notes"]`
- **입력 해상도**: 1280
- **데이터 분할**: train 80% / val 20%
- **증강**: HSV/scale/translate ON, **flip OFF** (도면 방향성 보존)
- **출력**: axis-aligned BBox `[x1, y1, x2, y2]`, conf, cls
- **체크포인트**: `checkpoints/yolo_det.pt`

### 4.2 Stage 2 — YOLOv11-obb
- **베이스 모델**: `yolo11m-obb.pt`
- **클래스**: `["Measure", "GDT", "Roughness"]`
- **입력 해상도**: 1024–1280
- **데이터 분할**: train 80% / val 20%
- **증강**: rotate ON (회전 텍스트 대응)
- **불균형 대응**: Roughness oversampling 또는 class-weighting
- **출력**: OBB 8-point `[x1,y1,...,x4,y4]`, angle, conf, cls
- **체크포인트**: `checkpoints/yolo_obb.pt`

### 4.3 Stage 3-A — Alphabetical VLM (Donut, zero-shot)
- **모델**: `naver-clova-ix/donut-base-finetuned-cord-v2` (또는 `donut-base`)
- **Fine-tuning 없음** (논문 기준; 자유 텍스트라 schema 부재)
- **프롬프트 토큰**: `<s_titleblock>`, `<s_notes>`
- **출력**: 자유 형식 JSON (drawing number, material, scale 등)
- **실패 시**: raw text fallback + 정규식 후처리

### 4.4 Stage 3-N — Numerical VLM (Donut, fine-tuned)
- **모델**: `naver-clova-ix/donut-base`
- **데이터 분할**: train 70% / val 20% / test 10%
- **에포크**: 30
- **Optimizer**: AdamW
- **LR scheduler**: cosine decay, init `1e-6`, no warm-up
- **Batch size**: 4
- **Precision**: FP16 mixed
- **추가 (RTX 5080 16GB 대응)**: gradient checkpointing ON
- **체크포인트**: `checkpoints/donut_numerical/`

---

## 5. JSON 스키마 정의

### 5.1 Stage 1 출력
```json
{
  "drawing_id": "drawing_001",
  "image_size": [W, H],
  "regions": [
    {"class": "View",       "bbox": [x1,y1,x2,y2], "conf": 0.97},
    {"class": "TitleBlock", "bbox": [x1,y1,x2,y2], "conf": 0.99},
    {"class": "Notes",      "bbox": [x1,y1,x2,y2], "conf": 0.98}
  ]
}
```

### 5.2 Stage 2 출력 (View 단위)
```json
{
  "view_id": "view_0",
  "parent_bbox": [x1,y1,x2,y2],
  "annotations": [
    {"class": "Measure",   "obb": [[x,y],[x,y],[x,y],[x,y]], "angle": 12.5, "conf": 0.93},
    {"class": "GDT",       "obb": [[...]], "angle": 0.0,  "conf": 0.96},
    {"class": "Roughness", "obb": [[...]], "angle": 90.0, "conf": 0.71}
  ]
}
```

### 5.3 Stage 3-A 출력 (자유 형식)
```json
{
  "title_block": {
    "drawing_no": "...", "title": "...", "material": "...",
    "scale": "1:2", "tolerance": "±0.1", "revision": "A", ...
  },
  "notes": ["1. UNLESS OTHERWISE SPECIFIED ...", "2. ..."]
}
```

### 5.4 Stage 3-N 출력 (스키마 고정)
```json
// Measure
{"type": "Measure", "nominal": 25.0,
 "tolerance": {"upper": 0.05, "lower": -0.05}, "unit": "mm"}

// GDT
{"type": "GDT", "symbol": "⏤", "tolerance": 0.02, "datum": ["A","B"]}

// Roughness
{"type": "Roughness", "Ra": 1.6, "unit": "μm"}
```

### 5.5 Stage 4 — 최종 통합 JSON
```json
{
  "drawing_id": "drawing_001",
  "image_path": "data/raw/drawing_001.jpg",
  "image_size": [W, H],
  "title_block": { ... },
  "notes": [ ... ],
  "views": [
    {
      "view_id": "view_0",
      "bbox": [x1,y1,x2,y2],
      "annotations": [
        {
          "class": "Measure",
          "obb_global": [[x,y]*4],
          "obb_local":  [[x,y]*4],
          "conf": 0.93,
          "parsed": { "type": "Measure", "nominal": 25.0, ... }
        }
      ]
    }
  ],
  "meta": {
    "model_versions": {
      "yolo_det": "yolo11m@v1",
      "yolo_obb": "yolo11m-obb@v1",
      "donut_alpha": "donut-base-cord-v2",
      "donut_num":   "donut_numerical@v1"
    },
    "timestamp": "ISO8601"
  }
}
```

---

## 6. 디렉터리 구조 (TARGET)
```
Drawing/
├── PROJECT_HANDOFF.md           # 본 문서 (✅ 존재)
├── PIPELINE.md                  # 초기 파이프라인 메모 (✅ 존재)
├── requirements.txt
├── configs/
│   ├── yolo_det.yaml
│   ├── yolo_obb.yaml
│   └── donut_numerical.yaml
├── data/
│   ├── raw/                     # 자체 수집 원본 JPG
│   ├── layout/                  # Stage 1 데이터셋
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}
│   ├── annotation/              # Stage 2 데이터셋
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}
│   └── vlm/                     # Stage 3 image–text pair
│       ├── alphabetical/        # title block + notes
│       └── numerical/           # 수치/심볼 패치
├── src/
│   ├── stage1_layout.py
│   ├── stage2_annotation.py
│   ├── stage3_alphabetical.py
│   ├── stage3_numerical.py
│   ├── pipeline.py
│   ├── prepare_vlm_dataset.py   # YOLO 추론 → VLM pair 자동 crop
│   └── utils/
│       ├── crop.py
│       ├── json_merge.py
│       └── metrics.py
├── checkpoints/
│   ├── yolo_det.pt
│   ├── yolo_obb.pt
│   └── donut_numerical/
└── outputs/
    └── <drawing_id>.json
```

---

## 7. 자체 데이터셋 구축 워크플로우

### 7.1 도면 수집
- 형식: JPG, 단변 ≥ 1500 px 권장
- 권장 수량 (논문 비례 최소치):
  - Stage 1: 200–300장
  - Stage 2: 300–500장
  - Stage 3 Numerical pair: 2,000–3,000쌍

### 7.2 라벨링 도구
- ★ **CVAT** (오픈소스, OBB 지원, docker 로컬 가능)
- 또는 Roboflow (클라우드, OBB·YOLO export 즉시)
- labelImg는 OBB 미지원 → Stage 2엔 부적합

### 7.3 VLM image–text 쌍 생성 절차
1. Stage 1, 2 학습 완료된 YOLO로 자체 도면을 자동 crop (`prepare_vlm_dataset.py`)
2. 각 패치에 대해 사람이 ground-truth JSON 작성·검수
3. `data/vlm/{alphabetical, numerical}/` 에 `image.jpg + label.json` 쌍으로 저장

---

## 8. 평가 지표

| 단계 | 지표 |
|---|---|
| Stage 1 | mAP@0.5, per-class accuracy |
| Stage 2 | mAP@0.5, per-class accuracy (OBB 기준) |
| Stage 3-A | Precision / Recall / F1 / Hallucination rate |
| Stage 3-N | Precision / Recall / F1 / Hallucination rate |
| End-to-end | JSON field-level F1, latency |

> **Hallucination rate** = (스키마에 없는/원문에 없는 필드 생성 비율).
> 구현은 `src/utils/metrics.py` 에서 일원화.

---

## 9. 구현 로드맵 (8 Steps)

| # | Step | 산출물 | 의존 |
|---|---|---|---|
| 1 | 환경 세팅 + 디렉터리 + `requirements.txt` + `configs/*.yaml` | 골격 | — |
| 2 | Stage 1 학습/추론 (`stage1_layout.py`) | `yolo_det.pt` | Step 1 + 라벨링 데이터 |
| 3 | Stage 2 학습/추론 (`stage2_annotation.py`) | `yolo_obb.pt` | Step 1 + 라벨링 데이터 |
| 4 | VLM pair 자동 생성기 (`prepare_vlm_dataset.py`) | `data/vlm/*` 시드 | Step 2, 3 |
| 5 | Stage 3-A 래퍼 (`stage3_alphabetical.py`, zero-shot) | 추론 함수 | Step 1 |
| 6 | Stage 3-N fine-tune (`stage3_numerical.py`) | `donut_numerical/` | Step 4 |
| 7 | end-to-end pipeline (`pipeline.py`) | JPG→JSON | Step 2,3,5,6 |
| 8 | 평가 (`utils/metrics.py`) | 성능 리포트 | Step 7 |

> Step 2/3 학습은 **사용자의 라벨링 완료**가 선행되어야 한다. 코드 작성과 라벨링은 병렬 진행 가능.

---

## 10. 작업 진행 현황 (LIVE STATUS)

> **AI는 작업을 끝낼 때마다 이 표를 갱신할 것.**
> 상태 값: `TODO` / `IN_PROGRESS` / `DONE` / `BLOCKED`

| # | Step | 상태 | 마지막 업데이트 | 비고 |
|---|---|---|---|---|
| 0 | 논문 분석 + 파이프라인 정리 (`PIPELINE.md`) | DONE | 2026-04-25 | |
| 0.5 | 인수인계 문서 작성 (`PROJECT_HANDOFF.md`) | DONE | 2026-04-25 | 본 파일 |
| 1 | 환경 세팅 + 디렉터리 + configs + README | DONE | 2026-04-25 | requirements.txt, configs/{yolo_det,yolo_obb,donut_numerical}.yaml, .gitignore, **README.md** 생성 / WSL2 마운트 경로로 configs 갱신 |
| 1.5 | TitleBlock 기준 데이터 분류기 (`src/sort_by_titleblock.py`) | DONE | 2026-04-25 | PyTesseract + OpenCV. 입력 `dataset/`, 출력 `data/{stage1_titleblock,stage2_no_titleblock,manual_review}/` + `outputs/sort_titleblock_manifest.csv` |
| 1.6 | 가공/조립 분류기 (`src/sort_by_drawing_type.py`) | DONE | 2026-04-28 | D-026 휴리스틱 자동 분류. OCR (5개 언어) + Hough Circles (풍선) + 격자 검출 (BOM). 출력 `data/{manufacturing,assembly,manual_review_type}/` + manifest CSV. 더미 데이터로 BOM/풍선 검출 정상 확인. 실제 OCR 임계값은 `--dim-min` CLI 로 튜닝. |
| 2 | Stage 1 학습/추론 코드 (`src/stage1_layout.py`) | DONE | 2026-04-25 | YOLOv11-det. 서브커맨드 `train` / `predict` / `crop`. 공개 함수 `predict_one()`, `crop_regions()` 으로 pipeline.py 가 import. flip OFF 적용. best.pt → `checkpoints/yolo_det.pt` 자동 복사. **2026-04-28 갱신**: D-028 5클래스 + D-029 매핑 (`CLASS_NAMES_RF`, `ROBOFLOW_TO_INTERNAL`, `to_internal_class()`) 추가. |
| 2.A | **★ Stage 1 Version A 학습 완료** | DONE | 2026-04-28 | Roboflow seed 100장 (80/20) → yolo11m.pt 50 epochs / 28.5분 / RTX 5080 cu128. **mAP@0.5 = 0.9364** (논문 0.96 근접). V2-A 라벨 검증 PASS, V2-B 모델 검증 PMI/Text 만 살짝 미달 (seed 데이터 부족). 상세: `history.md` §A. |
| 5.5 | **`src/auto_label_stage1.py` (Active Learning)** | DONE | 2026-04-28 | 신규 (~393 lines). Stage 1 Version A 모델로 5,739장 자동 라벨링. predict_to_yolo_txt() + Roboflow Pre-annotation Import 호환 출력 (YOLO txt + 이미지 symlink/복사) + manifest CSV (priority 4-tier: empty/low_conf/review/auto_pass). 다음 단계 = 사용자 실행 (~5분). |
| 5.5.A | **★ Auto-labeling 실행 완료** | DONE | 2026-04-29 | **5,839장** (seed 별개 파일이라 미제외) 자동 라벨링 / **5분 6초** / 19.05 img/s. 총 293,894 박스 검출 (PMI 245,462 = 도면당 42, seed 35.7 와 일관). manifest 분포: empty 2 / low_conf 1,106 (18.9%) / review 4,731 / auto_pass 0 (HIGH_CONF_THRESHOLD 0.85 너무 엄격). 1차 후 0.65 로 조정 + 재실행 (5분 45초) → auto_pass 127 (2.2%) 정상. |
| Phase 2 | **Pre-annotation 스킵 결정 (D-035)** | DONE | 2026-04-29 | 라이선스 + 비용 + 3일 timeline 검토. **Roboflow 추가 업로드 안 함** (5,839장 로컬 only = 안전). Stage 1 Version A 그대로 사용. Stage 2 ~ Step 8 = 3일 plan (Day 1~3). |
| 1.6.A | **★ sort_by_yolo_pmi.py + exclude_groups.py 작성** | DONE | 2026-04-29 | D-026 휴리스틱 폐기 후 대체. PMI 카운트 기반 분류 (~390 + 295 lines). 실행 5,839장 / ~3분 / mfg 5,349 / asm 441 / review 49. WSL2 호환 정책 (검수 폴더 자동 copy). |
| 1.6.B | **manufacturing sample 검증** | DONE | 2026-04-29 | random 100장 시각 검증 → 조립 0% / 부품 10~20% / 가공 80~90%. 분류기 정확도 검증. 부품도면 학습 유지 결정 (가공과 분간 어려움). |
| 1.6.C | **사용자 검수 + group_key 추출 + exclude_groups 실행** | DONE | 2026-04-29 | assembly + manual_review 시각 검수 → 18 group_keys 식별 (자동 분류 후보의 80%+ false positive 제거) → exclude_groups.py 실행 (~9초) → 46 images + 46 labels 이동. **dataset/ 5,793 / excluded/ 46** / D-024 정합성 ✅. |
| **Day 1** | Stage 2 PMI crop 라벨링 (CVAT 로컬, **★ 844 crops**) | **DONE** | 2026-05-02 | **★ 라벨링 완료**: Stage2_PMI_v3_upscaled3x_844 (upscale 3x 적용). 전체 1026 박스 (Measure 555 / Roughness 106 / GDT 88 / SKIP 277). Frame-level SKIP 32.82% (>30% 임계). v1→v2→v3 padding 진화 (D-037). **D-038 박제** (Stage 1 fp Notes Rescue): SKIP 277 중 stage1_fp_notes 23개 → Day 2 Donut OCR 처리. extract_skip_list.py + rescue_misclassified_notes.py 작성 완료, rescue 실행은 transformers 설치 후 Day 2 진행. **history.md §A.11.7 통계 박제**. |
| **Day 2** | Phase 2 마무리 + Stage 2 학습 + Stage 3-A | **IN_PROGRESS** | 2026-05-03 | (1) ✅ uv pip install -r requirements.txt (5.94s, transformers 5.6.2) (2) ⚠ **Donut DocVQA Rescue 실질 실패** (4%) → ★ **D-039 박제: Stage 3-A → PaddleOCR-VL-1.5 채택** (0.9B, OmniDocBench 94.5%, CJK SOTA, 2026-01-29 출시). Stage 3-N → Donut Numerical 유지 + V6 검증 (★ 신규). (3) 사용자 샘플로 PaddleOCR-VL-1.5 zero-shot 사전 검증 예정 (Day 2 학습 백그라운드 시간). (4) Phase 7~14 진행: CVAT YOLO OBB export → SKIP 제거 → group-aware split → V3-A → augmentation 강화 → 학습 ~5h → V3-B → V5. |
| **Day 3** | Stage 3 통합 + Step 7~8 | **IN_PROGRESS** | 2026-05-04 | (1) ✅ **K-fold 학습 완료** (5 folds × 9.0h, mean mAP 0.932 ± 0.062, Best Fold = 2 mAP 0.978). (2) ✅ V3-B 단일 모델 검증: Measure missing 0.101 ❌ (D-023 FAIL). (3) ✅ **★ D-040 박제: 5-Fold Ensemble 채택** — `src/ensemble_predict.py` 작성 + NMS resolver fix (manual shapely fallback). (4) ✅ **D-023 PASS**: Measure/GDT/Roughness missing = 0.000, drawing_recall = 1.000. (5) ✅ pipeline.py 통합 (use_ensemble default ON, lazy load, model_versions 메타 갱신). (6) **다음**: PaddleOCR-VL-1.5 통합 (Stage 3-A) → Donut Numerical fine-tune ~6h → V6 (★ D-023 critical) → V7 → Step 8. |
| 3 | Stage 2 학습/추론 코드 (`src/stage2_annotation.py`) | DONE | 2026-04-25 | YOLOv11-obb. 서브커맨드 `train` / `predict` / `crop`. 공개 함수 `predict_one()`, `crop_obb_regions()` (perspective-warp). 회전 augmentation ON. |
| 4 | VLM pair 자동 생성기 (`src/prepare_vlm_dataset.py`) | DONE | 2026-04-27 | Stage 1+2 추론 → de-rotation crop → JSON 템플릿 자동 생성 + manifest. `all`/`alphabetical`/`numerical` 3 서브커맨드. `--ocr-prefill` 옵션 (Pytesseract). Group key 추출 8 케이스 PASS, 5종 템플릿 빌더 검증, 숫자 hint 추출 (Ø25.4 / Ra 1.6 / M8 등) 정상 |
| 5 | Stage 3-A 래퍼 (`src/stage3_alphabetical.py`) | DONE | 2026-04-26 | Donut zero-shot. DocVQA(다중 질문) / CORD-v2(단일 패스) 두 모드. 공개 함수 `load_model()`, `predict_one()`, `predict_titleblock()`, `predict_notes()`. CLI: `predict` / `batch`. FP16 기본, RTX 5080 16GB 호환. |
| 6 | Stage 3-N fine-tune (`src/stage3_numerical.py`) | DONE | 2026-04-27 | Donut Numerical fine-tune + 추론. JSON↔token 변환기 (`json_to_donut` / `donut_to_json`, 5 케이스 round-trip 검증 PASS), 그룹-aware split (D-024), HF Trainer 기반 학습 루프, FP16/BF16/8bit 옵션, `train`/`predict`/`batch` CLI |
| 7 | end-to-end pipeline (`src/pipeline.py`) | DONE | 2026-04-27 | Pipeline 클래스 + run/batch CLI. 모든 모델 1회 로드, lazy import (CLI --help 시 ultralytics/transformers 불필요). View OBB → 글로벌 좌표 자동 변환. Donut 미존재 시 Stage 3-N auto-skip. `--skip-numerical/--skip-alphabetical` 단계 분리 가능. timing/error 로그 + 매니페스트 생성. |
| 8 | 평가 스크립트 (`src/utils/metrics.py`) | DONE | 2026-04-27 | 평가 지표 라이브러리 (~600 lines, 10 섹션). pr_f1, set_pr_f1, edit_distance, fuzzy_match, numerical_match, field_level_f1, hallucination_rate, compare_measure/gdt/roughness, compare_titleblock/notes, bbox_iou, polygon_iou, detection_metrics. 15개 sanity test 모두 PASS. |
| 9 | Stage 9 enrichment (`src/stage5_enrichment.py`) | DONE | 2026-04-27 | 4-tier cascade (deterministic → heuristic → llm → hitl) + provider 추상화 (Mock/Gemini/Qwen). 5개 카테고리 (material/tolerance/roughness/process/qc). KB 인라인 + data/kb/*.json override. 더미 unified JSON 으로 5종 모두 정상 분류 (det 2 / heur 2 / llm 1 / hitl 1). |
| V0 | 검증 프레임워크 골격 (`src/validate/common.py` + `configs/validation_thresholds.yaml`) | DONE | 2026-04-27 | CheckResult / ValidationReport / Severity / load_thresholds / matplotlib plot helpers / Jinja2 HTML 템플릿 |
| V1 | `check_step1_5_sorter.py` (분류기 검증) | DONE | 2026-04-27 | manifest CSV 파싱 + GT join + per-language accuracy + confusion matrix plot. 콘솔/HTML/JSON 3종 출력 검증 완료 |
| V2 | `check_labels_yolo.py` + `check_stage1_model.py` | DONE | 2026-04-27 | YOLO det 라벨 품질 검증기 (빈 라벨/BBox 유효성/클래스 분포/aspect ratio) + Stage 1 모델 mAP@0.5/per-class accuracy/confusion matrix/FP rate. 더미 데이터로 더트 케이스 (empty/parse error/invalid bbox/class imbalance/extreme aspect) 모두 정상 검출 확인 |
| V3 | `check_labels_obb.py` + `check_stage2_model.py` | DONE | 2026-04-27 | OBB 라벨 품질 (8 항목: self-intersecting 검출 / 회전 OBB 비율 / Roughness 부족 알림) + Stage 2 모델 ★ 누락률(D-023) 측정 (per-image GT-Pred 매칭, shapely polygon IoU, drawing-level recall, FN/GT 비율) |
| V5 | `check_stage3a_alphabetical.py` | DONE | 2026-04-27 | Donut zero-shot 사후 검증. 10개 항목 (region별 F1 / hallucination / empty / edit dist / 언어별 gap). TitleBlock field-level breakdown + per-language plot. 5건 더미 데이터로 모든 케이스 (완벽매칭/부분매칭/fuzzy notes/빈응답/unmatched) 정확 검출. |
| V6 | `check_stage3n_numerical.py` | DONE | 2026-04-27 | Donut fine-tuned ★ D-023 측정. 16개 항목 (per-class F1 / numerical_accuracy / tolerance_match / symbol/datum/Ra accuracy / rare_symbol / hallucination critical / empty_rate). 7건 더미 (Measure 완벽/오답 + GDT 완벽/오답 + Roughness 완벽/incomplete-GT skip + Hallucination) 모두 정확 검출. |
| V7 | `check_pipeline_e2e.py` | DONE | 2026-04-28 | end-to-end pipeline 사후 검증. 13개 항목 (field_f1 / TB+Notes / per-class detection ★ D-023 재측정 / drawing-level recall / numerical content F1 + accuracy / per-stage timing / failure_rate / 최악 10건 표). 2건 더미 (완벽/부분매칭) 모두 정확 검출. metrics.py + summary JSON 활용. |
| V9 | `check_enrichment.py` | DONE | 2026-04-28 | Step 9 enrichment 사후 검증. 5개 항목 (provenance_completeness=1.0 critical / llm_method_rate <0.40 / hitl_flag_rate <0.25 / empty_suggestion_rate <0.10 / cost_per_drawing ≤$0.005). 3건 더미 (Mock+Gemini+Provenance누락) 정확 검출. material_recommendation_accuracy 옵션 측정 (expert GT 제공 시). |

### 10.1 데이터셋 진행 현황
| 데이터셋 | 수집 수량 | 라벨링 진행 | 검수 완료 |
|---|---|---|---|
| **`dataset/` (원본 + 증강 JPG)** | **5,839** | — | — |
| `IMMA.v1i.yolov11/{train,valid}/` (Roboflow export v1, Stage 1 seed, 80/20 split) | **100** | **100** ✅ (Roboflow) | 100% ✅ |
| `outputs/auto_labels/` (auto_label_stage1.py 산출 예정) | 5,739 | 0 (예정) | 0% |
| `outputs/cvat_stage2_input_v2/` (extract_pmi_crops.py, per-axis adaptive padding, D-037 v2) | **844** | 0 (예정) | 0% |
| `outputs/cvat_stage2_input_v3/` (extract_pmi_crops_v3.py, aspect-aware padding, D-037 v3) | **844** | **라벨링 IN_PROGRESS** | — |
| `data/stage1_titleblock/` (sort 후) | TBD | — | — |
| `data/stage2_no_titleblock/` (sort 후) | TBD | — | — |
| `data/manual_review/` (sort 후) | TBD | — | — |
| `data/layout/` (Stage 1 라벨링) | TBD | 0% | 0% |
| `data/annotation/` (Stage 2 라벨링) | TBD | 0% | 0% |
| `data/vlm/alphabetical/` (Step 4 후) | 0 | — | — |
| `data/vlm/numerical/` (Step 4 후) | 0 | — | — |

**데이터셋 메모 (2026-04-28 갱신)**
- **5,839 JPG**가 `dataset/` 에 적재됨 (2026-04-27: 4,587 → 2026-04-28: 5,839, +1,252).
  일부 이미지는 **사용자 사전 증강** (뒤집기 + 회전) 적용된 상태
- 원본 + 증강이 같은 풀에 섞여 있으므로 **train/val split 시 데이터 누수 주의** (D-024 group-aware split)
- **5개 언어** (EN / KO / JP / RU / **CN**) 혼합. 도면 1장 = 단일 언어 가정 (D-010 / D-013 / D-025)
- 가공도면 + 조립도면 혼재 (D-026) — `src/sort_by_drawing_type.py` 로 자동 분류
- 약 **95% 도면에 TB 존재** (D-027) — TB 유무 분류 가치는 낮음. 분류 없이 Stage 1 라벨링 (D-019)
- 논문 비례: 5,839장 = Stage 1 논문(1,000)의 5.8× / Stage 2 논문(1,406)의 4.2× → 양 충분
- **Roboflow seed 100장 라벨링 완료** (`IMMA.v1i.yolov11/`, 2026-04-28). 80/20 split, group leak 0.
- **★ Stage 1 Version A 학습 완료** (2026-04-28, mAP@0.5 = 0.9364, 28.5분, RTX 5080 cu128). 상세: [`history.md`](./history.md) §A.
- **★ auto_label_stage1.py 작성 완료** (2026-04-28). 5,739장 자동 라벨링 + Active Learning priority 매니페스트. 다음 단계 = 실행.

---

## 11. 의사결정 로그 (DO NOT REOPEN)

확정된 결정. 새 세션이 다시 논의하지 말 것.

1. **D-001** 아키텍처는 논문(YOLOv11-det + YOLOv11-obb + Donut Alphabetical/Numerical) 그대로 재현. 모델 교체 금지.
2. **D-002** 데이터셋은 자체 수집·라벨링. 논문 데이터셋 사용/요청 금지.
3. **D-003** 입력은 JPG only. PDF / PNG 변환 모듈 추가 금지.
4. **D-004** Alphabetical VLM은 **zero-shot** 운용. (Title/Notes는 schema 변동성이 커서 fine-tune 안 함) — 단, 후속 confirm 시 변경 가능.
5. **D-005** Numerical VLM의 학습 하이퍼파라미터는 논문 그대로(epoch 30 / AdamW / cosine 1e-6 / batch 4 / FP16). RTX 5080(16GB) 대응으로 **gradient checkpointing 추가**만 허용.
6. **D-006** YOLO는 `yolo11m` 기준. 정확도/속도 trade-off가 안 맞을 때만 사용자 confirm 후 `yolo11s` 또는 `yolo11l`로 변경.
7. **D-007** 라벨링 도구는 CVAT 권장(특히 OBB).
8. **D-008** 워크스페이스 루트는 `C:\Users\user\github\Drawing`. 다른 경로에 산출물 생성 금지.
9. **D-009** Stage 1·Stage 2 YOLO 모델은 **언어별로 분리하지 않고 단일 모델**로 학습. YOLO는 시각 패턴을 학습하므로 언어 분리 시 데이터만 1/4로 줄어 성능 저하. 언어별 처리는 Stage 3 (Donut VLM) 에서만 적용.
10. **D-010** 자체 도면은 한 장당 단일 언어 가정. 언어 분류 작업은 차후 JSON 메타데이터 추출 이후 별도 단계로 수행. (**2026-04-28 갱신**: D-025 에 의해 5개 언어로 확장 — KO / EN / JP / RU / **CN**)
11. **D-011** TitleBlock 분류기는 PyTesseract `lang='kor+eng+rus+jpn'` 단일 호출 + 4개 언어 키워드 사전 + 하단 라인 밀도 보조 신호. 결과는 `data/{stage1_titleblock, stage2_no_titleblock, manual_review}/` 로 **이동(move)**, manifest CSV 생성.
12. **D-012** Stage 2 OBB crop은 **perspective-warp 으로 de-rotation 적용**. Donut VLM은 upright text 기반이라 회전 패치가 그대로 들어가면 Stage 3-N 성능 저하. `cv2.getPerspectiveTransform` + `cv2.warpPerspective` 사용.
13. **D-013 (사용자 재확인 2026-04-25)** 도면 언어는 EN / KO / JP / RU 네 가지로 시작했음. Stage 1·2는 D-009에 따라 언어 무관 단일 모델. 언어 메타데이터는 Stage 3 결과 JSON에 기록되며, Stage 1·2 코드는 다국어 파일명·경로를 안전하게 처리한다 (`np.fromfile` + `cv2.imdecode/imencode`, `utf-8` JSON). (**2026-04-28 갱신**: D-025 에 의해 **CN(중국어, 簡/繁體) 추가, 총 5개 언어**.)
14. **D-014** 작업 환경을 **Ubuntu 22.04 on WSL2 + Antigravity (VS Code 기반)** 로 확정. Windows 측 `C:\Users\user\github\Drawing` 폴더를 WSL2에서 `/mnt/c/Users/user/github/Drawing` 으로 마운트해 동일 워크스페이스를 양쪽에서 접근. configs/*.yaml 의 `path:` 는 WSL2 마운트 경로 기본값. Tesseract는 apt 패키지 (`tesseract-ocr-{eng,kor,jpn,rus}`) 사용, `/usr/bin/tesseract` 가 PATH에 자동 등록되므로 코드의 Windows fallback 분기는 그대로 유지 (Windows 직접 사용도 호환).
15. **D-015** **FCF 컴파트먼트 분리는 기본 미수행.** Stage 2 OBB → de-rotation 패치 → Stage 3-N Donut 통패치 입력. Stage 3-N fine-tune 후 GD&T F1 < 0.85 일 때만 `src/utils/fcf_split.py` 추가 검토 (OpenCV 세로 선 검출 → 컴파트먼트 분할 → 1번 컴파트먼트 14-class 분류기).
16. **D-016** **Dimensions 처리는 Stage 2 YOLOv11-obb (Measure 클래스) + Stage 3-N Donut Numerical VLM** 으로 통합. eDOCr2 (Pinquié 2025) 의 CRAFT + 패치 분할 + 20 px 클러스터링 + Pytesseract 필터 + 합성 CRNN + ∅ 템플릿 매칭 다단계 파이프라인은 **차용하지 않음**. Stage 2 OBB가 CRAFT+클러스터링+각도 보정을, Donut이 CRNN+공차 subdivision+심볼 인식을 모두 흡수.
17. **D-017** 단, 다음 백업 모듈은 **Step 7 (end-to-end 평가) 결과에 따라 조건부 추가** 검토:
    - `src/utils/symbol_postcorrect.py` — Measure F1 < 0.90 + ∅ 누락이 주원인일 때, 템플릿 매칭 후처리
    - `src/utils/synthetic_gen.py` — Roughness 또는 희귀 GD&T 심볼 F1 < 0.80 일 때, 합성 데이터 보강
    - Pytesseract pre-filter — 도면당 추론 속도 > 30s 일 때만
18. **D-018** **Stage 3 모델은 Donut 유지** (D-001 재확인). Qwen2.5-VL / Gemini / DeepSeek V4 검토는 Step 7 평가 후 Donut 성능이 논문 수치(Numerical F1 0.96, Alphabetical F1 0.67) 대비 현저히 미달일 경우에만 재논의.
19. **D-019** `sort_by_titleblock.py` 의 역할은 **선택 분석 도구** (필수 아님). 데이터 품질 점검 / stratified split 보조 / Stage 1 false negative 디버깅 용도. 기본 학습 흐름에선 dataset/ 의 모든 JPG 를 그대로 라벨링 후 Stage 1 학습. 두 유형(TB 있음/없음)이 섞여 있어야 모델이 더 강건.
20. **D-020 (검증 의무화)** 각 step 완료 시 해당 `src/validate/check_*.py` 를 실행하고 산출 리포트(`reports/<date>_<check>.html` + `.json`)를 보관. 사람이 OK 판정한 단계만 다음 step 진행 허용.
21. **D-021 (임계값 정책)** `configs/validation_thresholds.yaml` 의 severity 분류:
    - `critical` 미달 → 다음 step **차단**
    - `warning` 미달 → 경고 출력, 사용자 판단으로 진행
    - `info` → 추세 모니터링용
22. **D-022 (리포트 형식)** 검증 출력은 콘솔 + JSON + HTML 3종 동시 생성. JSON 은 시계열 추적 (성능 회귀 감지), HTML 은 시각 검수 (이미지 그리드 + 차트), 콘솔은 즉시 PASS/FAIL.
23. **D-023 (사용자 필수 임계값)**
    - Stage 2 어노테이션 누락률: Measure < 8%, GDT < 5% (critical) / Roughness < 30% (warning)
    - Stage 2 도면 단위 회수율: ≥ 0.85 (critical)
    - Stage 3-A Notes F1: ≥ 0.75 (critical, 논문 0.810)
    - Stage 3-A Hallucination: < 0.50 (warning)
    - Stage 3-N Measure F1: ≥ 0.90 (critical, 논문 0.923)
    - Stage 3-N Hallucination: < 0.10 (critical)
24. **D-024 (사전 증강 데이터셋 정책)** `dataset/` 에 적재된 **5,839장** (2026-04-28 갱신) 은 원본 + 사용자가 사전 증강(뒤집기·회전)한 이미지가 혼재. 이 정책상:
    - **파일명 형식: Roboflow export 패턴** — `{original_stem}.rf.{augmentation_hash}.jpg`
      예) `11_jpeg.rf.8b46c563....jpg`, `11_jpeg.rf.de99e140....jpg` 두 파일은 **같은 원본** `11_jpeg` 의 두 증강 변형
    - **Group key 추출 규칙**: `filename.split('.rf.')[0]` → 그 값이 같으면 같은 원본 그룹
    - **train/val split 시 group-aware split 필수**. 같은 group key 의 모든 변형이 train 또는 val 한쪽에 모두 들어가야 함 (data leakage 방지)
    - 권장 구현 (Step 4 / Step 6 학습 직전):
      ```python
      from sklearn.model_selection import GroupShuffleSplit
      groups = [fn.split('.rf.')[0] for fn in filenames]
      splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
      train_idx, val_idx = next(splitter.split(filenames, groups=groups))
      ```
    - Stage 1 augmentation 정책(D-001 / yolo_det.yaml: `fliplr=0.0, flipud=0.0`)은 **그대로 유지**. 이유: 사전 증강된 flip 이미지는 별개 학습 샘플로 들어가고, on-the-fly augmentation 추가 flip은 중복·과적합 위험.
    - Stage 2 회전 augmentation(yolo_obb.yaml: `degrees=15.0`)도 그대로 유지. 사전 회전과 결합해 다양성 ↑.
    - sort_by_titleblock.py manifest CSV 에 group key 컬럼 추가 권장 (후속 작업 시).
25. **D-025 (5개 언어 확장, 2026-04-28)** 라벨링 시작 시점에 **Chinese (中文) 추가** 발견. 도면 언어는 이제 **EN / KO / JP / RU / CN (簡/繁體 모두)** 5가지.
    - **Tesseract**: `lang='kor+eng+rus+jpn+chi_sim+chi_tra'` 단일 호출
    - 시스템 패키지 추가: `tesseract-ocr-chi-sim`, `tesseract-ocr-chi-tra`
    - `sort_by_titleblock.py` 키워드 사전에 中文 추가 필요 (圖紙/標題/比例/材質/張數/設計/圖號 등 — 簡/繁 모두)
    - `data/kb/material_catalog.json` 에 GB (Guobiao) 표준 grade 추가 필요 (Q235 / 45# / 0Cr18Ni9 등)
    - `validation_thresholds.yaml#step1_5.per_language_min` 에 CN 추가
    - Stage 1·2 YOLO 는 D-009에 따라 언어 무관 — 영향 없음
    - Stage 3-A Donut 은 영어 위주 사전학습 → CN 정확도 상대적으로 낮을 것 → Step 9 enrichment 의존도 ↑
    - 코드 수정 필요 항목은 차후 별도 작업 (이 D-025 박제 시점에서는 미실행)
26. **D-026 (가공/조립 도면 분류 필요, 2026-04-28)** Roboflow 데이터셋에 **가공도면 + 조립도면** 혼재 발견. 조립도면은 치수 표기가 거의 없어 Stage 2 (Measure/GDT/Roughness OBB) 학습 데이터로 무용.
    - **영향 범위**:
      - Stage 1 (View/TB/Notes 영역 검출): 가공/조립 모두 사용 가능
      - **Stage 2 (OBB)**: **가공도면만** 학습 데이터로 사용
      - Stage 3-A (TitleBlock VLM): 가공/조립 모두 사용
      - Stage 3-N (Numerical VLM): **가공도면만**
    - **분류 휴리스틱 (가공도면 시그널)**: 치수 표기 (Ø/R/M/±) 다수, 표면거칠기 심볼, GD&T 프레임, View 1~3개
    - **분류 휴리스틱 (조립도면 시그널)**: 부품번호 풍선 (작은 원 + 숫자), BOM 표 (큰 격자), 다수 부품, 치수 거의 없음
    - **권장 자산** (차후 작성):
      - `src/sort_by_drawing_type.py` — OpenCV (Hough Circles + 격자 검출) + Pytesseract (치수 빈도) 휴리스틱 자동 분류
      - 출력: `data/manufacturing/`, `data/assembly/`, `data/manual_review_type/`
    - **빠른 우회**: Roboflow Tag 기능으로 100장 seed 에 `manufacturing` / `assembly` 수동 분류 (10초/장 × 100 = 17분)
    - **라벨링 룰**: Stage 2 라벨링은 가공도면의 View crop 만. 조립도면 View crop 은 Stage 1 학습 데이터로만 사용 (또는 제외).
    - **🔴 갱신 2026-04-29 (휴리스틱 분류기 폐기)**: 5,839장 실측 결과 비현실적 분류 (mfg=0 / asm=5,313 / review=526). **분류기 폐기**.
      - **원인**:
        1. **OCR 치수 검출 실패** — Tesseract 5개 언어 OCR 가 도면당 평균 0~1개 치수만 검출 (실제 30+ 개 치수). 도면 해상도 (Roboflow Resize 1280) + 작은 글자 인식률 한계.
        2. **BOM 검출 false positive** — Hough Lines 격자 검출이 일반 표 (TitleBlock 등) 도 BOM 으로 오인. 91% 도면이 BOM=True.
        3. **Hough Circles balloon 정확** — 풍선 0개 = 가공도면 시그널은 맞음 (단, manufacturing 룰 dim≥5 미달로 무용).
      - **격리 범위**: OCR/BOM 실패는 `sort_by_drawing_type.py` 와 `sort_by_titleblock.py` (Step 1.5/1.6) 한정. **Stage 2 이후 모든 단계는 OCR 미사용**:
        - Stage 1/2: YOLO (pure CNN, 시각 패턴) — 무영향
        - Stage 3-A/N: Donut VLM (OCR-free transformer) — 무영향
        - Step 9 enrichment: Stage 3 JSON 만 사용 — 무영향
      - **대체**: Stage 1 Version A (mAP 0.9364) 의 **PMI 박스 카운트 기반 가공/조립 분류** 가능. PMI ≥ 5 → manufacturing / 0~1 + balloon → assembly. 차후 별도 스크립트 (필요 시).
      - **D-027 (TB 95:5) 재검토**: D-027 도 sort_by_titleblock OCR 의존 — 정확성 의심. 다만 D-032 시각 검증으로 **모든 도면에 표 (Table 클래스) 100% 존재** 확인됨 → D-027 가정 (95:5) 의 실측은 100:0 으로 정정 (모든 도면에 어떤 형태든 표 있음).
    - **🟢 갱신 2026-04-29 #2 (sort_by_yolo_pmi.py 작성 + 실행)**:
      - **신규 모듈**: `src/sort_by_yolo_pmi.py` (~390 lines) + `src/exclude_groups.py` (~295 lines)
      - **분류 규칙**:
        - PMI ≥ 5 → manufacturing
        - PMI < 5 + (Iso ≥ 1 OR Table ≥ 3) → assembly (★ 검수)
        - PMI < 5 + signal 없음 → manual_review_type (★ 검수)
      - **실행 결과 (5,839장 / ~3분)**: mfg 5,349 (91.6%) / asm 441 (7.6%) / review 49 (0.8%)
      - **WSL2 호환성**: 기본 정책 = manufacturing symlink + assembly/manual_review copy (Windows Explorer 호환)
      - **사용자 검수 워크플로**:
        1. 검수 폴더 (assembly + manual_review) 시각 확인
        2. 조립도면 / 학습부적합 → 폴더에 남김 / 가공·부품도면 → 폴더에서 삭제
        3. Python 한 줄로 group_key 자동 추출 → outputs/exclude_list.txt
        4. `exclude_groups.py` 실행 → dataset/ 에서 D-024 group 단위 일괄 이동
    - **🟢 갱신 2026-04-29 #3 (manufacturing sample 검증)**:
      - **5,349장 중 random 100장** 시각 검증
      - **조립도면 0%** ✅ — 분류기 정확도 검증 (false positive 없음)
      - **부품도면 10~20%** 🟡 — 가공도면과 분간 어려움
      - **부품 vs 가공도면 식별 기준** (참고용):
        - 1개 파트 단일 → 가공도면 확정
        - 여러 파트 + 치수만 (BOM 없음) → 부품도면
        - 평탄도 / 공차 / GD&T 표기 → 가공도면 확정 (단, 박스 라벨 누락 시 식별 곤란)
      - **결정**: 부품도면도 학습 유지 (분간 어려움 + 학습 영향 미미). D-035 ("학습 데이터에 부품도면 섞여도 OK") 와 일치.
      - **차후**: Stage 2 라벨링 시점에 PMI crop 단위에서 가공/부품 차이 자연 흡수 (PMI = 가공정보 영역, 부품도면도 PMI 일부 보유).
27. **D-027 (TB 핵심 필드 누락 정책, 2026-04-28)** 가공도면 TB 가 있어도 material / quantity 같은 핵심 필드가 자주 비어있음 (사용자 라벨링 시 발견).
    - **Stage 1 (TB 영역 검출)**: 영향 없음. 박스만 검출.
    - **Stage 3-A (DocVQA)**: 빈 필드를 Donut 이 hallucinate 할 위험 → V5 의 hallucination_rate 임계값 (0.50) 미달 가능. **임계값 조정 검토**: 0.50 → 0.55.
    - **Step 9 Enrichment**: **이 시나리오가 enrichment 의 존재 이유**. 4-tier cascade 가 빈 필드 보강.
    - **V9 임계값 재조정 검토**: `llm_method_rate_max` 0.40 → **0.55** (빈 필드 비율 높으면 LLM 호출 빈도 ↑, 비용 통제 vs 정확도 trade-off)
    - `prepare_vlm_dataset.py alphabetical` 의 GT JSON 작성 시 빈 필드는 명시적 `null` (사람이 "정보 없음"으로 검수 — 모델이 이를 학습)
    - 코드 수정은 차후 별도 작업 (이 D-027 박제 시점에서는 미실행)
28. **D-028 (Stage 1 클래스 재정의 5개, 2026-04-28)** 데이터셋 업로드자가 정한 **Roboflow data.yaml** 의 클래스 체계를 채택:
    - **이전**: `View / TitleBlock / Notes` (3 클래스)
    - **이후**: `Isometric / PMI / Table / Text / View` (5 클래스, `nc=5`)
    - 추가 의미:
      - **`Isometric`** — 3D 등각투영도. Stage 1 만 검출, Stage 2 OBB skip (치수 표기 거의 없음).
      - **`PMI`** (Product Manufacturing Information) — 가공정보 영역. **Stage 2 OBB 의 입력 영역** 으로 사용 (View 안에 PMI, PMI 안에 Measure/GDT/Roughness). 계층 구조 명확화.
    - **`Table` ≡ `TitleBlock`** (사용자 확인 2026-04-28). 데이터셋 업로드자가 표제란을 일반화된 "Table" 로 라벨링.
    - **`Text` ≡ `Notes`** (의미 동일). 자유 텍스트 어노테이션.
    - 영향:
      - `configs/yolo_det.yaml` — 5 클래스로 확장
      - `src/stage1_layout.py` — `CLASS_NAMES_RF` (Roboflow 5종) + `CLASS_NAMES` (내부 5종)
      - `configs/validation_thresholds.yaml` — Isometric / PMI per-class threshold 추가
    - **Stage 2 OBB 클래스는 변경 없음**: `Measure / GDT / Roughness` 그대로 유지 (사용자 확인 2026-04-28). 근거: Khan 2025 논문 동일 클래스 / ISO/ASME PMI 표준.
29. **D-029 (Roboflow→내부 클래스 매핑, 2026-04-28)** Stage 1 학습 모델은 Roboflow data.yaml 이름을 출력하나, 다운스트림 모듈 (`pipeline.py`, `prepare_vlm_dataset.py`, `stage3_alphabetical.py`, `stage5_enrichment.py`) 은 의미 보존을 위해 **내부 정규명** 을 사용한다.
    - **매핑** (`src/stage1_layout.ROBOFLOW_TO_INTERNAL`):
      ```
      Isometric → Isometric  (그대로)
      PMI       → PMI        (그대로)
      Table     → TitleBlock (의미 보존, Stage 3-A 토큰 호환)
      Text      → Notes      (의미 보존, Stage 3-A 토큰 호환)
      View      → View       (그대로)
      ```
    - **적용 지점**: `_result_to_schema()` 한 곳. 이후 모든 JSON `regions[*].class` 는 내부명.
    - **근거**:
      - "TitleBlock" 의 의미가 "Table" 보다 명확 (Table 은 BOM/revision 등 모호함)
      - Stage 3-A Donut 의 prompt 토큰 (`<s_titleblock>`, `<s_notes>`) 이 이미 학습된 토큰
      - 다운스트림 코드 (~7 모듈, ~20 라인) 일괄 변경 회피 → 매핑 1지점만 추가
    - **하위 호환성**: 기존 학습 모델 / 라벨 / JSON 모두 그대로 사용 가능.
30. **D-030 (RTX 5080 Blackwell PyTorch 빌드 정책, 2026-04-28)** RTX 50 시리즈는 Blackwell 아키텍처 (`compute_capability == (12, 0)`, `sm_120`).
    - **PyTorch cu124 빌드 비호환**: cu124 stable 휠은 `TORCH_CUDA_ARCH_LIST` 에 `sm_50/60/70/75/80/86/90` 만 포함 (Hopper H100 까지). RTX 5080 호출 시 `"sm_120 is not compatible"` UserWarning 발생, GPU kernel launch 실패 가능.
    - **cu128 빌드 사용 필수**:
      ```bash
      uv pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu128
      # Stable 미출시 시: --pre + nightly/cu128
      ```
    - **검증 명령** (CI / onboarding 필수 단계):
      ```python
      import torch
      assert torch.cuda.get_device_capability() == (12, 0), "Need cu128 build for sm_120"
      ```
    - **영향 받는 의존성**: `bitsandbytes`, `flash-attn` 도 모두 cu128 호환 빌드 필요.
    - **대안 미사용 사유**: cu126 빌드도 `sm_90` 까지만 지원 (확인 2026-04-28). cu128 이 Blackwell 의 first-class 지원.
    - **문서 갱신 위치**: `requirements.txt` 헤더, `MANUAL.md §1.4`, `README.md §2`, `docs/modules/*.md` 의존성 섹션.
    - **사용자 검증 결과 (2026-04-28)**: cu128 stable 재설치 후 `compute_capability=(12,0)` / `CUDA Version=12.8` 확인 ✓ — 호환성 경고 사라짐.
31. **D-031 (Stage 1 클래스 분포 임계값 재조정, 2026-04-28)** D-028 5클래스 체계의 **실측 분포** 기반 임계값 갱신.
    - **100장 seed 실측 (`IMMA.v1i.yolov11/train/labels` 3544 bboxes / 80 도면)**:

      | 클래스 | 비율 | 도면당 평균 | 비고 |
      |---|---|---|---|
      | PMI       | **80.59%** | 35.7 | ★ Dominant — 가공정보 영역이 가장 많음 |
      | View      | 9.65% | 4.3 | 도면당 여러 뷰 |
      | TitleBlock | 7.05% | 3.1 | TB + BOM + Revision 등 표 통합 (Roboflow `Table`) |
      | Notes     | 1.89% | 0.84 | |
      | Isometric | 0.82% | 0.36 | 도면당 0~1개 |

    - **임계값 갱신** (`configs/validation_thresholds.yaml::stage1_labels.class_distribution`):
      - 이전 (D-028 가정): View 0.30 / PMI 0.15 / TB 0.10 / Notes 0.05 / Iso 0.02
      - 신규 (실측 기반): **PMI 0.50 / View 0.05 / TB 0.04 / Notes 0.01 / Iso 0.005**
    - **근거**:
      - D-028 이전 3클래스 (View / TitleBlock / Notes) 에서는 View 가 dominant 였으나, 5클래스 추가 시 PMI 가 dominant 가 됨 (가공정보 영역이 도면당 평균 35.7개)
      - Roboflow data.yaml 의 `Table` 은 표제란 외에 BOM/Revision/부품번호 표 등 모든 표 포함 → 도면당 3.1개
    - **`extreme_aspect_ratio_count` WARN 정상 사유**: 65건 모두 Table 클래스 — BOM 표 (가로로 긴) / Revision 표 (세로로 긴) 의 **도메인 정상 특성**. 임계값 ≤ 0 은 너무 엄격하나 validator 코드에 하드코딩되어 있어 WARN 무시 가능 (차후 `extreme_aspect_ratio_max_ratio` 0.05 로 yaml 화 검토).
    - **5,839장 본격 작업 시 재측정 필요**: 100장 seed 와 분포가 크게 다르면 임계값 재조정.
    - **D-029 매핑 재검토**: `Table → TitleBlock` 매핑은 의미상 부정확 가능성 (Roboflow `Table` 은 모든 표 포함). 차후 `Table → Tables` (복수형) 또는 코드 내부에서 `Table` 그대로 유지 검토.
32. **D-032 (Roboflow `Table` 클래스 의미 확정, 2026-04-28)** 100장 seed 의 **사용자 직접 시각 검증** 결과 확정:
    - **Roboflow `Table` = 표 전체 통합 클래스** (단일 표제란이 아님):
      - **TitleBlock** (표제란) — 도번/제목/재질/척도/날짜
      - **BOM / Parts List** — 부품 목록 표
      - **Revision Table** — 개정 이력 표
      - **Notes Table** — 일반 노트 / 주서 표
    - **분포 통계** (80장 train):
      - 0개: 0% / 1개: 11.2% / 2개: 20% / 3개: 13.8% / **4개: 55%** ★ 최빈값
      - 평균 3.12 / 도면, 중앙값 4
    - **D-027 (TB 95:5 분포) 와의 관계**: D-027 의 "95% 도면에 TB 존재" 는 sort_by_titleblock 의 OCR 키워드 검출 기준 — 실제 라벨링은 모든 표 100% 커버.
    - **D-029 매핑 (Table → TitleBlock) 정책 결정**:
      - 코드 내부 매핑 **유지** (Stage 3-A Donut 토큰 `<s_titleblock>` 호환성)
      - 의미는 "**모든 표 콘텐츠 추출 일반화**" 로 해석 (표제란 한정 X)
    - **Stage 3-A 처리 정책** (Donut zero-shot):
      - 각 Table crop 에 대해 키워드 기반 자체 분류 (Donut 자체 능력 활용)
      - 출력 JSON 의 `table_type` 필드 추가 가능 (옵션, 차후 코드 수정 시)
      - `<s_titleblock>` 토큰을 표 콘텐츠 추출 일반 의도로 사용
    - **Stage 9 enrichment 영향 (D-027 정책 보강)**:
      - material/quantity 필드 추출은 `table_type == "TitleBlock"` 인 경우만
      - BOM 표에서 material 추출 시도하면 노이즈 (예: BOM 의 부품별 재질이 표제란의 도면 재질과 다름)
    - **5,839장 본격 작업 시 재측정 필요**: seed 분포 (4개/도면 최빈) 가 일관되는지 확인.
    - **러시아어 도면 해상도 이슈**: 사용자 발견 — Table 내 텍스트가 저해상도. Stage 3-A 시점에 영향 (D-033 후보, 미박제).
33. **D-033** (placeholder, 2026-04-28 미박제) — 도면 해상도별 처리 정책 (러시아어 저해상도 발견). Stage 3-A 진입 직전 박제 예정.
34. **D-034 (PMI 처리 계층 정책, 2026-04-28)** PMI 영역의 검출/처리 계층을 **2단계 hierarchical** 로 확정.
    - **계층 구조**:
      ```
      Stage 1 (axis-aligned):  PMI 영역 박스 (가공정보 영역 = "거친 후보")
                  ↓
      Stage 1 crop:            outputs/crops/<id>/PMI/*.jpg
                  ↓
      Stage 2 (OBB):           Measure / GDT / Roughness 의 정확한 회전 박스
                  ↓
      Stage 3-N:               각 OBB crop → 수치/심볼 schema JSON
      ```
    - **사용자 시각 검증 (100장 seed, 2026-04-28) 발견**:
      - 일부 도면에서 PMI 영역 자체가 회전 (회전 단면도 옆 치수군 등)
      - axis-aligned PMI 박스가 회전 영역을 덜 정확하게 cover
      - 다만 Stage 2 OBB 가 후속 단계에서 정확도 보정 가능 → 큰 문제 아님
    - **옵션 비교 및 결정**:
      | 옵션 | 변경 | 영향 | 결정 |
      |---|---|---|---|
      | A (★ 채택) | 현재 유지 — Stage 1 PMI = axis-aligned | 100장 seed 그대로 사용 | ✅ |
      | B | PMI 도 OBB | Roboflow 재라벨링 필요 (현실 X) | ❌ |
      | C | PMI 클래스 제거 + Stage 1 View → Stage 2 OBB 직접 | Roboflow 라벨 재정의, 4클래스 | 🟡 보류 |
    - **옵션 C 재검토 트리거 조건** (5,839장 본격 작업 시):
      - PMI bbox 의 회전 케이스 비율 > 20% 이면 옵션 C 검토
      - PMI 검출 mAP < 0.80 이면 옵션 C 검토
      - 그 외에는 옵션 A 유지
    - **누락 라벨 처리**: 100장 seed 의 일부 PMI 미라벨 — 5,839장 본격 라벨링 시 보강. Active Learning 검수 단계 (Step 7) 에서 catch.
    - **Roughness 라벨 누락 (사용자 발견)**: 100장 seed 에 표면거칠기 OBB 라벨 없음. Roughness 는 Stage 2 클래스이므로 정상. Stage 2 라벨링 단계에서 추가 — D-035 후보.
35. **D-035 (Pre-annotation 스킵 + Version A 활용 정책, 2026-04-29)**
    - **결정**: Active Learning Phase 2 (Roboflow Pre-annotation Import + 사람 검수 + Version B 학습) **보류**. Stage 1 **Version A (mAP 0.9364) 그대로 사용** → Stage 2 ~ Step 8 진행.
    - **사유** (라이선스 무관 — 비용 + 시간):
      1. **Roboflow Workspace Private 은 유료** ($50~$249/월) — 비용 부담
      2. **5,839장 검수 ~29시간** (3~4일) — 3일 timeline 초과
      3. **Version A 일반화 검증됨** — auto_label_stage1.py 결과의 클래스 분포 (PMI 84%, View 4.1/도면 등) 가 seed 와 매우 일관 (D-031)
    - **라이선스 평가** (병행 검토):
      - `dataset/` (5,839장) = **로컬 only** → 재배포 X → **개인 학습용 안전**
      - `IMMA.v1i.yolov11/` (Roboflow seed 100장) = 추가 업로드 안 함 → 추가 위험 X
      - 학습 결과 모델 weights = 로컬 보관 → 안전
    - **3일 plan** (Stage 2 ~ Step 8):
      - **Day 1**: Stage 1 PMI crop 추출 (20장 → 844 PMI) → CVAT 로컬 설치 → Stage 2 PMI crop 라벨링 **500 crops** (4~7h, ★ 2026-04-29 갱신) → V3-A 검증
      - **Day 2**: Stage 2 학습 (yolo11m-obb, ~5h) → V3-B → Stage 3-A zero-shot → V5
      - **Day 3**: Stage 3-N fine-tune (Donut Numerical, 6h) → V6 → Step 7 pipeline batch → V7 → Step 8 metrics 통합 검증
    - **Trade-off 명시**:
      - Stage 1 Version A 만 사용 (mAP 0.93 그대로, Version B 미달성)
      - Stage 2 seed **500 crops** (Active Learning 1단계, mAP 0.78~0.85 예상, ★ 2026-04-29 갱신: 200 → 500 — D-023 critical 통과 안정)
      - Stage 3-N 학습 데이터 부족 가능 (F1 lower 가능)
      - Step 7 pipeline 검증 = 부분 샘플 (100장)
    - **절대 양보 안 되는 항목**:
      - Group-aware split (D-024)
      - Class scheme (D-028 / D-029)
      - Validation framework (V0~V9 매 step 후 실행)
      - Provenance (D-022)
    - **차후 검토** (Step 8 완료 후):
      - 라이선스 + 비용 정리 후 Pre-annotation Phase 2 재개 → Version B 학습
      - Step 9 enrichment (Mock provider 동작 검증, Gemini 비용 후 검토)
36. **D-036 (Version A 회전 증강 변형 라벨 노이즈 — 옵션 B 채택, 2026-04-29)**
    - **발견**: low_conf priority 1,099 도면 시각 검수 (`visualize_labels.py --priority low_conf` + Explorer) 결과, **대부분이 회전 증강 변형**이며 다음 박스 오류 패턴:
      - View 박스가 도면 외곽 미덮음 (회전 외곽 인식 실패)
      - PMI 박스 중복 (NMS 가 회전 좌표에서 약함)
      - PMI 박스 누락 (회전된 작은 글자 인식 실패)
      - 박스 위치 어긋남 (확대 상세도)
    - **근본 원인**:
      - Stage 1 Version A = seed 100장 + flip OFF + rotation 0° (D-001) 학습
      - Roboflow dataset 의 사전 증강 = 회전 변형 다수 (~1.94×/group, ~50% 가 회전)
      - → Version A 가 회전 변형에서 검출 일반화 약함
    - **옵션 검토 + 결정**:
      - 옵션 A: variant-level 제외 (`exclude_files.py`) — 5,793→4,694 (-19%). ❌ 시간 부담 + D-024 정합성 손상
      - **옵션 B (★ 채택)**: 학습 데이터 그대로 + Stage 2 입력은 `auto_pass + review` priority 만 사용 (4,731장 candidate)
      - 옵션 C: Roboflow 재 export (rotation OFF) — ~50% 데이터 감소. ❌ 시간 부담
    - **옵션 B 채택 사유**:
      - 3일 plan 우선 (추가 작업 없음, Day 1 즉시 시작)
      - Stage 2 라벨링 입력은 auto_pass+review 만 사용 → 좋은 도면 4,731장에서 sample
      - Version A 의 의도된 한계 (D-001 flip OFF 정책) 의 자연 결과
      - Version B 학습 시 rotation augmentation 추가 → 자체 보강
    - **★ 차후 복기 트리거** — 다음 조건 만족 시 옵션 A 또는 C 재검토:
      | 조건 | 재검토 항목 |
      |---|---|
      | **Test 데이터 mAP < 0.85** | Version A 일반화 한계 의심 → 옵션 A 또는 C 검토 |
      | **Stage 2 OBB 학습 mAP < 0.80** | PMI crop 의 회전 변형 노이즈 영향 → 옵션 A 적용 후 재학습 |
      | **Stage 1 Version B 학습 직전** | rotation augmentation 추가 적용 검토 (D-001 정책 부분 갱신) |
      | **5,839장 본격 라벨링 시** | 회전 변형 라벨 정확도 측정 후 옵션 A vs C 결정 |
    - **복기 시 사용할 도구** (작성 보류 중):
      - `src/exclude_files.py` (variant-level 제외) — D-024 group 정합성 일부 손상 감수
      - Roboflow 재 export 가이드 (Generate New Version → Augmentation OFF)
    - **D-001 (flip OFF) 정책 일부 재검토**: rotation augmentation 은 별개 (D-001 은 좌우반전만 명시) — Version B 학습 시 rotation aug 추가 OK.

37. **D-037 (PMI Crop Padding 전략 — adaptive padding 도입, 2026-04-30)**
    - **제목**: PMI Crop Padding 전략 — adaptive padding 도입 (v1 fixed → v2 per-axis → v3 aspect-aware)
    - **배경**:
      - v1 (fixed 10px): 화살표/리더선 잘림. 큰 PMI 에는 인접 치수 침입 위험.
      - → 큰 padding (50~100px) 은 인접 치수 침입 (overlap) 야기 vs 작은 padding (10px) 은 화살표 잘림
      - **v1 한계**: 정사각형 bbox 와 가로/세로형 bbox 를 동일하게 취급 → 최적 padding 불가능
    - **평가 이력**:
      - **v1 → v2 (per-axis adaptive)** 전환:
        - v2 수식: `pad_x = clamp(bbox_w × ratio, [min, max])`, `pad_y = clamp(bbox_h × ratio, [min, max])`
        - 기본값: ratio 0.4, min 30, max 80
        - 결과 (20도면 844 crops): pad_x mean=33.2, pad_y mean=30.6, max=44
        - **만족도**: 비회전 텍스트 90% / 회전 텍스트 80% (잔존 20% 는 45° 대각선 화살표)
      - **v2 → v3 (aspect-aware)** 추가 보강:
        - aspect = max(w,h) / min(w,h) 로 정사각형 판정 (threshold 1.5)
        - 정사각형 (aspect < 1.5): uniform pad = long_side × 0.6 (45° 회전 화살표 보강)
        - 비정사각형 (aspect ≥ 1.5): per-axis 로직 유지 (인접 침입 최소화)
        - manifest 추가 컬럼: aspect_ratio, padding_strategy
        - **목표**: v2 의 정사각형 bbox (693개, 82.1%) 의 회전 텍스트 잘림 보강
        - **실측**: 라벨링 진행 중 — Strategy 분포/성능 비교는 라벨링 완료 후 측정 예정
    - **최종 채택 결정**:
      - **Stage 2 라벨링 입력**: v3 (aspect-aware) **권장** — Stage2_PMI_v3_844 task 라벨링 진행 중
      - v2 와 v3 모두 생성 완료 — manifest 비교로 strategy 분포 확인 가능
      - 라벨링 시점에 둘 다 사용 가능. v3 가 회전 텍스트 잘림 비율 더 낮음.
    - **관련 파일**:
      - `src/extract_pmi_crops.py` — v2 구현 (per-axis adaptive padding)
      - `src/extract_pmi_crops_v3.py` — v3 구현 (aspect-aware)
      - `docs/modules/extract_pmi_crops.md` — v2 문서
      - `docs/modules/extract_pmi_crops_v3.md` — v3 문서
    - **차후 검토 항목**:
      - 회전 잘림 잔여 케이스 (v2 20%, v3 측정 예정) → label_manual.md §3.5 라벨링 룰로 처리 (Rule G "잘린 텍스트", Rule H "과확대 케이스")
      - Stage 2 라벨링 완료 후 mAP@0.5 비교 (v2 vs v3) — 통계적 유의미성 판정
      - Stage 3 fine-tune 시점에 crop quality 최적화 재검토 가능
38. **D-038 (Stage 1 false positive Notes 의 Rescue 처리, 2026-05-01)**
    - **발견**: Stage 2 OBB 라벨링 중 Stage 1 Version A 가 일반 주석 (Notes/Text 클래스) 영역을 PMI 로 잘못 검출하는 케이스 다수 발견.
      - 예시 (실제 라벨링 중 발견된 케이스):
        - `材料は鉄かSUS403` (재질 명세, 일본어)
        - `+0.1以下のものは機械加工のこと` (가공 지시, 일본어, 참조 치수 12.3 포함)
        - `UNLESS OTHERWISE SPECIFIED ±0.1` 류의 일반 공차 주석
      - **이 내용들은 메타데이터 JSON 의 필수 항목** (재질/가공 지시/검사 기준 등)
    - **영향**: 단순 SKIP 처리하면 다음 정보 손실 발생:
      - Stage 1: Notes 영역 → PMI 로 오라우팅
      - Stage 2: Measure/GDT/Roughness 아님 → SKIP
      - Stage 3-A: Notes 영역 검출 안 됨 (이미 PMI 로 분류됨) → OCR 안 함
      - Stage 4 JSON merger: 정보 없음 → **메타데이터에 누락** ★
    - **해결**: SKIP `stage1_fp_notes` 마킹 + 별도 Rescue Path 추가
      - 라벨링: SKIP 라벨에 `reason=stage1_fp_notes` attribute 사용 (CVAT)
      - 추출: `src/extract_skip_list.py` → `outputs/skip_lists/stage1_fp_notes.txt`
      - Rescue: `src/rescue_misclassified_notes.py` → Donut zero-shot OCR → `outputs/rescued_notes.json`
      - 병합: pipeline.py / stage4 merger 에서 최종 JSON 의 `general_notes` 필드로 통합
    - **CVAT 라벨 설정 (필수)**: SKIP 라벨에 reason attribute 추가 (9개 카테고리)
      ```
      stage1_fp_other (default), unreadable, stage1_fp_section, stage1_fp_detail,
      stage1_fp_projection, stage1_fp_table, stage1_fp_notes (★ rescue 대상),
      stage1_fp_isometric, other
      ```
    - **관련 파일**:
      - `src/extract_skip_list.py` — CVAT XML → reason 별 분리 + summary
      - `src/rescue_misclassified_notes.py` — stage1_fp_notes → Donut OCR → JSON
      - `docs/modules/extract_skip_list.md` — 도구 문서
      - `docs/modules/rescue_misclassified_notes.md` — rescue 문서
      - `label_manual.md §3.5` — Rule O (stage1_fp_notes 의 중요성 + rescue 워크플로)
    - **차후 검토 항목**:
      - Stage 1 Version B 학습 시 Text 클래스 보강 → false positive 해소 (rescue 의존 최소화)
      - rescue 결과의 OCR 품질 검증 (다국어, 특수 기호, 손글씨 케이스)
      - `general_notes` 필드의 JSON schema 확정 (현재: items list + raw_text)
      - Stage 3 fine-tune 시 rescue 결과를 input domain 에 포함할지 결정
    - **★ 1차 시도 결과 (2026-05-03, Donut DocVQA — 실패)**:
      - 모델: `naver-clova-ix/donut-base-finetuned-docvqa`
      - 입력: 23개 stage1_fp_notes crops
      - 표면 통계: 23/23 success (에러 없음, 5.5초 / 4.17 crops/sec)
      - **실질 품질: 4% (1/23 만 의미 있는 결과)**
        - 단일 문자만: 11개 (48%, 예: `r`, `m`, `x`, `2`, `6`)
        - 환각 답변: 5개 (22%, 예: `let yourself`)
        - 부분 추출: 5개 (22%, 예: `gpi`, `iii`, `to ict`)
        - 의미 있는 결과: 1개 (4%, `d'sus403` ← `鉄かSUS403` 추정)
      - **원인**:
        1. **다국어 미스매칭** ★ — Donut DocVQA = 영어 학습, 우리 노트 = 일본어 다수 (`材料は鉄かSUS403`, `機械加工のこと`)
        2. **모델 부적합** — DocVQA = 문서 질문응답, 단순 OCR 아님
        3. **Crop 크기/컨텍스트 부족** — 작은 텍스트 fragment 처리 한계
      - **결정**: rescue 결과 폐기 (JSON 메타데이터 병합 안 함). 환각 텍스트 ("let yourself") 가 `general_notes`에 들어가면 메타데이터 오염 — 빈 필드 < 잘못된 정보.
      - **상세 박제**: `history.md §A.11.8`
    - **★ 2차 시도 (Day 3, 2026-05-04)** — **D-039로 흡수** (PaddleOCR-VL-1.5 채택)
    - **★ 적용 범위 명확화 (2026-05-03 박제)**:
      - **Phase 8 SKIP-only frame 제외 (Option B)** = **Stage 2 학습 데이터(`data/annotation/`) 에서만 적용**
      - 이미지 파일은 `outputs/cvat_stage2_input_v3_upscaled/` 에 그대로 보존
      - `outputs/skip_lists/stage1_fp_notes.txt` (23개) 는 Stage 3-A Rescue 경로로 별도 활용
      - 다른 SKIP 카테고리 (43 unreadable / 134 stage1_fp_other / 33 detail / 29 section / 13 table / 2 projection) 는 Stage 3 미사용
      - **stage1_fp_table 비활용 사유 (사용자 결정, 2026-05-03)**: Stage 1 이 PMI 로 false positive 한 표제란 일부는 정보 가치 거의 없음. 정상적인 Stage 3-A 입력은 **Stage 1 의 `Table` + `Text` 클래스 검출 결과** (논문 §4.3 정합)
39. **D-039 (Stage 3-A 모델 다국어 적응 — PaddleOCR-VL-1.5 채택, 2026-05-03)**
    - **배경**: D-038 1차 Rescue (Donut DocVQA zero-shot) 실질 4% 성공 → 다국어 SOTA 모델 재선정 필요
    - **사용자 환경**: 한국어/일본어/중국어/영어 도면 혼재 (러시아어 일부, 해상도 낮음)
    - **논문 (Khan et al. 2025)에서의 위치**:
      - Stage 3-A: Donut zero-shot (F1 0.672, 환각 39.9%) — 영어 데이터셋 가정
      - Stage 3-N: Donut fine-tune (F1 0.963)
      - "오픈소스 document loader 중 적합한 것 선정" 은 **사용자 자율 영역** (논문이 모델 미명시)
    - **2026 신규 논문 검증 결과** (`From Drawings to Decisions, arXiv 2506.17374`):
      - Donut vs Florence-2 비교 — Donut 우수 (F1 94%, 환각 10.8%)
      - Donut: Swin-B 비전 + BART 텍스트 디코더, OCR-free end-to-end
      - Florence-2: DaViT 비전 + multimodal token alignment
    - **모델 비교 (2026-04 SOTA 검색)**:

| 항목 | PaddleOCR-VL-1.5 | DeepSeek-OCR-2 | Donut DocVQA (zero-shot) |
|---|---|---|---|
| 출시일 | 2026-01-29 | 2026-01-27 | 2022 (영어만) |
| 모델 크기 | **0.9B** (~3GB VRAM) | 3B (16~24GB VRAM) | 200M |
| OmniDocBench v1.5 | **94.50%** ★ | 91.09% | (벤치 없음) |
| Table-TEDS | **92.76%** | 미공개 | 미공개 |
| Formula CDM | **94.21%** | 미공개 | 미공개 |
| CJK 다국어 | "industry-leading" ★ | 100언어 균등 | ❌ 영어만 |
| Seal 인식 | ✅ (1.5 신규) | ❌ | ❌ |
| JSON cell 좌표 | ✅ | △ | ❌ |
| 라이센스 | Apache 2.0 | MIT | MIT |

    - **★ PaddleOCR-VL-1.5 채택 사유 (8가지)**:
      1. **OmniDocBench 94.50%** (DeepSeek 91.09% 대비 +3.41%)
      2. **0.9B 모델** → RTX 5080 16GB 에서 Stage 2 모델 동시 로드 가능 (DeepSeek 3B 는 OOM 위험)
      3. **Table TEDS 92.76% 명시** → 우리 Title Block (표 형식) 직접 적용
      4. **Formula CDM 94.21% 명시** → Notes의 수식/공차 정확
      5. **Seal Recognition (1.5 신규)** → 우리 도면 도장/검도 도장 처리 (D-038에서 발견된 stage1_fp_table 케이스)
      6. **CJK industry-leading** → 일본어/한국어/중국어 도면 처리 (사용자 데이터셋 핵심)
      7. **JSON cell 좌표 제공** → Stage 4 merge 시 위치 정보 활용
      8. **2026-03-06 update (llama.cpp 추론 추가)** → 활발한 개발 + 배포 유연성
    - **Stage 3-N 정책 (논문 정합 유지)**:
      - Donut Numerical fine-tune 계속 사용 (Day 3 V6 검증)
      - **검증 단계 추가 (★ 신규)**:
        - V6 임계값: Measure F1 ≥ 0.80 / GDT F1 ≥ 0.85 / Roughness F1 ≥ 0.70 / Hallucination < 0.20
        - FAIL 시 폴백: Qwen3-VL (1순위) → PaddleOCR-VL (2순위) → DeepSeek-OCR-2 (3순위)
    - **하이브리드 아키텍처**:
      ```
      Stage 3-A: PaddleOCR-VL-1.5 (zero-shot, 다국어 SOTA)  ← 신규 D-039
                 ├─ Stage 1 Table 클래스 (TitleBlock + BOM 등 정상 검출 영역)
                 ├─ Stage 1 Text 클래스 (Notes 등 정상 검출 영역)
                 └─ D-038 Notes Rescue (stage1_fp_notes.txt 23개 — PMI로 오검출된 일반 주석)
                    ↓
                  general_notes 필드로 통합

      Stage 3-N: Donut Numerical fine-tune (논문 정합 유지)
                 ├─ Measure / GDT / Roughness 처리
                 └─ V6 검증 후 폴백 결정
      ```
    - **★ Rescue 범위 결정 (사용자 결정, 2026-05-03)**: stage1_fp_notes 만 Rescue
      - stage1_fp_table (13개) 는 Rescue 안 함 — 정보 가치 없음 (false positive 표제란 일부)
      - 정상 표제란/BOM 영역은 Stage 1 이 `Table` 클래스로 자체 검출 → Stage 3-A 직접 입력
    - **사전 검증 계획 (Day 2 학습 백그라운드 시간 활용)**:
      - 사용자 제공 샘플 (한국어/일본어/중국어/영어 각 1~3장) 으로 zero-shot 시범
      - 검증 통과 시 Day 3 본격 적용
      - 검증 실패 시 fine-tune 또는 폴백 (Qwen3-VL 등)
    - **관련 파일** (Day 3 작성 예정):
      - `src/stage3_alphabetical.py` 백엔드 교체 또는 신규 모듈
      - `src/rescue_misclassified_notes.py` 백엔드 옵션 추가 (--backend paddleocr-vl)
      - `requirements.txt` paddleocr 의존성 추가
      - `docs/modules/stage3_alphabetical.md` PaddleOCR-VL-1.5 통합 갱신
    - **차후 검토 항목**:
      - PaddleOCR-VL-2.0 또는 DeepSeek-OCR-3 출시 시 재평가
      - DeepSeek V4 multimodal 공식 출시 시 평가
      - Stage 3-N 폴백 시 Qwen3-VL fine-tune 가이드 사전 작성
    - **상세 박제**: `history.md §A.11.9`

40. **D-040 (Stage 2 5-Fold OBB Ensemble 채택, 2026-05-04)**
    - **결정**: Stage 2 (OBB 검출) default 추론 모드 = **5-Fold Ensemble**.
      - Pipeline 통합 시 5 fold best.pt 동시 추론 → class-wise rotated NMS (`iou_nms=0.5`) → §5.2 schema 반환.
      - legacy 단일 best.pt 모드는 `--no-ensemble` flag 로 fallback (디버깅/속도 비교용).
    - **배경**:
      - K-fold CV 학습 (Option F, 5 fold × 9.0h) 완료, mean mAP@0.5 = 0.932 (논문 +9.2 pp).
      - V3-B 단일 모델 (Best Fold = 2, mAP 0.978) 로 D-023 평가 시:
        - Measure missing_rate **0.101** (임계 0.08 초과) → ❌ FAIL
        - GDT 0.000 / Roughness 0.036 — PASS
      - conf 튜닝 (0.15/0.25 동일 recall) 효과 없음 → recall 회복은 추론 strategy 변경으로만 가능.
    - **5-Fold Ensemble 결과 (110장 val, 2026-05-04)**:
      | 클래스 | P_single | P_ens | R_single | R_ens | miss_single | miss_ens | 임계 | 판정 |
      |---|---|---|---|---|---|---|---|---|
      | **Measure** | 0.949 | 0.683 | 0.899 | 1.000 | 0.101 ❌ | **0.000** | <0.08 | ✅ PASS |
      | GDT | 0.945 | 0.848 | 1.000 | 1.000 | 0.000 | 0.000 | <0.05 | ✅ PASS |
      | Roughness | 0.957 | 0.846 | 0.964 | 1.000 | 0.036 | 0.000 | <0.30 | ✅ PASS |

      drawing_recall = **1.000** / D-023 overall = **★ PASS ★**.
    - **Trade-off**: Recall +0.101 (Measure) 대신 Precision -0.266 (Measure FP +46 / GDT +5 / Roughness +2 = 53 추가 detection). D-023 은 missing_rate 기반 → trade-off 수용 가능. Stage 3-A FP 처리량 5~10% ↑ — Phase 15 통합 시 모니터링.
    - **구현**:
      - `src/ensemble_predict.py` — 680+ lines, NMS resolver (다중 import + manual fallback), `predict_one_schema()` (pipeline 어댑터).
      - `src/pipeline.py` — `use_ensemble=True` default, 5 fold lazy load (`_ensure_ensemble`), `_run_stage2()` 분기, weights 검증 분기.
      - 의존성: `shapely>=2.0.0` (이미 등록), `torch` cu128.
    - **사용 예시**:
      ```bash
      # ensemble default (D-040 채택)
      python src/pipeline.py run --image dataset/sample.jpg --out outputs/sample.json

      # 단일 best.pt 모드 (legacy, 디버깅)
      python src/pipeline.py run --image dataset/sample.jpg --no-ensemble \
          --obb-weights checkpoints/yolo_obb.pt
      ```
    - **차후 검토 항목**:
      - Stage 3-A 에서 FP 부담 시: `--conf-obb 0.30` (Measure FP 50% ↓ 추정) 또는 top-3 fold ensemble
      - Weighted Box Fusion (WBF) 실험 — `manual_nms_rotated` 대체
      - cross-fold eval (Fold 0/1/3/4 val.txt) 로 일반화 재확인
      - ultralytics `nms_rotated` 향후 복원 시 `_resolve_nms_rotated()` 자동 사용 (코드 변경 X)
    - **관련 파일**:
      - `src/ensemble_predict.py` (★ 신규)
      - `src/pipeline.py` (수정)
      - `docs/modules/ensemble_predict.md` (★ 신규)
      - `docs/modules/pipeline.md` (수정)
      - `outputs/v3b_summary.txt` (V3-B + ensemble 결과)
      - `outputs/v3b_ensemble_eval.json` (raw evaluate)
    - **상세 박제**: `history.md §A.11.13` (.1 ~ .8)

---

## 12. 사용자 선호 (USER PREFERENCES)

> 새 세션에서도 반드시 준수.

1. **언어**: 응답·확인은 **한국어**.
2. **코드 작성 전 확인 룰** (★ 가장 중요):
   - 어떤 코드 파일이라도 작성·수정하기 직전에 반드시
     **“…를 작성해드릴까요?”** 라는 한국어 확인을 먼저 받는다.
   - 사용자가 “네/진행/ㅇㅇ” 등 명시적 동의를 표시한 뒤에만 코드를 작성한다.
   - 단, **문서(.md)** 와 **설정 템플릿(.yaml)** 은 사용자가 명시적으로 “파일로 출력/작성/만들어줘” 라고 지시한 경우 즉시 작성 가능.
3. **포맷**: 채팅 응답은 과한 bullet/header 지양, 자연스러운 문장 우선.
4. **파일 공유**: 산출물은 `C:\Users\user\github\Drawing` 에 저장하고 `computer://` 링크로 공유.

---

## 13. 새 AI 세션을 위한 부트스트랩 프롬프트 (그대로 복붙 가능)

> 다른 AI에게 이 프로젝트를 인계할 때 아래 블록을 그대로 붙여 넣으면 즉시 이어서 작업 가능.

```
당신은 "Multi-Stage Hybrid Framework for Engineering Drawings" 자체 구현
프로젝트를 이어받는다. 시작 전 반드시 다음을 수행하라.

1) 워크스페이스(`C:\Users\user\github\Drawing`)에서 PROJECT_HANDOFF.md 를 읽는다.
2) §0 체크리스트, §10 작업 진행 현황, §11 의사결정 로그, §12 사용자 선호를
   숙지한다.
3) §10 의 가장 위에 있는 TODO 항목이 다음 작업 후보다.
4) 사용자 선호에 따라, 코드 작성 전 반드시 한국어로
   "…를 작성해드릴까요?" 확인을 먼저 받아라.
5) 작업이 끝날 때마다 PROJECT_HANDOFF.md §10 표를 갱신해라.

데이터셋은 자체 수집(JPG only), 하드웨어는 RTX 5080 / i9-13900K / 128GB,
아키텍처는 논문 그대로(YOLOv11-det + YOLOv11-obb + Donut x2) 재현이다.
```

---

## 14. 검증 절차 (Validation Procedures)

각 step 완료 시 실행해야 하는 검증 매트릭스. 임계값은 `configs/validation_thresholds.yaml` 에서 관리.

### 15.1 검증 모듈 구조

```
src/validate/
├── __init__.py
├── common.py                       # CheckResult, ValidationReport, HTML/JSON 렌더
├── check_step1_5_sorter.py         # 분류기 정확도
├── check_labels_yolo.py            # YOLO det 라벨 (Stage 1)
├── check_labels_obb.py             # YOLO obb 라벨 (Stage 2)
├── check_stage1_model.py           # Stage 1 mAP / per-class
├── check_stage1_crops.py           # crop 시각/품질
├── check_stage2_model.py           # Stage 2 + ★ 누락률 (D-023)
├── check_stage2_warps.py           # perspective-warp 가독성
├── check_stage3a_alphabetical.py   # Donut zero-shot 성능
├── check_stage3n_numerical.py      # Donut fine-tuned 성능
├── check_pipeline_e2e.py           # end-to-end 통합
├── check_enrichment.py             # Step 9 enrichment 통계
└── run_all.py                      # 단계별 오케스트레이터

configs/validation_thresholds.yaml  # 임계값 + severity (critical / warning / info)
reports/                            # 출력 (gitignored)
data/validation_gt/                 # 사람 검수 ground truth (CSV/JSON)
```

### 15.2 단계별 검증 항목 (요약)

| Step | 핵심 검증 | 임계값 (critical) |
|---|---|---|
| 1.5 | 분류기 accuracy / 4개 언어 균형 / manual_review 비율 | accuracy ≥ 0.85 |
| 2 라벨 | 빈 라벨 / BBox 유효성 / 클래스 분포 / 인터-라벨러 IoU | bbox 유효 100% / 빈 라벨 < 5% |
| 2 모델 | per-class accuracy / mAP / FP rate | View 0.90 / TB 0.95 / Notes 0.90 / mAP ≥ 0.85 |
| 2 crop | 빈 crop / View 누락 / aspect ratio | 빈 crop = 0 / 누락 < 3% |
| 3 라벨 | OBB 유효성 / 각도 분포 / Roughness 수 | OBB 유효 100% |
| 3 모델 | per-class / **누락률** / **회수율** | Measure F1 ≥ 0.92 / 누락 Measure < 8%, GDT < 5% / 회수율 ≥ 0.85 |
| 3 warp | 빈 warp / OCR confidence / aspect | 빈 warp = 0 |
| 3-A | field F1 / Hallucination / 언어별 격차 | Notes F1 ≥ 0.75 |
| 3-N | schema F1 / 수치 정확도 / Hallucination | Measure ≥ 0.90 / GDT ≥ 0.95 / Hallucination < 0.10 |
| E2E | end-to-end F1 / 추론시간 / 실패율 | F1 ≥ 0.75 / 실패율 < 1% |
| 9 | method 분포 / HITL flag / provenance | provenance 100% |

### 15.3 검증 실행 방법

```bash
# 단일 단계
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv \
    --gt data/validation_gt/step1_5_titleblock_gt.