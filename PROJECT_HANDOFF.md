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
| 도면 언어 | **EN / KO / JP / RU / CN / DE** (도면 1장당 단일 언어, 6개 언어) | D-010 / D-013 / D-025 |
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
| **Day 3 (Phase 14 + 15a + 15b 작성 DONE)** | Stage 2 ensemble + GitHub + Phase 15a 환경 + 15b 스크립트 | **PARTIAL DONE** | 2026-05-04 | **Phase 14**: K-fold 학습 → **★ D-040 5-Fold Ensemble** → **D-023 PASS** → pipeline.py 통합 → smoke test → **★ D-041 GitHub push** (https://github.com/amadda0616-hash/IMMA). **Phase 15a**: `.venv-paddleocr` 분리 → **★ D-042 monkey-patch** (7차 시도 PASS) → `install_check.py` 작성 (393 lines) → **검증 PASS** (0.91B / 39.7s / 2.26~3.47s / 3.29GB). **Phase 15b 작성**: **★ D-043 도메인 한계 박제** (한국어 학습용 / 영어 부족 / 중국어 풍부 / CNC+기어 위주) + 사용자 sample 5장 확정 (한/영/일/중/러) + ★ **독일어 ~10장 발견 (D-025 6개 언어 확장)** + `stage3_paddleocr_zero_shot_test.py` 작성 (786 lines) + **★ D-044 TitleBlock Schema 23 필드** (ISO 7200 + KS A 0005 + 첨부 이미지 통합) + stage1_fp_notes 23개 박제 (CAD_Drawing219 14 + sample_01266 9). **다음 세션 (Phase 15b 실행 ~ 15e)**: [`docs/NEXT_SESSION_GUIDE.md`](./docs/NEXT_SESSION_GUIDE.md) 참조 — 5장 평가 → 정성 검토 → 15c 백엔드 교체 → 15d Notes Rescue → 박제 + commit. |
| **Day 4 (Phase 15b 1차~4차 + D-045~D-048 + Phase 16a 완료 + Phase 16b 학습 시작)** | Phase 15b 반복 + 4차 부분 PASS + Phase 16a/b | **IN_PROGRESS** | 2026-05-06 | (1)~(6) Phase 15b ~ D-048 (Day 4 이전 박제). (7) ✅ **★ Phase 16a 완료 (24분 04초)** — `prepare_vlm_dataset.py numerical --limit 500` 처리: 500/500 도면 → **11,470 region** 생성 (도면당 평균 22.94). manifest.csv 정상 작성. **★ D-049 sys.path bootstrap pattern 적용** (prepare_vlm_dataset.py + auto_fill_numerical_gt.py). 인자명 정정 (--input → --dataset, --stage1-weights → --det-weights, --stage2-ensemble → --obb-weights 단일 파일, --output 인자 없음). (8) ✅ **★ Auto-fill 결과 (2분 22초, 50.4%)** — Measure 61.5% (5,381/8,750) ★ baseline 충분 / GDT 0.2% (1/531) ★ 학습 불가 / Roughness 18.4% (402/2,189). **★ D-050 박제** Tesseract OCR 한계 (tolerance/GDT/Ra 인식 0%). **★ D-051 박제** 1차 baseline = Measure-only. **★ 5개 기준 4/5 통과** (#5 학습 안정성은 시작 후 검증). (9) **이제 진행**: Phase 16b overnight 학습 시작 (`stage3_numerical.py train --cfg configs/donut_numerical.yaml --device 0`, ~6h, 예상 종료 ~06:00). 다음 날 아침: V6 검증 + ja 영역별 Stage 3-A 평가. **★ 신규 박제**: `docs/KNOWN_LIMITATIONS.md` (376 lines, 한계 단일 source) + `outputs/workflow_diagram_v4.png` + `outputs/IMMA_progress_report_v4.docx` (동료 공유용 진행 보고서). |
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
- **6개 언어** (EN / KO / JP / RU / CN / **DE**) 혼합. 도면 1장 = 단일 언어 가정 (D-010 / D-013 / D-025).
  ★ DE (독일어) 약 10장 추가 (2026-05-04 사용자 확인) — Phase 15b 진입 전 발견.
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
25. **D-025 (6개 언어 확장, 2026-04-28 / 2026-05-04 갱신)** 라벨링 시작 시점에 **Chinese (中文) 추가** 발견. 추가로 Phase 15b 진입 전 **German (Deutsch) 약 10장 발견** (2026-05-04). 도면 언어는 이제 **EN / KO / JP / RU / CN (簡/繁體 모두) / DE** 6가지.
    - **★ 2026-05-04 갱신**: 독일어 도면 약 10장 추가 (사용자 확인). 비율은 적지만 D-013 4개 → D-025 5개 → 6개 언어 확장.
    - **Tesseract**: `lang='kor+eng+rus+jpn+chi_sim+chi_tra+deu'` 단일 호출
    - 시스템 패키지 추가: `tesseract-ocr-chi-sim`, `tesseract-ocr-chi-tra`, **`tesseract-ocr-deu`** (★ 2026-05-04 추가)
    - `sort_by_titleblock.py` 키워드 사전에 中文 + 獨語 추가 필요 (中文: 圖紙/標題/比例/材質/張數/設計/圖號 / DE: Zeichnung/Titel/Maßstab/Werkstoff/Stückzahl/Konstrukteur/Zeichnungsnummer)
    - `data/kb/material_catalog.json` 에 GB (Guobiao) + **DIN/EN (German/European)** 표준 grade 추가 필요 (DIN: St37/C45/X5CrNi18-10 등)
    - `validation_thresholds.yaml#step1_5.per_language_min` 에 CN + **DE** 추가
    - Stage 1·2 YOLO 는 D-009에 따라 언어 무관 — 영향 없음
    - Stage 3-A PaddleOCR-VL-1.5 (D-039 채택) 는 100+ 다국어 지원 → DE 자동 처리 (별도 fine-tune 불필요 예상)
    - 코드 수정 필요 항목은 차후 별도 작업 (이 D-025 박제 시점에서는 미실행)
    - **Phase 15b 평가 시 가중치**:
      - DE 1장 (or 활용 가능 시 일부) → mid confidence (sample 적음)
      - 독일어는 PaddleOCR-VL-1.5 의 라틴 알파벳 base 라 정확도 높을 것 예상
      - **★ 향후**: 독일어 도면 ~10장 별도 분석 후 V5 통합 평가
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

41. **D-041 (GitHub IMMA repo 워크플로 + Google Drive 자산 분리, 2026-05-04)**
    - **결정**: GitHub repo 와 Google Drive 의 자산 분리 정책 확립.
    - **분리 정책**:
      | 위치 | 용도 | 크기 |
      |---|---|---|
      | **GitHub IMMA** | 코드 + 문서 + 박제 (작은 산출물) | ~5 MB |
      | **Google Drive** | 모델 weights + dataset + 라벨 + 참고 자료 | ~10 GB |
    - **GitHub repo**: https://github.com/amadda0616-hash/IMMA
    - **Google Drive**: https://drive.google.com/drive/u/0/folders/1YweZCGEe8JbrRBaMSlSS7WIIx-yk_r8M
    - **`.gitignore` 핵심 패턴**:
      - `*.pt` `*.pth` `*.bin` `*.safetensors` `*.onnx` (모델 weights, GitHub 100MB 제한 회피)
      - `*.jpg` `*.jpeg` `*.png` (전역) + whitelist (`!docs/**/*.png`, `!workflow_*.png`)
      - `dataset/**` `IMMA.v1i.yolov11/` `articles/` `data/annotation/**` (저작권 + 외부 자산)
      - `*.docx` `*.xlsx` `*.pdf` (사용자 보고서)
    - **저작권 보호**: 라벨 파일명에 도면 ID 노출되므로 (`CAD_Drawing385_jpg.rf.<hash>__PMI_005.txt`) `data/annotation/**` 전체 ignore.
    - **Pull 전략 (★ 함정 박제)**:
      - **`git config pull.rebase true` 활성화 환경에서 `git pull --allow-unrelated-histories` 시 conflict 해결의 ours/theirs 의미 반대**
      - rebase 중 `--ours` = upstream/origin (들어오는 변경) ⚠️
      - 일반 merge 의 `--ours` = 현재 브랜치 (local)
      - **권장 설정**: `git config pull.rebase false` 로 명시적 merge 모드 사용
      - 또는 `git rebase --abort` 후 일반 merge 로 재시도
    - **이전 IMMA repo (16 commits) 보존 정책**:
      - 원격에 한글 논문 PDFs (`논문_kr/`), `erd.mermaid`, 메모 파일들 존재
      - `--allow-unrelated-histories` 로 병합 commit 작성 → 보존
      - README 충돌만 발생, 우리 commit 의 README 복원 후 push
    - **백업 자산 가이드**: [`docs/GOOGLE_DRIVE_ASSETS.md`](./docs/GOOGLE_DRIVE_ASSETS.md)
    - **Phase 15+ 체크리스트**: [`docs/PHASE15_CHECKLIST.md`](./docs/PHASE15_CHECKLIST.md)
    - **상세 박제**: `history.md §A.11.14`

42. **D-042 (PaddleOCR-VL-1.5 transformers monkey-patch + 별도 venv 분리, 2026-05-04)**
    - **결정**: Phase 15a 환경 설치 시 PaddleOCR-VL-1.5 호환성 워크어라운드 + 별도 venv 분리 정책 확립.
    - **★ Critical workaround (1줄)**:
      ```python
      from transformers import AutoConfig
      config = AutoConfig.from_pretrained('PaddlePaddle/PaddleOCR-VL-1.5', trust_remote_code=True)
      if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
          config.text_config = config.get_text_config()
      ```
      **모든 후속 코드** (`stage3_alphabetical.py`, `pipeline.py`, etc.) 가 이 patch 적용해야 함.
    - **별도 venv (`.venv-paddleocr`)**:
      - Python 3.10.20 / torch 2.11.0+cu128 / transformers 5.0.0
      - accelerate / sentencepiece / protobuf / einops / pillow
      - Phase 14 ultralytics venv (`.venv`) 와 분리 — 의존성 충돌 회피
    - **이슈 + 해결 매트릭스 (★ 7차 시도)**:
      | 시도 | 환경 | 에러 |
      |---|---|---|
      | 1 | transformers 5.6.2 + AutoModel | ROPE_INIT_FUNCTIONS['default'] KeyError |
      | 2 | transformers 4.49.0 + AutoModel | masking_utils 모듈 없음 |
      | 3 | transformers 4.50.0 + AutoModel | masking_utils 미도입 |
      | 4 | transformers 4.50.0 + AutoModelForImageTextToText | 4.x native paddleocr_vl 없음 |
      | 5 | transformers 5.0.0 + AutoModel | 같은 ROPE 에러 (잘못된 클래스) |
      | 6 | transformers 5.0.0 + AutoProcessor + AutoModelForImageTextToText | text_config AttributeError |
      | **★ 7** | **transformers 5.0.0 + monkey-patch + AutoProcessor + AutoModelForImageTextToText** | **PASS** ✅ |
    - **vLLM 미선택 사유**:
      | 옵션 | Phase 15 채택 | 사유 |
      |---|---|---|
      | transformers | ★ 채택 | drop-in 단순, debug 쉬움, 단일 호출 위주 |
      | vLLM | Phase 17 검토 | batch throughput 4~6× 빠름 (5,839 도면 시 7h vs 1.5h) |
    - **검증 PASS 결과 (`outputs/stage3a_install_check.json`)**:
      - Model params: 0.91B (논문 일치)
      - Load time: 39.7s (warm cache)
      - Inference: 2.26~3.47s / 더미 이미지
      - GPU used: 3.29 GB / 17.09 GB
    - **관련 파일**:
      - `src/stage3_paddleocr_install_check.py` (★ 신규, 393 lines)
      - `outputs/stage3a_install_check.json`
      - `docs/modules/stage3_paddleocr_install_check.md` (★ 신규)
    - **차후 검토**:
      - PaddleOCR-VL-2.0 출시 시 monkey-patch 폐기 가능 여부 재확인
      - transformers 5.x major 버전 업그레이드 시 ROPE/masking API 호환성 재검증
      - Phase 17 batch 단계에서 vLLM 도입 ROI 측정
    - **상세 박제**: `history.md §A.12.1` ~ `§A.12.2`

43. **D-043 (Stage 3-A 평가 데이터 도메인 한계 박제, 2026-05-04)**
    - **배경**: Phase 15b zero-shot 평가용 sample 7장 (한/영/일×2/중×2/러) 검토 중 **데이터셋 도메인 한계** 발견.
    - **★ 5가지 핵심 통찰 (사용자 분석)**:
      | # | 발견 | Phase 15b 영향 | Phase 16+ 영향 |
      |---|---|---|---|
      | 1 | 영어 비율 매우 적음 | V5 영어 신뢰도 낮음 (1장) | 영어 도면 추가 수집 |
      | 2 | 한국어 = 학습용 (TitleBlock 단순) | 실 산업 한국 도면 부족 | **★ 한국 산업 도면 별도 수집 필요** |
      | 3 | 중국어 자료 풍부 + 우수 (간체 + 번체) | zero-shot baseline | 중국어 가중치 ↑ 가능 |
      | 4 | CNC + 기어 가공 위주 (용접/판금/후처리 부재) | TitleBlock + Notes 어휘 편향 | **★ Stage 3-N fine-tune 시 도메인 한계 명시** |
      | 5 | 데이터 증강 비율 높음 (mirror/회전/색상) | group leak 검증 필수 | D-024 정책 유지 |
    - **결정**:
      - Phase 15b 평가 시 **언어별 confidence 가중치 적용**:
        - 영어 1장 → low confidence
        - 한국어 1장 → "학습용" 한정 (실 산업 도면 별도 검증 권장)
        - 일본어 2장 → high confidence
        - 중국어 2장 → high confidence (간체 + 번체)
        - 러시아 1장 → mid confidence
      - Phase 16 Stage 3-N fine-tune 시 **도메인 한계 명시** ("CNC + 기어 가공 도면 최적화" 박제)
      - 향후 데이터 수집 (Stage 1 V.B 단계): 한국어/영어 산업 도면 + 용접/판금/후처리 공정 보강
    - **상세 박제**: `history.md §A.12.0`

44. **D-044 (TitleBlock Standard Schema — ISO 7200 + KS A 0005 + 첨부 이미지 통합 23 필드, 2026-05-04)**
    - **결정**: Phase 15b zero-shot 평가 + 차후 Stage 3-A 통합 시 사용할 **23 필드 표준 schema** 확립.
    - **배경**: 사용자 첨부 "Structured JSON Output" 예시 + Web search (ISO 7200:2004 / ASME Y14 / KS A 0005) 비교 결과 기존 14 필드에서 9 필드 누락 발견.
    - **23 필드 분류**:
      | 카테고리 | 필드 |
      |---|---|
      | identification (5) | drawing_no / project_id ★ / title / sheet / revision |
      | descriptive (10) | part_name / material / mass ★ / scale / projection ★ / paper_size ★ / quantity / surface_treatment / heat_treatment ★ / general_tolerance ★ |
      | administrative (8) | company / department ★ / drawn_by / designed_by ★ / checked_by / approved_by / date / state ★ |
    - ★ = 본 박제로 추가된 9 필드.
    - **표준 출처**:
      - ISO 7200:2004 mandatory (8): legal_owner / drawing_no / date / sheet / title / approval_person / creator / document_type
      - ISO 7200 optional: revision / sheets_total / supplementary_title / department / classification / state / paper_size
      - ISO 7200 dynamic (외부 표기): scale / projection / general_tolerance / surface_texture
      - KS A 0005: 도면번호 / 명칭 / 재질 / 척도 / 설계자 / 검도자 / 승인자 / 일자
      - 첨부 이미지: project_id / mass / state / designed_by 분리
    - **구현**:
      ```python
      TITLEBLOCK_STANDARD_SCHEMA = {
          "identification": ["drawing_no", "project_id", "title", "sheet", "revision"],
          "descriptive": [
              "part_name", "material", "mass", "scale", "projection",
              "paper_size", "quantity", "surface_treatment",
              "heat_treatment", "general_tolerance",
          ],
          "administrative": [
              "company", "department", "drawn_by", "designed_by",
              "checked_by", "approved_by", "date", "state",
          ],
      }
      ```
    - **다국어 keyword hint**:
      - 한국어: 도면번호 / 명칭 / 재질 / 척도 / 일자
      - 일본어: 図番 / 部品名 / 材質 / 尺度 / 日付
      - 중국어 (간체/번체): 图号/圖號 / 名称/名稱 / 材料/材質
      - 러시아: Номер чертежа / Название / Материал / Масштаб
      - 독일어: Zeichnungsnummer / Bauteil / Werkstoff / Maßstab
    - **적용 범위**:
      - `src/stage3_paddleocr_zero_shot_test.py` (★ Phase 15b — 본 박제로 적용)
      - 차후 `src/stage3_alphabetical.py` (Phase 15c 백엔드 교체)
      - 차후 `src/pipeline.py` §5.5 unified JSON schema 갱신
    - **관련 파일**:
      - `src/stage3_paddleocr_zero_shot_test.py` (`TITLEBLOCK_STANDARD_SCHEMA` 상수)
      - `docs/modules/stage3_paddleocr_zero_shot_test.md` (★ 신규)
    - **상세 박제**: `history.md §A.12.3`

45. **D-045 (PaddleOCR-VL-1.5 generation parameters + Prompt 단순화, 2026-05-05)**
    - **결정**: PaddleOCR-VL 사용 시 **`repetition_penalty=1.2` + `no_repeat_ngram_size=4` + `max_new_tokens=512` 필수**. Prompt 의 schema list (23 필드 등) 는 hallucination 유발 → 제거.
    - **배경**: Phase 15b 1차 평가 (5장) — 모든 도면에서 **degenerate generation (반복 패턴)** 발생. 단 en `full_text` (단순 prompt) 만 정상.
    - **★ 1차 실패 사례 (★ 박제 가치)**:
      | 도면 | 출력 패턴 |
      |---|---|
      | en titleblock | `"Data Type \| Data Type \| ..." × 4096 chars` |
      | en notes | `"All notes are for the following:" × 102` |
      | ja titleblock | `"12345..." 1024 chars` |
      | ko titleblock | **"図番号" (일본어) 반복** — 한국어 → 일본어 fallback |
      | ru titleblock | `"100°N, 72°E, 140°W" GPS 좌표 hallucination` |
      | zh titleblock | `"\| 1 \| 2 \| 3 \|..." 숫자 표` |
      | en full_text | ✅ **정상 ("120, 23, 4 Holes #9 Thro', ALL DRILLED & TAPPED HOLES...")** |
    - **유일 baseline**: `full_text` prompt 가 정상 작동 → 모델 자체는 OCR 가능. 문제는 **prompt + generation parameters**.
    - **★ Fix 1 (generation parameters)**:
      ```python
      gen_kwargs = dict(
          max_new_tokens=512,            # 1024 → 512 (반복 패턴 길이 제한)
          do_sample=False,
          repetition_penalty=1.2,        # ★ 신규 — 반복 토큰 페널티
          no_repeat_ngram_size=4,        # ★ 신규 — 4-gram 반복 차단
          pad_token_id=processor.tokenizer.eos_token_id,
      )
      output_ids = model.generate(**inputs, **gen_kwargs)
      ```
    - **★ Fix 2 (Prompt 단순화)**:
      - 이전: `"Search for ALL of these fields when present: drawing_no, project_id, title, sheet, revision, part_name, material, mass, scale, projection, paper_size, quantity, surface_treatment, heat_treatment, general_tolerance, company, department, drawn_by, designed_by, checked_by, approved_by, date, state. ..."`
      - 신규: `"Read the title block (legend block) of this engineering drawing. The title block usually appears at the bottom-right or bottom of the drawing and contains structured fields such as drawing number, title, material, scale, sheet number, designer, date, etc. Extract the visible fields and return as a single JSON object. ..."`
      - **이유**: 23 필드 list 가 모델에게 schema 모방 패턴을 유도 → 표 형식 hallucination
      - `TITLEBLOCK_STANDARD_SCHEMA` 상수 (D-044) 는 **후처리/박제용으로 유지** — JSON parse 시 검증, 향후 fine-tune label 정의
    - **모든 후속 코드** (`stage3_alphabetical.py`, `pipeline.py` 등) 가 D-045 generation parameters 적용해야 함.
    - **차후 검토**:
      - 이미지 해상도 (640×640) 가 한계인 도면 → 원본 고해상도 (1280+) 로 교체 시 정확도 ↑
      - PaddleOCR-VL-2.0 출시 시 default generation parameters 재확인
      - vLLM 도입 시 동등 parameter 적용 (sampling_params)
    - **관련 파일**:
      - `src/stage3_paddleocr_zero_shot_test.py` (★ Fix 적용, 802 lines)
      - `outputs/stage3a_zero_shot_eval.{json,md}` (1차 실패 결과 — degenerate 박제)
    - **상세 박제**: `history.md §A.12.4`

46. **D-046 (PaddleOCR-VL-1.5 README 권장 호출 방식 — Critical Fix, 2026-05-05)**
    - **결정**: PaddleOCR-VL 사용 시 README sample code 의 정확한 호출 방식 채택. **자연어 prompt 폐기** + **task keyword** + **bfloat16** + **apply_chat_template 통합** + **max_pixels 명시**.
    - **배경**: D-045 (1차 fix) 이후 2차 평가에서도 부분 실패 — Layout token 누출 (`<|LOC_xxx|>`) / emoji hallucination / TitleBlock 인식 실패. 원인 추적: README 권장 방식과 우리 구현이 5가지 차이.
    - **★ 5가지 차이점**:
      | # | 항목 | 잘못된 (D-045) | 정정 (D-046) | 영향 |
      |---|---|---|---|---|
      | 1 | Prompt | 자연어 ("Read the title block...") | **task keyword** (`"OCR:"`, `"Table Recognition:"`, `"Spotting:"`) | ★★★ critical |
      | 2 | dtype | `torch.float16` | **`torch.bfloat16`** | ★★ |
      | 3 | Messages content | `{"type": "image"}` (binding 분리) | `{"type": "image", "image": image}` (직접) | ★★ |
      | 4 | Input pipeline | `processor(images=, text=)` 분리 | `apply_chat_template(... tokenize=True, return_dict=True, images_kwargs={...})` 통합 | ★★ |
      | 5 | max_pixels | 미설정 | `1280 * 28 * 28` (default) / `2048 * 28 * 28` (spotting) | ★ |
    - **★ 6개 Task Keyword (README BLOCK 3)**:
      ```python
      PROMPTS = {
          "ocr":      "OCR:",                   # 전체 텍스트 transcribe
          "table":    "Table Recognition:",     # ★ 표 형식 (TitleBlock!)
          "formula":  "Formula Recognition:",   # 수식
          "chart":    "Chart Recognition:",     # 차트
          "spotting": "Spotting:",              # 텍스트 + bbox
          "seal":     "Seal Recognition:",      # 도장
      }
      ```
    - **★ 우리 use case 매핑**:
      - `titleblock` → **`"Table Recognition:"`** (TitleBlock = 표)
      - `notes` → **`"OCR:"`**
      - `full_text` → **`"OCR:"`**
      - 향후: stage1_fp_table 도장 영역 → `"Seal Recognition:"`
    - **★ 정확한 호출 코드**:
      ```python
      from PIL import Image
      import torch
      from transformers import AutoProcessor, AutoModelForImageTextToText

      model = AutoModelForImageTextToText.from_pretrained(
          "PaddlePaddle/PaddleOCR-VL-1.5",
          torch_dtype=torch.bfloat16,                # ★ bfloat16
      ).to("cuda").eval()
      processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL-1.5")

      image = Image.open(image_path).convert("RGB")
      max_pixels = 1280 * 28 * 28   # ★ 명시 (1003520)

      messages = [{
          "role": "user",
          "content": [
              {"type": "image", "image": image},     # ★ image 직접 포함
              {"type": "text", "text": "Table Recognition:"},  # ★ task keyword
          ]
      }]

      # ★ apply_chat_template 통합 호출
      inputs = processor.apply_chat_template(
          messages,
          add_generation_prompt=True,
          tokenize=True,
          return_dict=True,
          return_tensors="pt",
          images_kwargs={
              "size": {
                  "shortest_edge": processor.image_processor.min_pixels,
                  "longest_edge": max_pixels,
              }
          },
      ).to(model.device)

      outputs = model.generate(**inputs, max_new_tokens=512)
      # ★ input 부분 슬라이스 + processor.decode
      result = processor.decode(
          outputs[0][inputs["input_ids"].shape[1]:],
          skip_special_tokens=True,
      )
      ```
    - **D-045 의 generation parameters 폐기**:
      - `repetition_penalty=1.2` / `no_repeat_ngram_size=4` → 제거 (README 미사용, 너무 보수적이라 정상 generation 도 차단됨)
      - `pad_token_id` → 제거
      - `max_new_tokens=512` 만 유지
    - **D-044 TitleBlock 23 필드 schema 처리**:
      - `Table Recognition:` 출력은 markdown table (자유 형식) — JSON 직접 X
      - 2단계 처리: ① raw markdown 추출 → ② 정규식/LLM 으로 schema 매핑
      - `TITLEBLOCK_STANDARD_SCHEMA` 상수는 후처리 reference 로 유지
    - **추가 옵션 (BLOCK 4)**: `attn_implementation="flash_attention_2"` — 추론 가속 (현재 미설치, 차후 검토)
    - **2차 결과의 모든 문제 — 원인 명확**:
      | 문제 | 원인 |
      |---|---|
      | Layout token `<|LOC_xxx|>` 누출 | 자연어 prompt → spotting mode confuse |
      | emoji hallucination "📊 🔄 💬" | 자연어 + float16 numerical instability |
      | TitleBlock 인식 실패 | `"OCR:"` 만으론 표 구조 X → `"Table Recognition:"` 필수 |
      | en full_text 1차 → 2차 퇴행 | 1차 우연히 `"OCR:"` 와 가까운 자연어, 2차 repetition_penalty 과도 |
    - **관련 파일** (Fix 후):
      - `src/stage3_paddleocr_zero_shot_test.py` (★ D-046 적용 예정)
      - `src/stage3_paddleocr_install_check.py` (★ dtype 변경 + 호출 방식 갱신 예정)
      - `outputs/stage3a_zero_shot_eval_v2_partial.{json,md}` (2차 결과 — 박제 가치)
    - **상세 박제**: `history.md §A.12.5`

48. **D-048 (Stage 1 V.A 일본어/복잡 도면 generalization 검증, 2026-05-05)**
    - **결정**: ja_drawing 같은 다중 도면 + 다국어 페이지는 **Stage 1 (yolo_det.pt) 가 자동 분리 → 영역별 Stage 3-A 적용** 이 표준 흐름. 사용자 가설 ("다중 도면 분리 후 처리") = 우리 architecture 의 핵심 design.
    - **근거 — Stage 1 V.A zero-shot 검증** (ja_drawing 3334×2375):
      | 클래스 | 수량 | 평균 conf | 평균 사이즈 |
      |---|---|---|---|
      | View | **6** | 0.83 | ~800×900 |
      | TitleBlock | **3** | 0.87 | ~1000×400 |
      | Notes | **3** | 0.65 | ~700×300 |
      | PMI | 98 | 0.55 | (small dim labels) |
    - **핵심 검출 (★ TitleBlock + Notes — Stage 3-A 입력)**:
      - TitleBlock #1 conf=0.921 — CRITICAL CONTROL DESIGNATION 표 (1025×685)
      - TitleBlock #2 conf=0.873 — BSBM TT-10CW 메인 (1106×282)
      - Notes #1 conf=0.824 — 注記 5개 (916×306)
    - **★ 사용자 가설 검증 결과**:
      - ✅ Stage 1 V.A (영어 100장 seed) → 일본어 도면도 잘 분리 (★ generalization 우수)
      - ✅ 분리 후 영역 사이즈 충분 (~800×900) — PaddleOCR-VL `max_pixels=1003520` 이내
      - ✅ "분리 후 해상도 하락" 우려 → **PaddleOCR-VL image_processor 자동 normalize** 로 해결
    - **시사점**:
      - Stage 1 V.A 가 다국어 + 복잡 도면 generalize 가능 → V.B 재학습 우선순위 ↓
      - ja_drawing 같은 합성 도면도 pipeline.py Stage 1 → Stage 3-A 흐름으로 처리 가능
      - 4차 평가 ja 실패 (단일 처리 시 "B" 반복) → 분리 후 처리 시 정상화 기대
    - **다음 단계 (Phase 15c 후속, 다음 날)**:
      - `outputs/crops/ja_drawing/{Table, Notes}/` 영역별 Stage 3-A 평가
      - 결과 PASS 시 → ja 도 부분 PASS 인정
      - FAIL 시 → 한계 명시 + Phase 17 e2e 종합 재평가
    - **관련 파일**:
      - `outputs/ja_drawing.det.json` (110 region)
      - `outputs/crops/ja_drawing/` (자동 분리 crop)
    - **상세 박제**: `history.md §A.12.8.4`

47. **D-047 (PaddleOCR-VL OTSL Table Format + 3차 평가 V5 미통과 박제, 2026-05-05)**
    - **결정**: PaddleOCR-VL `Table Recognition:` 출력은 **OTSL (Optimized Table Structure Language) 토큰**. `<fcel>` / `<lcel>` / `<nl>` token 으로 표 구조 인코딩 → 후처리로 markdown / HTML / JSON 변환 필요.
    - **3차 평가 (D-046 적용) 결과**:
      | 도면 | OCR (notes) | OCR (full) | Table Recognition | 평가 |
      |---|---|---|---|---|
      | en | ✅ 정확 | ✅ 정확 | ⚠ OTSL token | 부분 성공 |
      | **ru** | ✅ **80%+** (Notes 5개 정확) | ✅ 동일 | ⚠ | ★ 가장 성공 |
      | ja | ❌ "B" 반복 | ❌ 동일 | ⚠ | 실패 |
      | ko | ❌ "샌드자동화기술사" hallucination | ❌ "철선자동화기능사" | ⚠ "비윤리" | 실패 (640×640 한계) |
      | zh | ⚠ dimension 일부 | ⚠ 동일 | ⚠ | 부분 실패 |
    - **D-013 V5 임계값**:
      | 지표 | 측정값 | 임계값 | 판정 |
      |---|---|---|---|
      | 평균 char accuracy | ~0.50 | ≥ 0.85 | ❌ FAIL |
      | field-level F1 | ~0.48 | ≥ 0.80 | ❌ FAIL |
      | hallucination rate | ~0.30 | ≤ 0.05 | ❌ FAIL |
    - **★ D-046 효과 입증** (FAIL 이지만 향상):
      - 무한 반복 ~95% → 0%
      - Layout token 누출 → 사라짐
      - emoji hallucination → 사라짐
      - 한국어 → 일본어 fallback → 사라짐
      - en/ru 부분 성공 (baseline 확보)
    - **OTSL Token 명세**:
      ```
      <fcel> = first cell (테이블 셀 시작)
      <lcel> = list cell (다음 셀)
      <nl>   = new line (행 바꿈)
      ```
      예시 출력:
      ```
      <fcel>120<lcel><lcel><lcel><lcel><lcel>...<nl><fcel>...
      ```
      → **셀 시작 인식 OK / 셀 내 텍스트 추출 부족** (작은 이미지 한계)
    - **원인 3가지** (★ 박제):
      1. **이미지 해상도 부족 (★ 가장 큰 요인)** — en/ko/ru/zh: 640×640 → 작은 글자 한계
      2. **Table Recognition 후처리 부재** — OTSL → markdown 변환 미구현
      3. **모델 fine-tune 부재** — PaddleOCR-VL-1.5 는 일반 문서 (책/신문) 학습, 엔지니어링 도면 도메인 적응 부족
    - **다음 단계 4 옵션**:
      | Option | 작업 | 권장도 |
      |---|---|---|
      | **A. 이미지 고해상도 교체** | 5장 → 1280+ 재제공 | ★★ |
      | **B. OTSL → markdown 후처리** | 정규식 또는 PaddleOCR native package | ★★★ |
      | **C. Stage 3-N 우선** | Phase 16 진행 후 e2e 시 재평가 | ★★ (시간 효율) |
      | **D. 폴백 평가** | Qwen3-VL / DeepSeek-OCR-3 별도 평가 | ★ |
    - **관련 파일**:
      - `src/stage3_paddleocr_zero_shot_test.py` (D-046 fix 809 lines)
      - `src/stage3_paddleocr_install_check.py` (D-046 fix 404 lines)
      - `outputs/stage3a_zero_shot_eval_v3_partial.{json,md}` (3차 결과)
      - `outputs/stage3a_zero_shot_eval_v2_partial.{json,md}` (2차 보존)
    - **상세 박제**: `history.md §A.12.6`

49. **D-049 (sys.path bootstrap pattern 확장 — Phase 16a 진입, 2026-05-05)**
    - **배경**: `python src/prepare_vlm_dataset.py numerical ...` 실행 시 `ModuleNotFoundError: No module named 'src'` 발생. 원인: `from src.stage1_layout import ...` 구문이 직접 실행 시 sys.path 에 프로젝트 루트가 없음.
    - **해결**: Task #92 (`pipeline.py` bootstrap) 와 동일 패턴 적용:
      ```python
      _PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
      if str(_PROJECT_ROOT_BOOT) not in sys.path:
          sys.path.insert(0, str(_PROJECT_ROOT_BOOT))
      ```
    - **적용 파일**:
      - `src/prepare_vlm_dataset.py` (Phase 16a, 2026-05-05)
      - `src/auto_fill_numerical_gt.py` (신규, 2026-05-05)
    - **★ 후속 적용 권장**: `src/validate/check_*.py` 9개 파일 (`from src.validate.common import ...`) — 사용 시점에 일괄 적용
    - **★ 박제 (절대 금지)**: PyPI 의 `src==0.0.7` 은 본 프로젝트와 무관한 외부 패키지. `pip install src` / `uv pip install src` 절대 실행 X — `requirements.txt` / `pyproject.toml` 에 `src` 추가 X.
    - **상세 박제**: `history.md §A.12.9` + `docs/KNOWN_LIMITATIONS.md §4.3, §4.4`

50. **D-050 (Tesseract OCR 도면 patch 본질적 한계 — Phase 16a, ★ Critical, 2026-05-05)**
    - **발견**: Phase 16a `prepare_vlm_dataset.py --ocr-prefill` 표본 500 dry-run.
    - **현상** (Pytesseract `--psm 6` + `kor+eng+rus+jpn`):
      - tolerance regex 매칭률: **0%** (OCR 출력에 `±` 기호 자체가 없음)
      - GDT symbol 매칭률: **0%** (의미 있는 텍스트 X — 예: `'더'`, `'80000'`, `''`)
      - Roughness Ra 매칭률: **30%** (Ra 키워드 OCR 인식 거의 안 됨)
      - Measure nominal 매칭률: **62.2%** (first numeric 추출만 안정적)
    - **본질**: 도면 patch 의 작은 글자 (10~14 px) + 한자/일본어/한글 혼재 → Tesseract 한계. regex 보강 효과 ≈ 0.
    - **OCR hint 노이즈 표본**: `'020'` (정상), `'on'` (오인식), `'ーーの40 ['` (일본어 노이즈), `'„23 „|'` (특수문자 노이즈), `''` (빈 문자열).
    - **영향**: Phase 16b 1차 baseline 의 학습 데이터 GT 품질 ↓↓ (특히 GDT / Roughness).
    - **★ 후속 옵션** (우선순위 순):
      1. **(권장)** 검수 도구 작성 + 사람 검수 (Phase 17 후, ~3일)
      2. PaddleOCR-VL 을 patch OCR 에 활용 (Tesseract 대체 실험)
      3. 도메인 특화 OCR 모델 fine-tune (long-term)
    - **상세 박제**: `docs/KNOWN_LIMITATIONS.md §4.1` (★ 핵심) + `history.md §A.12.9`

51. **D-051 (Phase 16b 1차 baseline 정의 — Measure-only, 2026-05-05)**
    - **정책**: Phase 16b Donut numerical fine-tune 의 ★ **1차 baseline 은 Measure nominal extraction 에 한정**.
    - **데이터 분포** (Phase 16a 표본 500 기준):
      | 클래스 | 비중 | Auto-fill rate | 학습 가능 sample (표본 500 → 전체 ~13,000 환산) |
      |---|---|---|---|
      | Measure | 86.2% | 62.2% | ~7,000 ★ |
      | Roughness | 11.2% | 30.4% | ~440 (제한적) |
      | GDT | 2.6% | 0.0% | 0 ★ |
    - **근거**:
      - Stage 2 라벨 단계의 GDT 부족 (KNOWN_LIMITATIONS §2.1) + Tesseract OCR 한계 (D-050) 의 결합
      - GDT 학습 사실상 불가 — noisy GT 로 포함되지만 학습 효과 기대 X
      - Measure nominal 학습은 의미 있음 (Donut 도메인 적응)
    - **Phase 17 e2e 평가 정책**:
      - Stage 3-N edit_distance 기준 (D-013) 부분 PASS 인정
      - 종합 점수에 Stage 3-N 자리 채움 → 후속 개선 우선순위 정량화
    - **★ 후속 (Phase 18+)**:
      - GDT crop ~500 추가 라벨링 + 검수
      - Stage 3-N 재학습 (full GT)
    - **관련 파일**:
      - `src/auto_fill_numerical_gt.py` (★ 신규 452 lines)
      - `configs/donut_numerical.yaml` (학습 설정)
      - `outputs/auto_fill_numerical_report.md` (Phase 16a 완료 후 생성)
    - **상세 박제**: `docs/KNOWN_LIMITATIONS.md §4.2` + `history.md §A.12.9`

52. **D-052 (Donut data_collator 호환성 fix — Phase 16b 학습 통과, 2026-05-06)**
    - **배경**: Phase 16b 학습 시작 시 첫 batch 에서 `ValueError: You should supply an encoding ... that includes input_ids, but you provided ['pixel_values', 'labels', 'decoder_input_ids']` 발생.
    - **원인**: HF Trainer 의 default data_collator (`DataCollatorWithPadding`) 가 `tokenizer.pad()` 호출 → `input_ids` 키 기대. Donut batch (vision-encoder-decoder) 는 `pixel_values, labels, decoder_input_ids` 만 가져 mismatch.
    - **Fix** (`src/stage3_numerical.py`):
      ```python
      from transformers import default_data_collator
      trainer = DonutTrainer(
          ...,
          data_collator=default_data_collator,  # ★ 단순 stack, tokenizer.pad 안 부름
      )
      ```
    - **상태**: ✅ Resolved — Phase 16b 학습 끝까지 ValueError 0건
    - **상세 박제**: `history.md §A.12.11.2` + `docs/KNOWN_LIMITATIONS.md §6 Resolved`

53. **D-053 (DonutTrainer subclass — transformers 5.x num_items_in_batch 호환, 2026-05-06)**
    - **배경**: D-052 fix 후 학습 진행 시 step 0 에서 `TypeError: DonutSwinModel.forward() got an unexpected keyword argument 'num_items_in_batch'` 발생.
    - **원인**: transformers 5.x 의 `Trainer.compute_loss()` 가 `num_items_in_batch=...` kwargs 를 model.forward 에 전달. DonutSwinModel 의 forward 는 이 인자 미지원 (구 모델 호환성 이슈).
    - **Fix** (`src/stage3_numerical.py`, train() 내부 클래스 정의):
      ```python
      class DonutTrainer(Trainer):
          def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
              # transformers 5.x 의 num_items_in_batch 등 흡수
              outputs = model(**inputs)
              loss = outputs.loss
              return (loss, outputs) if return_outputs else loss
      ```
    - **상태**: ✅ Resolved — Phase 16b 학습 step 27,930/27,930 완료 (TypeError 0건)
    - **★ 후속 적용**: 다른 Donut 기반 학습 코드에서도 동일 패턴 적용 필요
    - **상세 박제**: `history.md §A.12.11.3` + `docs/KNOWN_LIMITATIONS.md §6 Resolved`

54. **D-054 (Phase 16b 1차 baseline 학습 성공 — Measure nominal extraction, 2026-05-06)**
    - **결과**:
      | 항목 | 값 |
      |---|---|
      | 학습 시간 | 6시간 23분 (01:06~07:29) |
      | Total steps | 27,930 (15 epochs × 1,862) |
      | 평균 speed | 2.01 it/s |
      | **최종 train loss** | ~0.92 (epoch 14.99 plateau) |
      | **★ 최종 eval_loss** | **0.9581** (epoch 15.0) |
      | Over-fit | ✅ 없음 (train ≈ eval) |
      | Final ckpt | `checkpoints/donut_numerical/final/` (809MB) |
    - **cfg 변경**: epochs 30 → 15 (첫 학습 ETA 11h → 6h, 1차 baseline 시간 단축)
    - **D-051 검증**: 학습 가능 sample 5,784 (Measure 93%) 로 1차 baseline 수렴 확인. LR cosine decay 가 1e-12 도달 → 추가 epoch 의미 없음.
    - **다음 단계**: V6 검증 (`check_stage3n_numerical.py`) → D-013 임계값 비교 → Phase 17 e2e
    - **상세 박제**: `history.md §A.12.11`

---

## 11.6 D-038 stage1_fp_notes — Phase 15d 본격 실행 대상 (★ 박제, 2026-05-04)

D-038 1차 Donut DocVQA Rescue 4% 실패 → ★ **PaddleOCR-VL-1.5 backend 재실행** 예정.

### 입력 자산

```
outputs/skip_lists/stage1_fp_notes.txt
# SKIP reason: stage1_fp_notes
# Count: 23
# Source: CVAT XML SKIP 라벨
```

**구성**:
- **CAD_Drawing219**: 14개 PMI crop (PMI_006 ~ PMI_024)
- **sample_01266**: 9개 PMI crop (PMI_018, 019, 024, 038, 039, 071, 075, 083, 085)
- 모두 PMI 영역으로 분류됐지만 **실제는 일반 주석 (Notes) 영역** — Stage 1 false positive

### Phase 15d 작업 흐름

1. `data/stage1_fp_notes_crops/` 디렉토리 생성 + 23개 PMI crop 복사
2. PaddleOCR-VL OCR 실행:
   ```bash
   python src/stage3_paddleocr_zero_shot_test.py \
       --samples-dir data/stage1_fp_notes_crops/ \
       --prompts notes,full_text \
       --output-json outputs/stage1_fp_notes_paddleocr_eval.json \
       --output-md   outputs/stage1_fp_notes_paddleocr_eval.md
   ```
3. 비교: Donut 4% vs PaddleOCR (★ 목표 80%+)
4. 통합: 추출 텍스트 → Stage 3-A `general_notes` 필드 입력
5. 박제: D-038 갱신 (PaddleOCR backend 채택, 결과 PASS/FAIL)

### Phase 15d 임계값 (D-013 대응)

- 23개 중 의미 있는 텍스트 추출 ≥ 80% → PASS
- 빈 응답 / hallucination ≤ 20%
- PASS 시 → general_notes 통합, FAIL 시 → 한계 명시 + Stage 3-A 후속 검토

---

## 11.5 외부 자산 가이드 (★ 신규, 2026-05-04)

### Google Drive 백업 (팀 공유)

GitHub 에 포함되지 않은 모든 자산은 [팀 Google Drive](https://drive.google.com/drive/u/0/folders/1YweZCGEe8JbrRBaMSlSS7WIIx-yk_r8M) 에서 관리:

| 자산 | 크기 | 위치 (`<repo>/`) | 우선순위 |
|---|---|---|---|
| `dataset.tar.gz` | 1.1 GB | `dataset/` | ★★★ 필수 (저작권) |
| `IMMA.v1i.yolov11.tar.gz` | 13 MB | `IMMA.v1i.yolov11/` | ★★★ 필수 |
| `checkpoints.tar.gz` | 7.5 GB | `checkpoints/` | ★★★ 필수 |
| `data_annotation.tar.gz` | 50 MB | `data/annotation/` | ★★★ 필수 |
| `articles.tar.gz` | 231 MB | `articles/` | ★ 옵션 |
| `dataset_excluded.tar.gz` | 5 MB | `dataset_excluded/` | ★ 옵션 |

상세 가이드: [`docs/GOOGLE_DRIVE_ASSETS.md`](./docs/GOOGLE_DRIVE_ASSETS.md) — 다운로드 절차, 검증 체크리스트, 트러블슈팅 포함.

### Phase 15 ~ 18 체크리스트

다음 4개 phase 의 단계별 체크리스트: [`docs/PHASE15_CHECKLIST.md`](./docs/PHASE15_CHECKLIST.md)

- **Phase 15** (~6h): Stage 3-A PaddleOCR-VL-1.5 통합 (15a~15e)
- **Phase 16** (~8h): Stage 3-N Donut Numerical fine-tune (16a~16d) — ★ V6 D-023 critical
- **Phase 17** (~4h): Pipeline E2E batch 통합 (17a~17c)
- **Phase 18** (~4h): Step 8 metrics + Step 9 enrichment (18a~18d)

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
    --gt data/validation_gt/step1_5_titleblock_gt.tage 1 V.A 가 영어 위주 학습이지만 일본어 / 다언어 / 다중 도면 페이지에 일반화됨
      - 분리 후 해상도 손실 우려는 PaddleOCR-VL 의 normalization 로 자동 해결
      - 다음 세션 옵션: ja_drawing 영역별 Stage 3-A 평가 (`outputs/crops/ja_drawing/` 활용)
    - **상세 박제**: `history.md §A.12.8`

49. **D-049 (sys.path bootstrap pattern 확장 — Phase 16a 진입, 2026-05-05)**
    - **배경**: `python src/prepare_vlm_dataset.py numerical ...` 직접 실행 시 `from src.stage1_layout import ...` 가 `ModuleNotFoundError: No module named 'src'`. sys.path 에 프로젝트 루트 없음.
    - **해결**: Task #92 (`pipeline.py`) 와 동일 패턴 — 파일 상단에 sys.path bootstrap 추가:
      ```python
      _PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
      if str(_PROJECT_ROOT_BOOT) not in sys.path:
          sys.path.insert(0, str(_PROJECT_ROOT_BOOT))
      ```
    - **적용**: `src/prepare_vlm_dataset.py`, `src/auto_fill_numerical_gt.py`
    - **★ 절대 금지 박제**: PyPI 의 `src==0.0.7` 패키지 설치 시도 — 본 프로젝트와 무관한 외부 패키지. `pip install src` / `uv pip install src` 절대 실행 X. requirements.txt / pyproject.toml 에도 추가 X.
    - **상태**: ✅ Resolved
    - **상세 박제**: `history.md §A.12.9` + `docs/KNOWN_LIMITATIONS.md §4.3, §4.4`

50. **D-050 (Tesseract OCR 도면 patch 본질적 한계 — Phase 16a, ★ Critical, 2026-05-05)**
    - **발견**: Phase 16a `--ocr-prefill` 후 dry-run + 실제 적용 결과:
      - tolerance regex 매칭률: **0%** (OCR 출력에 `±` 부호 없음)
      - GDT symbol 매칭률: **0%** (의미 있는 텍스트 없음)
      - Roughness Ra 매칭률: **18.4%** (Ra 키워드 인식 거의 안 됨)
      - Measure nominal 매칭률: **61.5%** (first numeric 추출만 안정적)
    - **본질**: Pytesseract `--psm 6` + `kor+eng+rus+jpn` 도면 patch (10~14 px) + 한자/일본어/한글 혼재 → OCR 노이즈 매우 큼. regex 보강 효과 ≈ 0.
    - **OCR hint 노이즈 표본**: `'020'` (정상), `'on'` (오인식), `'ーーの40 ['` (일본어 노이즈), `'„23 „|'` (특수문자), `''` (빈 문자열).
    - **영향**: Phase 16b 1차 baseline 학습 데이터 GT 품질 ↓↓ (특히 GDT / Roughness).
    - **★ 후속 옵션**:
      1. **(권장)** 검수 도구 작성 + 사람 검수 (Phase 17 후, ~3일)
      2. PaddleOCR-VL 을 patch OCR 에 활용 (Tesseract 대체)
      3. 도메인 특화 OCR 모델 fine-tune (long-term)
    - **상태**: Active Critical (regex 보강 효과 X, 검수 도구 필수)
    - **상세 박제**: `docs/KNOWN_LIMITATIONS.md §4.1` (★ 핵심) + `history.md §A.12.9, §A.12.10`

51. **D-051 (Phase 16b 1차 baseline 정의 — Measure-only, 2026-05-05)**
    - **정책**: ★ Phase 16b Donut numerical fine-tune 의 **1차 baseline 은 Measure nominal extraction 에 한정**.
    - **데이터 분포** (Phase 16a 11,470 region, auto-fill 후 5,784 completed):
      | 클래스 | Total | Filled | Rate |
      |---|---|---|---|
      | Measure | 8,750 | 5,381 | 61.5% ★ |
      | Roughness | 2,189 | 402 | 18.4% (제한적) |
      | GDT | 531 | 1 | 0.2% (학습 불가) |
    - **근거**: Stage 2 라벨 단계 GDT 부족 (KNOWN_LIMITATIONS §2.1) + Tesseract 한계 (D-050).
    - **Phase 17 e2e 정책**: Stage 3-N edit_distance 기준 (D-013) 부분 PASS 인정 → 후속 개선 우선순위 정량화.
    - **★ 후속 (Phase 18+)**: 검수 도구 + 사람 검수 ~3일 + GDT crop ~500 추가 라벨링 → full GT 재학습.
    - **상태**: Active (D-054 학습 결과로 검증 완료)
    - **상세 박제**: `docs/KNOWN_LIMITATIONS.md §4.2` + `history.md §A.12.9, §A.12.10`

52. **D-052 (Donut data_collator 호환성 fix — Phase 16b 첫 batch, 2026-05-06)**
    - **배경**: Phase 16b 학습 첫 batch 에서 `ValueError: You should supply an encoding ... that includes input_ids, but you provided ['pixel_values', 'labels', 'decoder_input_ids']`.
    - **원인**: HF Trainer 의 default data_collator (`DataCollatorWithPadding`) 가 `tokenizer.pad()` 호출 → `input_ids` 키 기대. Donut batch (vision-encoder-decoder) 는 `pixel_values, labels, decoder_input_ids` 만 가져 mismatch.
    - **Fix** (`src/stage3_numerical.py`):
      ```python
      from transformers import default_data_collator
      trainer = DonutTrainer(
          ...,
          data_collator=default_data_collator,  # ★ 단순 stack, tokenizer.pad 안 부름
      )
      ```
    - **상태**: ✅ Resolved — Phase 16b 학습 끝까지 ValueError 0건
    - **상세 박제**: `history.md §A.12.11.2` + `docs/KNOWN_LIMITATIONS.md §6 Resolved`

53. **D-053 (DonutTrainer subclass — transformers 5.x num_items_in_batch 호환, 2026-05-06)**
    - **배경**: D-052 fix 후 학습 진행 시 `TypeError: DonutSwinModel.forward() got an unexpected keyword argument 'num_items_in_batch'`.
    - **원인**: transformers 5.x `Trainer.compute_loss()` 가 `num_items_in_batch=...` kwargs 를 model.forward 에 전달. DonutSwinModel.forward 미지원 (구 모델 호환성).
    - **Fix** (`src/stage3_numerical.py`, train() 내부):
      ```python
      class DonutTrainer(Trainer):
          def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
              # transformers 5.x num_items_in_batch 등 kwargs 흡수
              outputs = model(**inputs)
              loss = outputs.loss
              return (loss, outputs) if return_outputs else loss
      ```
    - **상태**: ✅ Resolved — Phase 16b 학습 27,930/27,930 step 완료 (TypeError 0건)
    - **★ 후속 적용**: 다른 Donut 기반 학습 코드 동일 패턴 권장
    - **상세 박제**: `history.md §A.12.11.3`

54. **D-054 (Phase 16b 1차 baseline 학습 성공 — Measure baseline 확보, 2026-05-06)**
    - **결과**:
      | 항목 | 값 |
      |---|---|
      | 시작 / 종료 | 01:06:07 → 07:29:26 |
      | 소요 | **6시간 23분** (예상 5h 33분 + eval overhead) |
      | Total steps | 27,930 (15 epochs × 1,862) |
      | 평균 speed | 2.01 it/s |
      | **최종 train loss** | ~0.92 (epoch 14.99 plateau) |
      | **★ 최종 eval_loss** | **0.9581** (epoch 15.0) |
      | Over-fit 여부 | ✅ 없음 (train ≈ eval) |
      | Final ckpt | `checkpoints/donut_numerical/final/` (model.safetensors 809MB) |
      | Top-3 ckpt | checkpoint-24206 (e13), 26068 (e14), 27930 (e15=final) |
    - **cfg 변경**: `epochs: 30 → 15` (첫 학습 ETA 11h → 6h 단축, 1차 baseline + over-fit 방지).
    - **D-051 검증**: Loss plateau (~0.92) + LR 1e-12 도달 → 학습 사실상 종료. Measure 위주 baseline 으로 수렴.
    - **다음 단계**: V6 검증 (`check_stage3n_numerical.py`) → D-013 임계값 비교 → Phase 17 e2e 진입.
    - **상태**: ✅ Phase 16b 완료
    - **상세 박제**: `history.md §A.12.11`

---
