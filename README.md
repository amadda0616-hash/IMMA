# Multi-Stage Hybrid Framework for Engineering Drawings

엔지니어링 도면(JPG)을 구조화된 JSON으로 자동 변환하는 멀티스테이지 하이브리드 파이프라인.
Khan et al. 2025 *"Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View
Engineering Drawings Using Vision Language Model"* 아키텍처를 자체 데이터셋으로 재현하고,
Pinquié 2025 (eDOCr2) / Khan 2026 (Context-Aware Mapping) / Singh & Sadhu 2026 (LLM Survey)
등 후속 연구의 인사이트를 흡수해 **Metadata Enrichment 단계**까지 확장.

> **인수인계 / 작업 사양은 [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md) 참조.**
> 새 AI 세션이 작업을 이어받을 때 반드시 먼저 읽을 것 (§0 체크리스트 → §10 진행현황 → §11 의사결정).

---

## 1. 아키텍처

```
[입력: JPG 도면 (KO / EN / JP / RU / CN)]
       │
       ▼
Step 1.5: Sort by TitleBlock  (PyTesseract + 라인 밀도)
       │
       ├─ stage1_titleblock/   stage2_no_titleblock/   manual_review/
       │
       ▼
Stage 1: YOLOv11-det          (Layout: Isometric / PMI / Table=TB / Text=Notes / View, D-028)
       │
       ├─► PMI crop  (★ Stage 2 입력)
       │      ▼
       │   Stage 2: YOLOv11-obb  (Annotation: Measure / GDT / Roughness OBB)
       │      │
       │      ▼  perspective-warp de-rotation
       │   Stage 3-N: Donut Numerical  (fine-tuned, schema JSON)  [V6 검증 후 폴백 가능]
       │
       └─► TitleBlock + Notes crop
              ▼
           Stage 3-A: PaddleOCR-VL-1.5 (★ D-039, 0.9B, zero-shot, JSON + cell 좌표)
                  │
                  ▼
              Stage 4: JSON Merger
                  │
                  ▼
              Step 9 (확장): Metadata Enrichment
                  │  (4-tier: deterministic → heuristic → llm → hitl)
                  ▼
              Unified Enriched JSON
```

상세 다이어그램: [`workflow_diagram_v2.png`](./workflow_diagram_v2.png) / [`workflow_diagram_v2.svg`](./workflow_diagram_v2.svg) (Step 9 포함)
v1 (Step 9 미포함): [`workflow_diagram.png`](./workflow_diagram.png) / [`workflow_diagram.svg`](./workflow_diagram.svg)

---

## 2. 작업 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04 LTS on **WSL2** (Windows 11 host) |
| IDE | **Antigravity** (VS Code 기반, WSL Remote) |
| GPU | NVIDIA **RTX 5080** (Blackwell, 16GB VRAM) |
| CPU / RAM | Intel i9-13900K / 128GB |
| CUDA | 12.4+ |
| Python | 3.10+ |
| Workspace | `/mnt/c/Users/user/github/Drawing` (WSL2 마운트) |
| 도면 언어 | **EN / KO / JP / RU / CN / DE** (도면 1장 = 단일 언어 가정, D-025 — DE 2026-05-04 추가) |

---

## 3. 진행 현황 (LIVE)

> 기준: 2026-04-27 (V6 완료 시점). 정확한 최신은 `PROJECT_HANDOFF.md` §10 참조.

### 3.0 데이터셋 (★ 2026-04-27 적재 완료)

| 항목 | 값 |
|---|---|
| 적재 경로 | `dataset/` |
| 수량 | **5,839 JPG** (2026-04-28 갱신, +1,252 추가) |
| 언어 분포 | **EN / KO / JP / RU / CN / DE** 혼합 (도면 1장 = 단일 언어, D-025) — DE ~10장 (2026-05-04) |
| TitleBlock 분포 | 있음 / 없음 혼합 |
| **사전 증강** | **일부 이미지 뒤집기 + 회전 적용** (사용자 사전 처리) |
| 논문 대비 | Stage 1: 5.8× / Stage 2: 4.2× — 양 충분 |

**※ 데이터 누수 주의 (D-024):** 같은 원본의 증강 변형이 train/val 양쪽에 들어가지 않도록 **group-aware split 필수**.

파일명은 **Roboflow export 형식**: `{original_stem}.rf.{hash}.jpg`
- 예: `11_jpeg.rf.8b46c563....jpg` 와 `11_jpeg.rf.de99e140....jpg` 는 동일 원본 `11_jpeg` 의 두 증강 변형
- **Group key 추출**: `filename.split('.rf.')[0]`
- Train/val split 시 `sklearn.model_selection.GroupShuffleSplit` 사용 권장 (PROJECT_HANDOFF.md §11 D-024 참조)

### 완료 (DONE)

| Step | 모듈 / 산출물 | 비고 |
|---|---|---|
| 0   | `PIPELINE.md` | 논문 분석 + 초기 파이프라인 정리 |
| 0.5 | `PROJECT_HANDOFF.md` | 인수인계 사양서 (§0~§14, 의사결정 D-001~D-018) |
| 1   | `requirements.txt`, `configs/*.yaml`, `.gitignore`, `README.md` | 환경 + 디렉터리 골격 |
| 1.5 | `src/sort_by_titleblock.py` | 5개 언어 키워드 + 라인 밀도 분류기 |
| 2   | `src/stage1_layout.py` | YOLOv11-det · train/predict/crop |
| 3   | `src/stage2_annotation.py` | YOLOv11-obb · perspective-warp de-rotation |
| 5   | `src/stage3_alphabetical.py` | Donut zero-shot · DocVQA/CORD 듀얼 모드 |
| —   | `workflow_diagram.svg`, `.png` | eDOCr2 스타일 워크플로 다이어그램 |
| —   | `data/sample_enriched/` | 10개 가상 enriched JSON + README |

### 완료 (DONE) — 핵심 파이프라인

| Step | 모듈 | 비고 |
|---|---|---|
| 1.5 | `src/sort_by_titleblock.py` | TitleBlock 분류기 (5개 언어 OCR + 라인 밀도) |
| 1.6 | `src/sort_by_drawing_type.py` | **가공/조립 도면 자동 분류** (D-026, OCR + 풍선 + BOM) |
| 2   | `src/stage1_layout.py` | YOLOv11-det train/predict/crop |
| 3   | `src/stage2_annotation.py` | YOLOv11-obb + perspective-warp de-rotation |
| 4   | `src/prepare_vlm_dataset.py` | Stage 1·2 → VLM pair 자동 시드 (group key + OCR pre-fill) |
| 5   | `src/stage3_alphabetical.py` | Donut zero-shot (DocVQA/CORD 듀얼 모드) |
| 6   | `src/stage3_numerical.py` | Donut Numerical fine-tune + 추론 (5 케이스 round-trip PASS) |
| 7   | `src/pipeline.py` | end-to-end JPG → 통합 JSON (lazy import, auto-skip) |
| 8   | `src/utils/metrics.py` | 평가 지표 라이브러리 (15 sanity tests PASS) |
| 9   | `src/stage5_enrichment.py` | Metadata Enrichment 4-tier cascade (Mock/Gemini/Qwen) |

### 완료 (DONE) — 검증 프레임워크

| Step | 모듈 | 비고 |
|---|---|---|
| V0  | `src/validate/common.py` + `configs/validation_thresholds.yaml` | CheckResult/ValidationReport/HTML/JSON 3-종 출력 |
| V1  | `src/validate/check_step1_5_sorter.py` | 분류기 정확도, per-language |
| V2  | `src/validate/check_labels_yolo.py` + `check_stage1_model.py` | YOLO det 라벨 + Stage 1 모델 |
| V3  | `src/validate/check_labels_obb.py` + `check_stage2_model.py` | OBB 라벨 + ★ D-023 누락률 측정 |
| V5  | `src/validate/check_stage3a_alphabetical.py` | Donut zero-shot 사후 검증 (10 항목) |
| V6  | `src/validate/check_stage3n_numerical.py` | Donut fine-tuned 검증 (16 항목, ★ D-023 핵심) |
| V7  | `src/validate/check_pipeline_e2e.py` | end-to-end pipeline 검증 (13 항목, D-023 재측정) |
| V9  | `src/validate/check_enrichment.py` | Step 9 enrichment 검증 (provenance / cost / HITL / material) |

### Stage 1 학습 — Version A 완료 ✅ (2026-04-28)

| 항목 | 값 |
|---|---|
| 데이터셋 | Roboflow seed 100장 (`IMMA.v1i.yolov11/`, train 80 / valid 20) |
| 모델 | yolo11m.pt → checkpoints/yolo_det.pt (40.7MB) |
| 학습 | 50 epochs / 28.5분 / RTX 5080 cu128 |
| **mAP@0.5** | **0.9364** ★ (논문 0.96 근접) |
| V2-A 라벨 검증 | PASS (5/6 PASS, 1 WARN) |
| V2-B 모델 검증 | mAP=0.9364 PASS, PMI/Text 만 살짝 미달 (seed 데이터 부족) |

상세: [`history.md`](./history.md) §A — Version A 전체 기록.

### Auto-Labeling 완료 ✅ (2026-04-29)

`auto_label_stage1.py` 로 5,839장 자동 라벨링 (Stage 1 Version A 사용):

| 항목 | 값 |
|---|---|
| 처리 시간 | **5분 45초** (RTX 5080 cu128, ~17 img/s) |
| 처리 도면 | 5,839장 / Errors 0 |
| 총 검출 박스 | 293,894 (도면당 평균 50.3) |
| 클래스 분포 | PMI 245,462 (84%) / View 24,082 / Table 13,171 / Text 7,925 / Iso 3,254 |
| Manifest priority | empty 2 / low_conf 1,106 / review 4,604 / auto_pass 127 |

상세: [`history.md`](./history.md) §A.6.3.

### 🔴 sort_by_drawing_type.py (D-026 휴리스틱) 실패 → 대체 ✅ (2026-04-29)

**1차** (휴리스틱 OCR + Hough): 5,839장 / **mfg=0 / asm=5,313 / review=526** 비현실적 결과 → 폐기.

**원인**:
- OCR 치수 검출 실패 (Tesseract 5개 언어 작은 글자 한계)
- BOM 검출 false positive (Hough Lines 격자가 일반 표를 BOM 으로 오인)

**대체** ✅ — `src/sort_by_yolo_pmi.py` (D-026 갱신):
- Stage 1 Version A 자동 라벨 기반 PMI 카운트 분류
- 실행 ~3분 / mfg 5,349 / asm 441 / review 49 (분포 합리적)
- WSL2 호환 정책 (검수 폴더 자동 copy)

> **Stage 2 이후 모든 단계는 OCR 미사용 → OCR 실패 영향 격리**.

### 학습 데이터 정리 완료 ✅ (2026-04-29)

| 단계 | 결과 |
|---|---|
| sort_by_yolo_pmi.py 실행 | 5,839 → mfg 5,349 / asm 441 / review 49 |
| manufacturing/ sample 검증 (100장) | 조립도면 0% / 부품도면 10~20% / 가공도면 80~90% |
| 사용자 검수 (assembly + manual_review) | **18 unique groups** 식별 (false positive 80%+ 제거) |
| `exclude_groups.py` 실행 (~9초) | 46 images + 46 labels 이동 (D-024 group 단위) |
| **dataset/ 학습 잔여** | **5,793 images / 2,991 unique groups** |
| `dataset_excluded/` (보관) | 46 images / 18 groups |
| D-024 group 정합성 | overlap 0 ✅ |

부품도면도 학습 유지 (D-035): 가공도면과 분간 어려움 + Stage 1/2 학습 영향 미미.

### 진행 예정 — Day 1~3 Stage 2 ~ Step 8

| 단계 | 작업 | 비고 |
|---|---|---|
| **Day 1 ✅ DONE** (2026-04-30 ~ 2026-05-02) | Stage 2 PMI crop 라벨링 (CVAT 로컬, **844 crops**) | **★ 라벨링 완료**: Stage2_PMI_v3_upscaled3x_844 (upscale 3x). 전체 1026 박스 (Measure 555 / Roughness 106 / GDT 88 / SKIP 277). Frame-level SKIP 32.82% (>30% 임계 → Stage 1 V.B 보강 트리거). v1→v2→v3 padding 진화 (D-037). **D-038 Notes Rescue 박제**: SKIP 중 stage1_fp_notes 23개 → Day 2 Donut OCR 처리 예정. extract_skip_list.py + rescue_misclassified_notes.py 작성 완료 (Day 2 transformers 설치 후 실행). |
| **Day 2 IN_PROGRESS** (2026-05-03) | Phase 2 마무리 + Stage 2 학습 + Stage 3-A 모델 선정 | (1) ✅ uv pip install (5.94s). transformers 5.6.2 / torch cu128 (2) ⚠ Donut DocVQA Rescue 4% → **★ D-039 박제: PaddleOCR-VL-1.5 채택** (0.9B, OmniDocBench 94.5%, CJK SOTA, 8가지 사유). Stage 3-N Donut 유지 + V6 검증 (3) ✅ **Phase 7 완료** — CVAT YOLO OBB export 검증 통과 (844 frames / 1026 boxes / 8-point OBB) (4) Phase 8 정책 결정 — Option B (Stage 2 학습 제외) / Copy / 80-20 / Rescue α (stage1_fp_notes 23개만, stage1_fp_table은 Stage 1 Table 클래스 활용) (5) Phase 8~11 통합 스크립트 작성 예정 |
| **Day 3 DONE** (2026-05-04) | Stage 2 K-fold + Stage 3-A 환경 + GitHub | **Phase 14**: K-fold 학습 (mean mAP 0.932), **★ D-040 5-Fold Ensemble** → **D-023 PASS**, pipeline.py 통합, smoke test, **★ D-041 GitHub IMMA push**. **Phase 15a**: `.venv-paddleocr` venv, **★ D-042 monkey-patch**, install_check PASS (0.91B / 39.7s / 2.26s). **Phase 15b 작성**: **★ D-043 도메인 한계** + sample 5장 + **★ D-044 23 필드 schema**. |
| **Day 4 DONE** (2026-05-05 ~ 06) | Phase 15b 4차 + Phase 16a/b 완료 + 1차 baseline ★ | **(전반)** Phase 15b 1~4차 → **★ D-045/D-046/D-047** + **★ D-048 ja_drawing 110 region 분리**. **(중반)** Phase 16a 완료 (24분 04초, **11,470 region**) + **★ D-049 sys.path bootstrap** + **★ D-050 Tesseract OCR 한계** (Critical) + **★ D-051 1차 baseline = Measure-only** + Auto-fill 50.4%. **(후반 ★ 신규)** **Phase 16b 학습 성공** (01:06~07:29, **6h 23분**, 27,930 steps, **eval_loss 0.9581**) + **★ D-052 data_collator fix** + **★ D-053 DonutTrainer subclass** (transformers 5.x num_items_in_batch 호환) + **★ D-054 1차 baseline 수렴 성공** + cfg epochs 30→15 단축. **신규 박제**: `docs/KNOWN_LIMITATIONS.md` (376 lines) + `outputs/workflow_diagram_v4.png` + `outputs/IMMA_progress_report_v4.docx` (동료 공유용). **★ 다음 (Day 5)**: V6 검증 + ja 영역별 Stage 3-A 평가 + Phase 17 e2e 진입. |
| **차후** | Pre-annotation Phase 2 (Version B 학습) | 라이선스 + 비용 정리 후, D-035. **★ Stage 1 V.B 학습 시 PMI false positive 보강** (Day 1 SKIP 33% > 30% 임계). |

---

## 4. 디렉터리 구조

```
Drawing/
├── README.md                         ← 본 파일
├── PROJECT_HANDOFF.md                ← ★ 인수인계/사양서 (필독)
├── PIPELINE.md
├── MANUAL.md                         ← 단계별 작업 가이드 (사용자 친화)
├── label_manual.md                   ← 라벨링 매뉴얼 (Roboflow + CVAT)
├── history.md                        ← ★ 학습 이력 (Version A, B, C ...)
├── workflow_diagram_v3.svg / .png    ← 최신 파이프라인 다이어그램 (v3)
├── workflow_diagram_v2.svg / .png    ← v2 (Step 9 enrichment 추가)
├── workflow_diagram.svg / .png       ← v1 (논문 그대로)
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── yolo_det.yaml                 ← Stage 1 데이터셋 설정
│   ├── yolo_obb.yaml                 ← Stage 2 데이터셋 설정
│   ├── donut_numerical.yaml          ← Stage 3-N 학습 설정
│   └── validation_thresholds.yaml    ← ★ 검증 임계값 + severity
│
├── data/
│   ├── raw/                          ← 원본 JPG 적재
│   ├── stage1_titleblock/            ← sort_by_titleblock.py 결과
│   ├── stage2_no_titleblock/
│   ├── manual_review/
│   ├── layout/                       ← Stage 1 라벨링 데이터셋
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}
│   ├── annotation/                   ← Stage 2 라벨링 데이터셋
│   │   ├── images/{train,val}
│   │   └── labels/{train,val}
│   ├── vlm/
│   │   ├── alphabetical/             ← Stage 3-A 평가용 (옵션)
│   │   └── numerical/                ← Stage 3-N 학습 데이터 pair
│   └── sample_enriched/              ← ★ 10개 가상 enriched JSON 샘플
│       ├── README.md
│       └── sample_01..10_*.json
│
├── dataset/                          ★ 5,839 JPG 적재 완료 (사전 증강 일부 포함, 2026-04-28 갱신)
├── IMMA.v1i.yolov11/                 ★ Roboflow seed 100장 라벨링 완료 (D-028 5클래스, 80/20 split)
│   ├── data.yaml                     ← Stage 1 클래스: Isometric/PMI/Table/Text/View
│   ├── train/{images,labels}/         ← 80 JPG + 80 YOLO txt
│   └── valid/{images,labels}/         ← 20 JPG + 20 YOLO txt
│
├── checkpoints/                      ✅ Version A 학습 산출물
│   ├── yolo_det.pt                    ← Stage 1 best (40.7MB, mAP 0.9364)
│   └── yolo_det_runs/yolo_det_seed/   ← 학습 run (results.png / confusion_matrix.png 등)
│
├── dataset_excluded/                 ✅ 조립도면 보관 (46 images, 18 groups, 2026-04-29)
│
├── outputs/
│   ├── auto_labels/                   ✅ auto_label_stage1.py 산출 (5,793 labels + 동기 정리)
│   │   ├── labels/                     ← 5,793 .txt
│   │   ├── images/                     ← 5,793 (symlink/copy)
│   │   ├── manifest.csv                ← priority 정렬
│   │   └── labels_excluded/            ← 46 (조립도면 제외, D-024)
│   ├── sort_by_yolo_pmi/              ✅ D-026 분류 (2026-04-29)
│   │   ├── manifest.csv                ← per-class counts + decision
│   │   ├── README.md                   ← 검수 가이드 (자동 생성)
│   │   ├── manufacturing/              ← 5,349 (symlink)
│   │   ├── assembly/                   ← (검수 후 18장 잔여, copy)
│   │   └── manual_review_type/         ← (검수 후 0장)
│   ├── exclude_list.txt                ← 18 group_keys (사용자 검수 결과)
│   └── exclude_groups_manifest.csv     ← 46 files 이동 기록
│
├── src/
│   ├── sort_by_titleblock.py         ✅ Step 1.5
│   ├── sort_by_drawing_type.py       ⚠️ Step 1.6 (D-026 휴리스틱, 폐기 2026-04-29)
│   ├── sort_by_yolo_pmi.py           ✅ Step 1.6 대체 (D-026 신규, PMI 카운트 분류)
│   ├── exclude_groups.py             ✅ D-024 group 단위 일괄 제외 (신규)
│   ├── stage1_layout.py              ✅ Step 2 (D-028 5클래스 + D-029 매핑)
│   ├── auto_label_stage1.py          ✅ Step 5.5 (Active Learning, 2026-04-28)
│   ├── stage2_annotation.py          ✅ Step 3
│   ├── prepare_vlm_dataset.py        ✅ Step 4
│   ├── stage3_alphabetical.py        ✅ Step 5
│   ├── stage3_numerical.py           ✅ Step 6
│   ├── pipeline.py                   ✅ Step 7
│   ├── stage5_enrichment.py          ✅ Step 9 (4-tier cascade)
│   ├── utils/
│   │   ├── __init__.py               ✅
│   │   ├── metrics.py                ✅ Step 8
│   │   ├── crop.py                   (옵션, 미작성)
│   │   └── json_merge.py             (옵션, 미작성)
│   └── validate/                     ★ 검증 프레임워크
│       ├── __init__.py               ✅
│       ├── common.py                 ✅ V0
│       ├── check_step1_5_sorter.py   ✅ V1
│       ├── check_labels_yolo.py      ✅ V2
│       ├── check_stage1_model.py     ✅ V2
│       ├── check_labels_obb.py       ✅ V3
│       ├── check_stage2_model.py     ✅ V3 (★ D-023)
│       ├── check_stage3a_alphabetical.py ✅ V5
│       ├── check_stage3n_numerical.py    ✅ V6 (★ D-023)
│       ├── check_pipeline_e2e.py     ✅ V7
│       └── check_enrichment.py       ✅ V9
│
├── checkpoints/
│   ├── yolo_det.pt                   ⏳
│   ├── yolo_obb.pt                   ⏳
│   └── donut_numerical/              ⏳
│
├── reports/                          ★ 검증 리포트 출력 (gitignored)
│   ├── <date>_step1_5_sorter.html
│   ├── <date>_step1_5_sorter.json
│   └── ...
│
├── data/validation_gt/               ★ 사람 검수 ground truth
│   ├── step1_5_titleblock_gt.csv
│   ├── stage1_iou_check.json
│   ├── stage3a_titleblock_gt.json
│   └── stage3n_numerical_gt.json
│
└── outputs/
    ├── <drawing_id>.det.json
    ├── <drawing_id>.obb.json
    ├── <drawing_id>.alpha.json
    ├── <drawing_id>.json             ← Stage 4 통합
    └── <drawing_id>.enriched.json    ← Step 9 확장
```

---

## 5. 설치

### 5.0 ★ 외부 자산 다운로드 (Google Drive, 팀 공유)

> 저작권 + 용량 문제로 GitHub 에 포함되지 않은 자산은 **팀 Google Drive** 에서 다운로드.

**📁 [팀 Google Drive (IMMA 자산)](https://drive.google.com/drive/u/0/folders/1YweZCGEe8JbrRBaMSlSS7WIIx-yk_r8M)**

| 자산 | 압축 풀기 후 위치 | 크기 | 비고 |
|---|---|---|---|
| `dataset/` (원본 도면) | `<repo>/dataset/` | ~1.1 GB | 5,839 장, ★ 저작권 — 외부 공유 금지 |
| `IMMA.v1i.yolov11/` (Roboflow seed) | `<repo>/IMMA.v1i.yolov11/` | ~13 MB | Stage 1 학습 100장 |
| `checkpoints/` (학습 weights) | `<repo>/checkpoints/` | ~7.5 GB | yolo_det.pt + yolo_obb_runs/ (5 fold) |
| `articles/` (참고 논문) | `<repo>/articles/` | ~231 MB | 라이선스 확인 후 활용 |

```bash
# git clone 직후 (자산 포함 풀 작업 환경 구축)
cd /your/path
git clone https://github.com/amadda0616-hash/IMMA.git Drawing
cd Drawing

# Google Drive 에서 위 자산을 다운로드 후 압축 해제
# (rclone / gdown / 수동 다운 등)
ls -la dataset/ IMMA.v1i.yolov11/ checkpoints/   # 검증
```

> **주의**: `dataset/` 은 저작권 보호 자료. **외부 공유 금지** — 팀 내부 사용만.

---

### 5.1 시스템 패키지

```bash
sudo apt update
sudo apt install -y \
    python3-venv python3-pip \
    tesseract-ocr \
    tesseract-ocr-eng tesseract-ocr-kor tesseract-ocr-jpn tesseract-ocr-rus \
    libgl1 libglib2.0-0

tesseract --list-langs   # eng / kor / jpn / rus 확인
```

### 5.2 Python 가상환경

```bash
cd /mnt/c/Users/user/github/Drawing
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# PyTorch CUDA 12.8 (RTX 5080 Blackwell sm_120, ★ D-030)
# cu124 빌드는 sm_90 까지만 지원 → RTX 5080 비호환
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
# 검증: python -c "import torch; print(torch.cuda.get_device_capability())" → (12, 0)

# 프로젝트 의존성
pip install -r requirements.txt
```

### 5.3 GPU 인식 검증

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0))"
# 기대: CUDA: True / Device: NVIDIA GeForce RTX 5080
```

---

## 6. 사용법 (작성 완료된 모듈)

### 6.1 Step 1.5 — TitleBlock 기준 분류기 (선택, D-019)

> **D-019:** 학습 흐름의 필수 단계가 아님. 데이터 품질 점검 / stratified split 보조 / 디버깅 용도.

```bash
# dataset/ 의 5,839 JPG 에 대해:
python src/sort_by_titleblock.py --dryrun        # 매니페스트만 확인
python src/sort_by_titleblock.py                 # 실제 이동

# 결과:
# - data/stage1_titleblock/      (TitleBlock 키워드 ≥ 2)
# - data/stage2_no_titleblock/   (키워드 0 + 라인 밀도 낮음)
# - data/manual_review/          (애매)
# - outputs/sort_titleblock_manifest.csv

# 결과 검증 (V1):
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv \
    --gt data/validation_gt/step1_5_titleblock_gt.csv
```

### 6.2 Step 2 — Stage 1 (YOLOv11-det) Layout Detection

```bash
# 학습 (라벨링 완료 후)
python src/stage1_layout.py train \
    --data configs/yolo_det.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --device 0

# 추론 → JSON
python src/stage1_layout.py predict \
    --image data/raw/sample.jpg \
    --weights checkpoints/yolo_det.pt

# 영역 자동 crop (Stage 2 / Stage 3-A 입력 준비)
python src/stage1_layout.py crop \
    --image data/raw/sample.jpg \
    --weights checkpoints/yolo_det.pt \
    --padding 5
# → outputs/crops/sample/{View, TitleBlock, Notes, Isometric, PMI}/*.jpg
```

### 6.3 Step 3 — Stage 2 (YOLOv11-obb) Annotation Localization

```bash
# 학습
python src/stage2_annotation.py train \
    --data configs/yolo_obb.yaml \
    --epochs 150 --imgsz 1024 --batch 8 --device 0

# View crop 위에서 OBB 추론
python src/stage2_annotation.py predict \
    --image outputs/crops/sample/View/sample__View_00.jpg \
    --weights checkpoints/yolo_obb.pt

# OBB perspective-warp de-rotation 패치 저장
python src/stage2_annotation.py crop \
    --image outputs/crops/sample/View/sample__View_00.jpg \
    --weights checkpoints/yolo_obb.pt
# → outputs/crops/sample/annotations/{Measure, GDT, Roughness}/*.jpg

# ★ 5-Fold Ensemble 추론 (D-040, default in pipeline.py)
python src/ensemble_predict.py evaluate \
    --val-txt data/annotation_kfold/fold_2/val.txt \
    --conf 0.25 --iou-nms 0.5 --device cuda:0 \
    --output outputs/v3b_ensemble_eval.json

# 단일 이미지 ensemble 추론
python src/ensemble_predict.py predict \
    --image data/annotation/images/valid/sample.jpg \
    --output outputs/sample_pred.json
```

### 6.3.1 ★ Stage 3-A 환경 검증 (Phase 15a, D-042)

```bash
# 별도 venv (Phase 15 전용 — Phase 14 ultralytics 와 분리)
source .venv-paddleocr/bin/activate

# PaddleOCR-VL-1.5 환경 자동 검증 (★ monkey-patch 자동 적용)
python src/stage3_paddleocr_install_check.py

# 빠른 모드 (inference 생략)
python src/stage3_paddleocr_install_check.py --skip-inference

# 결과: outputs/stage3a_install_check.json
# 종료 코드: PASS = 0, FAIL = 1
```

상세: [`docs/modules/stage3_paddleocr_install_check.md`](./docs/modules/stage3_paddleocr_install_check.md) / [`docs/PHASE15_CHECKLIST.md`](./docs/PHASE15_CHECKLIST.md).

### 6.4 Step 5 — Stage 3-A (Donut Alphabetical, zero-shot)

```bash
# 단일 TitleBlock
python src/stage3_alphabetical.py predict \
    --image outputs/crops/sample/TitleBlock/sample__TitleBlock_00.jpg \
    --region titleblock --mode docvqa --language en

# Notes
python src/stage3_alphabetical.py predict \
    --image outputs/crops/sample/Notes/sample__Notes_00.jpg \
    --region notes

# 배치 (Stage 1 crop 폴더 통째로)
python src/stage3_alphabetical.py batch \
    --input-dir outputs/crops/sample
```

### 6.5 검증 (Validation, 각 step 사후 검증) ★

```bash
# Step 1.5 분류기 정확도 (사람 검수 GT 필요)
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv \
    --gt data/validation_gt/step1_5_titleblock_gt.csv

# Stage 1 라벨 품질 (학습 전)
python -m src.validate.check_labels_yolo \
    --labels-dir data/layout/labels/train

# Stage 1 모델 (학습 후)
python -m src.validate.check_stage1_model \
    --weights checkpoints/yolo_det.pt \
    --data configs/yolo_det.yaml

# Stage 2 모델 + 누락률 분석 (사용자 핵심 요구)
python -m src.validate.check_stage2_model \
    --weights checkpoints/yolo_obb.pt \
    --data configs/yolo_obb.yaml

# Stage 3-A 성능 (zero-shot Donut)
python -m src.validate.check_stage3a_alphabetical \
    --predictions outputs/alphabetical/ \
    --gt data/validation_gt/stage3a_titleblock_gt.json

# 전체 통합 (단계별 순차 실행)
python -m src.validate.run_all --from step1.5 --to step3a
```

검증 결과는 다음 3종 동시 출력:
- 콘솔: PASS/FAIL/WARN 즉시 표시
- `reports/<date>_<check>.html` — 시각 리포트 (이미지 그리드, 차트, confusion matrix)
- `reports/<date>_<check>.json` — 정량 메트릭 (시계열 추적용)

임계값은 `configs/validation_thresholds.yaml` 에서 관리. severity 분류 (`critical` 차단 / `warning` 경고 / `info` 모니터링).

---

## 7. 자산 (Assets)

### 7.1 문서

| 파일 | 역할 |
|---|---|
| [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md) | 인수인계 사양서 (자기완결형). §0 체크리스트, §3 아키텍처, §5 JSON 스키마, §10 진행현황(LIVE), §11 의사결정(D-001~D-018), §13 새 AI 부트스트랩 프롬프트 |
| [`PIPELINE.md`](./PIPELINE.md) | 초기 파이프라인 메모 |
| [`README.md`](./README.md) | 본 파일 |
| [`workflow_diagram_v2.svg`](./workflow_diagram_v2.svg) / [`.png`](./workflow_diagram_v2.png) | **현행** 워크플로 다이어그램 (Step 9 Metadata Enrichment 포함) |
| [`workflow_diagram.svg`](./workflow_diagram.svg) / [`.png`](./workflow_diagram.png) | v1 (Step 9 미포함, 기존 기록용) |
| [`docs/modules/`](./docs/modules/) | **모듈별 상세 문서** (구현 요약 / 핵심 설계 / 검증 결과 / 사용법) |
| [`MANUAL.md`](./MANUAL.md) | 처음부터 끝까지 작업 매뉴얼 (uv 환경 + 라벨링 + 학습 + 검증) |

### 7.2 설정 (configs/)

| 파일 | 용도 |
|---|---|
| `yolo_det.yaml` | Stage 1 데이터셋 (Isometric/PMI/Table/Text/View, 5클래스 D-028), flip OFF |
| `yolo_obb.yaml` | Stage 2 데이터셋 (Measure / GDT / Roughness), 회전 aug ON |
| `donut_numerical.yaml` | Stage 3-N 학습 (epoch 30 / AdamW / cosine 1e-6 / batch 4 / FP16 / gradient checkpointing) |
| `validation_thresholds.yaml` | 검증 임계값 (D-021/D-023). severity = critical / warning / info |

### 7.3 코드 (src/)

| 파일 | 클래스/함수 (공개) | 상태 |
|---|---|---|
| `sort_by_titleblock.py` | `process_one`, `count_keywords`, `compute_line_density` | ✅ |
| `sort_by_drawing_type.py` | `count_dimensions`, `detect_balloons`, `detect_bom_table`, `classify` (D-026) | ✅ |
| `stage1_layout.py` | `predict_one`, `crop_regions`, `train` | ✅ |
| `stage2_annotation.py` | `predict_one`, `crop_obb_regions`, `warp_obb_crop`, `train` | ✅ |
| `prepare_vlm_dataset.py` | `extract_group_key`, `build_*_template`, `process_drawing_*` (3 서브커맨드) | ✅ |
| `stage3_alphabetical.py` | `load_model`, `predict_one`, `predict_titleblock`, `predict_notes` | ✅ |
| `stage3_numerical.py` | `json_to_donut`, `donut_to_json`, `train`, `predict_one`, `load_inference_model` | ✅ |
| `pipeline.py` | `Pipeline` class, `obb_local_to_global`, `run`, `run_batch` | ✅ |
| `utils/metrics.py` | `pr_f1`, `field_level_f1`, `hallucination_rate`, `compare_*`, `polygon_iou`, `detection_metrics` (10 섹션) | ✅ |
| `stage5_enrichment.py` | `Pipeline`-style `enrich()`, `make_provider()` (Mock/Gemini/Qwen), 4-tier cascade, 5개 카테고리 | ✅ |
| `validate/common.py` | `CheckResult`, `ValidationReport`, `make_bar_chart`, `make_confusion_matrix`, HTML/JSON 렌더 | ✅ V0 |
| `validate/check_step1_5_sorter.py` | 분류기 정확도 + per-language + confusion matrix | ✅ V1 |
| `validate/check_labels_yolo.py` | YOLO det 라벨 8 항목 검증 | ✅ V2 |
| `validate/check_stage1_model.py` | mAP / per-class accuracy / FP rate | ✅ V2 |
| `validate/check_labels_obb.py` | OBB 자기교차 검출 / 회전각 분포 / Roughness 부족 알림 | ✅ V3 |
| `validate/check_stage2_model.py` | ★ **누락률 + drawing-level recall (D-023)** | ✅ V3 |
| `validate/check_stage3a_alphabetical.py` | Donut zero-shot 사후 검증 (10 항목, per-language) | ✅ V5 |
| `validate/check_stage3n_numerical.py` | ★ **D-023 핵심** Donut fine-tuned 검증 (16 항목) | ✅ V6 |
| `validate/check_pipeline_e2e.py` | end-to-end 통합 검증 (13 항목, D-023 재측정) | ✅ V7 |
| `validate/check_enrichment.py` | Step 9 enrichment 검증 (provenance / cost / HITL / material) | ✅ V9 |

### 7.4 데이터 (data/)

| 폴더 | 내용 |
|---|---|
| `raw/` | 자체 수집 원본 JPG |
| `stage1_titleblock/` `stage2_no_titleblock/` `manual_review/` | sort_by_titleblock.py 분류 결과 |
| `layout/{images,labels}/{train,val}` | Stage 1 라벨링 데이터셋 (CVAT export YOLO det) |
| `annotation/{images,labels}/{train,val}` | Stage 2 라벨링 데이터셋 (CVAT export YOLO obb) |
| `vlm/{alphabetical,numerical}` | Stage 3 image–text pair (Step 4가 자동 시드) |
| **`sample_enriched/`** | **10개 가상 enriched JSON 샘플 + README** (Step 9 fixture) |
| **`validation_gt/`** | ★ 사람 검수 ground truth (CSV/JSON). 검증기 입력 |
| **(루트) `reports/`** | ★ 검증 리포트 출력 (HTML + JSON, gitignored) |

### 7.5 가상 메타데이터 샘플 ([data/sample_enriched/](./data/sample_enriched/))

Step 9 출력 형태를 보여주는 fixture 10건. 모두 JSON 유효성 통과.

| # | 가공 조합 | 재질 | 언어 |
|---|---|---|---|
| 01 | 레이저 + 판금 | SUS304 No.2D 1.5t | KO |
| 02 | 워터젯 + CNC + 용접 | SUS316L 6t | EN |
| 03 | CNC + 아노다이징 | AL6061-T6 | EN |
| 04 | 선삭 + 열처리 + 연삭 | S45C HRC58-62 | JA |
| 05 | 프레스 + 분체도장 | SECC 1.0t | KO |
| 06 | 스탬핑 + 스폿용접 | SAPH440 (자동차) | JA |
| 07 | 주조 + CNC + 열처리 | FCD450 | EN |
| 08 | DMLS + CNC | Ti6Al4V (항공) | EN |
| 09 | 플라스마 + 용접 + 도장 | SM490A (교량) | RU |
| 10 | 와이어 EDM + 연삭 | SKD11 (사출 금형) | JA |

---


## 8. 주요 의사결정 요약

상세는 [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md) §11. 주요 항목만 (D-001 ~ D-034):

| ID | 결정 |
|---|---|
| D-001 | 아키텍처: Khan 2025 논문 그대로 (YOLOv11-det + YOLOv11-obb + Donut x2) |
| D-009 | Stage 1·2는 언어 무관 단일 모델 (YOLO는 시각 패턴 학습) |
| D-010 / D-013 / D-025 | 도면 = 단일 언어 (KO/EN/JP/RU/CN/**DE** 6개 중 하나). 언어 분류는 차후 |
| D-012 | Stage 2 OBB crop 은 perspective-warp de-rotation 적용 (Donut 정확도 결정적) |
| D-014 | 작업환경 Ubuntu WSL2 + Antigravity 확정 |
| D-015 | FCF 컴파트먼트 분리 기본 미수행 (GD&T F1 < 0.85 시 검토) |
| D-016 | eDOCr2 다단계 파이프라인 차용 안 함 (Stage 2+3-N 통합) |
| D-017 | 백업 모듈 (`symbol_postcorrect.py`, `synthetic_gen.py`) Step 7 평가 후 조건부 |
| D-018 | Stage 3 모델 = Donut 유지 (Qwen/Gemini 평가 후 재논의) |
| D-019 | `sort_by_titleblock.py` 는 선택 분석 도구 (학습 흐름 필수 아님) |
| D-020 | 각 step 완료 시 `src/validate/check_*.py` 실행 + reports/ 보관 의무 |
| D-021 | 임계값 severity: critical 차단 / warning 경고 / info 모니터링 |
| D-022 | 검증 리포트 = 콘솔 + JSON + HTML 3종 동시 |
| D-023 | 사용자 필수 임계값: Measure 누락 < 8%, GDT < 5%, 회수율 ≥ 0.85, Notes F1 ≥ 0.75, Stage 3-N Hallucination < 0.10 |
| D-024 | dataset/ 사전 증강 포함. **group-aware split 필수** (data leak 방지). flip OFF / 회전 aug 정책 |
| D-026 | ~~`sort_by_drawing_type.py` 휴리스틱~~ → **폐기** (2026-04-29). 대체 = **`sort_by_yolo_pmi.py`** (Stage 1 Version A PMI 카운트). 사용자 검수 후 18 group / 46 files 제외 → dataset/ 5,793장. Manufacturing sample 100장 검증: 조립 0% / 부품 10~20% (학습 유지) / 가공 80~90% |
| D-027 | 가공도면 TB 핵심 필드 (material/quantity) 누락 — Step 9 enrichment 가 보강 |
| **D-028** | **Stage 1 클래스 5종 재정의** (`Isometric/PMI/Table/Text/View`). PMI = Stage 2 OBB 입력 영역. Stage 2 OBB 클래스는 Measure/GDT/Roughness 그대로 |
| **D-029** | **Roboflow→내부 매핑** (`Table→TitleBlock`, `Text→Notes`). `src/stage1_layout._result_to_schema()` 1지점 매핑. Stage 3-A 토큰 호환성 보존 |
| **D-030** | **PyTorch cu128 (RTX 5080 Blackwell sm_120)**. cu124 비호환 — `compute_capability == (12, 0)` 검증 필수 |
| **D-031** | **Stage 1 클래스 분포 임계값 재조정** (실측 PMI 80.6% / View 9.65% / TB 7.05% / Notes 1.89% / Iso 0.82%). PMI dominant |
| **D-032** | **Roboflow `Table` = 모든 표 통합** (TB+BOM+Revision+Notes Table). 도면당 평균 3.12개. Stage 3-A 가 자체 분류 |
| **D-034** | **PMI 처리 hierarchical** — Stage 1 axis-aligned + Stage 2 OBB 계층. 옵션 A 채택 (옵션 C "PMI 제거" 보류) |

**최신 학습 결과**: [`history.md`](./history.md) §A — Version A (Stage 1 seed 100장, mAP 0.9364)
