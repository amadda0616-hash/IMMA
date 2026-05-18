# 도면 인식 시스템 — 통합 README (V.A → V.H' D-110.7 최종)

> 2D 엔지니어링 도면 (PNG/JPG) 을 구조화 JSON 으로 자동 변환하는 Multi-Stage Hybrid VLM 파이프라인.
> 최종 산출 JSON 은 IMMA (지능형 제조 가공 매칭 플랫폼) RAG 시스템의 raw data 로 사용된다.

---

## 0. 핵심 키워드 사전 (Glossary)

본 프로젝트에서 반복 등장하는 24개 핵심 용어 한 줄 요약. 처음 보는 분은 여기부터.

### A. Pipeline & Architecture

| 키워드 | 한 줄 정의 |
|---|---|
| **Stage 1** | 도면을 region 단위 (Title Block / View / Notes / Table / Figure) 로 분할하는 첫 단계 — Detection 작업. |
| **Stage 2** | 각 region 안의 세부 정보 (15 fields / GD&T / Measure / Roughness) 를 추출하는 두 번째 단계 — Parsing 작업. |
| **Detection (객체 검출)** | 이미지에서 객체의 위치를 찾는 작업. 출력 = bounding box 좌표 + class. "어디에 무엇이 있는가?" 까지만 답함. |
| **Parsing (내용 해석)** | 검출된 영역의 내용을 구조화 데이터로 변환. 출력 = JSON / text. "그 안에 무슨 정보가 있는가?" 까지 답함. |
| **Hybrid Routing (하이브리드 라우팅)** | 영역별로 가장 잘하는 small 모델을 분배하는 architecture — 단일 거대 모델 대신 4가지 model 을 routing. |
| **VLM (Vision Language Model)** | 이미지와 텍스트를 동시에 입력으로 받아 텍스트를 출력하는 멀티모달 모델 — Qwen3-VL, Florence-2, Donut 등. |

### B. Detection 모델 (YOLO 모드)

| 키워드 | 한 줄 정의 |
|---|---|
| **det (axis-aligned)** | 가로/세로축에 평행한 직사각형 bbox (회전 X). 출력 = (x_center, y_center, width, height) + class. |
| **obb (Oriented Bounding Box)** | 회전 가능한 직사각형 bbox. 출력 = (x_center, y_center, width, height, **θ**) + class. 회전된 GD&T symbol 검출에 정확. |
| **Region (region 단위)** | YOLO 가 검출한 도면의 시각적 분할 단위 — Stage 1 의 출력 단위 (1 도면 = 평균 4.4 region). |

### C. 데이터 / Annotation

| 키워드 | 한 줄 정의 |
|---|---|
| **Annotation (라벨링)** | 도면 이미지에 대한 정답 JSON (Ground Truth) 작성 작업 또는 그 결과물. |
| **Patch (Khan paper 단위)** | Khan paper 의 region crop 이미지 + JSON pair 단위 — 본 프로젝트 region 단위와 동일 개념, 1 도면당 평균 8.4 patches. |
| **Single GT annotation** | Multi-TB 도면 (TB 가 여러 개) 인데 정답 JSON 은 1개만 있는 데이터 구조 결함 — 4-Cause Cause 1 (30%). |
| **training_gt_v2 / v3** | V.E 단계 annotation 데이터셋 — v2 는 V.B Qwen output 84% 혼재, v3 는 Gemini Flash 100% 재정제 (V.G v3 학습 base). |

### D. 학습 / 추론

| 키워드 | 한 줄 정의 |
|---|---|
| **Training (학습)** | 데이터셋으로 모델 weight 를 업데이트하는 단계 — gradient backpropagation 계산. |
| **Fine-Tuning (FT, 미세조정)** | 사전학습된 모델을 도메인 데이터셋으로 추가 학습하는 작업 — Donut FT, Qwen3-VL FT 등. |
| **LoRA (Low-Rank Adaptation)** | Full FT 대신 작은 rank 행렬만 학습하는 PEFT 기법 — 본 프로젝트 r=8/16/32 사용. |
| **Zero-shot** | 학습 없이 사전학습 모델만으로 추론하는 방식 — 본 프로젝트 Nemotron-OCR-v2 가 zero-shot. |
| **Inference (추론)** | 학습된 모델로 새 입력에 대한 예측을 생성하는 단계 — 학습(training) 의 반대 개념. |
| **Val_loss (Validation Loss)** | 학습 중 모델이 보지 못한 holdout validation set 에서 측정한 loss — Overfitting 감지 + best epoch 선택 기준. |

### E. 한계 / 결함 현상

| 키워드 | 한 줄 정의 |
|---|---|
| **Memorization (암기 현상)** | 학습된 모델이 학습 데이터의 mode 토큰을 외워서 새 입력에도 그대로 출력하는 현상 — 정답지만 외운 학생 비유. |
| **Mode Collapse** | 모델 출력이 다양성 잃고 특정 패턴으로 수렴하는 현상 — GD&T 14종이 모두 ⊥ 로 출력되는 V.B Stage 2 사례. |
| **학습 데이터 Mode 출력** | 학습 데이터에 자주 등장한 특정 토큰 (PACRAFT 26회, 弁棒 254회 등) 을 모델이 그대로 반복 출력하는 현상 — 4-Cause Cause 2 (40%). |
| **Source contamination** | Hallucination 으로 인한 동일한 잘못된 출력이 학습 데이터 source 자체에 박혀 다음 phase 로 전파되는 현상 — 4-Cause Cause 3 (20%). |
| **Hallucination** | 도면에 없는 내용을 모델이 만들어내서 출력하는 현상 — V.A Tesseract 72%, Khan baseline Donut 10.8%. |
| **Schema 강제력** | Model 이 structured JSON 출력을 보장하는 정도 — Donut (task token) 은 강함, Florence-2 (prompt-conditioned) 는 약함. |

### F. 평가 & 분석

| 키워드 | 한 줄 정의 |
|---|---|
| **Pro Judge** | Gemini-2.5-Pro 를 외부 평가자로 사용하는 본 프로젝트 평가 framework — 10 sample × 영역별 0-100 점수 + Verdict. |
| **PASS / SUSPECT / FAIL (Verdict)** | Pro Judge 판정 — 모두 60+ = PASS, 일부 30-60 = SUSPECT, 30 미만 또는 mode 출력 = FAIL. |
| **4-Cause Root Analysis** | TB 영역 미달 원인을 4가지로 정량 분해한 분석 — Cause 1 (Multi-TB 30%) + 2 (Mode 40%) + 3 (Source 20%) + 4 (Mixing 10%). |
| **Fundamental Insight** | 본 프로젝트 110일간 박제된 7가지 본질적 학습 — Data Quality > Model Size, Hybrid > Single, Memo fix architecture 종속 등. |

---

## TL;DR (AI 전문가용 30초 요약)

- **목표**: 2D 도면 → 구조화 JSON (Title Block 15 fields + Notes + View annotations + Table)
- **운영 모델**: **V.F-2 LOCAL Hybrid Routing (3.5B 합계)** — YOLOv11 + Qwen3-VL-2B FT + Donut FT + Nemotron-OCR-v2 Zero-shot
- **Pro Judge 결과**: TB 46 / **Notes 65 ★ PASS 60 도달** / View 15 / Memorization 60% — Verdict FAIL 10/10 but PASS 가장 근접
- **핵심 evidence**: V.B Teacher 30B 합산 81점 vs V.F-2 LOCAL 3.5B 합산 111점 → **Hybrid 가 Single 30B 능가** 입증
- **이론 base**: Khan et al. (2024/2025/2026), Donut (ECCV 2022), Florence-2 (Microsoft 2023)
- **하드웨어**: RTX 5080 (Blackwell sm_120) + i9-13900K + WSL2 Ubuntu / Python 3.13 + uv venv / torch 2.11+cu128

---

## 1. 프로젝트 개요

본 시스템은 일본/한국 가공·조립용 2D 도면 (KO / EN / JP / 한자 / 기호 mixed) 을 입력받아 다음 schema 의 JSON 을 생성한다:

```python
{
  "schema_version": "v_f_2_local-1.0",
  "drawing_id": "...",
  "title_block": { "Drawing_No": "...", "Material": "...", ...15 fields },
  "notes": [ { "lines": [...] } ],
  "views": [ { "measures": [...], "gdt": [...], "roughness": [...] } ],
  "tables": [ { "rows": [...] } ]
}
```

최종 사용처는 IMMA RAG — 도면 메타 (Drawing_No / Material / Company_Name) 가 검색 키이고, View 의 GD&T / Measure / Roughness 가 가공 매칭 raw data. 본 프로젝트 V.A 초기부터 V.H' 까지 약 110일간의 phase 별 진화 박제.

---

## 2. Quick Stats (정량 핵심)

| 지표 | 값 |
|---|---|
| 데이터셋 | Roboflow JP 1,500 도면 (일본어 68.7%) |
| V.B Phase C parse-success | 1,469 / 1,500 (97.93%) |
| V.B Phase C 평균 sections/도면 | 7.17 |
| V.E annotation training_gt_v2 | 4,730 train + 1,252 val |
| V.G training_gt_v3 | 4,400 records (View+Notes Gemini Flash 재정제) |
| Stage 1 mAP@0.5 | **0.86** (YOLOv11 5-class, V.A ckpt) |
| 운영 모델 합계 파라미터 | 약 3.5B (Qwen3-VL-2B + Donut 143M + YOLOv11) |
| Pro Judge 최고 (Notes) | **65** ★ PASS 60 도달 |
| Memorization 최저 (V.G v3) | **9%** ★ (V.B Teacher 30B 90% 의 1/10) |
| 추론 latency | 약 10~15초/도면 (병렬 시 ~7초) |
| 1000 도면 처리 비용 | 약 $5 (All-API $30 대비 1/6) |

---

## 3. 운영 모델 = V.F-2 LOCAL Hybrid Routing

영역별로 가장 잘하는 모델을 분배하는 4-way Hybrid:

```
[Input 도면 PNG/JPG]
        │
        ▼
[Stage 1] YOLOv11 5-class Region Detection (V.A ckpt, mAP 0.86)
        │
        ├──→ Title Block region
        │         ├──→ Donut FT (Swin-B + BART, 영문/숫자) → 46 ★
        │         └──→ Nemotron-OCR-v2 Zero-shot (한자 후처리, JP NED 0.046)
        │
        ├──→ Notes region   ──→ Qwen3-VL-2B FT (Unsloth, LoRA r=8) → 65 ★ PASS
        │
        ├──→ View region    ──→ Donut FT (동일 ckpt, region prompt 만 다름) → 15
        │
        └──→ Table region   ──→ Qwen3-VL-2B FT (BOM rows)
                                 │
                                 ▼
                    [Hybrid JSON Output, schema v_f_2_local-1.0]
```

### Hyperparameter 정리

| 영역 | 모델 | LoRA | Optimizer | Epochs / Records |
|---|---|---|---|---|
| Notes | Qwen3-VL-2B-Instruct | r=8, α=16, dropout 0.1 | AdamW, lr 2e-4, bf16 | 2 epochs / training_gt_v2 |
| TB + View | Donut (naver-clova-ix/donut-base) | r=32, α=64, dropout 0.05 | AdamW, lr 2e-4, bf16 | 15 epochs / val 0.6596 |
| Detection | YOLOv11 (V.A ckpt) | — | — | 50 epochs / Roboflow JP 1,500 |
| TB 한자 | Nemotron-OCR-v2 (NVIDIA) | Zero-shot | — | — |

운영 채택 사유: 8개 phase 중 **TB+Notes 합산이 가장 높음** (46 + 65 = 111 vs V.B 30B 의 35.9 + 45.1 = 81). V.G v3 가 Memo 9% 로 best 이지만 TB 3.2 / Notes 18.6 운영 불가.

---

## 4. 데이터셋

| 단계 | 데이터셋 | 수량 | 비고 |
|---|---|---|---|
| V.A 초기 | Roboflow 자체 수집 | 5,839 JPG | EN/KO/JP/RU/CN/DE mixed |
| V.A Stage 1 | IMMA.v1i.yolov11 (Roboflow seed) | 100장 (80/20) | mAP 0.9364 |
| V.B | Roboflow JP 1,500 | unique 1,515 / Avg 1.41× | 일본어 68.7% |
| V.B Phase C | Teacher 30B 자동 라벨링 | 1,469 (97.93%) | 평균 7.17 sections |
| V.E annotation | training_gt_v2 | 4,730 + 1,252 | Gemini Flash 16% + V.B Qwen 84% |
| V.G training_gt_v3 | View+Notes 재정제 | 4,400 records | Gemini Flash $0.38, Cause 2 해결 |

데이터셋 정책 (D-024): group-aware split 필수 (`filename.split('.rf.')[0]` → `GroupShuffleSplit`), data leak 방지. 46 images / 18 groups (조립도면) 제외 → 잔여 5,793 images / 2,991 groups.

### Khan paper 비교

| 항목 | 본 프로젝트 | Khan paper |
|---|---|---|
| 도면 수 | 1,500 / 1,469 | 1,367 |
| 도면당 patch | 4.4 (region 단위) | 8.4 (annotation 단위) |
| Total patches | 약 6,500 region | 11,469 patches |
| Numerical F1 | 0.3786 (V.A V.6) | 0.963 (Khan 2026) |

→ Khan 과의 fundamental gap = patch 단위 annotation 부재 (V.A KNOWN_LIMITATIONS).

---

## 5. Stage 1 + Stage 2 Pipeline

### Stage 1 — Region Detection (YOLOv11 5-class)

- **모델**: `checkpoints/yolo_det.pt` (40.7 MB), V.A D-001~D-058 학습
- **클래스**: Title Block / View / Notes / Table / Figure
- **성능**: mAP@0.5 = 0.86, precision 0.91, recall 0.89, F1 0.90
- **출력**: axis-aligned bbox + class

V.B Day 1 에 multi-view crop 모듈 (`preprocess_split_views_ver.B.py`, 363줄) 을 시도했으나 V.E 이후 폐기. YOLO26 OBB 학습 데이터 부족과 4-Cause Cause 1 의 fundamental 한계 때문.

### Stage 2 — Annotation Extraction (VLM-based)

각 region 안의 세부 정보 추출:

| Region | 추출 대상 | Khan 카테고리 매핑 |
|---|---|---|
| Title Block | Drawing_No, Material, Mass, Scale, Engineer, Designer, Company_Name 등 15 fields | §3.5 Title Block + §3.3 Material |
| View | GD&T 14 Unicode symbols, Measures (치수 + tolerance), Surface Roughness, Radii, Thread | §3.1 GD&T + §3.2 General Tolerances + §3.4 Measures + §3.6 Radii + §3.7 Roughness + §3.8 Thread |
| Notes | Structured list (표면조도 / 열처리 / 지시사항) | Notes |
| Table | BOM rows | §3.3 General Tolerances (확장) |

### View 영역 PASS 미달의 fundamental 원인

V.A 시점 데이터셋의 절대 부족:

| 카테고리 | 본 프로젝트 V.A 비율 | Khan 목표 | Gap |
|---|---|---|---|
| GD&T | 2.6% (auto-fill 0.2%) | 30%+ | -27.4pt |
| Roughness | 11.2% (fallback 18.4%) | 60%+ | -48.8pt |
| Measure | 86.2% | 100% | -13.8pt |

→ V.B Stage 2 학습 결과 GD&T 14종 중 ⊥ 만 7% 정확, 나머지 13종 mode collapse → ⊥. V.G v3 가 fix 시도해도 View Pro Judge 17.7 (PASS 60 미달).

---

## 6. 버전별 진화 (V.A → V.H')

### V.A (D-001~058) — Khan 2025 재현 시도
YOLOv11-det + YOLOv11-obb + Donut x2. Stage 1 mAP 0.9364, but Stage 3-N **numerical 3.43% / hallucination 72%**. 종료 사유 = D-050 Tesseract auto-fill GT 의 noise 가 Donut FT 에 학습되어 paper 96.3% 대비 28배 gap.

### V.B (D-059~085) "Pragmatic Closure"
Teacher Qwen3-VL-30B-A3B-Instruct-FP8 직접 production. Phase C 1,469 자동 라벨링 97.93% 성공. Student distillation (Qwen2.5-VL-7B LoRA r=32) 시도 → eval_loss 4.03 plateau (D-083) → distillation 폐기. **Pattern 11~17** (Material 오인, view 영역 혼합, view 완전 중복, Notes copy 등) 발견. V.B Teacher Pro Judge = TB 35.9 / Notes 45.1 / View 11.8 / Memo 90%.

### V.D (D-086~091) — Prompt + Inference fix 시도
Lost in the Middle hypothesis 기반 Mirror prompt (`teacher_prompts_v_d_task2_ver.D.py`). Task 3 (FP8 → bf16 + positive overcorrection) overnight 200 sample 검증: V.B 진짜 baseline = view 52.3% / table 38.0% 확인. Task 2/3 모두 marginal → prompt level fix 한계 도달, **Qwen VL family domain weakness 확정**.

### V.E (D-092~099) — Annotation Quality 명제
> "V.A 실패 원인은 architecture 가 아니라 annotation quality"

V.E-1 Hybrid (사용자 spot-check + GPT-4o, 41% pass) STOP → **V.E-4-flash** (Gemini 2.5 Flash, $0.38). 4가지 annotation 원칙 — OCR auto-fill 금지 / Human verify ≥1 / 빈 cell `null` vs `""` 구분 / Pydantic strict. **Donut FT v2** 학습 (Swin-B + BART, LoRA r=32, lr 2e-4, 15 epochs, val 0.6596). 결과: TB 영문 75 / 한자 13.9 (Donut Swin-B ceiling) / View 60. 한자 한계 발견 → V.F Hybrid 진입.

### V.F (D-100~108.1) — Hybrid Routing 완성
- **Donut FT v2** (영문 TB / View symbol) + **Nemotron-OCR-v2** (한자) + **Qwen3-VL-2B FT** (Notes / view 자연어)
- V.F-1 LoRA r=64 → Memorization 100% → V.F-2 LoRA r=8 + dropout 0.1 + 2 epochs → V.F-2 LOCAL (RTX 5080 재학습, eval_loss 0.180, A100 0.353 대비 -49%)
- **V.F-2 LOCAL Pro Judge**: TB 46 / Notes 65 ★ / View 15 / Memo 60%
- V.F v2 (LoRA r=16 + augmented + 4 epoch) **실패**: placeholder fake "Company A/B Inc." → TB 42→28 악화

### V.G (D-109~) — Mode Collapse Fundamental 해결 시도
Florence-2-base (DaViT + BART, 0.23B) + training_gt_v3 (Gemini Flash 재정제). HF blog finetune-florence2 + Roboflow Issue #162 best config. **LoRA r=8, α=16, dropout 0.05, use_rslora=True, lr 1e-6, AdamW + linear (no warmup), bf16, 7 epochs, vision encoder freeze**. v3 결과: TB 3.2 / Notes 18.6 / **View 17.7 ★** / **Memo 9% ★★★** (V.F-1 100% → 9%, 극적 개선). 단 TB/Notes 운영 불가 → V.F-2 LOCAL 유지.

### V.H' (D-110.7) — V.F-3 + V.G v3 통합 시도 (마지막 phase)
Donut TB val_loss 0.4433 (V.E v2 0.6596 대비 -33% 개선) BUT Pro Judge TB **1.5** (V.F-2 LOCAL 46 의 1/30). Memorization 70% 회귀 — Continue training (LoRA on base ckpt) 으로 base 의 memorization 이 LoRA 위에 잔존. 본 프로젝트의 마지막 phase, **운영 채택 V.F-2 LOCAL 변경 없음**.

---

## 7. Pro Judge 평가 framework

### 평가 방식

- **평가 모델**: Google Gemini-2.5-Pro (외부 평가자)
- **입력**: 도면 이미지 1장 + 본 프로젝트 출력 JSON
- **샘플 수**: 10 도면 (다국어 분포)
- **비용**: 10 sample 당 약 $0.30~0.50

### 영역별 채점 (0-100점)

| 영역 | 평가 대상 |
|---|---|
| **Titleblock** | Drawing_No / Title / Material / Mass / Scale / Date / Engineer 등 15 fields, 실제 도면 영역과 비교 |
| **Notes** | 일본어/한자/한글 OCR 정확도, Pattern 14 (다른 notes 영역 복사) 확인 |
| **View** | 치수 (measures) / 기하공차 14종 / 표면 거칠기, Pattern 12 (다른 view 의 measure 혼입) 확인 |

### Verdict 판정

| Verdict | 기준 |
|---|---|
| **PASS** | 3 영역 모두 score 60+ |
| **SUSPECT** | 일부 영역 30~60 사이 |
| **FAIL** | score 30 미만 또는 mode token (PACRAFT, 弁棒, P110-8302) 의심 |

### Memorization Check

각 sample 마다 학습 데이터 mode token 의심 여부 (titleblock_mode / notes_mode / view_pool) Boolean 판정. 전체 10 sample 중 mode 의심 비율 = Memorization Rate.

---

## 8. Pro Judge / Memorization 종합 비교 (8 phase)

| Phase | TB | Notes | View | Memorization | Verdict |
|---|---|---|---|---|---|
| V.A (Tesseract+YOLO+Donut) | — | — | — | — | numerical 3.43% / hallucination 72% |
| V.B Teacher 30B (zero-shot FP8) | 35.9 | 45.1 | 11.8 | 90% | FAIL 10/10 |
| V.D (prompt fix) | — | view 56.8% recall | table 30.7% | — | marginal |
| V.E v2 (Donut FT, training_gt_v2) | 75 영문 / 13.9 한자 | 75 | 60 | — | Hybrid 구성요소 |
| **★ V.F-2 LOCAL (3.5B Hybrid)** | **46** | **65 ★ PASS** | 15 | 60% | FAIL 10/10 (★ 운영 채택) |
| V.F v2 (r=16 + augmented) | 28 ⚠ | 60 | 20 | 56% | placeholder fake 실패 |
| V.G v1/v2 (Florence + training_gt_v2) | 0~3.2 | 15~18.6 | 10 | 50% | FAIL 11/11 |
| **★ V.G v3 (Florence + training_gt_v3)** | 3.2 | 18.6 | **17.7 ★** | **9% ★★★** | FAIL 11/11 (Memo 1위) |
| V.H' (V.F-3 + V.G v3 통합) | 1.5 | 59.0 | 5.5 | 70% | FAIL 10/10 (마지막) |

**영역별 1위**:
- TB → V.F-2 LOCAL 46 (V.B Teacher 30B +10.1)
- Notes → V.F-2 LOCAL 65 (V.B Teacher +19.9, **★ PASS 60 도달**)
- View → V.G v3 17.7
- Memorization → V.G v3 9%

---

## 9. 4-Cause Root Analysis (TB 미달 정량 분해)

TB Pro Judge 가 PASS 60 미달인 원인 4가지 (D-110.3 박제):

| Cause | 비율 | 원인 | 영향 |
|---|---|---|---|
| **Cause 1: Multi-TB + Single GT** | 30% | 한 도면에 TB 가 여러 개인데 GT 는 1개만 | V.B YOLO26 outer bbox 만 |
| **Cause 2: 학습 데이터 Mode 출력** | 40% | training_gt_v2 의 84% V.B Qwen output. PACRAFT 26회, 弁棒 254회 등 mode token 잔존 | annotation 한계 |
| **Cause 3: View Source Contamination** | 20% | V.C v_b_phase_c (Teacher 30B inference) 의 mode collapse 가 view 학습 데이터로 전파 | source 한계 |
| **Cause 4: Annotation Source 혼재** | 10% | Gemini Flash 16% + V.B Qwen 84% mapping 모호 | mixing 한계 |

**핵심**: Cause 1+2 = 70% 가 dataset/annotation 의 fundamental issue. Architecture 변경으로 해결 불가능. → 향후 데이터셋 변경 (Khan paper Multi-TB 별도 annotation) 이 fundamental fix.

---

## 10. 7가지 Fundamental Insight

### ★★★ 1. Data Quality > Model Size
V.B Teacher 30B Memo 90% vs V.G v3 0.23B + training_gt_v3 Memo **9%**. 10배 차이. 작은 모델 + 정제 데이터 > 큰 모델 + 노이즈.

### ★★★ 2. Hybrid Routing > Single Model
V.F-2 LOCAL 3.5B 합산 111 vs V.B 30B 합산 81 (+30pt). **30B 도 못 한 Notes PASS 60 도달**.

### ★★ 3. Memorization Fix 의 Architecture 종속성 (V.H' 신규)
V.G v3 (from-scratch + training_gt_v3) Memo 9% vs V.H' (continue training + 동일 training_gt_v3) Memo 70%. **같은 data 인데 학습 방식만 달라 7.8배 차이**. → "Data fix 만으로 부족, from-scratch + data 동시 필요".

### ★★ 4. Val_loss ≠ Inference Quality (V.H' 신규)
V.H' Donut TB val_loss 0.4433 (V.E v2 -33% 개선) BUT Pro Judge TB 1.5 (V.F-2 LOCAL 46 의 1/30). 원인: 학습량 부족 + new schema token 50개 + continue training. → "Training metric 의 inference quality 부정확성 정량 입증".

### ★ 5. 학습 데이터 Quality 가 Mode Collapse 의 Fundamental
V.F-1/V.F-2/V.G v1/V.G v2 모두 동일 training_gt_v2 사용 → 모두 mode collapse. Architecture 변경이 해결 못한 이유 = 학습 데이터 동일. 해결 = training_gt_v3 재정제.

### ★ 6. View 영역 본질 한계 = V.A 데이터셋 Chain 누적
V.A GD&T 2.6% + Roughness 18.4% → V.B YOLO26 outer bbox → V.E LIST 단위 → V.E v2/V.F-2/V.G v3 REGION 단위 학습 → **모든 phase View Pro Judge 5.5~17.7 (PASS 60 미달)**.

### ★ 7. TB 4-Cause Root Analysis
§9 참조 — TB 미달은 4 cause 정량 합 (30/40/20/10%), 모델 한계 X 데이터 한계.

---

## 11. 미해결 결함 + 향후 개선 방향

### 미해결 결함 4가지

1. **TB Pro Judge 46 (PASS 60 미달, -14pt)** — Cause 1+2 = 70% dataset fundamental
2. **View Pro Judge 최고 17.7 (PASS 60 미달, -42.3pt)** — V.A 데이터셋 chain 한계
3. **TB 한자 13.9 ceiling** — Donut Swin-B pretrained ~1000~2000 종 vs 도면 5000~8000 종. Nemotron 검증 D-103 보류.
4. **Continue Training 의 Memorization 회귀** — LoRA continue 로는 base ckpt memorization 잔존. From-scratch 필요.

### 향후 개선 방향

**1순위 — 데이터셋 변경 (fundamental fix)**:
- GD&T 라벨 추가 (2.6% → 30%+ direct annotation)
- Roughness Ra direct annotation (18.4% fallback → 60%+ direct)
- Multi-TB 도면 별도 처리 (Cause 1 해결)
- Patch 단위 image-text pair 생성 (Khan paper 11,469 patches base)
- 예상 비용: CVAT 수동 annotation + domain expert 2명 verify

**2순위 — Khan Numerical VLM (V.H Option B)**:
- 코드 박제 완료 (D-110.6)
- Auto OBB labeling (Gemini Pro Vision $14.43, ~12h) + YOLO26-obb 학습 (~1-2h) + Donut Numerical-specific FT (~3-4h)
- 예상 View 17.7 → 35-50 (Khan F1 0.963 reference)

**3순위 — V.H' learnings 적용**:
- Sample-level inference verify 필수
- Florence-2 + from-scratch + 충분한 학습량 권장
- Nemotron-OCR-v2 검증 (TB 한자 13.9 → 95% 개선 기대, JP NED 0.046)

---

## 12. 기술 스택

| Layer | Stack |
|---|---|
| OS | Ubuntu 22.04 LTS on WSL2 (Windows 11 host) |
| GPU | **NVIDIA RTX 5080 (Blackwell sm_120, 16 GB VRAM)** + Colab Pro+ A100 40 GB |
| CUDA | **12.8** (sm_120 호환 필수, cu124 비호환 D-030) |
| Python | **3.13.11** (V.F~V.H 기준) |
| Env manager | **uv** (lockfile 보존: `.venv-vf` / `.venv-vb` / `.venv-paddleocr` 분리) |
| Deep learning | **torch 2.11+cu128**, transformers 5.x, peft, **Unsloth**, bitsandbytes 0.49.x |
| Detection | **ultralytics** YOLOv11 + YOLO26 + YOLOv11-obb + 5-Fold Ensemble |
| VLM | Donut (clovaai), Qwen3-VL-2B-Instruct, Qwen3-VL-30B-A3B-Instruct-FP8, Florence-2-base, Nemotron-OCR-v2 |
| Inference | vLLM 0.20.1 (Teacher 30B FP8), transformers (Donut/Florence-2), Unsloth (Qwen3-VL FT) |
| OCR | PaddleOCR-VL-1.5 (0.9B subprocess wrapper, OmniDocBench 94.5%), Tesseract (보조 only) |
| API server | **FastAPI** + **Cloudflare Tunnel** (Server_VF demo) |
| Schema | Pydantic strict (V.B `schema_ver.B.py`) |
| 평가 | Gemini-2.5-Pro (Pro Judge) |
| Annotation | Gemini-2.5-Flash ($0.38 total), CVAT (수동 검수) |

---

## 13. 디렉토리 구조

```
Drawing/
├── README.md                          ← 본 문서 (통합본)
├── README_ver.A/B/D/E/F/G.md         ← 버전별 baseline 박제
├── PROJECT_HANDOFF_ver.A/B/D/E/F.md  ← 의사결정 D-001~D-110.7
├── src/
│   ├── stage1_layout.py / stage2_annotation.py / ensemble_predict.py
│   ├── stage3_alphabetical.py / stage3_numerical.py
│   ├── pipeline.py (V.A) / pipeline_ver.B.py (V.B Stage 0~6)
│   ├── stage5_enrichment.py (4-tier cascade)
│   ├── teacher_prompts_ver.B/D.py
│   ├── preprocess/multi_drawing_detector_ver.B.py
│   ├── inference/teacher_ver.B.py / student_ver.B.py / nemotron_ocr_inference_ver.F.py
│   ├── annotation/annotation_pipeline_ver.E.py
│   ├── donut/donut_ft_ver.E.py
│   ├── training/florence2_ft_ver.G.py / qwen3vl_ft_ver.F.py / donut_ft_v_h_prime_ver.G.py
│   ├── validate/{check_*, schema_ver.B}.py
│   └── utils/metrics.py
├── docs/
│   ├── V_B/V_D/V_E/V_F/V_G/V_H_*_ver.*.md
│   ├── V_F_2_LOCAL_FINAL_DECISION_ver.G.md
│   ├── V_H_PRIME_PRO_JUDGE_RESULTS_ver.G.md
│   ├── PROJECT_FUNDAMENTAL_INSIGHTS_ver.G.md (7 insight 종합)
│   ├── RESEARCH_REFERENCES_CONSOLIDATED_ver.G.md (100여개 reference)
│   ├── KNOWN_LIMITATIONS_ver.A/B/D/F.md
│   └── presentation_assets/
│       ├── V_A_TO_V_G_Presentation_ver.G.docx (17 슬라이드, 15분)
│       ├── V_A_TO_V_G_Presentation_Script_5min_*.docx
│       └── svgs/ (24개 — pipeline / architecture / 4-cause / comparison 등)
├── Server_VF/                         ← V.F-2 LOCAL 운영 demo (FastAPI + Cloudflare Tunnel)
│   ├── server.py + start_server.sh + start_tunnel.sh
│   ├── README.md + manual.md
│   └── requirements.txt
├── Server_VB/                         ← V.B Web Service 팀 hand-off package
├── colab_sync/                        ← Colab Pro+ A100 sync
│   ├── src/ (PC ↔ colab 양방 sync, B.14.3 정책)
│   ├── dataset/JP/images/
│   ├── outputs/{phase1_instruct_phaseB, v_e_annotation, v_e_donut_ft, v_f_qwen3vl_ft, ...}/
│   └── colab_*.ipynb
├── checkpoints/                       ← yolo_det.pt (40.7MB) + yolo_obb_runs/ 5 fold + donut_numerical/
├── outputs/
│   ├── v_g_florence2_local_train_v1/v2/v3/
│   ├── v_f_2_local_retrain/
│   ├── v_f_v2_local_train/
│   ├── v_e_annotation/training_gt_v2/
│   └── auto_labels/ / exclude_groups_manifest.csv
├── dataset/                           ← 5,839 JPG (V.A) / Roboflow JP 1,500 (V.B)
├── IMMA.v1i.yolov11/                  ← Roboflow seed 100장
├── dataset_excluded/                  ← 조립도면 46 images / 18 groups
├── configs/{yolo_det, yolo_obb, donut_numerical, validation_thresholds}.yaml
├── data/{raw, layout, annotation, vlm, sample_enriched, validation_gt}/
└── articles/                          ← Khan 2024/2025/2026, Donut, Florence-2 등 ~231 MB
```

---

## 14. 주요 Reference

### 학술 논문 (핵심 7개)

| 출처 | 핵심 |
|---|---|
| **Khan et al. 2024** — Florence-2 GD&T (arXiv 2411.03707) | Florence-2 FT vs GPT-4o → F1 +52.4%, hallucination -43.15% (V.G baseline) |
| **Khan et al. 2025** — Hybrid VL Framework (arXiv 2506.17374) | Donut 89.2% precision / 94% F1 (V.E v2 채택 정당화) |
| **Khan et al. 2026** — Multi-Stage Hybrid Numerical (arXiv 2510.21862) | Numerical F1 0.963 (V.H Option B 목표 baseline) |
| **Donut** (Kim et al., ECCV 2022, arXiv 2111.15664) | Swin-B + BART OCR-free document understanding |
| **Florence-2** (Microsoft 2023, arXiv 2311.06242) | DaViT + BART, FLD-5B pretraining (V.G architecture) |
| **VLM Posterior Collapse** (arXiv 2502.16842) | V.F view 5.5 fail 의 본질 원인 |
| **Lost in the Middle** (TACL 2024, arXiv 2307.03172) | V.D Task 1 FAIL 의 dominant 원인 |

### 도구 / 라이브러리

| 출처 | 용도 |
|---|---|
| **roboflow/maestro Issue #162** | Florence-2 best LoRA config (mAP50 0.20→0.71, V.G hyperparameter 100% 채택) |
| **HF blog "Fine-tuning Florence-2"** | 공식 lr 1e-6, vision freeze, AdamW + linear (V.G base) |
| **Unsloth Qwen3-VL docs** | Qwen3-VL-2B FT 가속 + 4-bit LoRA |
| **OmniDocBench** (CVPR 2025) | PaddleOCR-VL-1.5 94.5% CJK SOTA (V.A D-039 채택) |
| **NVIDIA Nemotron-OCR-v2** | JP NED 0.046 (Donut 0.86 대비 95% 개선) |
| **vLLM #41985 + #27157** | CJK token-level sampling penalty (D-071 근거) |

전체 reference (학술 31개 + HF 모델 13개 + GitHub 30+ + 기술 blog 22개 등 약 100개): `docs/RESEARCH_REFERENCES_CONSOLIDATED_ver.G.md` 참조.

---

## 15. 학습 환경 + 운영 가동

### 학습 환경

| 항목 | 값 |
|---|---|
| GPU | RTX 5080 Blackwell sm_120 (16 GB) + Colab Pro+ A100 40 GB |
| CPU / RAM | Intel i9-13900K / 128 GB |
| OS | Ubuntu 22.04 on WSL2 |
| Python | 3.13.11 (V.F~V.H), 3.10+ (V.A) |
| Env | `uv venv .venv --python 3.13` |
| CUDA | 12.8 (Blackwell 호환 필수) |
| 학습 시간 (RTX 5080) | YOLOv11 5-class: 28.5분 / Donut FT v2: ~30분 / Qwen3-VL-2B LoRA r=8: ~60분 / Florence-2 7 epochs: ~7-8h overnight |
| VRAM 점유 | Donut FT ~2GB / Qwen3-VL-2B LoRA ~4-10GB / Florence-2 vision freeze + bf16 ~10-12GB |
| 안정화 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`, nohup + nice + taskset, watcher polling |

### 운영 가동 (Server_VF)

```bash
# 학습 venv 활성화 (V.F-2 LOCAL 학습 시 사용한 .venv-vf)
source /mnt/c/Users/user/github/Drawing/.venv-vf/bin/activate

# Terminal 1 — FastAPI
cd /mnt/c/Users/user/github/Drawing/Server_VF
bash start_server.sh         # uvicorn server:app on 0.0.0.0:8000

# Terminal 2 — Cloudflare Tunnel
bash start_tunnel.sh         # 출력의 https://xxxx.trycloudflare.com URL → IMMA 측 전달
```

- Endpoint: `POST /analyze` (multipart image)
- Response schema: `v_f_2_local-1.0`
- Latency: 약 10-15초/도면 (병렬 ~7초)
- 비용: 1000 도면 처리 ~$5 (Nemotron API + 전기료)

상세: `Server_VF/README.md` + `Server_VF/manual.md`

---

## 16. 결론

본 프로젝트는 V.A 의 Khan 2025 직접 재현 시도 (numerical 3.43% 실패) 부터 V.H' 의 통합 시도 (Val_loss 0.44 → Pro Judge 1.5 회귀) 까지 110일간 8개 phase 의 박제다. 최종 운영 채택 **V.F-2 LOCAL** 은 3.5B 합산으로 V.B Teacher 30B 를 모든 영역에서 능가하며 (TB +10.1, Notes +19.9), Notes 영역은 PASS 60 도달한 유일한 모델이다.

남은 7가지 fundamental insight 와 미해결 결함 4가지는 모두 dataset 과 학습 방식의 fundamental issue 로 확인됨 — Architecture 변경으로 해결 불가능. 향후 개선은 Khan paper Multi-TB + Patch 단위 데이터셋 재구축 (1순위) + Khan Numerical VLM (V.H Option B, 코드 박제 완료) 적용이 fundamental fix.

본 README 와 함께 다음 문서를 권장:
- 운영 모델 선정 근거: `docs/V_F_2_LOCAL_FINAL_DECISION_ver.G.md`
- 7가지 insight 종합: `docs/PROJECT_FUNDAMENTAL_INSIGHTS_ver.G.md`
- V.H' 최종 결과: `docs/V_H_PRIME_PRO_JUDGE_RESULTS_ver.G.md`
- 100여개 reference: `docs/RESEARCH_REFERENCES_CONSOLIDATED_ver.G.md`
- 발표 자료 (15분 + 5분): `docs/presentation_assets/V_A_TO_V_G_Presentation*.docx`
- 서버 운영 가이드: `Server_VF/README.md` + `Server_VF/manual.md`

---

*본 README 는 V.A → V.H' 8개 phase 의 6개 README (ver.A/B/D/E/F/G) 통합본이다. AI 전문가가 처음 본 프로젝트를 이해할 수 있도록 핵심 키워드 사전 (§0) + 정량 수치 + Hyperparameter + 결정 근거를 모두 포함.*

*마지막 업데이트: D-110.7 — V.F-2 LOCAL 운영 채택 확정.*
