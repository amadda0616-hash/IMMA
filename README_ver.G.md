# Engineering Drawing OCR — Version G (Florence-2 Phase)

> **작성일**: 2026-05-16 (D-109, V.G phase 시작 직전 새벽)
> **선행 phase**: V.F (Qwen3-VL-2B FT, D-100 ~ D-108.1) — mode collapse fundamental 한계 확정 후 종료
> **신규 phase**: V.G (Florence-2 + V.E annotation 재활용)
> **목표**: VLM posterior collapse 회피 + mode collapse fundamental 해결 + Khan paper baseline (F1 +52.4%) 달성

---

## §1 Project 진화 (V.A → V.G)

```
V.A (Florence-2 region + Donut OTSL + Tesseract, D-001~D-058)
    ↓ char_acc 0.69 < 0.85 (Tesseract OCR 한계, hallucination 72%)
V.B (Pretrained Qwen2-VL-7B prompt-only, D-061~D-091)
    ↓ Pattern 11~17 (mode collapse, GDT ⊥ collapse 7%, Notes 99% ★)
V.D (Prompt enhancement, D-088~D-091)
    ↓ PARTIAL (table 38%, view 52%, notes 99%)
V.E v2 (Donut FT + V.A YOLO 회귀 + Gemini annotation, D-092~D-099)
    ↓ Donut Swin-B 한자 OCR ceiling (titleblock 13.9%)
V.F-1/2 (Qwen3-VL-2B + Unsloth FT + Hybrid Routing, D-100~D-108)
    ↓ Notes 90%+ but Titleblock memorization 100% → mode collapse 60%
V.F v2 (LoRA r=16 + augmented + 4 epoch 시나리오 B, D-108.1)
    ↓ Augmented synthetic data quality 문제 (placeholder fake) → TB 28 (악화)
★ V.G (Florence-2 + V.E annotation 재활용, D-109~)
```

---

## §2 V.G phase 명제

> **Florence-2 (encoder-decoder + region prompt) + V.E annotation 재활용 으로 VLM posterior collapse 회피 + mode collapse fundamental 해결 + Khan paper baseline F1 +52.4% 달성**

### 2.1 V.F → V.G 핵심 변경

| 항목 | V.F (실패) | V.G (신규) |
|---|---|---|
| **Architecture** | Decoder-only LLM (Qwen3-VL-2B) | **Encoder-decoder (Florence-2-base, DaViT+BART)** |
| **모델 크기** | 2B | **0.23B (10x 작음)** |
| **Posterior collapse** | ★ 발생 (이미지 무시) | **회피** (encoder-decoder + region prompt) |
| **학습 데이터** | training_gt_v2_aug (placeholder fake) | **training_gt_v2 (원본, 4730 records)** |
| **Mode collapse** | 60% (V.F-2 LOCAL) ~ 56% (V.F v2) | **목표 < 40%** |
| **Hybrid 위치** | Notes only (Pro Judge 65) | TBD (학습 결과 후) |

### 2.2 V.G 핵심 출처 (3가지)

| 출처 | 핵심 |
|---|---|
| **[arxiv 2411.03707 Khan Florence-2 GD&T](https://arxiv.org/abs/2411.03707)** | Florence-2 FT vs GPT-4o → **F1 +52.4%, hallucination -43.15%** |
| **[Roboflow Issue #162](https://github.com/roboflow/maestro/issues/162)** | Florence-2 best LoRA config (mAP50 0.20 → 0.71, 3.5x 개선) |
| **[HF blog finetune-florence2](https://huggingface.co/blog/finetune-florence2)** | 공식 lr 1e-6, vision encoder freeze, AdamW + linear |

---

## §3 V.G phase 학습 계획 (Roboflow Issue #162 + HF 공식 통합)

### 3.1 학습 hyperparameter (검증 완료)

```yaml
model:
  base: microsoft/Florence-2-base  # 0.23B
  trust_remote_code: True
  torch_dtype: bfloat16

vision_encoder:
  freeze: True  # HF 공식 권장

lora:  # Roboflow Issue #162 best
  r: 8
  alpha: 16  # = 2r
  dropout: 0.05
  target_modules:
    - q_proj
    - o_proj
    - k_proj
    - v_proj
    - linear
    - Conv2d
    - lm_head
    - fc2
  use_rslora: True       # ★ rank-stabilized LoRA
  init_lora_weights: gaussian  # ★ Gaussian init
  task_type: CAUSAL_LM
  bias: none

training:  # HF 공식
  epochs: 7
  batch_size: 2  # RTX 5080 16GB
  grad_accum_steps: 8  # effective batch 16
  learning_rate: 1.0e-6  # 매우 낮음 (overfit 회피)
  weight_decay: 0.01
  grad_clip: 1.0
  optimizer: AdamW
  scheduler: linear (no warmup)
  precision: bf16
  seed: 42

dataset:
  train: outputs/v_e_annotation/training_gt_v2/train.jsonl  # 4730 records
  val: outputs/v_e_annotation/training_gt_v2/val.jsonl      # 1252 records
  # ★ training_gt_v2_aug 사용 안 함 (D-108.1 placeholder 문제 회피)
```

### 3.2 학습 시간 + VRAM 예상

| 항목 | 예상값 |
|---|---|
| 학습 시간 (RTX 5080) | **~7~8h** (overnight) |
| VRAM | **~10~12 GB (60~75%)** |
| GPU 온도 | 50~60°C |
| 종료 예상 | 오전 10~11시 |

---

## §4 산출물 (V.G phase 학습 + 평가)

### 4.1 학습 스크립트
- **[src/training/florence2_ft_ver.G.py](src/training/florence2_ft_ver.G.py)** ★ (~430 lines, D-109 작성)
  - Roboflow Issue #162 best LoRA config 100% 적용
  - HF 공식 lr 1e-6 + AdamW + linear scheduler
  - Vision encoder freeze (~3 GB VRAM 절감)
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments (단편화 완화)

### 4.2 박제 문서 (작성 완료 또는 예정)

| 파일 | 상태 | 용도 |
|---|---|---|
| **`docs/V_G_PHASE_START_ver.G.md`** | ✅ D-109 작성 | V.G phase plan |
| **`docs/V_G_RESEARCH_REFERENCES_ver.G.md`** | ✅ D-109 작성 | Florence-2 references 종합 |
| **`README_ver.G.md`** | ✅ D-109 작성 (본 문서) | V.G phase overview |
| `docs/V_G_FLORENCE2_TRAINING_RESULTS_ver.G.md` | ⏳ D-110 예정 | 학습 결과 박제 |
| `docs/V_G_PRO_JUDGE_RESULTS_ver.G.md` | ⏳ D-110 자동 생성 | Pro Judge 평가 |
| `PROJECT_HANDOFF_ver.G.md` | ⏳ D-110 후 baseline | V.G phase baseline handoff |

### 4.3 학습 산출물 (예정)
```
outputs/v_g_florence2_local_train/
├── checkpoints/
│   ├── checkpoint-epoch-1/  ~ checkpoint-epoch-7/
│   └── final/  (best val_loss)
└── train_*.log
```

---

## §5 의사결정 framework (D-110 학습 결과 후)

### 5.1 V.E v2 Donut + V.F-2 LOCAL 와의 비교

| 영역 | V.E v2 Donut | V.F-2 LOCAL | **V.G Florence-2 (목표)** |
|---|---|---|---|
| Title Block 영문 | 75 | 42 | **> 75** (V.E v2 보다 우수) |
| Title Block 한자 | 13.9 | 60% mode | **> 50** (Donut ceiling 돌파) |
| View | 60 | 15 | **> 60** (V.E v2 동등 또는 우수) |
| Notes | 75 | 65 | **> 70** |

### 5.2 분기 결정

| Case | 조건 | 결정 |
|---|---|---|
| **A 우수** | V.E v2 보다 우수 (모든 영역 > V.E v2) | ★ **Florence-2 채택** + Path B-3 v3 |
| **B 동등** | V.E v2 수준 | Hybrid Routing 정밀 튜닝 |
| **C 부족** | V.E v2 보다 못함 | Florence-2-large 시도 (~3~4h 추가) |
| **D 완전 실패** | mode collapse 잔존 | V.F-2 LOCAL Hybrid 운영 유지 |

---

## §6 V.G phase 작업 단계 (D-109 ~ D-115)

### Phase 1 — D-109 (학습 직전, 본 시점)
- [x] **D-109 hyperparameter 검증** (Roboflow Issue #162 + HF 공식)
- [x] **`src/training/florence2_ft_ver.G.py` 작성**
- [x] **V.G 박제 3 파일 작성** (PHASE_START, RESEARCH_REFERENCES, README)
- [ ] **D-110 학습 실행** (overnight, ~7~8h)

### Phase 2 — D-110 (학습 + 평가, 오늘 오전)
- [ ] Florence-2-base 학습 (lr 1e-6, 7 epoch, ~7~8h)
- [ ] V.G inference 10 sample (V.F-2/V.F v2 와 동일 sample, ~10분)
- [ ] Pro Judge 평가 (~10분)
- [ ] D-110 결과 박제 + 의사결정 분기

### Phase 3 — D-111~115 (분기별 후속, 1~2주)
- [ ] (Case A) Path B-3 v3 cog 패키지
- [ ] (Case B) Hybrid Routing 정밀 (영역별 best 모델)
- [ ] (Case C) Florence-2-large 시도
- [ ] (Case D) V.F-2 LOCAL Hybrid 운영
- [ ] FastAPI demo server V.G 채택
- [ ] PROJECT_HANDOFF_ver.G.md baseline 작성

---

## §7 학습 시작 명령 (overnight 실행)

```bash
cd /mnt/c/Users/user/github/Drawing
source .venv-vf/bin/activate

mkdir -p outputs/v_g_florence2_local_train

# Florence-2 학습 시작 (overnight, ~7~8h)
nohup nice -n 10 taskset -c 0-15 \
    python src/training/florence2_ft_ver.G.py \
        --train-jsonl outputs/v_e_annotation/training_gt_v2/train.jsonl \
        --val-jsonl outputs/v_e_annotation/training_gt_v2/val.jsonl \
        --image-dir colab_sync/dataset/JP/images \
        --phase-c-dir colab_sync/outputs/phase1_instruct_phaseB \
        --output-dir outputs/v_g_florence2_local_train \
        --base-model microsoft/Florence-2-base \
        --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
        --epochs 7 --batch-size 2 --grad-accum-steps 8 \
        --lr 1e-6 --seed 42 \
    > outputs/v_g_florence2_local_train/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Watcher (자동 알림)
(while ps -p $TRAIN_PID > /dev/null 2>&1; do sleep 120; done; \
 echo ""; \
 echo "🔔🔔🔔 V.G Florence-2 학습 종료! [$(date '+%H:%M:%S')] 🔔🔔🔔"; \
 tail -50 /mnt/c/Users/user/github/Drawing/outputs/v_g_florence2_local_train/train_*.log; \
 ls -la /mnt/c/Users/user/github/Drawing/outputs/v_g_florence2_local_train/checkpoints/; \
 powershell.exe -c "[System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep 1; [System.Media.SystemSounds]::Asterisk.Play(); Start-Sleep 1; [System.Media.SystemSounds]::Asterisk.Play()" 2>/dev/null) &
echo "Watcher PID: $!"
```

---

## §8 V.G phase 의 주요 risk + 대응책

### 8.1 잠재 risk

| Risk | 발생 확률 | 대응책 |
|---|---|---|
| **OOM (VRAM 초과)** | 낮음 (vision freeze + bf16) | --batch-size 1 + --max-seq-length 768 |
| **trust_remote_code 실패** | 낮음 | 인터넷 연결 확인 + HF cache 재시도 |
| **Vision encoder freeze 패턴 미일치** | 중간 | 학습 로그에서 trainable params % 확인 |
| **lr 1e-6 너무 낮음 (학습 안 됨)** | 중간 | 학습 끝까지 후 평가, 부족 시 lr 1e-5 로 재시도 |
| **한자 OCR 성능 부족** | 중간 | Florence-2 의 일본어 pretraining 양 미확인 |
| **Mode collapse 잔존** | 낮음 (architecture 변경) | Florence-2-large 또는 Hybrid 운영 |

### 8.2 V.F phase 의 known issues 회피

| V.F 문제 | V.G 회피 방법 |
|---|---|
| Memorization 100% (V.F-1) | encoder-decoder + region prompt → image 무시 회피 |
| Mode collapse 60% (V.F-2) | Florence-2 의 작은 모델 + capacity 적정 |
| eval_loss ≠ 품질 | Pro Judge 평가로 결정 |
| Augmented data placeholder | **training_gt_v2 원본 사용** (D-108.1 교훈) |

---

## §9 V.A 의 Florence-2 사용 vs V.G 의 Florence-2 사용 (재확인)

| 항목 | V.A Florence-2 (D-001~D-058) | V.G Florence-2 (D-109~) |
|---|---|---|
| **사용 목적** | Region detection (작은 영역 검출) | **OCR + structured JSON extraction** |
| 학습 데이터 | YOLO bbox + class | **V.E annotation (15 fields + view + notes)** |
| 출력 | bbox + class | **structured JSON** |
| 종료 사유 | F1 0.81 (Khan baseline 미달) | TBD (학습 후 평가) |
| Reference | (V.A 시점 paper 없음) | **arxiv 2411.03707 Khan F1 +52.4%** baseline |

→ **V.A 부분 실패 ≠ V.G 결과 예측 불가** (사용 목적 + 학습 데이터 모두 다름).

---

## §10 박제 체크리스트

### D-109 (학습 직전, 본 시점) — 완료
- [x] `docs/V_G_PHASE_START_ver.G.md` ★
- [x] `docs/V_G_RESEARCH_REFERENCES_ver.G.md` ★
- [x] `README_ver.G.md` (본 문서) ★
- [x] `src/training/florence2_ft_ver.G.py` ★
- [x] `PROJECT_HANDOFF_ver.F.md` §6 D-108.1 + V.G 결정 entry

### D-110 (학습 + 평가) — 학습 종료 후 예정
- [ ] `docs/V_G_FLORENCE2_TRAINING_RESULTS_ver.G.md`
- [ ] `docs/V_G_PRO_JUDGE_RESULTS_ver.G.md` (Pro Judge 자동)
- [ ] `src/inference/v_g_florence2_inference_random10_ver.G.py` (V.F-2 동일 sample 처리)
- [ ] `PROJECT_HANDOFF_ver.G.md` baseline (V.G phase 박제 시작)

### D-111~115 (분기별 후속) — 의사결정 후 예정
- [ ] Hybrid Routing 갱신 (`V_F_MODEL_SELECTION_CRITERIA_ver.F.md` 또는 V.G 신규)
- [ ] Path B-3 v3 cog 패키지 (Case A 시)
- [ ] Florence-2-large 시도 (Case C 시)
- [ ] FastAPI demo server V.G 채택

---

## §11 발표 자료 갱신 (V.G 결과 후)

학습 + 평가 완료 후 갱신할 파일:
- `docs/presentation_assets/PROJECT_PRESENTATION_SLIDES_ver.F.md` (V.G 결과 추가)
- `docs/presentation_assets/V_F_MODEL_SELECTION_CRITERIA_ver.F.md` (Florence-2 채택 시 영역 갱신)
- `docs/presentation_assets/performance_comparison.svg` (V.G column 추가)

---

**작성일**: 2026-05-16 (D-109, V.G phase 시작 직전 새벽 ~03:30)
**상태**: V.G phase plan 완료. Hyperparameter 검증 완료. 학습 스크립트 작성 완료. **학습 시작 대기** (overnight, ~7~8h, 오전 10~11시 종료 예상).
**다음 단계**: §7 학습 명령 실행 → D-110 결과 박제 + 의사결정
