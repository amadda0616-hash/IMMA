# 금일 저녁 (2026-05-05) — Phase 16b Fine-tune 시작 전까지 체크리스트

> **목표**: 3~4시간 안에 Phase 16b (Donut Numerical fine-tune ~6h overnight) 시작 명령 실행.
>
> **현재 상태 (15:30 추정)**: Phase 15b 3차 V5 미통과, D-046/D-047 박제 완료, Real-ESRGAN 진행 결정.
>
> **저녁 흐름**: 작업 마무리 → 16b 학습 명령 실행 → 취침 → 다음날 아침 학습 결과 확인.

---

## ⏱️ 시간 예산

| Block | 작업 | 시간 | 누적 |
|---|---|---|---|
| **B1** | Real-ESRGAN 설치 + 5장 4x upscale | 30분 | 30분 |
| **B2** | Phase 15b 4차 평가 + 정성 검토 | 30분 | 1h |
| **B3** | 4차 결과 박제 + Phase 15c 백엔드 교체 (선택) | 30분~1h | 1h~2h |
| **B4** | Phase 16a — VLM pair 학습 데이터 준비 | 1h | 2h~3h |
| **B5** | Phase 16b 학습 명령 검증 + overnight 시작 | 30분 | 2.5h~3.5h |
| **여유** | 트러블슈팅 / 박제 마무리 | 30분 | **~4h** |

---

## 🟢 Block 1 — Real-ESRGAN (30분)

### Step 1.1 — 설치 (10분)

```bash
cd /mnt/c/Users/user/github/Drawing
source .venv-paddleocr/bin/activate

# Real-ESRGAN + dependencies
uv pip install realesrgan basicsr facexlib gfpgan
# 약 200MB 다운, 5분 소요 예상

# 검증
python -c "from RealESRGAN import RealESRGAN; print('Real-ESRGAN OK')"
```

### Step 1.2 — 5장 upscale 스크립트 작성 + 실행 (15분)

작성할 파일: `src/upscale_images_realesrgan.py` (신규, ~120 lines)

기능:
- 입력 디렉토리 자동 스캔
- 4x model 자동 다운 (`weights/RealESRGAN_x4plus.pth`, ~64MB)
- min(W, H) ≥ 1280 인 이미지는 skip (이미 충분)
- 그 외는 4x upscale + 저장
- 출력 디렉토리 자동 생성

사용:
```bash
python src/upscale_images_realesrgan.py \
    --input data/stage3a_eval_samples/ \
    --output data/stage3a_eval_samples_realesrgan/
```

처리 시간: **~30s/이미지 × 4장 = 2분** + 모델 다운 5분.

### Step 1.3 — 검증 (5분)

```bash
python -c "
from PIL import Image
from pathlib import Path
for p in sorted(Path('data/stage3a_eval_samples_realesrgan').glob('*.jpg')):
    with Image.open(p) as im:
        w, h = im.size
        mark = '★' if min(w,h) >= 1280 else '❌'
        print(f'{mark} {p.name}: {w}x{h}')
"
```

5장 모두 ★ 확인.

---

## 🟢 Block 2 — Phase 15b 4차 평가 (30분)

### Step 2.1 — 평가 실행 (5~10분)

```bash
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir data/stage3a_eval_samples_realesrgan/ \
    --output-json outputs/stage3a_zero_shot_eval_v4_realesrgan.json \
    --output-md   outputs/stage3a_zero_shot_eval_v4_realesrgan.md
```

예상 시간: ~5~10분 (5장 × ~60s).

### Step 2.2 — 정성 검토 (사용자, 15~20분)

```bash
cat outputs/stage3a_zero_shot_eval_v4_realesrgan.md | head -100
```

3차 vs 4차 비교 검토:
- en (1280x1280 upscaled) → TitleBlock 인식 향상?
- ko (1280x1280 upscaled) → "수도전기공업고등학교" 인식?
- ru (1280x1280 upscaled) → Notes 정확도 향상?
- zh (1280x1280 upscaled) → 표 인식?

판정 (D-013):
- char acc 향상 시 PASS 가능성
- FAIL 시 → 한계 인지하고 Phase 16 진행

---

## 🟢 Block 3 — 박제 + Phase 15c (선택, 30분~1h)

### Step 3.1 — 4차 결과 박제 (15분)

`history.md §A.12.7` 추가:
- Real-ESRGAN 결정 사유
- 4차 결과 (3차 vs 4차 비교)
- D-013 V5 최종 판정

`PROJECT_HANDOFF.md` Day 4 업데이트.

### Step 3.2 — Phase 15c 백엔드 교체 (선택, ~1h)

만약 4차 결과 충분히 좋으면:
```bash
# stage3_alphabetical.py 백업
cp src/stage3_alphabetical.py src/stage3_alphabetical.donut.py.bak

# PaddleOCR-VL 백엔드로 교체 (D-046 적용)
# (수동 코드 작업 또는 별도 세션)
```

**시간 부족 시 → 다음 날 진행** (16b 학습이 더 우선).

---

## 🔴 Block 4 — Phase 16a VLM Pair 학습 데이터 준비 (★ 1h)

### Step 4.1 — venv 전환

```bash
deactivate    # paddleocr venv 빠져나오기
source .venv/bin/activate    # Phase 14 venv (ultralytics + Donut)

# 환경 검증
python -c "
import torch
import ultralytics
import transformers
print(f'torch:        {torch.__version__}')
print(f'ultralytics:  {ultralytics.__version__}')
print(f'transformers: {transformers.__version__}')
"
```

### Step 4.2 — 사전 검증

```bash
# Stage 1 / Stage 2 ensemble checkpoints 존재 확인
ls -la checkpoints/yolo_det.pt
ls checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_*/weights/best.pt | wc -l   # 5 이어야 함

# prepare_vlm_dataset.py CLI 확인
python src/prepare_vlm_dataset.py --help
```

### Step 4.3 — VLM pair 생성 (★ 메인)

```bash
# 500장 sample (full 5,839 는 후속)
# ★ 2026-05-05 인자명 수정:
#   --input → --dataset, --stage1-weights → --det-weights,
#   --stage2-ensemble (디렉토리) → --obb-weights (단일 파일),
#   --output 인자 없음 (코드 내 고정 경로 data/vlm/numerical/ 사용)
python src/prepare_vlm_dataset.py numerical \
    --dataset dataset/ \
    --det-weights checkpoints/yolo_det.pt \
    --obb-weights checkpoints/yolo_obb.pt \
    --device 0 \
    --ocr-prefill \
    --limit 500

# 진행 상황 모니터링 (~30~50분)
# Stage 1 + Stage 2 단일 fold 추론 → de-rotation crop → JSON template 자동 생성
# (K-fold ensemble 은 prepare_vlm_dataset 미지원 — Phase 17 e2e 에서 적용)
```

**예상 결과**:
```
data/vlm/numerical/
├── train/
│   ├── images/      # ~400장 (80%)
│   └── jsons/       # ~400 JSON
├── valid/
│   ├── images/      # ~100장 (20%)
│   └── jsons/       # ~100 JSON
└── manifest.json    # group leak 0 검증 포함
```

### Step 4.4 — 데이터 검증 (10분)

```bash
# 파일 수 확인
echo "Train images: $(ls data/vlm/numerical/train/images/*.jpg 2>/dev/null | wc -l)"
echo "Train jsons:  $(ls data/vlm/numerical/train/jsons/*.json 2>/dev/null | wc -l)"
echo "Valid images: $(ls data/vlm/numerical/valid/images/*.jpg 2>/dev/null | wc -l)"
echo "Valid jsons:  $(ls data/vlm/numerical/valid/jsons/*.json 2>/dev/null | wc -l)"

# manifest 검증
python -c "
import json
with open('data/vlm/numerical/manifest.json') as f:
    m = json.load(f)
print('Train pairs:', m.get('n_train', '?'))
print('Valid pairs:', m.get('n_valid', '?'))
print('Group leak:', m.get('group_leak', '?'))
print('Class distribution:', m.get('class_distribution', {}))
"
```

**검증 항목**:
- [ ] Train + Valid 합계 ≥ 500 pairs (또는 OBB 검출 비율 따라 다름)
- [ ] Group leak = 0 (★ D-024 보장)
- [ ] Class 균형: Measure / GDT / Roughness 모두 일정 수량

---

## 🔴 Block 5 — Phase 16b Fine-tune 시작 명령 (★ overnight)

### Step 5.1 — 학습 명령 검증 (10분)

```bash
# stage3_numerical.py train 인자 확인
python src/stage3_numerical.py train --help
```

### Step 5.2 — 학습 시작 (★ 5~10분 setup → 6h 학습)

```bash
# Overnight 학습 시작
nohup python src/stage3_numerical.py train \
    --data data/vlm/numerical/ \
    --epochs 30 \
    --batch 4 \
    --device 0 \
    --output checkpoints/donut_numerical/ \
    --save-period 5 \
    > outputs/stage3n_train.log 2>&1 &

# PID 저장
echo $! > outputs/stage3n_train.pid
echo "Stage 3-N training started, PID: $(cat outputs/stage3n_train.pid)"
```

**Tip**:
- `nohup` + `&` → 터미널 닫아도 학습 지속
- log: `outputs/stage3n_train.log`
- 진행 모니터링: `tail -f outputs/stage3n_train.log`

### Step 5.3 — 시작 확인 (5분)

```bash
# 첫 1~2 epoch 까지 watch
sleep 60
tail -50 outputs/stage3n_train.log
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
```

확인 포인트:
- [ ] GPU VRAM 사용 ~12~14 GB
- [ ] GPU utilization ~80~95%
- [ ] log 에 epoch 1 시작 메시지

### Step 5.4 — 박제 + 취침 준비

```bash
# 학습 시작 박제
echo "Phase 16b started at $(date)" >> history.md

# 취침
echo "Good night! Check tomorrow morning."
```

---

## 🌅 다음 날 아침 (2026-05-06) — 학습 결과 확인

```bash
# 학습 종료 여부 확인
ps -p $(cat outputs/stage3n_train.pid) 2>/dev/null && echo "Still running" || echo "Finished"

# 마지막 로그
tail -100 outputs/stage3n_train.log

# 결과 weights 존재 확인
ls -la checkpoints/donut_numerical/final/

# Phase 16c V6 검증 (~30분)
python -m src.validate.check_stage3n_numerical \
    --weights checkpoints/donut_numerical/final \
    --val data/vlm/numerical/valid/
```

---

## 🚨 트러블슈팅

| 이슈 | 원인 | 대응 |
|---|---|---|
| CUDA OOM (16b) | batch 4 가 너무 큰 모델 | `--batch 2` 다시 시도 |
| `prepare_vlm_dataset.py` 실패 | Stage 2 weights path 오류 | `--obb-weights checkpoints/yolo_obb.pt` 단일 파일 검증 (K-fold 미지원) |
| 학습 epoch 1 만에 정지 | 데이터 부족 또는 loss NaN | log 확인 + `--epochs 5 --batch 2` 작은 시작 |
| log 갱신 안 됨 | nohup buffer | `python -u src/stage3_numerical.py ...` (unbuffered) |

---

## 📋 우선순위 (시간 부족 시)

**필수 (Must)**:
- ✅ Block 1 (Real-ESRGAN, 30분)
- ✅ Block 4 (VLM pair 준비, 1h)
- ✅ Block 5 (16b 학습 시작, 30분)

**Total: ~2h** — 충분히 가능.

**선택 (Should)**:
- Block 2 (4차 평가, 30분)
- Block 3.1 (4차 박제, 15분)

**선택 (Could)**:
- Block 3.2 (Phase 15c 백엔드 교체, 1h) — 다음 날 진행 가능

---

## ✅ 체크리스트 (★ 진행 상태 갱신 2026-05-05 22:30)

```
[x] B1.1 Real-ESRGAN 설치 (10분) — uv pip install + basicsr monkey-patch
[x] B1.2 src/upscale_images_realesrgan.py 작성 (450 lines) + 실행 (4.9s, 매우 빠름)
[x] B1.3 5장 ★ 검증 — en/ko/ru/zh: 640→2560, ja: 그대로

[x] B2.1 Phase 15b 4차 평가 (1043s ~ 17분 — 큰 이미지 처리)
[x] B2.2 정성 검토 — avg char acc 0.50 → 0.69 (★ ko/zh 큰 향상, ja "B" 변화 X)

[x] B3.0 ★ 추가 — Stage 1 ja_drawing 분리 검증 (D-048)
        - 110 region (View 6 / TitleBlock 3 / Notes 3 / PMI 98)
        - 사용자 가설 PASS (분리 후 해상도 하락 우려 X)
[x] B3.1 4차 + Stage 1 검증 결과 박제 (history §A.12.8 + D-048)

[ ] B4.1 venv 전환 (.venv-paddleocr → .venv)            ← 다음 작업
[ ] B4.2 환경 검증 (torch / ultralytics / transformers)
[ ] B4.3 prepare_vlm_dataset.py 실행 (~30~50분)
[ ] B4.4 manifest 검증 (group leak 0)

[ ] B5.1 stage3_numerical.py train --help 확인
[ ] B5.2 nohup overnight 학습 시작
[ ] B5.3 첫 epoch 시작 확인
[ ] B5.4 박제 + 취침
```

## 📊 진행 시간 (실측)

| Block | 예상 | 실측 | 상태 |
|---|---|---|---|
| B1 (Real-ESRGAN) | 30분 | ~10분 | ✅ |
| B2 (4차 평가) | 30분 | ~17분 | ✅ |
| B3.0 (Stage 1 ja 검증, ★ 보너스) | — | ~1분 | ✅ |
| B3.1 (박제) | 15분 | ~15분 | ✅ |
| **누적** | — | **~45분** | **앞으로 ~3h 여유** |
| B4 (Phase 16a) | 1h | TBD | ⏳ 다음 |
| B5 (Phase 16b 시작) | 30분 | TBD | ⏳ |
| **목표 16b 시작** | ~24시 | **~23시 30분 예상 (★ 일정 충분)** | — |

---

**Last updated**: 2026-05-05 (Phase 15b 3차 후, Real-ESRGAN 결정 + Phase 16 진입 준비)
