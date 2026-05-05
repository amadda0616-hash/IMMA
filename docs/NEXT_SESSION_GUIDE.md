# 다음 세션 시작 가이드 (★ 2026-05-06 아침)

> **현재 상태 (2026-05-05 EOD, ★ Phase 16b overnight 학습 진행 중)**:
> - Phase 14 ✅ / Phase 15a ✅ / Phase 15b 1차~4차 ✅ (V5 부분 PASS)
> - Stage 1 ja_drawing 분리 검증 ✅ (D-048 박제, 110 region)
> - Phase 16a VLM pair 준비 ✅ / **Phase 16b 학습 overnight 진행 중**
>
> **다음 세션 (아침 8~9시 가정)**:
> 1. **Phase 16b 학습 결과 확인 + V6 검증** (★ D-023 critical) — 30분
> 2. ja 영역별 Stage 3-A 평가 (옵션) — 30분
> 3. Phase 15c stage3_alphabetical.py 백엔드 교체 — 1h
> 4. Phase 15d Notes Rescue 재실행 — 30분
> 5. Phase 17 e2e batch 시작 (옵션) — 5,839장 batch ~14h

---

## 🔴 [필수 1] Phase 16b 학습 결과 확인 (★ 가장 우선)

```bash
cd /mnt/c/Users/user/github/Drawing

# 학습 종료 여부
ps -p $(cat outputs/stage3n_train.pid 2>/dev/null) 2>/dev/null && echo "Still running" || echo "Finished"

# 학습 로그 확인 (마지막 100줄)
tail -100 outputs/stage3n_train.log

# best.pt 확인
ls -la checkpoints/donut_numerical/final/ 2>/dev/null || ls -la checkpoints/donut_numerical/
```

**확인 포인트**:
- [ ] Epoch 30/30 완료 (또는 early stopping)
- [ ] Best epoch validation loss
- [ ] checkpoints/donut_numerical/final/ 존재
- [ ] log 에 OOM 또는 NaN 발생 안 함

### V6 검증 (★ D-023 critical, ~30분)

```bash
source .venv/bin/activate

python -m src.validate.check_stage3n_numerical \
    --weights checkpoints/donut_numerical/final \
    --val data/vlm/numerical/valid/
```

**임계값 (D-023)**:
- Measure F1 ≥ 0.85
- GDT F1 ≥ 0.90
- Roughness F1 ≥ 0.50
- numerical_accuracy ≥ 0.85
- tolerance_match ≥ 0.80
- hallucination_rate ≤ 0.05 ★ critical

**FAIL 시**: Plan B
- Qwen3-VL fine-tune 검토
- PaddleOCR-VL Stage 3-N 활용 검토
- Plan B 박제 + Phase 17 시작 (Stage 3-N 부분 성공 인정)

---

## 🟡 [옵션 2] ja_drawing 영역별 Stage 3-A 평가 (30분)

D-048 검증 결과 활용 — 110 region 중 Stage 3-A 영역 (Table + Notes) 만 별도 평가.

```bash
deactivate
source .venv-paddleocr/bin/activate

# 1. Stage 1 crop 결과 확인
ls outputs/crops/ja_drawing/

# 2. TitleBlock 영역 평가 (3개)
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir outputs/crops/ja_drawing/Table/ \
    --prompts titleblock \
    --output-json outputs/stage3a_ja_table_split.json \
    --output-md   outputs/stage3a_ja_table_split.md

# 3. Notes 영역 평가 (3개)
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir outputs/crops/ja_drawing/Notes/ \
    --prompts notes \
    --output-json outputs/stage3a_ja_notes_split.json \
    --output-md   outputs/stage3a_ja_notes_split.md
```

**평가 포인트**:
- 4차 ja 단일 처리 시 "B" 무한 반복 → 분리 처리 시 정상화 여부
- TitleBlock 3개 모두 BSBM/CRITICAL/ロストワックス 구분 인식
- Notes 注記 5개 정확 추출

**결과 박제**: `history.md §A.13.x` (Phase 16 결과와 별개로 추가).

---

## 🟢 [필수 3] Phase 15c — `src/stage3_alphabetical.py` 백엔드 교체 (1h)

V5 부분 PASS 받아들이고 PaddleOCR-VL 백엔드로 교체:

```bash
# 백업
cp src/stage3_alphabetical.py src/stage3_alphabetical.donut.py.bak

# 교체 코드 작성 (수동 또는 별도 세션)
# - load_model() → AutoConfig + monkey-patch + AutoModelForImageTextToText (D-042)
# - predict_titleblock() → "Table Recognition:" task keyword
# - predict_notes() → "OCR:" task keyword
# - 호출: apply_chat_template 통합 + processor.decode 슬라이스
```

**상세 절차**: `docs/PHASE15_CHECKLIST.md §15c` 참조.

---

## 🟢 [필수 4] Phase 15d — Notes Rescue 재실행 (30분)

`outputs/skip_lists/stage1_fp_notes.txt` (23개) 에 PaddleOCR-VL backend 적용:

```bash
mkdir -p data/stage1_fp_notes_crops
while read -r fname; do
    [[ "$fname" =~ ^# ]] && continue
    [[ -z "$fname" ]] && continue
    cp "data/annotation/images/train/${fname}" \
       "data/annotation/images/valid/${fname}" \
       "data/stage1_fp_notes_crops/" 2>/dev/null
done < outputs/skip_lists/stage1_fp_notes.txt

source .venv-paddleocr/bin/activate
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir data/stage1_fp_notes_crops/ \
    --prompts notes,full_text \
    --output-json outputs/stage1_fp_notes_paddleocr_eval.json \
    --output-md   outputs/stage1_fp_notes_paddleocr_eval.md
```

목표: Donut 4% → PaddleOCR ≥ 80%.

---

## 🟡 [옵션 5] Phase 17 — Pipeline E2E batch 시작

V6 PASS + Stage 3-A backend 교체 완료 시 → Phase 17 e2e:

```bash
source .venv/bin/activate

# 50장 sample batch 먼저
python src/pipeline.py batch \
    --input-dir dataset/ \
    --out-dir outputs/json \
    --device cuda:0 \
    --limit 50

# 5,839장 full batch (overnight)
nohup python src/pipeline.py batch \
    --input-dir dataset/ \
    --out-dir outputs/json_full \
    --device cuda:0 \
    > outputs/pipeline_batch.log 2>&1 &
```

**임계값 (D-021)**: avg ≤ 30s/도면.

---

## 📋 30초 체크 (작업 시작 전)

```bash
cd /mnt/c/Users/user/github/Drawing

# 1. Phase 16b 학습 종료 확인
ps -p $(cat outputs/stage3n_train.pid 2>/dev/null) 2>/dev/null && echo "🔴 학습 진행 중 (대기)" || echo "✅ 학습 종료"

# 2. checkpoints 확인
ls checkpoints/donut_numerical/ 2>/dev/null

# 3. Stage 1 ja crop 확인 (D-048)
ls outputs/crops/ja_drawing/ 2>/dev/null

# 4. 4차 결과 확인
ls -la outputs/stage3a_zero_shot_eval_v4_realesrgan.* 2>/dev/null
```

---

## 핵심 박제 위치 (Quick Reference)

- **Phase 14 (Stage 2 ensemble)**: `history.md §A.11.13` / D-040
- **Phase 15a (env)**: `history.md §A.12.1~2` / D-042
- **Phase 15b 1~3차**: `history.md §A.12.4~6` / D-045/D-046/D-047
- **Phase 15b 4차 + Stage 1 ja 검증**: `history.md §A.12.8` / D-048
- **Phase 16 진행**: `outputs/stage3n_train.log`
- **NEXT 작업**: 본 가이드

---

## 1️⃣ 30초 컨텍스트 복원

| 상태 | 내용 |
|---|---|
| **Stage 1 (Layout)** | DONE — V.A seed 학습 완료 (`yolo_det.pt`) |
| **Stage 2 (OBB)** | DONE — 5-Fold Ensemble (D-040, D-023 PASS) — `pipeline.py` 통합 |
| **Stage 3-A (PaddleOCR-VL)** | 환경 설치 PASS (★ Phase 15a) — `.venv-paddleocr` venv |
| **Stage 3-A 평가 스크립트** | 작성 완료 (★ Phase 15b 작성) — 미실행 |
| **Stage 3-N (Donut Numerical)** | TODO (Phase 16) |
| **GitHub** | https://github.com/amadda0616-hash/IMMA — Phase 14 까지 push |

**핵심 박제** (꼭 알아둘 것):
- **D-042**: PaddleOCR-VL 사용 시 `config.text_config = config.get_text_config()` monkey-patch 필수
- **D-044**: TitleBlock 23 필드 표준 schema (ISO 7200 + KS A 0005 + 첨부 이미지)
- **D-043**: 데이터 도메인 한계 (한국어 학습용 / 영어 부족 / 중국어 풍부 / CNC+기어 위주)
- **D-038 stage1_fp_notes**: 23개 (CAD_Drawing219: 14 + sample_01266: 9) — Phase 15d PaddleOCR-VL Rescue 대상

---

## 2️⃣ 환경 활성화 (1분)

```bash
cd /mnt/c/Users/user/github/Drawing
source .venv-paddleocr/bin/activate

# 환경 검증
python -c "
import torch, transformers
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
# 예상: torch 2.11.0+cu128 / transformers 5.0.0 / RTX 5080
```

만약 venv 가 없으면:
```bash
uv venv --python 3.10 .venv-paddleocr
source .venv-paddleocr/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==5.0.0" accelerate sentencepiece protobuf einops pillow
```

---

## 3️⃣ 작업 체크리스트 (★ 우선순위 순)

### 🔴 [필수 1] Phase 15b 실행 — 5장 zero-shot 평가 (예상 5분)

```bash
# 사전 검증
ls -la data/stage3a_eval_samples/   # 5장 파일 확인 (en/ja/ko/ru/zh_drawing.jpg)

# 환경 검증 (생략 가능)
python src/stage3_paddleocr_install_check.py --skip-inference

# 평가 실행 (Cold start 40s + 5장 × 3 prompt × ~3s ≈ 90~120s)
python src/stage3_paddleocr_zero_shot_test.py
```

**확인 포인트**:
- [ ] 종료 코드 0
- [ ] `outputs/stage3a_zero_shot_eval.json` 생성
- [ ] `outputs/stage3a_zero_shot_eval.md` 생성
- [ ] 모든 도면 처리 (n_errors = 0)

---

### 🔴 [필수 2] 결과 정성 평가 (예상 30분, 사용자 작업)

```bash
# MD 보고서 검토
cat outputs/stage3a_zero_shot_eval.md
# 또는 VS Code / Antigravity 에서 열기
```

**5개 항목 1~5점 평가** (각 도면별):
1. TitleBlock 필드 정확도 (key + value 정확하게 추출?)
2. Notes 의미 보존 (원문 누락/왜곡 없이?)
3. 다국어 정확도 (해당 언어 글자 정확한 transcription?)
4. Hallucination 여부 (없는 텍스트 생성 안 함?)
5. JSON 형식 준수 (TitleBlock 만)

**임계값 (D-013)**:
- 평균 char accuracy ≥ 0.85 → PASS
- field-level F1 ≥ 0.80 → PASS
- hallucination ≤ 0.05

**결과 박제**:
- `outputs/stage3a_zero_shot_eval.md` 끝에 사용자 정성 평가 추가 (또는 별도 `outputs/stage3a_v5_summary.txt`)
- `history.md §A.12.4` 결과 박제 (PASS/FAIL + 점수)

---

### 🟢 [필수 3] Phase 15c — `src/stage3_alphabetical.py` 백엔드 교체 (예상 1h)

15b PASS 시 Donut 코드를 PaddleOCR-VL 로 교체:

**현재 구조** (Donut):
```python
# src/stage3_alphabetical.py
def load_model(device): ...               # DonutProcessor + VisionEncoderDecoderModel
def predict_one(image_path, region_type, processor, model, device, language_hint): ...
def predict_titleblock(...): ...
def predict_notes(...): ...
```

**교체 방향** (PaddleOCR-VL):
- `load_model()` → AutoProcessor + AutoModelForImageTextToText (★ D-042 monkey-patch 적용)
- `predict_titleblock()` → 23 필드 prompt (★ D-044 표준 schema)
- `predict_notes()` → 다국어 keyword hint
- 함수 시그니처 100% 유지 (drop-in)

**작업 흐름**:
- [ ] `src/stage3_alphabetical.py` 백업 (`cp src/stage3_alphabetical.py src/stage3_alphabetical.donut.py.bak`)
- [ ] PaddleOCR-VL 통합 코드 작성
- [ ] CLI 검증: `python src/stage3_alphabetical.py predict --image <patch> --region titleblock`
- [ ] V5 검증기 (`src/validate/check_stage3a_alphabetical.py`) 호환 테스트

---

### 🟢 [필수 4] Phase 15d — D-038 Notes Rescue 재실행 (예상 30분)

**입력**: `outputs/skip_lists/stage1_fp_notes.txt` (23개 — CAD_Drawing219: 14 / sample_01266: 9)

```bash
# 1. 23개 PMI crop 디렉토리 준비
mkdir -p data/stage1_fp_notes_crops
while read -r fname; do
    [[ "$fname" =~ ^# ]] && continue
    [[ -z "$fname" ]] && continue
    cp "data/annotation/images/train/${fname}" \
       "data/annotation/images/valid/${fname}" \
       "data/stage1_fp_notes_crops/" 2>/dev/null
done < outputs/skip_lists/stage1_fp_notes.txt
ls data/stage1_fp_notes_crops/ | wc -l   # 23 확인

# 2. PaddleOCR-VL OCR 실행
source .venv-paddleocr/bin/activate
python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir data/stage1_fp_notes_crops/ \
    --prompts notes,full_text \
    --output-json outputs/stage1_fp_notes_paddleocr_eval.json \
    --output-md   outputs/stage1_fp_notes_paddleocr_eval.md
```

**결과 비교**:
| Backend | 성공률 | 판정 |
|---|---|---|
| Donut DocVQA (1차) | 4% | ❌ |
| **PaddleOCR-VL (2차)** | **목표 ≥ 80%** | ★ |

**박제**:
- `outputs/stage1_fp_notes_paddleocr_eval.md` 정성 검토
- `history.md §A.12.5` 결과 박제
- `PROJECT_HANDOFF.md D-038` 갱신 (PaddleOCR backend 채택, 결과)

---

### 🟢 [필수 5] Phase 15e — 박제 + commit + push (예상 30분)

```bash
cd /mnt/c/Users/user/github/Drawing

# 1. 변경 사항 확인
git status

# 2. add + commit
git add -A
git commit -m "Phase 15b~15d: Stage 3-A PaddleOCR-VL 통합 (D-013 V5)

- stage3_paddleocr_zero_shot_test.py 실행 + 결과
- 5장 + stage1_fp_notes 23개 평가
- stage3_alphabetical.py 백엔드 교체 (Donut → PaddleOCR-VL)
- D-038 Notes Rescue 재실행 (4% → 80%+)

박제:
- history.md §A.12.4 ~ §A.12.6
- PROJECT_HANDOFF.md D-038 갱신
- outputs/stage3a_v5_summary.txt
- docs/PHASE15_CHECKLIST.md 갱신"

# 3. push
git push origin main
```

---

### 🟡 [옵션 1] 독일어 ~10장 별도 평가 (예상 30분)

사용자가 독일어 도면 ~10장 재검색 후:

```bash
mkdir -p data/stage3a_eval_samples_de
# (사용자가 독일어 10장 복사)

python src/stage3_paddleocr_zero_shot_test.py \
    --samples-dir data/stage3a_eval_samples_de/ \
    --output-json outputs/stage3a_zero_shot_eval_de.json \
    --output-md   outputs/stage3a_zero_shot_eval_de.md
```

DE 가중치 = mid confidence (D-025 갱신 시 박제됨).

---

### 🟡 [옵션 2] Phase 16 — Donut Numerical fine-tune 시작 준비

15c 완료 후 시간 여유 있으면:

```bash
# Phase 14 venv 로 복귀 (ultralytics + Donut)
deactivate
source .venv/bin/activate

# Stage 3-N 학습 데이터 준비
# ★ 2026-05-05 인자명 수정 (실제 prepare_vlm_dataset.py CLI 일치):
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
```

**참고**: K-fold ensemble (5 fold) 은 prepare_vlm_dataset 에서 미지원 → 단일 `yolo_obb.pt` 사용.
Ensemble 적용은 Phase 17 e2e (pipeline.py + ensemble_predict.py) 에서.

★ Phase 16 학습 자체는 ~6h 이므로 다음 세션 또는 overnight 권장.

---

## 4️⃣ 예상 시간

| 단계 | 시간 | 누적 |
|---|---|---|
| 환경 활성화 + 검증 | 1분 | 1분 |
| 15b 실행 | 5분 | 6분 |
| 15b 정성 평가 (사용자) | 30분 | 36분 |
| 15c 백엔드 교체 | 1h | 1h 36분 |
| 15d Notes Rescue | 30분 | 2h 6분 |
| 15e 박제 + commit + push | 30분 | **2h 36분** |

**전체 ~2.5h** — 다음 세션 1회로 Phase 15 전체 마무리 가능.

---

## 5️⃣ 트러블슈팅 (자주 만나는 에러)

| 에러 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | 직접 실행 시 sys.path 미등록 | 스크립트에 bootstrap 있음 — 그대로 사용 |
| `AttributeError: text_config` | D-042 monkey-patch 미적용 | `load_model_and_processor()` 사용 (자동 적용) |
| `Cannot instantiate this tokenizer` | sentencepiece 누락 | `uv pip install sentencepiece protobuf einops` |
| `KeyError: 'default'` (ROPE) | transformers 5.6+ | `uv pip install "transformers==5.0.0"` 다운그레이드 |
| CUDA OOM | 다른 프로세스 점유 | `nvidia-smi` 확인 후 정리, `--max-new-tokens 512` |

---

## 6️⃣ 핵심 박제 위치 (Quick Reference)

- **`history.md §A.12.0 ~ A.12.3`** — Phase 15 시작 + 도메인 한계 + 환경 + 평가 스크립트
- **`PROJECT_HANDOFF.md §11 D-039 ~ D-044`** — 의사결정 박제
- **`PROJECT_HANDOFF.md §11.5`** — Google Drive 자산
- **`PROJECT_HANDOFF.md §11.6`** — D-038 stage1_fp_notes (Phase 15d 입력)
- **`docs/PHASE15_CHECKLIST.md`** — 단계별 체크리스트
- **`docs/GOOGLE_DRIVE_ASSETS.md`** — 외부 자산
- **`docs/modules/stage3_paddleocr_install_check.md`** — 환경 검증 모듈
- **`docs/modules/stage3_paddleocr_zero_shot_test.md`** — 평가 모듈

---

## 7️⃣ 마지막 점검 사항 (★ 작업 시작 전)

```bash
# 빠른 4-검증
echo "=== 1. git status ===" && git status | head -3
echo ""
echo "=== 2. venv 활성화 ===" && which python
echo ""
echo "=== 3. 모델 cache ===" && du -sh ~/.cache/huggingface/hub/models--PaddlePaddle--PaddleOCR-VL-1.5/ 2>/dev/null
echo ""
echo "=== 4. sample 도면 ===" && ls -la data/stage3a_eval_samples/
```

모두 정상이면 **Phase 15b 실행** (위 [필수 1]) 으로 시작!

---

**Last updated**: 2026-05-04 EOD (Phase 15b 작성 완료, 다음 세션 = 실행 + 박제)
