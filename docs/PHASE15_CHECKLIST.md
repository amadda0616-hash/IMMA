# Phase 15 ~ Phase 18 작업 체크리스트

> **Last updated**: 2026-05-04 (Phase 14 완료, GitHub IMMA 첫 push)
>
> Phase 14 (Stage 2 D-040 5-Fold Ensemble, D-023 PASS) 완료 → 다음 4개 phase 까지 남은 작업.
>
> 관련 문서: [`MANUAL.md`](../MANUAL.md) / [`PROJECT_HANDOFF.md §10`](../PROJECT_HANDOFF.md) / [`history.md`](../history.md)

---

## 진행 흐름 요약

```
Phase 14 (DONE) ─→ Phase 15 ─→ Phase 16 ─→ Phase 17 ─→ Phase 18
Stage 2 OBB        Stage 3-A    Stage 3-N    Pipeline     Step 8 + 9
ensemble PASS      PaddleOCR    Donut FT     E2E batch    Metrics + Enrichment
```

| Phase | 주제 | 예상 시간 | 핵심 검증 |
|---|---|---|---|
| **15** | Stage 3-A (PaddleOCR-VL-1.5 통합) | ~6h | V5 (D-013 다국어) |
| **16** | Stage 3-N (Donut Numerical fine-tune) | ~8h | V6 (★ D-023 critical) |
| **17** | Pipeline E2E batch 통합 | ~4h | V7 (D-021 ≤30s) |
| **18** | Step 8 metrics + Step 9 enrichment | ~4h | V9 (D-019/D-020) |

총 약 **22h** (학습 시간 포함, 실 작업 시간 ~12h).

---

## Phase 15 — Stage 3-A (PaddleOCR-VL-1.5 통합)

> **목적**: Donut DocVQA (D-018) 폐기 → PaddleOCR-VL-1.5 (D-039) 채택. 다국어 OCR + structured output.

### 15a. 환경 설치 + import 검증 (실측 ~ 1시간 시행착오 포함, **★ 2026-05-04 DONE**)

- [x] **★ 별도 venv 분리** (`.venv-paddleocr`) — Phase 14 ultralytics 환경과 충돌 회피
- [x] `src/stage3_paddleocr_install_check.py` 작성 (393 lines)
- [x] HuggingFace transformers 로 **`PaddlePaddle/PaddleOCR-VL-1.5`** 로드 — ★ 7차 시도 끝에 PASS
- [x] **★ Critical workaround (D-042)**: `config.text_config = config.get_text_config()` monkey-patch
- [x] 더미 이미지 inference 검증 — 2.26~3.47s
- [x] GPU 메모리 측정 — 3.29 GB used / 17.09 GB total

**실측 환경**:
- Python 3.10.20 / torch 2.11.0+cu128 / transformers **5.0.0**
- accelerate 1.13.0 / sentencepiece / protobuf / einops / pillow
- Model params: **0.91B** (논문 일치)

**Load + Inference 측정**:
| 항목 | 시간 |
|---|---|
| Load (warm cache) | 39.4 ~ 39.9s |
| Cold start (첫 다운로드, 1.92GB) | ~3분 |
| Inference (256×128 더미) | 2.26 ~ 3.47s |

**🎉 Phase 15a PASS** — 다음 단계 (15b) 진입 가능.

**박제 산출물**:
- `src/stage3_paddleocr_install_check.py`
- `outputs/stage3a_install_check.json`
- `history.md §A.12.1` ~ `§A.12.2` (시도 매트릭스 + 결과)
- `PROJECT_HANDOFF.md D-042` (monkey-patch 박제)

**Tip — 향후 다른 환경에서 재현**:
```bash
cd /mnt/c/Users/user/github/Drawing
uv venv --python 3.10 .venv-paddleocr
source .venv-paddleocr/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==5.0.0" accelerate sentencepiece protobuf einops pillow
python src/stage3_paddleocr_install_check.py
# 종료 코드 0 = PASS
```

### 15b. Sample 5장 zero-shot 정성 평가 (~1h) — ★ 2026-05-04 sample 7장 + DE 10장 확보

#### 사용자 보유 도면 (★ 2026-05-04)

| # | 언어 | 도면 식별 특징 | 평가 우선순위 |
|---|---|---|---|
| 1 | 영어 | "MOTOR MTG. PLATE" / 인도 SV ROBOTICS / MS / W.NO. 1087 | ★ 5장 평가 |
| 2 | 일본어 | 그리퍼 (SUS440C) / 東洋自動機 | 추가 평가 |
| 3 | 일본어 | 브쉬 (BSBM) / TT-10CW / **CRITICAL CONTROL DESIGNATION** 포함 | ★ 5장 평가 (더 풍부) |
| 4 | 한국어 | "수도전기공업고등학교 [42 과제]" / Ø80 | ★ 5장 평가 (유일, 학습용) |
| 5 | 러시아어 | FNINI.732214.001 / "Корпус" | ★ 5장 평가 |
| 6 | 중국어 | "规格零件图" / 간체 / JS-718 | ★ 5장 평가 (대륙 표준) |
| 7 | 중국어 | 700bar2 / 嵐統 (대만, 번체) | 추가 평가 |
| **★ 추가** | **독일어** | **~10장 보유 (D-025 6개 언어 확장)** | **별도 평가 (15b 후속)** |

#### 작업 흐름

- [x] `data/stage3a_eval_samples/` 디렉토리 생성 (★ 2026-05-04)
- [x] 5장 선별 저장 (각 언어 1장): `en/ja/ko/zh/ru_drawing.jpg`
- [x] **`src/stage3_paddleocr_zero_shot_test.py` 작성 (786 lines, ★ 2026-05-04)**
- [x] **★ TitleBlock Schema 23 필드 (D-044 박제)**: ISO 7200 + KS A 0005 + 첨부 이미지 통합
- [ ] **다음 세션 (스크립트 실행)**:
  - 5장 → 3 prompt (titleblock 23필드 / notes / full_text)
  - 결과 → `outputs/stage3a_zero_shot_eval.{json,md}`
- [ ] **★ 추가 평가 (15b 후속)**:
  - 독일어 ~10장 별도 batch 평가
  - 추가 일본어/중국어 (제외된 2장) 평가
- [ ] **임계값** (D-013):
  - 다국어 평균 char accuracy ≥ 0.85 → PASS
  - field-level F1 ≥ 0.80 → PASS
  - hallucination rate ≤ 0.05
- [ ] **★ 언어별 신뢰도 가중치** (D-043 박제):
  - 영어 1장 → low confidence (단일 sample)
  - 한국어 1장 → "학습용 한정" — 실 산업 도면 별도 검증 필요
  - 일본어 2장 → high confidence
  - 중국어 2장 → high confidence (간체 + 번체)
  - 러시아 1장 → mid confidence
  - **독일어 ~10장 → mid confidence** (sample 더 많지만 별도 검증)
- [ ] PASS 시 → 15c 진행
- [ ] FAIL 시 → fine-tune 검토 또는 폴백 (Qwen3-VL / DeepSeek-OCR-2)

### 15c. `src/stage3_alphabetical.py` 백엔드 교체 (~1h)

- [ ] 기존 Donut zero-shot 로직 → PaddleOCR-VL 통합
  - `load_model()` 시그니처 유지 (드롭인 호환)
  - `predict_one()` / `predict_titleblock()` / `predict_notes()` 동일 인터페이스
  - region_type 별 prompt 템플릿 (TitleBlock fields / Notes items)
- [ ] `pipeline.py` 와 호환성 검증 (lazy load + skip 인자)
- [ ] V5 검증기 (`src/validate/check_stage3a_alphabetical.py`) 호환 테스트

### 15d. D-038 Notes Rescue 재실행 (~30분)

**입력 자산 (★ 2026-05-04 박제)**: `outputs/skip_lists/stage1_fp_notes.txt`
- CAD_Drawing219 (14개) + sample_01266 (9개) = **23개**
- 모두 PMI 영역으로 분류됐지만 실제는 일반 주석 (Notes) 영역

**작업 흐름**:
- [ ] `data/stage1_fp_notes_crops/` 디렉토리 생성 + 23개 PMI crop 복사
  ```bash
  mkdir -p data/stage1_fp_notes_crops
  while read -r fname; do
      [[ "$fname" =~ ^# ]] && continue
      [[ -z "$fname" ]] && continue
      cp "data/annotation/images/train/${fname}" \
         "data/annotation/images/valid/${fname}" \
         "data/stage1_fp_notes_crops/" 2>/dev/null
  done < outputs/skip_lists/stage1_fp_notes.txt
  ls data/stage1_fp_notes_crops/ | wc -l   # 23 확인
  ```
- [ ] PaddleOCR-VL OCR 실행:
  ```bash
  source .venv-paddleocr/bin/activate
  python src/stage3_paddleocr_zero_shot_test.py \
      --samples-dir data/stage1_fp_notes_crops/ \
      --prompts notes,full_text \
      --output-json outputs/stage1_fp_notes_paddleocr_eval.json \
      --output-md   outputs/stage1_fp_notes_paddleocr_eval.md
  ```
- [ ] **결과 비교**:
  | Backend | 성공률 | 판정 |
  |---|---|---|
  | Donut DocVQA (D-038 1차) | 4% | ❌ deprecated |
  | PaddleOCR-VL (D-038 2차) | 목표 ≥ 80% | ★ |
- [ ] (선택) `src/rescue_misclassified_notes.py` 에 `--backend paddleocr-vl` 옵션 추가 (코드 정식 통합)
- [ ] 통합: 추출 텍스트 → Stage 3-A `general_notes` 필드
- [ ] 박제: `history.md §A.12.x` + `PROJECT_HANDOFF.md D-038` 갱신

### 15e. 박제 + commit + push (~20분)

- [ ] `outputs/stage3a_paddleocr_summary.txt` 작성 (V5 통과 보고서)
- [ ] `history.md §A.12 (Phase 15 시작)` 추가
- [ ] `PROJECT_HANDOFF.md §10 Day 3` 갱신
- [ ] `docs/modules/stage3_alphabetical.md` 갱신 (PaddleOCR-VL 통합)
- [ ] git commit + push

---

## Phase 16 — Stage 3-N (Donut Numerical fine-tune)

> **목적**: Donut Numerical fine-tune (논문 정합 유지) — Measure / GDT / Roughness 의 OBB crop → JSON 변환.
>
> ★ D-023 critical (numerical_accuracy / tolerance_match / per-class F1) 검증 핵심.

### 16a. 학습 데이터 준비 (~1h)

- [ ] `src/prepare_vlm_dataset.py numerical` 실행
  - Stage 1 + Stage 2 ensemble 추론 → de-rotation crop → JSON 템플릿 자동 생성
  - 입력: `dataset/` (5,839 장) 또는 일부 (~500장)
  - 출력: `data/vlm/numerical/{train,valid}/{images,jsons}/`
- [ ] manifest 검증 (group leak 0 확인)
- [ ] 사용자 검수 (~30분, sample 50건 정성 검증)

### 16b. fine-tune 학습 (~6h)

- [ ] `src/stage3_numerical.py train` 실행
  - 기본: epoch 30, batch 4, FP16 (RTX 5080 16GB)
  - resume 기능 활용 (학습 도중 중단 시)
- [ ] 학습 로그 모니터링 (loss curve, val F1)
- [ ] best checkpoint 저장 — `checkpoints/donut_numerical/final/`

### 16c. V6 검증 (★ D-023 critical) (~30분)

- [ ] `python -m src.validate.check_stage3n_numerical \
    --weights checkpoints/donut_numerical/final \
    --val data/vlm/numerical/valid/`
- [ ] **임계값** (D-023):
  - Measure F1 ≥ 0.85 (논문 0.92 목표)
  - GDT F1 ≥ 0.90 (논문 0.95 목표)
  - Roughness F1 ≥ 0.50 (논문 baseline)
  - numerical_accuracy ≥ 0.85
  - tolerance_match (±공차 동시 추출) ≥ 0.80
  - hallucination_rate ≤ 0.05 (★ critical)
- [ ] PASS 시 → 17 진행
- [ ] FAIL 시 → 폴백 결정 (Qwen3-VL / PaddleOCR-VL / DeepSeek-OCR-2)

### 16d. 박제 (~20분)

- [ ] `outputs/stage3n_v6_summary.txt`
- [ ] `history.md §A.13` 추가
- [ ] PROJECT_HANDOFF.md 갱신
- [ ] git commit + push

---

## Phase 17 — Pipeline E2E Batch 통합

> **목적**: 전체 도면 → 통합 JSON e2e 검증. D-021 (≤30s/도면) 충족 확인.

### 17a. Pipeline batch 실행 (~2h)

- [ ] sample batch (10 ~ 50장) 으로 정상 동작 확인
  ```bash
  python src/pipeline.py batch \
      --input-dir dataset/ \
      --out-dir outputs/json \
      --device cuda:0 \
      --limit 50
  ```
- [ ] timing 분석:
  - cold start (첫 도면) — ~45s
  - warm batch — 평균 ~5-10s/도면 예상
  - D-021 (≤30s) 평균 통과 검증
- [ ] error rate 측정 (log + summary JSON)

### 17b. V7 (e2e 검증) (~1h)

- [ ] `src/validate/check_pipeline_e2e.py` 실행
  - 13개 항목 (field_f1 / TB+Notes / per-class detection D-023 재측정 / numerical / per-stage timing / failure_rate)
  - GT 데이터 (사용자 수동 라벨링) 5~10건 입력
- [ ] **임계값**:
  - field_level_f1 ≥ 0.80
  - drawing-level recall ≥ 0.85 (Stage 2 수준 재현)
  - per-stage avg timing ≤ 30s/도면
- [ ] FAIL 항목 분석 + 수정

### 17c. 박제 (~30분)

- [ ] `outputs/v7_summary.txt`
- [ ] `history.md §A.14` 추가
- [ ] PROJECT_HANDOFF.md 갱신
- [ ] git commit + push

---

## Phase 18 — Step 8 (Metrics) + Step 9 (Enrichment)

> **목적**: 평가 지표 통합 + Step 9 metadata enrichment 사후 검증.

### 18a. Step 8 — `src/utils/metrics.py` 통합 검증 (~1h)

- [ ] 기존 metrics 라이브러리 (15 sanity test PASS 완료) 가 V7 결과 활용 가능한지 확인
- [ ] 필요 시 metrics 추가 (예: per-language gap)
- [ ] V7 → metrics CSV 자동 생성

### 18b. Step 9 — `src/stage5_enrichment.py` 실행 (~1h)

- [ ] V7 통과 한 unified JSON 들 → enrichment 실행
- [ ] 4-tier cascade (deterministic → heuristic → LLM → HITL)
- [ ] provider: Mock 만 사용 (D-019 / D-020 — LLM 의존성 옵션)

### 18c. V9 검증 (~30분)

- [ ] `src/validate/check_enrichment.py` 실행
- [ ] **임계값** (D-019 / D-020):
  - provenance_completeness = 1.0 (★ critical)
  - llm_method_rate < 0.40
  - hitl_flag_rate < 0.25
  - empty_suggestion_rate < 0.10
  - cost_per_drawing ≤ $0.005

### 18d. 최종 박제 (~1h)

- [ ] `outputs/v9_summary.txt`
- [ ] `outputs/final_summary.md` (전체 V0 ~ V9 종합)
- [ ] `history.md §A.15` (최종 박제)
- [ ] PROJECT_HANDOFF.md §10 → DONE 표시
- [ ] README.md 갱신 (모든 단계 ✅)
- [ ] git tag `v1.0-phase18-complete`
- [ ] git push --tags

---

## 차후 검토 항목 (★ 박제, 우선순위 낮음)

| 항목 | 트리거 | 액션 |
|---|---|---|
| Stage 1 Version B 재학습 | Day 1 SKIP 33% > 30% 임계 | Active Learning + 5,839 풀 데이터 학습 |
| PaddleOCR-VL-2.0 출시 | 2026 H2 추정 | 평가 + 폴백 결정 |
| Roboflow auto-label 추가 | 라벨링 부담 시 | Step 5.5 활성화 |
| Weighted Box Fusion (WBF) | Stage 2 ensemble 정밀도 개선 시 | `manual_nms_rotated` 대체 |
| top-3 fold ensemble | 추론 속도 부족 시 | `--n-folds 3` |

---

## 진행 상황 추적

| Phase | 시작일 | 완료일 | V_X 검증 | 비고 |
|---|---|---|---|---|
| 14 (Stage 2 ensemble) | 2026-05-03 | **2026-05-04** | V3-B PASS ★ | D-040, D-023 PASS |
| **15a (Stage 3-A 환경)** | **2026-05-04** | **2026-05-04** | install_check PASS | **D-042 monkey-patch 박제** |
| **15b 작성 (script + schema)** | **2026-05-04** | **2026-05-04** | code ready | **D-043 도메인 한계 + D-044 23 필드 박제** |
| **15b 1차 실행** | **2026-05-05** | **2026-05-05** | ❌ degenerate | **D-045 박제 — repetition_penalty + prompt 단순화** |
| **15b 2차 실행 (D-045 적용)** | **2026-05-05** | **2026-05-05** | ⚠ 부분 성공 | **D-046 박제 — README sample code 발견 (task keyword + bfloat16)** |
| **15b 3차 실행 (D-046 적용)** | **2026-05-05** | **2026-05-05** | ⚠ V5 미통과 (FAIL) | **D-047 박제 — OTSL token + en/ru 부분 성공 (avg 0.50)** |
| **15b 4차 실행 (Real-ESRGAN)** | **2026-05-05** | **2026-05-05** | ⚠ V5 부분 PASS (avg 0.69) | **★ ko/zh 큰 향상** — Real-ESRGAN 동아시아 효과 입증 |
| **15b ★ Stage 1 ja 분리 검증** | **2026-05-05** | **2026-05-05** | ✅ 110 region | **D-048 박제 — 사용자 가설 검증 PASS** |
| 15c ja 영역별 Stage 3-A 평가 | TBD (내일) | TBD | V5 (ja 부분) | outputs/crops/ja_drawing/ 활용 |
| 15c stage3_alphabetical.py 백엔드 교체 | TBD (내일~) | TBD | V5 | Donut → PaddleOCR-VL |
| 15d Notes Rescue 재실행 | TBD | TBD | — | stage1_fp_notes 23개 PaddleOCR backend |
| 15c (백엔드 교체) | TBD | TBD | V5 | stage3_alphabetical.py |
| 15d (Notes Rescue 재실행) | TBD | TBD | — | stage1_fp_notes 23개 PaddleOCR backend |
| 16 (Stage 3-N) | TBD | TBD | V6 ★ | Donut fine-tune ★ D-023 critical |
| 17 (E2E pipeline) | TBD | TBD | V7 | D-021 timing |
| 18 (Step 8+9) | TBD | TBD | V9 | 최종 |

---

## 참고 자료

- [`MANUAL.md`](../MANUAL.md) — 단계별 작업 가이드
- [`PROJECT_HANDOFF.md`](../PROJECT_HANDOFF.md) — 사양 + 의사결정 박제
- [`history.md`](../history.md) — 모든 학습/실험 시도 기록
- [`docs/GOOGLE_DRIVE_ASSETS.md`](./GOOGLE_DRIVE_ASSETS.md) — 외부 자산 가이드
- [`docs/modules/`](./modules/) — 모듈별 상세 문서

---

**Last updated**: 2026-05-04 (Phase 14 완료, Phase 15 진입 대기)
