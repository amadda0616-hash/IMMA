# IMMA — Version B (★ Pragmatic Closure 완료, 2026-05-06 ~ 2026-05-10)

> ★★★ **버전 B Pragmatic Closure 완료 (2026-05-10)** — 1280+ ko/ja 도면 (~2,133개) + YOLO26 + Teacher 30B 직접 production.
> 버전 A 종료 (2026-05-06) 기록은 [`README_ver.A.md`](./README_ver.A.md) 참조.
>
> **2026-05-10 V.B Pragmatic Closure 완료 시점**: Stage 6 multi_drawing detect 통합 완료 (D-085) + Step 14 Web Service 팀 hand-off 문서 4건 작성 + V7 Fallback E2E ALL PASS (4/4). distillation 가설 부분 실패 (D-083) → Teacher 30B 직접 production 채택 (D-084) → pipeline_ver.B.py Stage 0~6 통합 (Phase 2) → colab A100 환경 V7 검증 (Phase C 1469 결과 활용, multi_drawing conf 0.90~0.95 정확 분류, Pydantic violations=0). 부수 발견 3건 (sync 누락 + dataclass fix + truncate 사고) §B.14 박제. 다음 단계: Step 14 외부 팀 + V.D (detector v3 / larger Student / View 인식 보강).

---

## 0. 버전 B 개요 (★ TBD — 진행 중 갱신)

| 항목 | 버전 A | 버전 B |
|---|---|---|
| 데이터셋 | 5,839장 (6개 언어) | ★ **1280+ ko/ja (~2,133개)** |
| YOLO 모델 | YOLOv11 | ★ **YOLO26** |
| OCR (Stage 3-A) | PaddleOCR-VL-1.5 only | ★ **멀티 OCR 비교** (TBD) |
| VLM (Stage 3-N) | Donut Numerical | TBD (멀티 OCR 결과 후 결정) |
| 환경 | 로컬 RTX 5080 16GB | ★ **로컬 + 코랩 Pro+** (600 컴퓨팅) |
| GDT 라벨 | 88개 (2.6%) | ★ TBD (데이터셋 분석 후 추가 라벨링) |
| ★ 색상 정규화 | wrapper 자동 통합 (D-058 ★ Resolved) | ★ **모델 단계 통합** (CV-mask + augmentation, D-063) |

---

## 1. 진행 상태 (★ LIVE — 갱신 중)

| Step | 작업 | 상태 |
|---|---|---|
| Step 1~3 | 버전 A 종료 + 파일명 rename + 신규 문서 | ✅ 완료 |
| Step 4-pre / 5-pre | ★ §4 V.B 전략 + §5 웹서비스 아키텍처 (PROJECT_HANDOFF_ver.B.md) | ✅ 2026-05-06 |
| ★ ★ **Step 4** | ★ ★ 데이터셋 분석 (`analyze_dataset_ver.B.py`) — unique 1,515 / Avg 1.41× / PMI 127,477 | ✅ **2026-05-07** |
| **Step 4-post** | ★ ★ ★ ★ ★ **D-061 Resolved** (옵션 X / 방식 D / 500 unit 풀활용 / 1주 ETA) + ★ ★ **D-066 ★ ★ JP 1,500 만 (Q24)** + D-067~069 Pipeline 재구성 | ✅ **2026-05-07** |
| Day 1 (Step 5/7 + Stage 1.5 crop + Streamlit) | uv `.venv-vb` 셋업 + YOLO26 환경 + 신규 모듈 작성 | ✅ 2026-05-06 |
| Day 2 | Teacher prompt + Pydantic schema + Streamlit 마무리 + Pipeline 확장 | ✅ 2026-05-07 |
| Day 3 | colab 셋업 + Multi-OCR 비교 30장 (zero-shot) | ✅ 2026-05-07 |
| **Day 4** (Phase 1 1차) | Teacher 풀라벨링 1,515장 시도 → 1/5 fail (Thinking 모델) | ⚠️ **2026-05-07** |
| **★ ★ Day 5** (Phase 1 재시도) | Day 5-1/2 prompt + parser 강화 → Day 5-3 Instruct 회귀 → B-fix6~10 backend chain → ★ ★ **D-070 Schema trade-off** (Resolved) → **B-fix11 옵션 B 회귀** | ⚠️ 2026-05-08 |
| **★ ★ Day 5-4** (Phase A 1차 + B-fix12) | B-fix11 Phase A 5장 → 0/5 (CJK 부작용) → **D-071 Resolved** (B-fix12: penalty 0) | ⚠️ 2026-05-08 |
| **★ ★ ★ Day 5-4** (Phase A 2차 + B-fix13) | B-fix12 Phase A → 1/5 (list 폭주) → D-072 (B-fix13 3 phase plan) | ⚠️ 2026-05-08 |
| **★ ★ ★ ★ Day 5-4** (B-fix13a Phase 1 → X2) | B-fix13a (prompt) → 1/5 (Agent 가설 효과 0%) → **D-073** (X2 가설 E parser 강화) → ★ ★ ★ **4/5 partial 압도적 검증** | ✅ 2026-05-08 |
| **★ ★ ★ ★ ★ Day 5-4** (X1 max_tokens 8192) | sample_00003 fail = string truncate → **D-074 Resolved** (max_tokens 8192) → ★ ★ ★ ★ **5/5 success 100% 검증** (1 normal + 4 partial) | ✅ 2026-05-08 |
| **★ ★ ★ ★ ★ Day 5-4** (Plan B 채택) | 잔여 518 unit → Plan B 결정 (max_pixels 4M, 단일 Phase C 18.75h, 7B Student, 마진 118) → ★ ★ ★ **D-075 Resolved** | ⚠️ **2026-05-08 (Phase B 30장 sanity 대기)** |
| **★ ★ ★ ★ Day 5-4** (Phase B 90% + D-076) | Phase B 27/30 (90% / normal 33%) → 3 fail = parser X2 한계 + 시간 23.7h → ★ ★ ★ **D-076 Resolved** (parser-v2 alternation 확장 + 한국어 처리 + max_tokens 6144 = ETA 17.8h) | ⚠️ **2026-05-08 (옵션 3 5장 sanity 대기)** |
| **★ ★ ★ ★ Day 5-4** (옵션 3 + D-077 parser-v3) | 옵션 3 = 53/53 success (3 fail 회수 + normal 40%) → 새 4 fail 발견 → ★ ★ ★ **D-077 parser-v3** (pat5/pat6 + stack 기반 repair) | ⚠️ **2026-05-08 (parser-v3 retest 대기)** |
| ★ ★ ★ ★ ★ Day 5-4 (Phase C 1500 종료) | parser-v3 retest 100% 회수 → Phase C 1차 13.5h (1124) → resume 4.6h (345 success + 31 fail) → ★ ★ ★ **D-078 Resolved** (1469/1500 = 97.93%, Phase 2 quality units 1,191) | ✅ **2026-05-10 01:37 KST** |
| ★ ★ ★ ★ ★ Day 5-5 (Phase 2 LoRA 학습 ★ 완료) | Phase 2 dataset 변환 (train 1166 + val 129) → Qwen2.5-VL-7B + LoRA r=32 + bf16 + 4h 31분 → ★ **eval_loss 4.03 (best ckpt-400, adapter 380MB)** → ★ ★ ★ **D-079 Resolved** | ✅ **2026-05-10 07:25 KST** |
| Day 5-6+ (Phase 1.5 + Phase 3) | ★ ★ ★ Phase 1.5 Quick scan 5/5 = 100% multi_drawing → ★ ★ ★ **D-081 Resolved** (★ 38개 + sample_00003 = 39개 ★ 일괄 제외, 수동 검수 ★ ★ 완전 스킵) → 학습 데이터 ★ ★ **1500 → 1461** → Phase 3 V10 validation (Student vs Teacher) | ✅ **2026-05-10 박제 / Phase 3 진행 중** |
| ★ ★ ★ ★ ★ ★ Day 5-6 V.B 종료 | Phase 3 Hot-fix 1~6 누적 + Phase 2 재학습 실패 (eval_loss 4.03→5.33) → ★ ★ **distillation 가설 부분 실패** 박제 → ★ ★ ★ V.C Pragmatic 전환 결정 (D-083 + D-084 Resolved) | ✅ **2026-05-10 V.B 종료** |
| ★ ★ ★ ★ ★ ★ V.B Pragmatic Closure (V.C 명명 폐기) | Teacher 30B 직접 production + Stage 6 multi_drawing 통합 + Step 14 hand-off 문서 4건 | ✅ **2026-05-10 완료** |
| ↳ Phase 1 (Validation 1~6) | Teacher 30B + parser-v5 + Stage 1 + schema + multi_drawing v1 + pipeline 통합 + Drive 환경 | ✅ V1~V6 PASS |
| ↳ Phase 2 (Stage 6 통합 + E2E) | pipeline_ver.B.py 4곳 patch (시그니처/Stage 6 block/CLI flag/호출) → multi_drawing 3/3 PASS conf 0.90~0.95 | ✅ |
| ↳ Phase 3 (박제 + Step 14) | `docs/V_B_OUTPUT_SCHEMA.md` + `docs/V_B_PRODUCTION_GUIDE.md` + D-085 박제 + 본 README 갱신 | ✅ |
| ★ Step 14 (★ Web Service 팀 hand-off) | 출력 schema + 운영 가이드 hand-off 완료 → FastAPI / PostgreSQL 통합 = Web Service 팀 작업 | 🔄 (hand-off 완료, 통합은 별도) |
| V7 Fallback E2E test | colab A100 40GB 환경 — Phase C 결과 + Pipeline E2E + Stage 6 검증 (4 sample ALL PASS, conf 0.90~0.95) | ✅ **2026-05-10 완료** (B.14) |
| Step 15 / V.D | larger Student 32B 재학습 / KL distillation / detector v3 / 회사명 (logo OCR) / View 인식 보강 | ⏳ (별도) |

---

## 2. 핵심 문서 (★ 2026-05-10 V.B Pragmatic Closure 갱신)

### V.B Pragmatic Closure 산출물 (Step 14 hand-off, 2026-05-10 신규)
- [`docs/V_B_OUTPUT_SCHEMA.md`](./docs/V_B_OUTPUT_SCHEMA.md) — **JSON 출력 schema 명세** (Web Service 팀 hand-off, 8개 섹션 + 예시 + 한계)
- [`docs/V_B_PRODUCTION_GUIDE.md`](./docs/V_B_PRODUCTION_GUIDE.md) — **운영 환경 가이드** (셋업 + CLI + 트러블슈팅 10개 섹션)
- [`colab_sync/colab_validation_v_b_closure.ipynb`](./colab_sync/colab_validation_v_b_closure.ipynb) — Validation 1~7 노트북 (V1~V6 PASS)
- [`colab_sync/colab_v7_e2e_test_ver.B.ipynb`](./colab_sync/colab_v7_e2e_test_ver.B.ipynb) — **V7 Fallback E2E** 노트북 (21 cells, 4 sample ALL PASS)
- [`src/preprocess/multi_drawing_detector_ver.B.py`](./src/preprocess/multi_drawing_detector_ver.B.py) — Stage 6 multi_drawing detector v1 (5 기준)
- [`src/pipeline_ver.B.py`](./src/pipeline_ver.B.py) — Stage 0~6 통합 pipeline + `_import_dot_module()` fix (D-085, 16,101 bytes)

### 인수인계 / 시간순 / 가이드
- [`PROJECT_HANDOFF_ver.B.md`](./PROJECT_HANDOFF_ver.B.md) — D-059~085 Resolved + §13 변경 로그 (2026-05-10 갱신, D-085 신규)
- [`history_ver.B.md`](./history_ver.B.md) — §B.1 ~ §B.13 (V.B 종료 + Pragmatic Closure 박제)
- [`MANUAL(local)_ver.B.md`](./MANUAL(local)_ver.B.md), [`MANUAL(colab)_ver.B.md`](./MANUAL(colab)_ver.B.md)
- [`instruction_ver.B.md`](./instruction_ver.B.md) — Quick Start
- [`label_manual_ver.B.md`](./label_manual_ver.B.md) — 라벨링 매뉴얼

### V.B Pipeline / 분석 / 한계
- [`docs/VERSION_B_PIPELINE_DETAILED.md`](./docs/VERSION_B_PIPELINE_DETAILED.md) — V.B 1주 작업 master
- [`docs/VERSION_B_PIPELINE.md`](./docs/VERSION_B_PIPELINE.md), [`docs/VERSION_B_BACKLOG.md`](./docs/VERSION_B_BACKLOG.md), [`docs/VERSION_B_DATASET_ANALYSIS.md`](./docs/VERSION_B_DATASET_ANALYSIS.md)
- [`docs/KNOWN_LIMITATIONS_ver.B.md`](./docs/KNOWN_LIMITATIONS_ver.B.md), [`docs/V_B_WORKFLOW_DAILY_ver.B.md`](./docs/V_B_WORKFLOW_DAILY_ver.B.md)

### Day 4~5 박제 (Phase 1 시도 + Schema trade-off)
- [`docs/V_B_DAY4_SMOKE_TEST_5_RESULTS_ver.B.md`](./docs/V_B_DAY4_SMOKE_TEST_5_RESULTS_ver.B.md) — Day 4 1/5 fail 박제
- [`docs/V_B_DAY5_PROMPT_FIX_STRATEGY_ver.B.md`](./docs/V_B_DAY5_PROMPT_FIX_STRATEGY_ver.B.md) — Day 5-1/2 prompt + parser 시방서
- [`docs/V_B_DAY5_MODEL_SWITCH_ver.B.md`](./docs/V_B_DAY5_MODEL_SWITCH_ver.B.md) — Thinking → Instruct 회귀 의사결정
- ★ ★ ★ [`docs/V_B_DAY5_SCHEMA_ENFORCEMENT_ANALYSIS_ver.B.md`](./docs/V_B_DAY5_SCHEMA_ENFORCEMENT_ANALYSIS_ver.B.md) — **D-070 Schema 강제 vs No-schema trade-off** (2026-05-08 신규)
- ★ ★ [`docs/V_B_DAY5_SCHEMA_RESEARCH_FINDINGS_ver.B.md`](./docs/V_B_DAY5_SCHEMA_RESEARCH_FINDINGS_ver.B.md) — **D-070 외부 검증** (Agent 위임, GitHub Issues + OpenAI 정책 + Qwen EOS 일치 확인, 2026-05-08 신규)
- ★ ★ [`docs/V_B_DAY5_BFIX11_VALIDATION_PLAN_ver.B.md`](./docs/V_B_DAY5_BFIX11_VALIDATION_PLAN_ver.B.md) — **B-fix11 검증 plan** (Phase A 5장 → Phase B 30장 → Phase C 1500장, 2026-05-08 신규)
- ★ ★ ★ [`docs/V_B_DAY5_JP_SAMPLING_RESEARCH_ver.B.md`](./docs/V_B_DAY5_JP_SAMPLING_RESEARCH_ver.B.md) — **D-071 외부 검증** (Agent 위임, vLLM #41985 직접 증거 + token-level CJK 메커니즘, 2026-05-08 신규)
- ★ ★ ★ [`docs/V_B_DAY5_LIST_REPETITION_RESEARCH_ver.B.md`](./docs/V_B_DAY5_LIST_REPETITION_RESEARCH_ver.B.md) — **D-072 외부 검증** (Agent 위임, list 무한 반복 4 가설 검증 — sampling 만의 문제 X, prompt + max_tokens 결합 본질, 2026-05-08 신규)
- [`docs/V_B_PHASE2_FINETUNE_PLAN_ver.B.md`](./docs/V_B_PHASE2_FINETUNE_PLAN_ver.B.md) — Phase 2 LoRA / QLoRA 잠정 계획

### 리서치 / 실행
- ★ [`RESEARCH_SOURCES.md`](./RESEARCH_SOURCES.md) — 권장 검색 사이트 + Exa/Firecrawl 우회 정책
- ★ [`colab_run_ver.B.ipynb`](./colab_run_ver.B.ipynb) — Instruct 회귀 + smoke + 본 추론 통합 노트북

## 3. 버전 A 참조 (★ 비교 / 박제 보존)

- [`README_ver.A.md`](./README_ver.A.md) — 버전 A 결과 + Day 1~4 + ★ D-058 자동 색상 정규화 통합
- [`PROJECT_HANDOFF_ver.A.md`](./PROJECT_HANDOFF_ver.A.md) — D-001 ~ D-058 의사결정 (★ D-058 색상 정규화 Resolved)
- [`docs/KNOWN_LIMITATIONS_ver.A.md`](./docs/KNOWN_LIMITATIONS_ver.A.md) — 버전 A 한계 + 추천 해결
- [`history_ver.A.md`](./history_ver.A.md) — 버전 A 시간순 이력 (3700+ lines, §A.1 ~ §A.12.20)
- ★ `src/preprocess_color_normalize.py` — ★ **버전 A/B 공통 모듈** (D-058 색상 정규화)

---

## 4. 갱신 정책

- 본 문서는 ★ LIVE 진행 현황. 버전 B 작업 시작 시 즉시 갱신.
- 의사결정 박제: `PROJECT_HANDOFF_ver.B.md` D-059 부터 (★ D-058 은 버전 A 종료 직전 Resolved).
- 시간순 이력: `history_ver.B.md` §B.1 부터.
- 버전 B 종료 시 본 문서는 final 버전 → `README.md` (버전 통합) 또는 그대로 유지.

---

> ★ 본 문서는 placeholder 입니다. 버전 B 진�