# IMMA — Version D (V.D Phase, 2026-05-10 ~)

> **버전 D 시작 (2026-05-10)** — V.B Pragmatic Closure 종료 후 발견된 view ~37% / table 0% recall 한계 해결을 위한 V.D phase.
> 버전 B Pragmatic Closure 종료 시점 기록: [`README_ver.B.md`](./README_ver.B.md)
> 버전 A 종료 (2026-05-06): [`README_ver.A.md`](./README_ver.A.md)
>
> **V.D 시작 시점 (2026-05-10)**: V.B 종료 시점 V7 Fallback E2E (4 sample) 검증에서 view ~37% / table 0% recall 발견 → V.D Task 1 (Prompt 강화) 부터 단계별 진행. 동일 모델 family 의 known cases (8건, 핵심: Qwen3-VL #1257 + vLLM #27157 + Businessware engineering drawing benchmark) 검색 근거로 우선순위 결정.
>
> **V.D Task 1 결과 (2026-05-10, D-087)**: **FAIL** — view 36.7% → **24.6%** (-12.1%p, **악화**), table 0% → 0%. Qwen3-VL-30B-A3B-Instruct-FP8 의 prompt 강화로 region recall 개선 불가 확인 — model-level 변경 (Task 3 FP16 / Task 6 fine-tune / Task 7 Teacher 교체) 필요. 또는 prompt 재설계 + image-level 시도 (Opt-1 / Opt-2 / Opt-3) 중 사용자 선택 대기.
>
> **V.D Task 1 Diagnostics + Task 2 코드 (2026-05-10, D-088)**: 검증 1+2+3 종합 → **가설 1 (Lost in the middle) STRONG dominant 확정**. 검증 2 sample_00270 mini-test 에서 prompt 위치 변경 (mirror 패턴) 만으로 **view 1→5, table 0→2 (100% recall)** 완전 회복.
>
> **V.D Task 3 (D-090, 2026-05-11 overnight 200 sample)**: V.D Task 2 200 sample (view 58.2%, table 34.4%) + V.D Task 3 200 sample (view 56.8%, table 30.7%). **결정적 발견**: V.B Phase C 진짜 baseline 이 4 sample 측정 (36.7%/0%) 과 큰 차이 — **200 sample 평균 52.3%/38.0%/99.3%** (운영 가치 수준). prompt + inference level fix 한계 도달 → **Qwen VL family domain weakness 확정** → **V.D phase 잠정 종료**. 다음 단계 5 옵션 (A/B/C/D/E) 은 `docs/V_D_PHASE_CLOSURE_ver.D.md` 참조, 사용자 재부팅 후 결정.

---

## 0. V.D Phase 개요

| 항목 | V.B (종료 baseline) | V.D (진행 중) |
|---|---|---|
| Phase status | ✅ Pragmatic Closure 완료 | 🔄 진행 중 |
| 핵심 문제 | distillation 가설 부분 실패 (D-083) | view ~37% + table 0% recall (V.B 종료 시 발견) |
| 해결 전략 | Teacher 30B 직접 production (D-084) | Prompt 강화 (Task 1) → 단계별 |
| Pipeline | `src/pipeline_ver.B.py` (Stage 0~6 통합) | V.B 그대로 사용 (재사용 매트릭스 §3) |
| Teacher prompt | `src/teacher_prompts_ver.B.py` (Day 5) | `src/teacher_prompts_ver.D.py` (V.B + 변경 A+B+C) |
| 검증 환경 | colab A100 40GB + Drive 15GB | 동일 (V.D 검증 노트북 신규) |

---

## 1. V.D 진행 상태 (LIVE — 갱신 중)

| Task | 작업 | 상태 |
|---|---|---|
| Task 1 (Prompt 강화) | `teacher_prompts_ver.D.py` 변경 A+B+C — region 갯수 강제 | ✅ 코드 완료 |
| Task 1 검증 | colab Teacher 30B + V.D prompt 4 sample 추론 + V.B 비교 | ❌ **2026-05-10 FAIL** (D-087): view 36.7% → **24.6%** (-12.1%p) |
| Task 1 Diagnostics | 검증 1 (raw_len) + 검증 2 (sample_00270 v2 100%) + 검증 3 (14 known cases) | ✅ **2026-05-10 STRONG** (D-088): 가설 1 (Lost in the middle) dominant 확정, mirror 패턴 100% 입증 |
| **Task 2 (Mirror 패턴)** | `teacher_prompts_v_d_task2_ver.D.py` 변경 D — prompt 시작 + 끝 양쪽 mirror | ✅ 코드 + 검증 완료 |
| Task 2 검증 (4 sample) | sample_00001/00193/00270/00678 (D-088) | △ PARTIAL (view 50.8%, 4 sample biased) |
| **Task 3 (FP8 우회 + 표현 완화)** | `teacher_v_d_task3_ver.D.py` (FP8→bf16 compute) + `teacher_prompts_v_d_task3_ver.D.py` (positive overcorrection) | ✅ 코드 + overnight 검증 완료 |
| **Task 3 검증 (200 sample, overnight)** | V.B 52.3% → Task 2 58.2% (+5.9%p) → Task 3 56.8% (+4.5%p), table V.B 38% → Task 3 30.7% **악화** (D-090) | ❌ marginal + table 악화 |
| **V.D phase 잠정 종료 (D-090)** | prompt + inference level fix 한계 도달 → Qwen domain weakness 확정 | ✅ **2026-05-11 잠정 종료** |
| **다음 결정 (D-091 예정)** | `docs/V_D_PHASE_CLOSURE_ver.D.md` 5 옵션 (A: V.B 운영 / B: padding / C: fine-tune / D: Gemini / E: hybrid) | 🔄 **재부팅 후 결정** |
| Task 4 (정사각 padding) | 사례 4, 후순위 (효과 가능성 낮음) | 옵션 B (보류) |
| Task 6 (Fine-tuned variant) | Glazkov-style, 장기 | 옵션 C |
| Task 7 (Teacher 교체) | Gemini Pro/Flash 검증 | 옵션 D (가장 결정적) |
| Phase C 1500 재추론 | 옵션 결정 후 (필요 시) | 보류 |

---

## 2. V.D 신규 산출물 (2026-05-10)

### 코드
- [`src/teacher_prompts_ver.D.py`](./src/teacher_prompts_ver.D.py) — V.D Task 1 핵심 변경 (V.B base + 변경 A+B+C, 23,377 bytes)
- [`colab_sync/src/teacher_prompts_ver.D.py`](./colab_sync/src/teacher_prompts_ver.D.py) — colab sync (B.14.3 정책)

### 노트북
- [`colab_sync/colab_v_d_task1_validation_ver.D.ipynb`](./colab_sync/colab_v_d_task1_validation_ver.D.ipynb) — V.D Task 1 검증 (21 cells, Teacher 30B 다운로드 + V.D prompt 추론 + V.B 비교)

### 문서
- [`docs/V_D_TASK1_PROMPT_ENHANCEMENT_ver.D.md`](./docs/V_D_TASK1_PROMPT_ENHANCEMENT_ver.D.md) — Task 1 변경 사항 + 검증 plan + 결과 박제 슬롯
- 본 [`README_ver.D.md`](./README_ver.D.md) — V.D phase baseline
- [`PROJECT_HANDOFF_ver.D.md`](./PROJECT_HANDOFF_ver.D.md) — V.D 의사결정 박제 (D-086 V.D 시작)

---

## 3. V.A / V.B 재사용 매트릭스 (V.D 에서 그대로 사용하는 파일)

V.D 작업은 V.A / V.B 의 검증된 모듈을 최대한 재사용. 신규 작성 불필요한 파일은 그대로 유지.

### 3.1 V.B 그대로 사용 (수정 없음)

| 파일 | 용도 | 사유 (V.D 신규 작성 불필요) |
|---|---|---|
| `src/pipeline_ver.B.py` | Pipeline Stage 0~6 통합 + Stage 6 multi_drawing | V.D Task 1 변경은 prompt 만 — pipeline 은 teacher_gt JSON 만 받아서 처리하므로 prompt 변경에 영향 X |
| `src/preprocess/multi_drawing_detector_ver.B.py` | Stage 6 multi_drawing 감지 | V.B 에서 검증 완료 (V7 4 sample ALL PASS, conf 0.90~0.95) |
| `src/validate/schema_ver.B.py` | Pydantic schema (TitleBlock 13 + Notes + View + Table) | V.D 도 동일 schema 사용 (출력 구조 일관) |
| `src/stage1_figure_note_table_ver.B.py` | Stage 1 figure/note/table split | V.D Task 1 무영향 (Roboflow 라벨 그대로 사용) |
| `src/stage2_layout_in_figure_ver.B.py` | Stage 2 figure 안 V.A 5-class | V.D 무영향 |
| `src/inference/teacher_ver.B.py` | TeacherVLM wrapper (vLLM 0.20.1 + FP8) | V.D Task 1 은 prompt 만 변경 — wrapper 자체는 V.B 그대로. Hot-fix 2/4/5 (parser-v5) 적용 상태 유지 |
| `src/inference/student_ver.B.py` | Student LoRA 추론 (Phase 2 결과) | V.B Phase 2 1차 결과 사용 (Hot-fix 1/6 + LoRA r=32 eval_loss 4.03) |
| `src/preprocess_color_normalize.py` | Stage 0 색상 정규화 | V.A/B 공통 모듈 |
| `src/teacher_prompts_ver.B.py` | V.B prompt baseline | **수정 X — V.D 가 별도 `teacher_prompts_ver.D.py` 로 신규** |

### 3.2 V.B 그대로 사용 (문서)

| 파일 | 용도 |
|---|---|
| `docs/V_B_OUTPUT_SCHEMA.md` | Web Service 팀 hand-off (V.D 도 동일 schema) |
| `docs/V_B_PRODUCTION_GUIDE.md` | 운영 가이드 (§7.5/§7.6 V/T 한계 + 모니터링) |
| `docs/KNOWN_LIMITATIONS_ver.B.md` (§1.7 포함) | V.B 한계 박제 (V.D 가 §1.7 의 V/T 한계 해결 시도) |
| `docs/V_D_VIEW_TABLE_EXTRACTION_PLAN_ver.B.md` | V.B 종료 시점 V.D plan (V.D Task 들의 base) |
| `RESEARCH_SOURCES.md` | 검색 정책 (4 사이트 우선) |
| `history_ver.B.md` (§B.0~B.14) | V.B 시간순 박제 |
| `PROJECT_HANDOFF_ver.B.md` (D-001~D-085) | V.A + V.B 의사결정 박제 |
| `README_ver.B.md` | V.B Pragmatic Closure 완료 상태 |

### 3.3 V.A 그대로 사용

| 파일 | 용도 |
|---|---|
| `README_ver.A.md` | V.A 종료 baseline |
| `PROJECT_HANDOFF_ver.A.md` | V.A 의사결정 (D-001~D-058) |
| `history_ver.A.md` | V.A 시간순 박제 |
| `docs/KNOWN_LIMITATIONS_ver.A.md` | V.A 한계 (V.B 에서 다수 Resolved) |

### 3.4 V.D 신규 작성 정책

V.D 작업의 새 산출물 (.md / .py / .ipynb 등) 은 모두 `_ver.D` 또는 `_ver.D.py` 파일명으로 신규 작성 (V.A → V.B 정책과 동일). V.B 기존 파일들은 baseline 으로 영구 보존.

---

## 4. V.D 작업 정책

### 4.1 파일명 정책

- 신규 파일 → `_ver.D` (예: `teacher_prompts_ver.D.py`, `colab_v_d_task1_validation_ver.D.ipynb`, `V_D_TASK1_PROMPT_ENHANCEMENT_ver.D.md`)
- V.B 기존 파일 → 수정 X (baseline 보존)
- V.B 의 V.D 관련 plan (`V_D_VIEW_TABLE_EXTRACTION_PLAN_ver.B.md`) 는 V.B 종료 시점 박제로 그대로 유지

### 4.2 sync 정책 (B.14.3 정책 준수)

PC 작업 시 양쪽 동기화 필수:
```bash
cp /sessions/.../mnt/Drawing/src/teacher_prompts_ver.D.py \
   /sessions/.../mnt/Drawing/colab_sync/src/teacher_prompts_ver.D.py
```

향후 file watcher / pre-commit hook 검토 (V.B 에서 sync 누락 발견 후 정책).

### 4.3 박제 정책

- 의사결정 → `PROJECT_HANDOFF_ver.D.md` D-086 부터
- 시간순 → `history_ver.D.md` (Task 1 검증 시 신규 작성 검토)
- 한계 발견 → `docs/KNOWN_LIMITATIONS_ver.D.md` (Task 별 한계 발견 시 신규 작성 검토)

---

## 5. V.D 핵심 문서

### V.D 신규 (2026-05-10)
- 본 [`README_ver.D.md`](./README_ver.D.md) — V.D phase baseline
- [`PROJECT_HANDOFF_ver.