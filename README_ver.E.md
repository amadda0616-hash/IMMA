# Drawing — Version E (V.E Phase, 2026-05-11 ~)

> **V.E phase baseline**: V.A architecture (Donut+YOLO) + V.B 데이터 (Phase C 1469) + V.D 분석 (Qwen 한계) + 신규 annotation
> **시작**: 2026-05-11 (V.D phase 잠정 종료 + V.A 회귀 분석 직후, D-092)
> **잔여 budget**: Colab Pro+ compute unit **239.42 unit**

---

## §0 V.E phase 의 핵심 명제

> **V.A 가 fail 한 원인은 architecture 가 아니라 annotation quality 였다.**
> V.E = V.A architecture + 진짜 annotation quality 보장 + V.B/V.D 산출물 재사용

| Phase | Architecture | 측정 결과 | 근본 원인 |
|---|---|---|---|
| **V.A (2026-04~05)** | YOLOv11-det/obb + Donut zero-shot/FT | Numerical 3.43% / Hallucination 72% | **D-050 Tesseract auto-fill GT** |
| **V.B (2026-05~)** | Qwen3-VL-30B-A3B-FP8 zero-shot Teacher | view 52% / table 38% / notes 99% | annotation 우회 — Qwen domain weakness |
| **V.D (2026-05~)** | V.B + prompt fix (Task 1~3) | view +5.9%p marginal / table -7.3%p | Qwen 본질 한계 확정 |
| **V.E (2026-05-11~)** | V.A architecture + 신규 annotation + V.B 재사용 | (V.E-1 시작) | annotation quality 본질 해소 |

---

## §1 V.E phase 진행 표

| Day | 작업 | 상태 | 산출물 |
|---|---|---|---|
| D-092 | V.E phase 시작 결정 (V.A 회귀 분석) | ✅ Resolved | `PROJECT_HANDOFF_ver.D.md §D-092`, `docs/V_E_PHASE_START_ver.E.md`, 본 README |
| D-093 | V.E annotation 옵션 결정 (**V.E-1 Hybrid 채택**) | ✅ Resolved | `docs/V_E_ANNOTATION_ANALYSIS_ver.E.md`, `MANUAL_ver.E.md`, `PROJECT_HANDOFF_ver.E.md §D-093` |
| D-093.5 | V.E server demo v1 (서버 팀 hand-off package + 4 가이드) | ✅ Resolved | `server/server_demo_ver_E/`, `server/V_E_SERVER_DEMO_V1_README_ver.E.md`, `server/V_E_SERVER_INTEGRATION_GUIDE_ver.E.md`, `server/V_E_SCHEMA_CONVERSION_GUIDE_ver.E.md`, `server/V_E_SERVER_STRUCTURE_CHECK_ver.E.md` |
| D-094 | annotation pipeline 코드 작성 | ✅ Resolved | `src/annotation/annotation_pipeline_ver.E.py`, `docs/V_E_ANNOTATION_GUIDE_ver.E.md` |
| D-094.5~22 | 검수 + spot-check 진행 + 누적 박제 (Pattern 11~16, GDT/measures evidence) | ✅ Resolved | 가이드 갱신 + HALLUCINATION_EVIDENCE |
| **D-095** ★ | **사용자 spot-check 100 STOP (41% pass) + V.E-1 → V.E-4-flash 변경** | ✅ Resolved | `PROJECT_HANDOFF §D-095`, `MANUAL_ver.E.md` 갱신 |
| D-095.5 | Gemini API key + 100 sample 무료 검증 | ⏳ Pending | 정확도 측정 (≥80% / 70~80% / <70%) |
| D-096 | `gemini_annotation_ver.E.py` Code 작성 | ⏳ Pending | Gemini 2.5 Flash API client |
| D-097 | Flash 1469 전체 annotation | ⏳ Pending | `outputs/v_e_annotation/gemini_flash_gt/` ($0~3) |
| D-098 | 사용자 verify random 100 + Donut FT 학습 | ⏳ Pending | `outputs/v_e_donut_ft/` (ckpt) |
| D-099 | 검증 + Phase C 재추론 | ⏳ Pending | `outputs/v_e_phase_c_rerun/`, `docs/V_E_DONUT_FT_RESULT_ver.E.md` |
| D-100 | V.E demo v2 작성 | ⏳ Pending | `src/demo_server_v_e_v2_ver.E.py` |
| D-101+ | 운영 적용 + iterate | ⏳ Pending | (운영 measure) |

---

## §2 V.E unit budget (V.E-1 추천)

| Step | unit | 누적 |
|---|---|---|
| Annotation (외부) | 0 | 0 |
| Donut FT 환경 setup | ~5 | 5 |
| Donut FT 학습 (A100 ~7h) | ~80 | 85 |
| 검증 (held-out 100) | ~10 | 95 |
| Phase C 재추론 (300 sample) | ~25 | **120** |

→ **V.E-1 총 ~120 unit** (잔여 239 unit, 마진 119)

| 옵션 | 설명 | unit | 채택 |
|---|---|---|---|
| ~~V.E-1 Hybrid~~ | 사용자 10h + GPT-4o assist $20 | ~120 | ❌ **D-095 보류** (spot-check 41% STOP) |
| V.E-2 GPT-4o | $73 + verify 5h | ~150 | 비추 |
| V.E-3 Manual | 사용자 50h | ~150 | (시간 부담) |
| **V.E-4-flash** ★★ | **Gemini 2.5 Flash $0~3 + verify 3~5h** | ~120 | **✅ D-095 채택** |
| V.E-4-pro | Gemini 2.5 Pro $30~60 | ~150 | Flash 부족 시 fallback |
| V.E-4-hybrid | Flash 1469 + Pro 100 검증 ~$2 | ~125 | (옵션) |

상세 옵션 비교: `docs/V_E_ANNOTATION_ANALYSIS_ver.E.md`
전체 작업 가이드: `MANUAL_ver.E.md`

---

## §3 V.A / V.B / V.D 재사용 매트릭스

상세: `docs/V_E_PHASE_START_ver.E.md §2`

### 3.1 V.A 재사용 (그대로)

- YOLOv11-det/obb ckpt (Stage 1, 2, 3-A) — bbox 만 학습이라 annotation 영향 없음
- Stage 1, 2, 3-A pipeline script
- D-050 박제 (Tesseract auto-fill 실패 회피 base)

### 3.2 V.B 재사용 (그대로)

- `src/pipeline_ver.B.py` (Stage 0~6 통합)
- `src/preprocess/multi_drawing_detector_ver.B.py`
- `src/validate/schema_ver.B.py` (Pydantic schema)
- `outputs/phase1_instruct_phaseB/` (1469 도면 결과 — annotation seed)
- `docs/V_B_OUTPUT_SCHEMA.md`

### 3.3 V.D 재사용 (그대로)

- `src/demo_server_v_d_v1_ver.D.py` (Donut FT 후 model swap)
- `outputs/v_d_task2_overnight/` (annotation priority sample 선별)
- `docs/V_D_TABLE_38_ROOT_CAUSE_ver.D.md` (IJCAI 2025 빈 cell finding)

### 3.4 V.E 신규

- 본 `README_ver.E.md`, `PROJECT_HANDOFF_ver.E.md`, `docs/V_E_PHASE_START_ver.E.md`
- `src/annotation/annotation_pipeline_ver.E.py` (D-094)
- `src/donut/donut_ft_ver.E.py` (D-097)
- `outputs/v_e_annotation/`, `outputs/v_e_donut_ft/`, `outputs/v_e_phase_c_rerun/`

---

## §4 V.E annotation 원칙 (V.A D-050 회피)

V.A 의 D-050 박제: **Tesseract OCR auto-fill 로 GT 채움 → Donut FT 가 Tesseract noise 학습 → paper 96.3% vs 실측 3.43%**

V.E 의 annotation 원칙:
1. **OCR auto-fill 절대 금지** (Tesseract, EasyOCR, PaddleOCR 모두)
2. **Human verification ≥1회** 모든 sample 필수
3. **빈 cell 명시 표기** (`null` / `""` 구분) — IJCAI 2025 finding
4. **Schema 일관성 검증** — Pydantic strict, fail sample 분리
5. **annotation seed = V.B Phase C 1469 결과** (high-confidence 만)

상세: `docs/V_E_PHASE_START_ver.E.md §3`

---

## §5 V.E expected 결과 (Khan paper 매칭 시)

| Metric | V.A 실측 | V.B baseline | **V.E-1 예상** |
|---|---|---|---|
| Numerical accuracy | 3.43% | (해당 없음) | **70~90%** |
| view recall | (해당 없음) | 52.3% | **65~80%** |
| table recall | (해당 없음) | 38.0% | **60~85%** |
| notes recall | (해당 없음) | 99.3% | 99%+ |
| Hallucination rate | 72% | (해당 없음) | **<10%** |

**보수 예상**: annotation quality ~70% 시 회복률 ~50~60% — 그래도 V.B baseline 대비 향상.

---

## §6 V.E phase 산출물 (D-092 시작 시)

- 본 `README_ver.E.md`
- `PROJECT_HANDOFF_ver.E.md` (D-093+ 박제 baseline)
- `docs/V_E_PHASE_START_ver.E.md` (V.E plan + V.A/B/D 재사용 매트릭스 + annotation 구현)

진행 시 추가:
- `docs/V_E_ANNOTATION_GUIDE_ver.E.md`
- `docs/V_E_DONUT_FT_RESULT_ver.E.md`
- `docs/V_E_DEMO_V2_GUIDE_ver.E.md`
- `src/annotation/annotation_pipeline_ver.E.py`
- `src/donut/donut_ft_ver.E.py`
- `colab_sync/colab_v_e_donut_ft_ver.E.ipynb`

---

## §7 이전 phase 박제 (참조)

| Phase | README | Handoff | 종료 사유 |
|---|---|---|---|
| V.A | `README_ver.A.md` | `PROJECT_HANDOFF_ver.A.md` (D-001~D-058) | D-050 Tesseract auto-fill GT 한계 → V.B 우회 |
| V.B | `README