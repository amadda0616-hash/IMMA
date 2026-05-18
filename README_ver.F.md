# V.F Phase — V.E annotation + 한자 OCR Hybrid

> **작성일**: 2026-05-13
> **상태**: V.F phase 시작 (D-100 진입 직전)
> **선행 박제**: V.A (Donut+YOLO + Tesseract auto-fill 3.43%), V.B (Qwen3-VL 30B FP8 zero-shot, view 52% / table 38% / notes 99%), V.D (prompt fix marginal, Qwen domain weakness), V.E (Donut FT + V.E annotation, titleblock 13.9% — 한자 OCR ceiling 입증)

## 명제

> **annotation quality (V.E 입증) + region crop 독립 추론 (Khan 2510 본질) + 영역별 best model Hybrid routing**

V.E phase 의 V.E annotation quality 효과 (symbol_rate 5배 + Pattern 11~17 자동 회피) 보존 + Donut Swin-B 의 한자 OCR ceiling 해결 = **Hybrid architecture** (Donut FT + Qwen3-VL-2B FT + Nemotron-OCR-v2 zero-shot).

## V.F Hybrid architecture

| Region / Field | Best Model | 근거 |
|---|---|---|
| 영문 field (Drawing_No, Mass, Scale, Sheet, Date, Quantity) | **Donut FT v2** (V.E D-099 — Drawing_No char_acc 67.1%) | 영문 OCR 강점 + V.E annotation effect |
| 한자 field (Title, Part_Name, Material, Engineer, Designer, Approved, Company_Name) | **Nemotron-OCR-v2** (zero-shot recognize) → **Qwen3-VL-2B FT** (schema mapper) | JP NED 0.046 + JSON schema |
| view symbol (Ø/⊥/▽ GDT/roughness) | **Donut FT v2** (symbol_rate 32.6% ★) | V.E annotation 효과 보존 |
| view name + measures | **Qwen3-VL-2B FT** | 일본어 한자 비중 |
| notes lines (일본어 자연어) | **Qwen3-VL-2B FT** | 자연어 강점 |

## 예상 성과

| Metric | V.E D-099 (Donut FT) | V.F 예상 |
|---|---|---|
| titleblock match_rate (15 fields) | 13.9% | **55~75%** (+40~60%p) |
| 한자 field match | 0~6% | **40~80%** |
| 영문 field match | 20~30% | Donut FT 유지 |
| view symbol_rate | 32.6% (V.B ⊥ 7% 의 5배 ★) | 유지 |
| view char_acc | 18.9% (noise mismatch) | Qwen3-VL FT = 40~60% |
| notes char_acc | 18.2% | Qwen3-VL FT = 60~80% |

## D-100+ 작업 list

| Day | 작업 |
|---|---|
| D-100 | V.F phase 박제 + qwen3vl_ft_ver.F.py 작성 |
| D-101 | Qwen3-VL-2B FT 학습 (Unsloth, A100, ~60 unit) |
| D-102 | Phase C 재추론 + Donut vs Qwen3-VL 영역별 비교 |
| D-103 | Nemotron-OCR-v2 zero-shot 검증 |
| D-104 | Hybrid pipeline + demo v2 (FastAPI) |
| D-105 | V.F phase 박제 종료 + 운영 적용 |
| (조건부) V.F-1.5 | Gemini view annotation 보강 (+12K원, +30 unit) |

## 잔여 budget
- Compute unit: 176 → V.F 사용 ~95 → 잔여 ~81 (V.F-1.5 + 안전 마진)
- API 비용: V.E D-097.3 까지 ~10,975원 사용, 잔여 cap ~139,000원

## 박제 매트릭스

- `docs/V_F_PHASE_START_ver.F.md` — V.F plan
- `PROJECT_HANDOFF_ver.F.md` — D-100+ 박제
- `history_ver.F.md` — V.F history
- 본 `README_ver.F.md`

## 연계 박제 (V.E 종료)
- `docs/V_E_PHASE_C_RESULTS_ver.E.md` — D-099 결과 분석 + V.B baseline 재해석 + V.F 진입 결정
- `PROJECT_HANDOFF_ver.E.md §D-099` — V.E phase 박제 종료
- `history_ver.E.md` — V.E history 종료
