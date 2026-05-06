# Known Limitations & Improvement Backlog

> 본 문서는 IMMA 프로젝트 진행 중 발견된 ★ 차후 개선 필요 / 미해결 / 본질적 한계 사항을 단일 위치에 박제한다.
> Stage 별 (1 / 2 / 3-A / 3-N / Pipeline) 로 구분하며 각 항목은 다음 정보를 포함한다.
> - 발견 시점 (Phase / Day / Decision ID)
> - 본질 (data / model / OCR / labeling / infra)
> - 영향도 (Critical / High / Medium / Low)
> - 후속 조치 옵션 + 예상 우선순위
>
> **갱신 정책**: 새로운 한계 발견 시 본 문서 + `history.md` + `PROJECT_HANDOFF.md` (관련 D-XXX) 동시 박제.
> 해결되면 § Resolved 로 이동 (삭제 X — 박제 보존).

---

## 0. 개요 — 우선순위 매트릭스 (★ 2026-05-06 V6 검증 결과 반영)

| Priority | Stage | 항목 | 영향 | ★ 추천 해결 방법 (구체) | ETA |
|---|---|---|---|---|---|
| ★★★ Critical | 3-N | **D-050 Tesseract OCR 도면 patch 한계** | nominal acc 3.4% / hallucination 72% (V6) | **Streamlit/CVAT 검수 도구 + 사람 검수 ~3일 → Stage 3-N 재학습** | 1주 |
| ★★★ Critical | Pipeline | **검수 도구 (web UI) 부재** | 사람 검수 효율 X / Phase 16a noisy GT | Streamlit 자체 작성 (~1주, 이미지 + JSON dual-pane + 단축키) **OR** CVAT JSON adapter | 1주 |
| ★★★ Critical | 3-N | **★ V6 numerical_accuracy 3.43%** (D-055) | Donut baseline 사용 불가 | (1) 검수 GT 로 재학습 (2) PaddleOCR-VL patch OCR 대체 시도 (D-050) | 2주 |
| ★★★ Critical | 3-N | **★ V6 hallucination_rate 72%** (논문 11배) | 신뢰도 저하 | constrained generation (regex grammar) + post-filter + 검수 GT 재학습 | 2주 |
| ★★★ Critical | 2 | GDT 라벨 절대 부족 2.6% | GDT 학습 사실상 불가 (auto-fill 0.2%) | **extract_gdt_crops.py 작성 + CVAT 라벨링 ~500 crop ~3일** + Stage 2 재학습 (overnight) | 1주 |
| ★★ High | 3-A | ja_drawing 다중 도면 영역별 미검증 | 6개 언어 중 ja 실패 | Stage 1 110 region 분리 → 영역별 PaddleOCR-VL 평가 (옵션, 30분, 코드 변경 X) | 1일 |
| ★★ High | 3-A | V5 char accuracy 0.69 < 0.85 | 부분 PASS | (1) Phase 15c vLLM 백엔드 (속도 7x ↑) (2) Phase 15d Notes Rescue (3) PaddleOCR-VL fine-tune (Phase 18+) | 2주 |
| ★★ High | 3-N | Roughness Ra fallback 18.4% | 학습 데이터 부족 | D-050 검수 도구로 함께 해결 (Ra 영역만 ~50개 검수) | 1주 |
| ★ Medium | 1 | stage1_fp_notes (false positive) | Notes 정확도 ↓ | rescue_misclassified_notes.py + PaddleOCR-VL backend (Phase 15d) | 1일 |
| ★ Medium | 1 | D-026 가공/조립 자동 분류 실패 | 데이터 정리 효율 ↓ | sort_by_yolo_pmi (PMI 카운트 기반) 부분 대체 — ★ 현 상태 유지 권장 | (보류) |
| ★ Medium | Pipeline | Phase 17 e2e 부분 진행 | 종합 점수 측정 X | smoke test PASS (D-056 Phase 15c 완료) → 5장 batch + V7 검증 | 1일 |
| ★ High | Pipeline | multi-TitleBlock merge 정책 (D-056 후속) | 7개 TB 중 마지막만 보존 | confidence-based 선택 또는 모든 raw_text 합침 (~3일) | Phase 18 |
| ☆ Low | 1 | D-036 V.A 회전 증강 한계 | mAP@50 0.86 — V.A 통과 | V.B 라벨링 시 albumentations ±90° 추가 (Phase 18+) | (Phase 18+) |
| ☆ Low | 2 | V3-B Ensemble 추가 fold 효과 | 통과 후 추가 효과 ↓ | 10-fold 시도 — ROI 낮음 (★ 보류) | (보류) |

---

## 1. Stage 1 (`yolo_det.pt`, 5 클래스 detection)

### 1.1 D-026 — 가공/조립 도면 자동 분류 실패 (Medium)

**발견**: Phase 6 (2026-04-29), `sort_by_drawing_type.py` 실행 결과 정확도 ~50%.

**본질**:
- "가공도면 = 단일 부품 + 1~2 view + 치수" vs "조립도면 = 다부품 + BOM" 의 시각적 차이가 모호한 도면 다수
- Roboflow 5,839 장 중 약 30% 가 경계 케이스 (e.g. "1 부품 + 작은 BOM")

**영향**:
- 데이터 정리 (Stage 1 학습용 가공 도면 추출) 자동화 실패
- 부분 대체: `sort_by_yolo_pmi.py` (PMI 풍부도 기반 3-tier 분류) → 효과 ★★

**상태**: 부분 해결 (Stage 1 학습 데이터로 충분, 100 seed 로 V.A mAP@50 0.86 달성)

**후속 옵션**:
- 1) 추가 라벨 컬럼 도입 (`drawing_type` 별도 라벨링) — 비용 ↑, ROI 낮음
- 2) **(권장) 현 상태 유지** — Phase 17 e2e 평가 후 필요 시 재고

### 1.2 stage1_fp_notes — Notes 클래스 False Positive (Medium)

**발견**: Phase 14 (2026-05-03), V.A predict 결과 분석.

**본질**:
- TitleBlock / Table 일부가 `Notes` 로 오분류 (특히 텍스트 위주 영역)
- 100 seed 라벨링 시 Notes vs TitleBlock vs Table 경계가 모호한 케이스 다수

**영향**:
- Stage 3-A Notes 추출 정확도 ↓ (불필요한 영역 OCR)
- 학습 손실 < 0.05 mAP — 본질적 큰 영향 X

**상태**: 미해결, Phase 15d Notes Rescue 에서 처리 예정

**후속 옵션**:
- Phase 15d: `rescue_misclassified_notes.py` (D-038, 작성됨) 으로 영역별 후처리
- 추가 라벨링: TitleBlock/Notes/Table 경계 ~50장 보강 → V.B 학습

### 1.3 D-036 — Version A 회전 증강 한계 (Low)

**발견**: Phase 7 (2026-04-30), V.A 학습 후 회전 도면 일반화 평가.

**본질**:
- ultralytics YOLOv11 의 default rotation augmentation 이 ±15° 만 지원
- 90° / 180° 회전 도면 (다중 view) 에서 mAP 약간 ↓

**영향**:
- ja_drawing 같은 다중 도면 (110 region) 도 D-048 검증 시 정상 분리됨 → 실제 영향 미미
- 일반 manufacturing 도면 회전 < 30° → 본질 영향 X

**상태**: V.A 통과 (mAP@50 0.86)

**후속 옵션**: V.B 라벨링 시 회전 도면 추가 → albumentations 으로 ±90° 증강

### 1.4 ★ 추가 (2026-05-05) — 다중 도면 영역별 OCR 미검증 (High)

**발견**: D-048 (2026-05-05), `ja_drawing.jpg` 110 region 분리 성공했지만 영역별 Stage 3-A 평가 미진행.

**본질**:
- Stage 1 V.A 가 다중 도면 (1 페이지에 여러 도면 합성) 자동 분리 성공
- 분리된 영역별 PaddleOCR-VL 적용 시 정확도 미검증

**영향**:
- 6개 언어 중 ja 만 실패한 4차 평가 (avg 0.69) 의 추가 개선 가능성
- 한자/일본어 OCR 실제 한계 vs 다중 도면 합성 한계 분리 검증 필요

**상태**: 다음 날 (2026-05-06) 옵션 작업, 30분

**후속 옵션**:
```bash
# Phase 15c 후속
python src/stage3_paddleocr_zero_shot_test.py \
    --input outputs/crops/ja_drawing/ \
    --output outputs/stage3a_ja_per_region.json
```

---

## 2. Stage 2 (`yolo_obb.pt`, 3 클래스 OBB detection)

### 2.1 ★ Critical — GDT 라벨 절대 부족 (Critical)

**발견**: 2026-05-05 Phase 16a dry-run, Measure 86.2% / GDT 2.6% / Roughness 11.2%.

**본질**:
- Roboflow 라벨 단계에서 GDT 어노테이션 비중이 낮음 (Measure 위주 라벨링)
- 500 도면 표본 중 GDT region 13개 = 학습 데이터로 불충분

**영향**:
- ★ Stage 3-N GDT 학습 사실상 불가 (1차 baseline)
- Phase 16b 학습은 Measure 위주 baseline 으로 정의됨 (D-051)

**상태**: ★ 미해결, **후속 라벨링 라운드 필요**

**후속 옵션**:
- 1) **(권장)** GDT crop ~500 추가 라벨링 (extract_gdt_crops.py 작성 + CVAT)
   - Phase 17 e2e 평가 후 진행
   - 예상 비용: 라벨링 ~3일 + 학습 6h
- 2) 외부 GDT 데이터셋 합성 (e.g. Sec.PMI dataset) — 도메인 차이 위험
- 3) Synthetic GDT generation (Cycle GAN / Diffusion) — 실험적

### 2.2 V3-B Ensemble 통과 후 추가 개선 여지 (Low)

**발견**: Phase 14 (2026-05-03), K-fold ensemble V3-B PASS (D-023).

**본질**:
- 5-fold Ensemble + WBF NMS → mAP@50 모두 통과
- 단일 fold mAP 와 ensemble mAP 차이 ~5% — 추가 fold (예: 10-fold) 효과 ↓

**상태**: 통과, 추가 개선 우선순위 ★ Low

---

## 3. Stage 3-A (PaddleOCR-VL-1.5 Zero-shot)

### 3.1 ★ ja_drawing 다중 도면 처리 (High)

**발견**: 1차~3차~4차 평가 (Phase 15b, 2026-05-04~05).

**본질**:
- ja_drawing.jpg = 1 페이지에 다수 도면 합성 (BSBM TT-10CW, ロストワックス 등)
- PaddleOCR-VL 단일 페이지 입력 → "B" 무한 반복 출력 (degenerate)
- D-048 검증: Stage 1 V.A 가 110 region 자동 분리 가능

**상태**: ★ 미해결, 영역별 OCR 적용 미검증 (다음 날 작업)

**후속 옵션**: §1.4 와 동일 — 영역별 평가 → ja avg 향상 기대

### 3.2 V5 Char Accuracy ~0.69 < 0.85 (High)

**발견**: 4차 평가 (2026-05-05).

**본질**:
- Real-ESRGAN 4x upscale 후 ko/zh 큰 향상 (0.50 → 0.69)
- 여전히 D-013 V5 임계값 (0.85) 미달
- en/ru 알파벳은 이미 인식 잘 됨 (~0.65)

**상태**: 부분 PASS 인정 → Phase 17 e2e 에서 종합 재평가

**후속 옵션**:
- 1) Phase 15c — 백엔드 교체 (transformers → vLLM)
   - 속도 ↑ (도면당 ~70s → ~10s 기대)
   - char accuracy 영향 X (모델 동일)
- 2) Phase 15d — Notes Rescue 도입
- 3) **(★ 본질)** PaddleOCR-VL fine-tune (도메인 적응)
   - 비용 ↑↑↑ (~500 검수된 도면 GT)
   - Phase 18+ 후속

### 3.3 PaddleOCR-VL 호환성 — transformers 5.0.0 monkey-patch (Resolved)

**발견**: Phase 15a (2026-05-03), 7차 환경 시도.

**본질**:
- PaddleOCR-VL `config.text_config` attribute 누락
- transformers 5.x ROPE / masking_utils 호환성 이슈

**해결 (D-042)**:
```python
config.text_config = config.get_text_config()  # monkey-patch
```

**상태**: ✅ Resolved — 박제 D-042

### 3.4 README BLOCK 3 호출 방식 (Resolved)

**발견**: Phase 15b 1~2차 평가 (2026-05-04).

**본질**:
- 자연어 prompt + max_new_tokens=1024 → degenerate generation
- repetition_penalty 같은 sampling 매개변수 부분 효과만

**해결 (D-046)**:
- task keyword (`"OCR:"`, `"Table Recognition:"`) + bfloat16 + apply_chat_template + processor.decode 슬라이스

**상태**: ✅ Resolved — 박제 D-046

---

## 4. Stage 3-N (Donut Numerical Fine-tune)

### 4.1 ★ Critical — Tesseract OCR 도면 patch 본질적 한계 (Critical)

**발견**: 2026-05-05 Phase 16a dry-run 표본 분석.

**본질** (★ 박제 D-050):
- Pytesseract `--psm 6` + `kor+eng+rus+jpn` 4개 언어 사용
- 도면 patch 의 작은 글자 (10~14 px) + 한자/일본어/한글 혼재 → OCR 노이즈 매우 큼
- 표본 OCR hint 분석:

| ocr_hint 예시 | 분석 |
|---|---|
| `'020'` | 정상 (leading 0 포함) |
| `'on'` | 노이즈 (`20` 오인식) |
| `'ーーの40 ['` | 일본어 노이즈 + 40 추출 |
| `'„23 „|'` | 특수문자 노이즈 + 23 추출 |
| `'더'` | 한글 단편 (의미 없음) |
| `''` | 빈 문자열 (인식 실패) |

**영향**:
- ★ tolerance regex 매칭 0% (OCR 출력에 `±` 기호 자체가 없음)
- ★ GDT symbol 매칭 0% (의미 있는 텍스트 X)
- ★ Roughness Ra 매칭 18.4% (전체 데이터 적용 결과, dry-run 30% 보다 낮음)
- **★ V6 검증 (D-055, 2026-05-06)**: Phase 16b baseline 학습 후 numerical_accuracy **3.43%** / hallucination **72%** — 학습 자체는 진행되지만 정확도는 사실상 사용 불가 수준 → ★ 검수 GT 필수성 검증 완료

**상태**: ★★★ Critical, 미해결 (regex 보강 효과 ≈ 0, V6 검증 결과로 검수 필요성 확정)

**★ 추천 해결 방법** (우선순위순, 정량 효과 + 비용):

| # | 방법 | 예상 효과 | 비용 | 우선순위 |
|---|---|---|---|---|
| 1 | **Streamlit/CVAT 검수 도구 작성 + 사람 검수 ~3일** | nominal acc 3.4% → 70%+ / hallucination 72% → 10% 이하 | 도구 ~1주 + 검수 ~3일 | ★★★ 1순위 |
| 2 | **PaddleOCR-VL patch OCR 대체** (Tesseract → PaddleOCR-VL `OCR:` task) | OCR hint 정확도 ↑↑ → auto-fill 50% → 80%+ | ~1~2일 실험 | ★★ 2순위 (#1 병행) |
| 3 | **constrained generation (regex grammar)** | hallucination 72% → 30% 이하 (GT 학습 없이) | ~1주 (transformers logits processor) | ★★ 단기 완화 |
| 4 | 도메인 특화 OCR 모델 fine-tune (e.g. TrOCR-printed) | 모든 한계 해결 가능 but 비용 큼 | ~1개월 (~5,000 검수 patch) | ★ Phase 18+ |

**구체 명령** (#1 검수 도구):
```bash
# 1. Streamlit 검수 도구 작성 (~1주, 작성 후 git commit)
# src/review_tool_streamlit.py
# - Phase 16a 산출 JSON 11,470 region 검수
# - 좌측 이미지 + 우측 JSON form (단축키 1/2/3 = Measure/GDT/Roughness)
# - manifest.csv 자동 갱신 + completed=True 마킹

# 2. 검수 진행 (~3일, 사용자)
streamlit run src/review_tool_streamlit.py -- --root data/vlm/numerical/

# 3. Stage 3-N 재학습 (~6h overnight)
python src/stage3_numerical.py train --cfg configs/donut_numerical.yaml --device cuda:0

# 4. V6 재평가 (목표: numerical_accuracy ≥ 70%, hallucination ≤ 10%)
```

### 4.2 D-051 — Phase 16b 1차 baseline 정의 + ★ D-055 V6 검증 결과 (Active)

**정책**:
- ★ Phase 16b Donut numerical fine-tune = **Measure nominal extraction baseline only**
- GDT / Roughness 는 noisy GT 로 포함되지만 학습 효과 기대 X
- Phase 17 e2e 검증의 "Stage 3-N 자리만 채우기" 목적

**근거**:
- Measure auto-fill 성공률 61.5% × 데이터 11,470 → nominal 학습 가능 sample 5,381
- GDT 자동 매핑 0.2% → 학습 데이터 사실상 부재
- Roughness 18.4% × 2,189 → 402 sample (제한적)

**★ V6 검증 결과 (D-055, 2026-05-06)**:

| 항목 | 결과 | 임계 | 판정 | 비고 |
|---|---|---|---|---|
| field_f1[Measure] | 0.3786 | ≥0.90 | ❌ | 논문 0.923 의 41% |
| numerical_accuracy ★ | **0.0343** | ≥0.95 | ❌ | nominal ±0.01mm 매칭 3.4% |
| tolerance_match | 0.9982 | ≥0.90 | ✅ | null 도 매칭 (구조 학습 OK) |
| hallucination_rate ★ | **0.7201** | ≤0.10 | ❌ | 논문 0.067 의 11배 |
| empty_rate | 0.0000 | ≤0.05 | ✅ | 응답 정상 |

**가설 검증**:
- ✅ "학습 자체 진행 가능" — loss plateau 0.92, empty 0%
- ✅ "구조 (JSON / null) 학습 OK" — tolerance_match 99.82%
- ❌ "정확한 값 예측" — numerical_accuracy 3.4% (사실상 사용 불가)
- ❌ "Hallucination 제어" — 72% (매우 심각)

**★ 추천 해결 방법**:

| # | 방법 | 예상 효과 | 비용 | 우선순위 |
|---|---|---|---|---|
| 1 | **검수 GT 로 재학습** (§4.1 #1 과 연계) | numerical 3.4% → 70%+ | 도구 ~1주 + 검수 ~3일 + 재학습 6h | ★★★ |
| 2 | **D-051 baseline 인정** (Phase 17 e2e 자리 채움) | 종합 평가 + 후속 우선순위 정량화 | 즉시 (별도 비용 X) | ★★★ (현 결정) |
| 3 | **GDT/Roughness 추가 라벨링** (§2.1 결합) | per-class F1 보강 | ~3일 라벨링 + 6h 학습 | ★★ Phase 18+ |

**현 상태 (사용자 결정, 2026-05-06)**: 옵션 #2 — Phase 17 e2e 진입 후 정량화 → #1 우선순위로 진행.

### 4.3 D-049 — sys.path bootstrap (Resolved)

**발견**: 2026-05-05 Phase 16a 첫 실행.

**본질**:
- `prepare_vlm_dataset.py` 가 `from src.stage1_layout import ...` 사용
- 직접 `python src/prepare_vlm_dataset.py` 실행 시 sys.path 에 프로젝트 루트 없음

**해결**:
```python
# pipeline.py 와 동일 패턴 (Task #92)
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))
```

**상태**: ✅ Resolved (`prepare_vlm_dataset.py`, `auto_fill_numerical_gt.py` 적용)

**후속 적용 권장**: `src/validate/check_*.py` 9개 파일 (`from src.validate.common import ...`) — 사용 시점에 일괄 적용

### 4.4 PyPI `src` 패키지 오설치 시도 차단 박제

**발견**: 2026-05-05, ModuleNotFoundError 'src' 후 `uv pip install src` 시도 → 빌드 실패 (다행).

**본질**:
- PyPI 의 `src==0.0.7` 은 ★ 본 프로젝트와 무관한 외부 패키지
- 로컬 `src/` 폴더 import 와 PyPI 패키지 이름 충돌 가능

**예방**:
- 본 프로젝트 어떤 환경에서도 `pip install src` / `uv pip install src` 금지 박제
- requirements.txt / pyproject.toml 에 `src` 절대 추가 X
- sys.path bootstrap (D-049) 으로 충분

---

## 5. Pipeline 종합

### 5.1 검수 도구 부재 (High)

**발견**: 2026-05-05, Phase 16a 산출 JSON 의 GT field null 상태 확인.

**본질**:
- Phase 16a 가 생성하는 JSON 템플릿은 `_review.completed = false` 상태
- `nominal`, `tolerance`, `symbol` 등 GT field 사람 검수 필요
- 현재는 console / IDE 에서 수동 편집 — UX ↓

**영향**:
- 후속 검수 라운드 (~3일 estimate) 의 작업 효율 ↓
- 1차 baseline 학습은 auto_fill 로 우회

**후속 옵션**:
- 1) **(권장)** Streamlit 기반 자체 검수 도구 (~1주 작성 + 검수 ~3일)
   - 이미지 + JSON 동시 표시 + 단축키
   - manifest.csv 자동 갱신
- 2) CVAT 활용 (이미 Stage 1 / Stage 2 라벨링에 사용 — 친숙함)
   - JSON template 변환 → CVAT XML → 검수 → 역변환
- 3) Label Studio (오픈소스, JSON 친화적)

**예상 우선순위**: Phase 17 e2e 평가 후 결정

### 5.2 Phase 17 e2e 미진행 (Active)

**현재 상태**: Stage 1 / 2 / 3-A / 3-N 개별 PASS or 부분 PASS

**Phase 17 목적**:
- pipeline.py 로 도면 1장 → 통합 JSON 생성 → 평가
- 각 Stage 의 noisy 결과가 종합 점수에 어떤 영향 미치는지 측정
- 후속 개선 우선순위 정량화

**예상 시점**: 2026-05-06 ~ 05-07 (Phase 16b 학습 결과 확보 후)

### 5.3 다국어 6개 PASS 미달 (High)

**현재 상태**:
- en / ru / zh / ko: 부분 PASS (~0.65~0.78 char accuracy)
- ja: 미PASS (다중 도면 한계)
- de: 평가 미진행 (~10장 박제됨)

**후속**:
- de 도면 4차 평가 추가 (Phase 15c)
- ja 영역별 평가 (D-048 후속)

---

## 6. § Resolved (해결 완료, 박제 보존)

다음 항목은 해결되었으나 박제 보존 차원에서 유지:

| ID | 항목 | 해결 시점 |
|---|---|---|
| D-013 | Stage 3-A V5 임계값 정의 | Phase 15a |
| D-023 | V3-B Ensemble PASS | Phase 14 |
| D-028 | Stage 1 5 클래스 확장 | Phase 1 |
| D-030 | cu124 → cu128 + FA2 | Phase 1 |
| D-031 | validation_thresholds 재조정 | Phase 1 |
| D-037 | extract_pmi_crops adaptive padding | Phase 8 |
| D-038 | rescue_misclassified_notes 작성 | Phase 13 |
| D-039 | Stage 3-A 채택 (Donut → PaddleOCR-VL) | Day 2 |
| D-042 | PaddleOCR-VL transformers 5.0.0 monkey-patch | Phase 15a |
| D-046 | README BLOCK 3 호출 방식 | Phase 15b |
| D-047 | OTSL Table Format 박제 | Phase 15b |
| D-048 | Stage 1 V.A generalization (ja 110 region) | Phase 15b |
| D-049 | sys.path bootstrap (prepare_vlm_dataset / auto_fill) | Phase 16a |
| D-052 | Donut data_collator 호환성 (default_data_collator 적용) | Phase 16b |
| D-053 | DonutTrainer subclass — transformers 5.x num_items_in_batch 호환 | Phase 16b |
| D-054 | Phase 16b 1차 baseline 학습 성공 (eval_loss 0.9581, 6h 23분) | Phase 16b |
| D-055 | V6 baseline 검증 (numerical 3.4% / hallucination 72%) — D-051 가설 검증 완료 | Phase 16c |
| D-056 | Phase 15c — pipeline.py PaddleOCR-VL backend 통합 (subprocess wrapper + 4 fix) | Phase 15c |

---

## 7. 갱신 정책

새로운 한계 / 미해결 발견 시:
1. 본 문서 § (해당 Stage) 에 신규 항목 추가
2. `history.md` 에 발견 시점 박제 (§A.X.Y)
3. `PROJECT_HANDOFF.md` 의 D-XXX 신규 박제 (필요 시)
4. 우선순위 매트릭스 (§0) 갱신
5. 해결 시 § Resolved 로 이동 + 해결 시점 / D-XXX 추가

본 문서는 ★ 단일 source of truth 로 유지되며, 다른 문서는 본 문서를 참조.
