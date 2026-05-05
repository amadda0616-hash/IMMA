# 모듈별 상세 문서

이 폴더의 각 `.md` 파일은 `src/` 내 단일 `.py` 모듈에 대한 **구현 요약 / 핵심 설계 결정 / 검증 결과 / 사용법 / 출력 형식** 을 다룹니다.

> **참조 문서**: 전체 사양은 [`PROJECT_HANDOFF.md`](../../PROJECT_HANDOFF.md), 빠른 참조는 [`README.md`](../../README.md), 단계별 작업 가이드는 [`MANUAL.md`](../../MANUAL.md).

## 목차

### Stage 모듈 (`src/`)

| 모듈 | 역할 | Step | 상태 |
|---|---|---|---|
| [sort_by_titleblock.md](./sort_by_titleblock.md) | TitleBlock 기준 데이터 분류기 (선택 도구) | 1.5 | ✅ |
| [sort_by_drawing_type.md](./sort_by_drawing_type.md) | 가공/조립 도면 자동 분류 (D-026, **tqdm 추가 2026-04-28**) | 1.6 | ✅ |
| [stage1_layout.md](./stage1_layout.md) | YOLOv11-det Layout Segmentation (**D-028 5클래스 + D-029 매핑**) | 2 | ✅ |
| [stage2_annotation.md](./stage2_annotation.md) | YOLOv11-obb Annotation Localization (PMI crop 입력 D-034) | 3 | ✅ |
| [ensemble_predict.md](./ensemble_predict.md) | **★ Stage 2 5-Fold OBB Ensemble** (D-040, V3-B Measure missing 0.101→0.000 PASS, 신규 2026-05-04) | 3.5 | ✅ |
| [auto_label_stage1.md](./auto_label_stage1.md) | **★ Stage 1 자동 라벨링 + Active Learning priority** (신규 2026-04-28) | 5.5 | ✅ |
| [extract_pmi_crops.md](./extract_pmi_crops.md) | **★ Stage 2 PMI crop 추출** (per-axis adaptive padding, v2, 신규 2026-04-30) | 5.5 | ✅ |
| [extract_pmi_crops_v3.md](./extract_pmi_crops_v3.md) | **★ Stage 2 PMI crop 추출 v3** (aspect-aware padding, 신규 2026-04-30) | 5.5 | ✅ |
| [extract_skip_list.md](./extract_skip_list.md) | **★ SKIP 라벨 분석** (reason 카테고리 분리, D-038 rescue 입력, 신규 2026-05-02) | 5.8 | ✅ |
| [rescue_misclassified_notes.md](./rescue_misclassified_notes.md) | **★ Notes Rescue** (D-038 구현, Donut OCR, 신규 2026-05-02) | 5.9 | ✅ |
| [stage3_alphabetical.md](./stage3_alphabetical.md) | Donut Alphabetical (zero-shot) | 5 | ✅ |
| [stage3_paddleocr_install_check.md](./stage3_paddleocr_install_check.md) | **★ Phase 15a — PaddleOCR-VL-1.5 환경 검증** (D-042 monkey-patch, 신규 2026-05-04) | 5a | ✅ |
| [stage3_paddleocr_zero_shot_test.md](./stage3_paddleocr_zero_shot_test.md) | **★ Phase 15b — 다국어 zero-shot 평가** (TitleBlock 23 필드 + Notes + full_text, 신규 2026-05-04) | 5b | ✅ |
| [stage3_numerical.md](./stage3_numerical.md) | Donut Numerical (fine-tune + 추론) | 6 | ✅ |
| [prepare_vlm_dataset.md](./prepare_vlm_dataset.md) | VLM image-text pair 자동 시드 생성 | 4 | ✅ |
| [pipeline.md](./pipeline.md) | end-to-end JPG → 통합 Structured JSON (**★ D-040 Stage 2 ensemble default**) | 7 | ✅ |
| [utils_metrics.md](./utils_metrics.md) | 평가 지표 라이브러리 (모든 검증기 공통) | 8 | ✅ |
| [stage5_enrichment.md](./stage5_enrichment.md) | Metadata Enrichment (4-tier cascade, 확장) | 9 | ✅ |

### 검증 프레임워크 (`src/validate/`)

| 모듈 | 역할 | Step | 상태 |
|---|---|---|---|
| [validate_common.md](./validate_common.md) (= `validate/common.py`) | 검증 프레임워크 공통 인프라 | V0 | ✅ |
| [check_step1_5_sorter.md](./check_step1_5_sorter.md) | 분류기 정확도 검증 | V1 | ✅ |
| [check_labels_yolo.md](./check_labels_yolo.md) | YOLO det 라벨 품질 검증 | V2-A | ✅ |
| [check_stage1_model.md](./check_stage1_model.md) | Stage 1 모델 성능 검증 | V2-B | ✅ |
| [check_labels_obb.md](./check_labels_obb.md) | YOLO OBB 라벨 품질 검증 | V3-A | ✅ |
| [check_stage2_model.md](./check_stage2_model.md) | Stage 2 모델 + ★ 누락률 (D-023) | V3-B | ✅ |
| [check_stage3a_alphabetical.md](./check_stage3a_alphabetical.md) | Donut Alphabetical zero-shot 사후 검증 | V5 | ✅ |
| [check_stage3n_numerical.md](./check_stage3n_numerical.md) | Donut Numerical fine-tuned 검증 (★ D-023) | V6 | ✅ |
| [check_pipeline_e2e.md](./check_pipeline_e2e.md) | end-to-end pipeline 검증 (D-023 e2e 재측정) | V7 | ✅ |
| [check_enrichment.md](./check_enrichment.md) | Step 9 Metadata Enrichment 사후 검증 (마지막) | V9 | ✅ |

## 공통 문서 구조

각 모듈 문서는 다음 7 섹션:

1. **구현 요약** — 모듈 역할 + 주요 컴포넌트
2. **핵심 설계 결정** — 의도적 선택과 근거 (D-XXX 의사결정 ID 표기)
3. **사용법** — CLI / 공개 함수 import 예
4. **검증 결과** — 단위 테스트 / 더미 데이터 검증 결과
5. **출력 형식** — JSON 스키마 / 산출 파일
6. **의존성** — 외부 라이브러리
7. **관련 의사결정** — PROJECT_HANDOFF.md §11 참조

## 다음 단계 모듈 (작성 예정)

**모든 핵심 모듈 + 검증기 작성 완료 ✅**. 추가 작업은 사용자의 라벨링 / 학습 / 실제 데이터 검증 단계에서 결정.

옵션 (필수 아님):
- `src/utils/{crop,json_merge}.py` — 별도 helper 분리 시
- KB JSON 파일 (`data/kb/material_catalog.json` 등) — 도메인 전문가 작성
