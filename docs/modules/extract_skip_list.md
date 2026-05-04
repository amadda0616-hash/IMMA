# `src/extract_skip_list.py`

> **Step 5.8 — Stage 2 SKIP 라벨 분석** (D-038) — CVAT XML 에서 SKIP 라벨 추출 + reason 카테고리별 분리 + rescue 대상 마킹

## 1. 구현 요약

Stage 2 OBB 라벨링 완료 후, CVAT XML export 에서 SKIP 라벨을 파싱하여 `reason` attribute 기준으로 9개 카테고리로 분리. 각 카테고리별 PMI crop 파일명 리스트를 생성하고, 특히 **stage1_fp_notes** 리스트는 D-038 Notes Rescue 의 입력으로 직접 사용됨. 최종적으로 모든 SKIP 통합 리스트 + 카테고리별 통계 CSV 생성.

**워크플로 (Stage 2 라벨링 완료 후 분석 단계)**

```
Stage 2 OBB 라벨링 완료 (✅)
        ↓
CVAT export → annotations.xml
        ↓
★ extract_skip_list.py — SKIP 라벨 파싱 + 분류 (~30초)
        ↓
outputs/skip_lists/
  ├── stage1_fp_notes.txt        ← D-038 rescue 입력
  ├── unreadable.txt
  ├── stage1_fp_*.txt (7개)
  ├── other.txt
  ├── all_skip.txt               ← Stage 3 제외 list
  └── summary.csv                ← 통계
```

**핵심 컴포넌트** (~400 lines)

| 함수 | 역할 |
|---|---|
| `parse_cvat_xml(xml_path)` | CVAT XML 파싱 → reason 별 crop filename set 반환 |
| `extract_reason(box_elem)` | `<box>` element 에서 reason attribute 추출 (기본값: stage1_fp_other) |
| `write_reason_files(by_reason, output_dir)` | reason 별 .txt 파일 생성 (alpha sort) |
| `write_all_skip(by_reason, output_dir)` | 통합 all_skip.txt 생성 (Stage 3 자동 제외용) |
| `write_summary_csv(by_reason, output_dir, total_crops)` | summary.csv 생성 (비율 포함) |
| `print_summary(by_reason, total_crops)` | 콘솔 요약 + rescue 다음 단계 안내 |

**1개 CLI 서브커맨드 (단일 실행)**

```bash
python src/extract_skip_list.py --xml <annotations.xml> [--output-dir <outputs/skip_lists>]
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| reason attribute | CVAT SKIP 라벨의 `<attribute name="reason">` | Stage 2 라벨링 시 각 SKIP 박스마다 reason 선택 (반드시 명시) |
| 9개 reason 카테고리 | Type A (unreadable) + Type B (stage1_fp_*) + Type C (other) | D-038 및 향후 Stage 1 Version B 학습 시 가독성 ↓ / false positive 카테고리별 보강 우선순위 결정용 |
| **stage1_fp_notes** | ★ rescue 대상 명시 | 일반 주석(재질/가공 지시/공차) 영역이 PMI 로 오검출된 경우만 이 카테고리 사용 (stage1_fp_other 금지) |
| 파일명 리스트 | 각 reason 별 .txt (1행 = 1개 crop 파일명) | 차후 Python 스크립트에서 line-by-line 읽기 용이 (stage1_fp_notes.txt → rescue_misclassified_notes.py 입력) |
| Alphabetic sort | 각 이유별 파일명 정렬 | 재현성 + 수동 검증 시 일관된 순서 |
| summary.csv | category, count, ratio_of_skip, ratio_of_total | 라벨링 진행률 추적 + Version B 학습 시 카테고리별 보강 우선순위 결정 |
| 콘솔 출력 | reason 별 count + marker (RESCUE/Type A/Type B) | 라벨러 피드백: rescue 건수 즉시 가시화 |
| 에러 처리 | 알 수 없는 reason 값도 파일 생성 (warning) | 라벨 정의 업데이트 시에도 호환성 유지 |

## 3. SKIP reason 9개 카테고리

| reason | 설명 | 예시 | 다음 처리 |
|---|---|---|---|
| `unreadable` | **Type A**: 가독성 한계 (회전/노이즈/야라 흐림) | 45° 회전 화살표 / 저해상도 텍스트 | → all_skip.txt (Stage 3 제외) |
| `stage1_fp_section` | **Type B**: 단면도 기호 | A-A 단면 선 / 단면 제목 | → all_skip.txt |
| `stage1_fp_detail` | **Type B**: 상세도 기호 | 원형/사각 상세 프레임 | → all_skip.txt |
| `stage1_fp_projection` | **Type B**: 제3각법 기호 | 정면/평면/측면 뷰 레이블 | → all_skip.txt |
| `stage1_fp_table` | **Type B**: 표제란/BOM/도장 | 도면 정보, 부품 목록, 도장 스펙 | → all_skip.txt |
| `stage1_fp_notes` | **Type B+**: ★ 일반 주석 (rescue 대상) | 재질, 가공, 공차, 검사 기준 | → rescue_misclassified_notes.py (Donut OCR) |
| `stage1_fp_isometric` | **Type B**: 등각도 | 3D 아이소메트릭 뷰 기호 | → all_skip.txt |
| `stage1_fp_other` | **Type B**: 기타 Stage 1 false positive | 라벨 정의 미포함 오검출 | → all_skip.txt |
| `other` | **Type C**: 그 외 (SKIP 의 전혀 다른 이유) | 사용자 정의 (권장 안 함) | → all_skip.txt |

## 4. 사용법

### CLI

```bash
# 기본 (모든 reason 별 분리)
python src/extract_skip_list.py \
    --xml outputs/cvat_stage2_v3_final.xml \
    --output-dir outputs/skip_lists/

# 입력: CVAT export XML 만 (다른 인자 불필요)
python src/extract_skip_list.py \
    --xml outputs/annotations.xml \
    --output-dir outputs/skip_lists/

# 전체 crop 수를 미리 알 때 (비율 계산용)
python src/extract_skip_list.py \
    --xml outputs/annotations.xml \
    --output-dir outputs/skip_lists/ \
    --total-crops 844
```

### CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--xml` | (필수) | CVAT export XML 파일 경로 (annotations.xml) |
| `--output-dir` | `outputs/skip_lists/` | 출력 폴더 |
| `--total-crops` | 0 (자동 계산) | 전체 crop 수 (비율 계산용, 0 = XML 내 image 개수 사용) |

### Python import (선택)

```python
from src.extract_skip_list import parse_cvat_xml, load_skip_list

# XML 파싱
by_reason = parse_cvat_xml(Path("outputs/annotations.xml"))
# by_reason → {"unreadable": {...filenames...}, "stage1_fp_notes": {...}, ...}

# 특정 reason 리스트 확인
rescue_files = by_reason.get("stage1_fp_notes", set())
print(f"Rescue 대상: {len(rescue_files)} 개")
```

## 5. 산출 형식

### 파일 구조

```
outputs/skip_lists/
├── unreadable.txt                  # Type A: 가독성 한계
├── stage1_fp_section.txt           # Type B: 단면도
├── stage1_fp_detail.txt            # Type B: 상세도
├── stage1_fp_projection.txt        # Type B: 제3각법
├── stage1_fp_table.txt             # Type B: 표제란/BOM
├── stage1_fp_notes.txt             # ★ Type B+: 일반 주석 (rescue 입력)
├── stage1_fp_isometric.txt         # Type B: 등각도
├── stage1_fp_other.txt             # Type B: 기타 FP
├── other.txt                       # Type C: 그 외
├── all_skip.txt                    # 통합 (Stage 3 제외용)
└── summary.csv                     # 통계
```

### 각 파일 형식 (예: stage1_fp_notes.txt)

```
# SKIP reason: stage1_fp_notes
# Count: 47
# Source: CVAT XML SKIP 라벨
DwgFoo_001__PMI_023.jpg
DwgFoo_001__PMI_045.jpg
...
```

### summary.csv 형식

```
category,count,ratio_of_skip,ratio_of_total
stage1_fp_notes,47,12.45%,5.57%
stage1_fp_section,89,23.54%,10.55%
stage1_fp_table,76,20.11%,9.00%
...
unreadable,34,9.00%,4.03%
TOTAL_SKIP,378,100.00%,44.79%
TOTAL_CROPS,844,—,100.00%
```

### 콘솔 출력 예

```
18:30:45 | INFO    | ============================================================
18:30:45 | INFO    | SKIP 카테고리 분포
18:30:45 | INFO    | ============================================================
18:30:45 | INFO    |   stage1_fp_table           :   76 (20.11%) (Type B: Stage 1 FP)
18:30:45 | INFO    |   stage1_fp_section        :   89 (23.54%) (Type B: Stage 1 FP)
18:30:45 | INFO    |   stage1_fp_notes          :   47 (12.45%) ★ RESCUE 대상 (D-038)
18:30:45 | INFO    | ============================================================
18:30:45 | INFO    |
18:30:45 | INFO    | [★ 다음 단계 — D-038 Notes Rescue]
18:30:45 | INFO    |   47 개의 stage1_fp_notes crop 을 Donut OCR 로 처리:
18:30:45 | INFO    |   python src/rescue_misclassified_notes.py \
18:30:45 | INFO    |       --skip-list outputs/skip_lists/stage1_fp_notes.txt \
18:30:45 | INFO    |       --crops-dir outputs/cvat_stage2_input_v3_upscaled/ \
18:30:45 | INFO    |       --output outputs/rescued_notes.json
```

## 6. 의존성

| 라이브러리 | 버전 | 역할 |
|---|---|---|
| `xml.etree.ElementTree` | stdlib | CVAT XML 파싱 |
| `csv` | stdlib | summary.csv 출력 |
| `pathlib` | stdlib | 파일 경로 처리 |
| `logging` | stdlib | 콘솔 로깅 |

## 7. 관련 의사결정

- **D-037** — adaptive padding v3 (PMI crop 품질)
- **D-038** — ★ Stage 1 false positive Notes Rescue (본 도구가 rescue 입력 제공)
  - 전체 흐름: extract_skip_list.py → stage1_fp_notes.txt → rescue_misclassified_notes.py → Donut OCR → rescued_notes.json → pipeline.py stage4 병합
- **차후 검토**:
  - Stage 1 Version B 학습 시 reason 별 false positive 패턴 분석 → Text 클래스 보강 (rescue 의존 최소화)
  - reason 카테고리 추가 필요성 (예: 손글씨, 인쇄체 구분) — 라벨링 진행 중 피드백 수집
