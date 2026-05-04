# `src/validate/check_labels_yolo.py`

> **V2-A** — YOLO det 라벨 품질 검증 (Stage 1: **Isometric / PMI / Table=TB / Text=Notes / View**, 5 클래스 D-028)

## 1. 구현 요약

YOLO det `.txt` 라벨 폴더를 스캔해 8가지 품질 항목을 검사. **Stage 1 학습 전 필수 게이트**.

**8개 검증 항목**

| # | 항목 | 임계값 | Severity |
|---|---|---|---|
| 1 | `empty_label_rate` | < 0.05 | critical |
| 2 | `parse_error_rate` | = 0 | critical |
| 3 | `bbox_validity_rate` | = 1.0 | critical |
| 4 | `small_bbox_rate` | < 0.05 | warning |
| 5 | `class_ratio[5 classes, D-028]` | View 0.30+ / PMI 0.15+ / TB 0.10+ / Notes 0.05+ / Isometric 0.02+ | warning |
| 6 | `labels_without_images` / `images_without_labels` | = 0 | warning |
| 7 | `extreme_aspect_ratio_count` (W/H < 0.1 또는 > 10) | = 0 | warning |
| 8 | 에러 샘플 표 (parse / bbox 문제 첫 20건) | — | info |

**YOLO det 형식**: `<class_id> <x_center> <y_center> <width> <height>` (모두 [0, 1] 정규화)

**검증 함수**

```python
parse_label_file(path) -> (rows, errors)
validate_bbox(cid, cx, cy, w, h, n_classes) -> issues_list
discover_pairs(labels_dir, images_dir) -> (label_files, image_map)
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 라벨 형식 | YOLO det `<cid> <cx> <cy> <w> <h>` | ultralytics 표준 |
| 정규화 범위 | [0, 1] 엄격 | 학습 시 좌표 변환 안전 |
| 빈 라벨 의미 | "객체 없음" 정상 case (사용자 의도면) | warning 처리, 비율로 검증 |
| 작은 BBox 임계값 | 면적 < `1e-4` (1000x1000 에서 ~100 px²) | 너무 작은 BBox 는 라벨 실수 의심 |
| 클래스 ID 범위 | YAML names 길이 기준 | 동적 로드 |
| 라벨-이미지 매칭 | 파일명 stem 일치 | YOLO 표준 convention |
| 에러 샘플 첨부 | 첫 20건만 (HTML report) | 사용자 확인 보조 |
| 에러 라인 처리 | 한 파일에 에러 여러 개 → 첫 3개만 기록 | 노이즈 감소 |
| 클래스 분포 | View 다수 + TB ≥ 30% + Notes ≥ 20% 권장 | 논문 §3.1 비례 |

## 3. 사용법

```bash
# 기본
python -m src.validate.check_labels_yolo \
    --labels-dir data/layout/labels/train \
    --cfg configs/yolo_det.yaml

# 이미지 폴더 지정 (label-image 매칭 검사)
python -m src.validate.check_labels_yolo \
    --labels-dir data/layout/labels/train \
    --images-dir data/layout/images/train \
    --cfg configs/yolo_det.yaml

# val 분할도 검증
python -m src.validate.check_labels_yolo \
    --labels-dir data/layout/labels/val \
    --images-dir data/layout/images/val
```

## 4. 검증 결과

### 4.1 더미 데이터 단위 검증 (6개 라벨 파일)

| 파일 | 내용 | 기대 결과 |
|---|---|---|
| `drawing_001.txt` | 정상 4 BBox | OK |
| `drawing_002.txt` | 정상 3 BBox | OK |
| `drawing_003.txt` | 빈 라벨 (객체 없음) | empty 카운트 |
| `drawing_004.txt` | 정상 (큰 W/H 비율 1개 포함) | aspect outlier 1 |
| `drawing_005_bad.txt` | 좌표 1.5 (out of range) + class_id=3 (out of [0,2]) | bbox invalid 2 |
| `drawing_006_parse_err.txt` | "not a number" + 5필드 미만 | parse error |

**실행 결과** (모두 정상 검출):

```
[ 1/8] empty_label_rate           0.3333  ✗ FAIL  ≤ 0.0500   (2/6 files)
[ 2/8] parse_error_rate           0.1667  ✗ FAIL  ≤ 0.0000   (1 file)
[ 3/8] bbox_validity_rate         0.8182  ✗ FAIL  ≥ 1.0000   (9 valid / 2 invalid)
[ 4/8] small_bbox_rate            0.0000  ✓ PASS  ≤ 0.0500
[ 5/8] class_ratio[View]          0.5556  ✓ PASS  ≥ 0.5000
[ 6/8] class_ratio[TitleBlock]    0.2222  ! WARN  ≥ 0.3000   (TB 부족)
[ 7/8] class_ratio[Notes]         0.2222  ✓ PASS  ≥ 0.2000
[ 8/8] extreme_aspect_ratio_count      1  ! WARN  ≤ 0

Overall: FAIL   PASS=3 WARN=2 FAIL=3 INFO=0
```

## 5. 임계값 (`validation_thresholds.yaml#stage1_labels`)

```yaml
stage1_labels:
  empty_label_rate_max:  {threshold: 0.05, severity: critical}
  bbox_validity_rate:    {threshold: 1.0,  severity: critical}
  small_bbox_rate_max:   {threshold: 0.05, severity: warning}
  class_distribution:
    View_min_ratio:       0.50
    TitleBlock_min_ratio: 0.30
    Notes_min_ratio:      0.20
    severity: warning
  inter_annotator_iou:   {threshold: 0.80, severity: warning}
```

## 6. 출력 형식

### 6.1 콘솔

위 §4.1 참조.

### 6.2 HTML 첨부

- 클래스 분포 표 (class_id / name / count / ratio)
- 클래스 분포 막대 차트
- "Sample: labels without images (first 10)" 표 (있을 시)
- "Parse errors (first 20)" 표
- "BBox errors (first 20)" 표 (file / row / issues)

### 6.3 JSON

리포트 파일명: `reports/<parent>_<dir>_yolo_labels.json`
(예: `layout_train_yolo_labels.json`)

## 7. 의존성

```
PyYAML
matplotlib  (via common.py)
jinja2      (via common.py)
```

(라벨 파싱은 stdlib 만)

## 8. 관련 의사결정

- **D-020** 검증 의무화 (학습 전 게이트)
- **D-021** severity (critical 미달 시 학습 차단)
- **D-024** 사전 증강 데이터셋 처리 (group-aware split 시점에서 검증)

## 9. 검증 대상 모듈

[`stage1_layout.md`](./stage1_layout.md) — Step 2 학습 전 게이트로 사용

## 10. 흔한 FAIL 케이스 + 해결

| FAIL 항목 | 원인 | 해결 |
|---|---|---|
| `empty_label_rate > 5%` | 라벨링 누락 도면 많음 | 빈 라벨 파일 stem 확인 → 재라벨링 |
| `parse_error_rate > 0` | Roboflow export 형식 오류 또는 수동 편집 실수 | 첫 20건 에러 샘플 보고 수정 |
| `bbox_validity_rate < 1.0` | 좌표 정규화 실패 (절대 좌표 사용) | export 설정 재확인 / 수동 정규화 |
| `class_ratio[TitleBlock] < 30%` | TB 있는 도면 라벨링 부족 | TB 있는 도면 의식적 추가 라벨링 |
