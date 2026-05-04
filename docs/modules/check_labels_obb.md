# `src/validate/check_labels_obb.py`

> **V3-A** — YOLO OBB 라벨 품질 검증 (Stage 2: Measure / GDT / Roughness)

## 1. 구현 요약

YOLO OBB `.txt` 라벨 (8-point 회전 사각형) 의 품질을 8가지 항목으로 검사. **자기교차 폴리곤 검출** + **회전각 분포 측정** + **Roughness 부족 알림** 이 핵심.

**8개 검증 항목**

| # | 항목 | 임계값 | Severity |
|---|---|---|---|
| 1 | `empty_label_rate` | < 0.05 | critical |
| 2 | `parse_error_rate` (9-field check) | = 0 | critical |
| 3 | `obb_validity_rate` (자기교차 / 좌표 / 면적) | = 1.0 | critical |
| 4 | `roughness_min_count` | ≥ 50 | warning (D-017 trigger) |
| 5 | `non_axis_aligned_ratio` (axis-aligned 외 ±3°) | ≥ 0.20 | warning |
| 6 | `small_obb_rate` | < 0.05 | warning |
| 7 | `labels_without_images` / `images_without_labels` | = 0 | warning |
| 8 | OBB 회전각 분포 히스토그램 (-90° ~ 90°, 15° bin) | — | info |

**YOLO OBB 형식**: `<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>` (모두 [0, 1] 정규화)

**기하 검증 함수**

```python
polygon_area(pts) -> float           # shoelace 공식
is_simple_quad(pts) -> bool          # cross product 부호 일관성 (볼록/비교차)
obb_long_edge_angle_deg(pts) -> float  # 가장 긴 변 각도 [-90, 90)
is_axis_aligned(angle, tol=3.0) -> bool  # ±3° 내 0/90°
validate_obb(cid, pts, n_classes) -> issues_list
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 라벨 형식 | YOLO OBB 8-point | ultralytics 표준 |
| 자기교차 검출 | Cross product 부호 일관성 | 외부 의존성 없음 (shapely 미사용) |
| 회전각 정의 | 가장 긴 변의 각도, [-90°, 90°) 정규화 | 표준 OBB 정의 |
| Axis-aligned 판정 | ±3° tolerance | 라벨링 노이즈 허용 |
| 작은 OBB 임계값 | 면적 < `5e-5` | YOLO det 보다 엄격 (어노테이션은 작음) |
| Roughness 임계값 | ≥ 50 (warning) | 논문 152개 / 자체 데이터 부족 시 D-017 trigger |
| 회전 다양성 임계값 | non-axis-aligned ≥ 20% | 사전 회전 증강 + 실제 회전 텍스트 |
| 각도 히스토그램 | 13 bins (-90° ~ 90°, 15° step) | 분포 시각화 |
| Roughness ID 자동 매핑 | YAML class names 에서 "Roughness" 인덱스 동적 검색 | yaml 변경 대응 |

## 3. 사용법

```bash
# 기본
python -m src.validate.check_labels_obb \
    --labels-dir data/annotation/labels/train \
    --cfg configs/yolo_obb.yaml

# 이미지 매칭 검사
python -m src.validate.check_labels_obb \
    --labels-dir data/annotation/labels/train \
    --images-dir data/annotation/images/train \
    --cfg configs/yolo_obb.yaml

# val 분할
python -m src.validate.check_labels_obb \
    --labels-dir data/annotation/labels/val
```

## 4. 검증 결과

### 4.1 더미 데이터 단위 검증 (6개 라벨 파일)

| 파일 | 내용 | 기대 |
|---|---|---|
| `view_001.txt` | 정상 4 OBB (View+Measure+GDT+Roughness) | OK |
| `view_002.txt` | 정상 + 회전 OBB 2개 | non-axis-aligned 검출 |
| `view_003_empty.txt` | 빈 라벨 | empty 카운트 |
| `view_004_bad.txt` | 좌표 1.5 (out of range) + self-intersecting (0.1,0.1)(0.4,0.4)(0.1,0.4)(0.4,0.1) | 2 invalid |
| `view_005_parse.txt` | 9 필드 미만 | parse error |
| `view_006.txt` | Roughness 1개 | Roughness 부족 알림 |

**실행 결과**:

```
[ 1/6] empty_label_rate           0.3333  ✗ FAIL  ≤ 0.0500   (2/6 files)
[ 2/6] parse_error_rate           0.1667  ✗ FAIL  ≤ 0.0000   (1 file)
[ 3/6] obb_validity_rate          0.8000  ✗ FAIL  ≥ 1.0000   (8 valid / 2 invalid)
[ 4/6] roughness_min_count             2  ! WARN  ≥ 50       (D-017 trigger)
[ 5/6] non_axis_aligned_ratio     0.2500  ✓ PASS  ≥ 0.2000   (2/8)
[ 6/6] small_obb_rate             0.0000  ✓ PASS  ≤ 0.0500

Overall: FAIL   PASS=2 WARN=1 FAIL=3
```

자기교차 폴리곤 (view_004_bad.txt 의 hourglass 형태) 정확히 검출 ✓

## 5. 임계값 (`validation_thresholds.yaml#stage2_labels`)

```yaml
stage2_labels:
  obb_validity_rate:    {threshold: 1.0, severity: critical}
  class_distribution:
    Roughness_min_count: 50
    severity: warning
    note: "논문 152개. 50개 이하면 D-017 synthetic_gen 검토"
  angle_diversity:
    non_axis_aligned_min: 0.20
    severity: warning
  inter_annotator_iou: {threshold: 0.75, severity: warning}
```

## 6. 출력 형식

### 6.1 HTML 첨부

- 클래스 분포 표 + 막대 차트
- "OBB long-edge angle distribution" 히스토그램 (회전 다양성 시각화)
- "Parse errors (first 20)" 표
- "OBB errors (first 20)" 표 (file / class_id / issues)

### 6.2 JSON

리포트 파일명: `reports/<parent>_<dir>_obb_labels.json`

## 7. 의존성

```
PyYAML, numpy
matplotlib, jinja2  (via common.py)
```

shapely 사용 안 함 (자기교차 검출은 cross product 로 자체 구현).

## 8. 관련 의사결정

- **D-009** 단일 모델 (언어 무관)
- **D-012** 회전 OBB → perspective-warp (다음 단계 stage2_annotation.py)
- **D-017** Roughness < 50 시 synthetic_gen 검토 (조건부)
- **D-020** 검증 의무화
- **D-024** 사전 증강 (회전 OBB 다양성 자연 증가)

## 9. 검증 대상 모듈

[`stage2_annotation.md`](./stage2_annotation.md) — Step 3 학습 전 게이트

## 10. 흔한 FAIL + 해결

| FAIL 항목 | 원인 | 해결 |
|---|---|---|
| `obb_validity_rate < 1.0` (자기교차) | 라벨링 도구에서 점 순서 잘못 | 라벨 재작성 (TL→TR→BR→BL 순서) |
| `obb_validity_rate < 1.0` (좌표 범위) | 절대 좌표 export | 정규화 export 옵션 확인 |
| `roughness_min_count < 50` | Roughness 라벨 부족 | D-017 trigger — Stage 2 학습 후 V3-B 결과 보고 synthetic_gen 결정 |
| `non_axis_aligned_ratio < 0.20` | 회전 어노테이션 부족 | 회전 텍스트 있는 도면 추가 라벨링, 또는 사전 증강 활용 |
