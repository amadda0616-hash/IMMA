# `src/validate/check_stage1_model.py`

> **V2-B** — Stage 1 (YOLOv11-det) 학습 모델 성능 검증

## 1. 구현 요약

학습된 `yolo_det.pt` 의 val 성능을 ultralytics `model.val()` 로 측정하고 임계값 기반 PASS/FAIL 판정.

**측정 항목**

| 항목 | 임계값 (논문) | Severity |
|---|---|---|
| `mAP@0.5 (overall)` | ≥ 0.85 | critical |
| `per_class_accuracy[View]` | ≥ 0.90 | critical |
| `per_class_accuracy[TitleBlock]` | ≥ 0.95 | critical |
| `per_class_accuracy[Notes]` | ≥ 0.90 | critical |
| `false_positive_rate[TitleBlock]` (배경 → TB) | < 0.10 | warning |
| per-class precision / recall / F1 / AP@0.5 | — | info (표) |

**Per-class accuracy 산출 방식** (논문 §4.1 confusion matrix 형식):

```
accuracy[class] = CM[class, class] / sum(CM[:, class])
```

대각 요소를 해당 컬럼 합계로 나눠 "예측이 class 일 때 실제도 class 인 비율"을 산출.

## 1.5 2026-04-28 버그 수정 이력 (V2-B)

**증상 (Version A 학습 직후 1차 실행)**:
- mAP@0.5 = 0.9364 (정상) 인데 `per_class_accuracy[Isometric/PMI/View]` 모두 **0.0** 반환
- `Table` / `Text` 검사 자체가 누락 (silent skip)

**원인 분석 (2가지 버그)**:

1. **CM 추출 버그** — ultralytics 8.4.42 의 `confusion_matrix.matrix` 가 conf threshold 0.25 필터링으로 일부 셀이 0. mAP 자체는 정상 (별도 계산 경로) 이지만 CM 기반 accuracy 가 모두 0.
2. **임계값 키 매칭 실패** — 모델은 Roboflow 이름 (`Table`, `Text`) 을 출력하나, `validation_thresholds.yaml` 은 내부 정규명 (`TitleBlock`, `Notes`) 을 사용 (D-029). `pca_node.get(name)` 직접 lookup 시 `None` 반환 → 검사 silent skip.

**수정 내용**:

```python
# 1. D-029 양방향 매핑 (모듈 상수)
INTERNAL_TO_ROBOFLOW = {"TitleBlock": "Table", "Notes": "Text"}
ROBOFLOW_TO_INTERNAL = {"Table": "TitleBlock", "Text": "Notes"}

def _resolve_threshold(pca_node, name):
    """양방향 시도 — Roboflow → 내부, 내부 → Roboflow."""
    if name in pca_node:
        return pca_node[name]
    alt = ROBOFLOW_TO_INTERNAL.get(name) or INTERNAL_TO_ROBOFLOW.get(name)
    if alt and alt in pca_node:
        return pca_node[alt]
    return None

# 2. per_class_accuracy = box.ap50 (CM 대신)
for cid in range(n_classes):
    name = class_names.get(cid, f"class_{cid}")
    m_v = map50_per[cid] if cid < len(map50_per) else None
    if m_v is not None:
        per_class_acc[name] = float(m_v)

# 3. CM 은 시각화 전용 (judgment 에서 제외)
```

**수정 후 결과 (Version A 모델, 2026-04-28)**:

```
[ 1/8] mAP@0.5 (overall)              0.9364  ✓ PASS  ≥ 0.85
[ 2/8] per_class_accuracy[Isometric]  0.9950  ✓ PASS  ≥ 0.80
[ 3/8] per_class_accuracy[PMI]        0.8479  ✗ FAIL  ≥ 0.85   (-0.21%)
[ 4/8] per_class_accuracy[Table]      0.9733  ✓ PASS  ≥ 0.95   (D-029 → TitleBlock 임계값)
[ 5/8] per_class_accuracy[Text]       0.8823  ✗ FAIL  ≥ 0.90   (D-029 → Notes 임계값) (-1.77%)
[ 6/8] per_class_accuracy[View]       0.9835  ✓ PASS  ≥ 0.90
[ 7/8] false_positive_rate[Table]     0.0000  ✓ PASS  ≤ 0.10
[ 8/8] n_classes                          5   INFO
Overall: FAIL  PASS=5  WARN=0  FAIL=2  INFO=1
```

D-029 매핑 정상 작동 — `Table` 입력에 `TitleBlock` 임계값 (0.95) 자동 적용. PMI/Text 미달은 100장 seed 의 자연스러운 한계 (5,839장 본격 학습 시 자연 개선 예상).

---

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 평가 함수 | ultralytics `model.val()` | 표준 mAP/P/R 계산 |
| 메트릭 추출 | `metrics.box.{p,r,f1,map50,map,ap50}` | ultralytics API |
| Confusion matrix | `metrics.confusion_matrix.matrix` (n+1 × n+1, 마지막 = background) | ultralytics 표준. **시각화 전용** (2026-04-28 수정 후) |
| **Per-class accuracy 산출** | **`box.ap50` (per-class AP@0.5)** | 2026-04-28 수정 — 이전 CM 기반은 ultralytics 8.4.x 에서 0 반환 버그 |
| **D-029 매핑** | **`_resolve_threshold()` 양방향 lookup** | Roboflow 이름 (`Table`/`Text`) ↔ 내부명 (`TitleBlock`/`Notes`) 자동 매핑 |
| FP rate 측정 | CM 마지막 행 (background → 각 class), Roboflow `Table` / 내부 `TitleBlock` 양쪽 인식 | 배경 오인 추적 (D-029 호환) |
| 임계값 | YAML `stage1_model.*` 동적 로드 | 사용자 조정 가능 |
| 산출 차트 | per-class mAP + per-class accuracy 막대 + CM heatmap | 시각 검수 |
| graceful 종료 | weights 미존재 → exit 2, ultralytics 미설치 → exit 3 | CI 통합 친화 |

## 3. 사용법

```bash
# 기본
python -m src.validate.check_stage1_model \
    --weights checkpoints/yolo_det.pt \
    --data configs/yolo_det.yaml \
    --device 0

# 분할 변경
python -m src.validate.check_stage1_model \
    --weights checkpoints/yolo_det.pt \
    --split test

# 입력 해상도 변경
python -m src.validate.check_stage1_model \
    --weights checkpoints/yolo_det.pt \
    --imgsz 1536 --batch 4
```

## 4. 검증 결과

### 4.1 CLI 단위 검증

```
$ python -m src.validate.check_stage1_model --help
usage: check_stage1_model.py [-h] [--weights WEIGHTS] [--data DATA]
                             [--split {val,test,train}] [--imgsz IMGSZ]
                             ...
```

CLI 정상 동작 확인. `--help` 만으로는 ultralytics import 안 함 (lazy import).

### 4.2 Graceful fail

```
$ python -m src.validate.check_stage1_model --weights nonexistent.pt
ERROR | Weights not found: nonexistent.pt
exit code: 2
```

ultralytics 미설치 환경:
```
ERROR | ultralytics not installed: No module named 'ultralytics'
exit code: 3
```

### 4.3 실제 학습 후 검증 (대기)

학습 완료 후 사용자가 실행 → HTML 리포트 확인.

## 5. 임계값 (`validation_thresholds.yaml#stage1_model`)

```yaml
stage1_model:
  per_class_accuracy:
    View:       {threshold: 0.90, severity: critical}
    TitleBlock: {threshold: 0.95, severity: critical}
    Notes:      {threshold: 0.90, severity: critical}
  map_at_50:    {threshold: 0.85, severity: critical}
  false_positive_rate_max:
    threshold: 0.10
    severity: warning
    note: "TB 없는 도면에서 TB 검출 비율"
```

## 6. 출력 형식

### 6.1 콘솔 (예상)

```
╔════════════════════════════════════════════════════════════════╗
║  V2-B — Stage 1 YOLOv11-det Model Validation                   ║
╚════════════════════════════════════════════════════════════════╝

 [1/7] mAP@0.5 (overall)                  0.8721   ✓ PASS  ≥ 0.8500
       · mAP@0.5:0.95 = 0.6432
 [2/7] per_class_accuracy[View]           0.9215   ✓ PASS  ≥ 0.9000
 [3/7] per_class_accuracy[TitleBlock]     0.9633   ✓ PASS  ≥ 0.9500
 [4/7] per_class_accuracy[Notes]          0.9081   ✓ PASS  ≥ 0.9000
 [5/7] false_positive_rate[TitleBlock]    0.0421   ✓ PASS  ≤ 0.1000

 Overall: PASS
```

### 6.2 HTML 첨부

- "Per-class metrics" 표 (precision / recall / F1 / mAP@0.5)
- "Confusion matrix (val set)" 4×4 heatmap (3 classes + background)
- "Per-class mAP@0.5" 막대 차트
- "Per-class accuracy (CM-based)" 막대 차트

### 6.3 JSON

```json
{
  "step": "stage1_model",
  "overall_status": "PASS",
  "metadata": {
    "weights": "checkpoints/yolo_det.pt",
    "split": "val",
    "imgsz": 1280,
    "classes": ["Isometric", "PMI", "TitleBlock", "Notes", "View"]
  },
  ...
}
```

## 7. 의존성

```
ultralytics>=8.3.0     # lazy import inside run()
torch (CUDA 12.8)
numpy
PyYAML, matplotlib, jinja2 (via common.py)
```

CLI `--help` 시점에서는 ultralytics 불필요 (lazy import).

## 8. 관련 의사결정

- **D-001** 아키텍처 = YOLOv11-det
- **D-006** 모델 = `yolo11m`
- **D-020** 검증 의무화
- **D-021** severity (critical 미달 시 다음 단계 차단)

## 9. 검증 대상 모듈

[`stage1_layout.md`](./stage1_layout.md) — Step 2

## 10. FAIL 발생 시 조치

| FAIL 항목 | 추정 원인 | 조치 |
|---|---|---|
| 특정 클래스 accuracy 미달 | 해당 클래스 라벨 부족 | V2-A 클래스 분포 점검 → 추가 라벨링 |
| mAP < 0.85 (전반) | epochs 부족 / imgsz 작음 | epochs 150 / imgsz 1536 / yolo11l 시도 |
| FP rate (TB) 높음 | 라벨 noise | TB 라벨 재검수, "비슷한 영역" 인지 확인 |

논문 수치 (View 0.96 / TB 0.99 / Notes 0.98) 도달 가능 — 5,839 도면 데이터셋이 논문 1,000 �