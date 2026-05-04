# `src/validate/check_stage2_model.py`

> **V3-B** ★ — Stage 2 (YOLOv11-obb) 모델 + **D-023 사용자 필수 임계값** 측정

## 1. 구현 요약

★ **사용자 필수 요구사항 (D-023)** 측정 모듈. ultralytics `model.val()` 의 표준 mAP 외에, **per-image GT-Pred greedy 매칭** 으로 클래스별 누락률 + 도면 단위 회수율을 별도 산출.

**측정 항목**

| 항목 | 임계값 | Severity |
|---|---|---|
| `mAP@0.5 (overall)` | ≥ 0.80 | critical |
| **`missing_rate[Measure]`** | **< 0.08** | **critical** ★ |
| **`missing_rate[GDT]`** | **< 0.05** | **critical** ★ |
| `missing_rate[Roughness]` | < 0.30 | warning |
| **`drawing_level_recall`** | **≥ 0.85** | **critical** ★ |
| `per_class_accuracy[Measure/GDT/Roughness]` | 0.92 / 0.95 / 0.50 | critical / critical / warning |
| `class_confusion_rate` | < 0.05 | warning |
| per-class TP / FP / FN / P / R / F1 표 | — | info |

★ = 사용자 필수 요구 (다른 검증기에서는 없음)

**측정 흐름**

```
1. ultralytics model.val()
   → mAP@0.5, per-class P/R, confusion matrix 추출
   
2. val 이미지마다 model.predict() 실행
   → GT obb 로딩 + Pred obb 파싱 + IoU greedy 매칭

3. 클래스별 TP/FP/FN 누적
   → missing_rate[c] = FN[c] / (TP[c] + FN[c])

4. 이미지별 recall 계산
   → drawing_level_recall = mean(per_image_recall)
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 매칭 IoU | **0.5** (CLI override 가능) | 표준 OBB 검출 평가 |
| 매칭 알고리즘 | **Greedy (conf desc)** + **same-class** | 단순·재현성 |
| Polygon IoU | **shapely** 우선 / axis-aligned bbox IoU fallback | 정확도 + 의존성 graceful |
| Per-image GT 무 처리 | recall = 1.0 (vacuous truth) | 빈 GT 도면은 평가 제외 |
| Drawing-level recall | TP / GT_count per image, 평균 | "도면 단위 보장" 사용자 요구 |
| Confusion matrix | ultralytics 표준 사용 | 클래스 혼동 측정 |
| Class confusion | off-diagonal sum / total | 5% 미만이 정상 |
| BG → TB FP | Stage 1 모델과 달리 별도 산출 안 함 (Stage 2 는 View 안에서 동작이라 BG 의미 다름) | 스코프 한정 |
| 산출 차트 | per-class missing rate + drawing recall histogram + CM heatmap | 시각 검수 |

## 3. 사용법

```bash
# 기본 (★ D-023 측정)
python -m src.validate.check_stage2_model \
    --weights checkpoints/yolo_obb.pt \
    --data configs/yolo_obb.yaml \
    --device 0

# IoU 임계값 / confidence 조정
python -m src.validate.check_stage2_model \
    --weights checkpoints/yolo_obb.pt \
    --iou 0.5 --conf 0.25

# test 분할로 평가
python -m src.validate.check_stage2_model \
    --weights checkpoints/yolo_obb.pt \
    --split test
```

## 4. 검증 결과

### 4.1 CLI 단위 검증

```
$ python -m src.validate.check_stage2_model --help
usage: check_stage2_model.py [-h] [--weights WEIGHTS] [--data DATA]
                             [--split {val,test,train}] [--imgsz IMGSZ]
                             [--batch BATCH] [--device DEVICE] [--iou IOU]
                             [--conf CONF] ...
```

CLI 정상.

### 4.2 Graceful fail

ultralytics 미설치 시 exit 3, weights 미존재 시 exit 2.

### 4.3 실제 학습 후 검증 (대기)

학습 완료 → 사용자 실행 → HTML 리포트로 누락률·회수율 확인.

## 5. 임계값 (`validation_thresholds.yaml#stage2_model`)

```yaml
stage2_model:
  per_class_accuracy:
    Measure:   {threshold: 0.92, severity: critical}
    GDT:       {threshold: 0.95, severity: critical}
    Roughness: {threshold: 0.50, severity: warning}    # 논문 0.54
  map_at_50:    {threshold: 0.80, severity: critical}
  
  missing_rate_max:      # ★ 사용자 필수 (D-023)
    Measure:   {threshold: 0.08, severity: critical}
    GDT:       {threshold: 0.05, severity: critical}
    Roughness: {threshold: 0.30, severity: warning}
  
  drawing_recall_min:    # ★ 사용자 필수 (D-023)
    threshold: 0.85
    severity: critical
  
  class_confusion_max:
    threshold: 0.05
    severity: warning
```

## 6. 출력 형식

### 6.1 콘솔 (예상)

```
╔════════════════════════════════════════════════════════════════╗
║  V3-B — Stage 2 YOLOv11-obb Model Validation (★ D-023)         ║
╚════════════════════════════════════════════════════════════════╝

 [ 1/12] mAP@0.5 (overall)                 0.8421   ✓ PASS  ≥ 0.8000
 [ 2/12] missing_rate[Measure]             0.0521   ✓ PASS  ≤ 0.0800   (FN=120 / GT=2300)
 [ 3/12] missing_rate[GDT]                 0.0314   ✓ PASS  ≤ 0.0500
 [ 4/12] missing_rate[Roughness]           0.4200   ! WARN  ≤ 0.3000   (D-017 trigger)
 [ 5/12] drawing_level_recall (★)          0.8721   ✓ PASS  ≥ 0.8500   (mean over 920 images)
 [ 6/12] per_class_accuracy[Measure]       0.9281   ✓ PASS  ≥ 0.9200
 [ 7/12] per_class_accuracy[GDT]           0.9612   ✓ PASS  ≥ 0.9500
 [ 8/12] per_class_accuracy[Roughness]     0.5823   ✓ PASS  ≥ 0.5000
 [ 9/12] class_confusion_rate              0.0421   ✓ PASS  ≤ 0.0500

 Overall: WARN   PASS=8 WARN=1 FAIL=0
```

### 6.2 HTML 첨부

- "Per-class TP / FP / FN / missing rate (★ D-023)" 표
- "Per-class missing rate" 막대 차트 (★)
- "Drawing-level recall histogram" (이미지별 recall 분포)
- "Confusion matrix (val set)" 4×4 heatmap

### 6.3 JSON

```json
{
  "step": "stage2_model",
  "overall_status": "WARN",
  "metadata": {
    "iou_threshold": 0.5,
    "conf_threshold": 0.25,
    "classes": ["Measure", "GDT", "Roughness"]
  },
  ...
}
```

## 7. 의존성

```
ultralytics>=8.3.0    # lazy import
torch (CUDA 12.8)
shapely>=2.0.0        # OBB polygon IoU (fallback 가능)
opencv-python         # imread for size
numpy, PyYAML, matplotlib, jinja2 (via common.py)
```

shapely 미설치 시 axis-aligned bbox IoU 로 자동 fallback (정확도 약간 ↓).

## 8. 관련 의사결정

- **D-001** 아키텍처 = YOLOv11-obb
- **D-016** Stage 2 = OBB 검출 + Stage 3-N 으로 통합 처리 (eDOCr2 다단계 pipeline 차용 안 함)
- **D-017** Roughness 부족 / 누락 시 synthetic_gen.py 백업 (조건부)
- **D-020** 검증 의무화
- **D-021** severity 분류
- **D-023** ★ 사용자 필수 임계값 — 본 검증기가 직접 측정
- **D-024** 사전 증강 (group-aware split 후 평가)

## 9. 검증 대상 모듈

[`stage2_annotation.md`](./stage2_annotation.md) — Step 3

## 10. FAIL 발생 시 액션 (Plan A → Plan B 전환 결정)

| FAIL 항목 | 1차 조치 | 2차 (Plan B) |
|---|---|---|
| `missing_rate[Measure] > 8%` | epochs 200 / imgsz 1280 | **`utils/symbol_postcorrect.py` 추가** (∅ 템플릿 매칭, eDOCr2 차용, D-017) |
| `missing_rate[GDT] > 5%` | GDT 라벨 추가, fcf_split.py 검토 (D-015) | LLM 후처리 |
| `missing_rate[Roughness] > 30%` | **`utils/synthetic_gen.py`** (D-017) | LoRA fine-tune 또는 추가 데이터 |
| `drawing_level_recall < 0.85` | 모든 클래스 누락률 점검 | 모델 교체 검토 |
| `class_confusion_rate > 5%` | 라벨 재검수 | hard negative mining |

**Plan B 발동 결정점** = V3-B 결과. 임계값 미달이 critical 이면 Plan A 진행 차단 → Plan B 개발 시작.

## 11. 측정 시간 예상

- 5,839 도면 × 80% (val) = 약 920 도면
- per-image predict + IoU 매칭: ~0.5초/도면 (RTX 5080 기준)
- 총 약 8~10분
