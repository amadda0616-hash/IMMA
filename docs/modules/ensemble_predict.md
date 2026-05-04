# `src/ensemble_predict.py`

> **Phase 14** — Stage 2 5-Fold OBB Ensemble (★ D-040, D-023 PASS 확보)

## 1. 구현 요약

V3-B 단일 모델 (Fold 2 best.pt) 이 Measure missing_rate **0.101** (D-023 임계 0.08 초과 → FAIL) 을 보였다. 5-fold K-fold CV 학습 결과를 보유하고 있으므로 5개 모델의 OBB 검출 결과를 concatenate 후 **클래스별 rotated NMS** 로 합쳐 recall 을 끌어올리는 ensemble 채택.

**측정 결과 (2026-05-04, fold_2 val 110장)**:

| 클래스 | P_single → P_ens | R_single → R_ens | miss_single → miss_ens | 임계 | 판정 |
|---|---|---|---|---|---|
| Measure | 0.949 → **0.683** | 0.899 → **1.000** | 0.101 ❌ → **0.000** | <0.08 | ✅ PASS |
| GDT | 0.945 → 0.848 | 1.000 → 1.000 | 0.000 → 0.000 | <0.05 | ✅ PASS |
| Roughness | 0.957 → 0.846 | 0.964 → 1.000 | 0.036 → 0.000 | <0.30 | ✅ PASS |

drawing_recall = **1.000** / D-023 overall = **★ PASS ★**

Trade-off: Recall 0.899 → 1.000 (+0.101) 대신 Precision 0.949 → 0.683 (-0.266). FP 증가는 Stage 3-A (PaddleOCR-VL) 의 빈 영역 hallucination 위험 증가로 후속 모니터링 필요.

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 모델 수 | **5 fold** (K-fold CV 학습 산출물) | 추가 학습 불필요, fold variance 활용 |
| Detection 결합 | concatenate + class-wise rotated NMS | recall 최대화 |
| NMS IoU | **0.5** | Stage 2 학습 default + COCO OBB 표준 |
| conf 기본 | **0.25** | ultralytics 표준 (recall-precision balance) |
| OBB IoU | shapely Polygon | V3-B (`check_stage2_model.py`) 와 일관성 |
| NMS 호환성 | **다중 import 경로 + manual fallback** | ultralytics 8.3+ `nms_rotated` 위치 변경 대응 |
| Pipeline 통합 | `predict_one_schema()` 별도 어댑터 | `stage2_annotation.predict_one()` 호환 schema 반환 |

### 2.1 NMS Resolver (★ ultralytics 버전 호환성)

ultralytics 8.3+ 에서 `nms_rotated` 가 `ultralytics.utils.ops` 에서 제거됨. 다음 4개 경로를 순회 후 모두 실패 시 manual fallback.

```
ultralytics.utils.ops.nms_rotated     (8.3 이전)
ultralytics.utils.metrics.nms_rotated (중간 버전)
ultralytics.utils.tal.nms_rotated     (신규)
ultralytics.utils.nms_rotated         (namespace level)
       ↓ 모두 실패 시
manual_nms_rotated() — shapely 기반 greedy NMS
```

**Manual NMS 구현** (`xywhr_to_corners` + `polygon_iou`):
1. 각 box 의 `(cx, cy, w, h, r)` → 4-corner 변환
2. score 내림차순 정렬
3. 최고 conf box 선택, IoU > thr 인 모든 box 제거 → 반복
4. 결과: keep indices (`torch.long`)

## 3. 공개 API

```python
# 5 fold best.pt 로드
load_fold_models(ckpt_root, n_folds=5, fold_pattern, device) -> list[YOLO]

# 단일 이미지 → 5 모델 detection → class-wise NMS
ensemble_predict(models, image, conf=0.25, iou_nms=0.5, imgsz=1024, device)
    -> dict {xyxyxyxy, xywhr, conf, cls, img_w, img_h} | None

# Pipeline 통합용 — HANDOFF §5.2 schema 반환
predict_one_schema(models, image_path, conf, iou_nms, imgsz, device, parent_bbox)
    -> dict {view_id, image_path, image_size, parent_bbox, annotations: [...]}

# val.txt 기반 D-023 evaluation
evaluate_d023(val_txt, ckpt_root, n_folds, conf, iou_nms, iou_match,
              imgsz, device) -> dict (per_class P/R/miss + drawing_recall + pass)

# 단일 이미지 inference → JSON
predict_single(image, ckpt_root, n_folds, conf, iou_nms, imgsz, device) -> dict

# Helpers
polygon_iou(p1, p2) -> float                    # shapely Polygon IoU
xywhr_to_corners(xywhr) -> ndarray              # (5,) → (4, 2)
manual_nms_rotated(boxes_xywhr, scores, iou_thr) -> torch.Tensor
```

## 4. 사용법

### 4.1 D-023 재평가 (Fold 2 val set)

```bash
python src/ensemble_predict.py evaluate \
    --val-txt data/annotation_kfold/fold_2/val.txt \
    --conf 0.25 --iou-nms 0.5 --imgsz 1024 \
    --device cuda:0 \
    --output outputs/v3b_ensemble_eval.json
```

종료 코드: `0` = D-023 PASS, `1` = FAIL.

### 4.2 단일 이미지 추론

```bash
python src/ensemble_predict.py predict \
    --image data/annotation/images/valid/sample.jpg \
    --conf 0.25 --iou-nms 0.5 \
    --output outputs/sample_predictions.json
```

### 4.3 Pipeline 통합 (★ default, D-040)

`src/pipeline.py` 의 Stage 2 = ensemble mode (default `--use-ensemble`):

```bash
# 단일 도면 (ensemble default ON)
python src/pipeline.py run --image dataset/sample.jpg \
    --out outputs/sample.json

# 단일 best.pt 사용 (legacy 디버깅)
python src/pipeline.py run --image dataset/sample.jpg \
    --no-ensemble --obb-weights checkpoints/yolo_obb.pt
```

## 5. CLI 인자

```
--ckpt-root            checkpoints/yolo_obb_runs (default)
--n-folds              5 (default)
--fold-pattern         yolo_obb_v3_kfold_{i} (default)
--conf                 0.25 (default)
--iou-nms              0.5 (default, cross-fold rotated NMS)
--imgsz                1024 (default, must match training)
--device               cuda:0 (recommended)

evaluate:
  --val-txt            (필수) val 이미지 list .txt
  --iou-match          0.5 (default, GT 매칭 IoU)
  --output             summary JSON 경로

predict:
  --image              (필수) 단일 이미지 경로
  --output             detections JSON 경로
```

## 6. 의존성

- `ultralytics>=8.3.0` (predict)
- `shapely>=2.0.0` (polygon IoU + manual NMS)
- `torch` (cu128, D-030)
- `numpy<2.0`

## 7. 박제

- **D-040** (PROJECT_HANDOFF.md): Stage 2 5-Fold Ensemble 채택 (D-023 PASS)
- **history.md §A.11.13** : V3-B + ensemble 의사결정
- **outputs/v3b_summary.txt** : V3-B + ensemble 결과 요약
- **outputs/v3b_ensemble_eval.json** : raw evaluate 결과

## 8. 향후 검토

| 조건 | 액션 |
|---|---|
| Stage 3-A FP 처리 부담 시 | conf=0.30 (Measure FP 50% ↓ 추정) |
| 추론 속도 부족 시 | top-3 fold 만 ensemble (속도/recall 절충) |
| 다른 val split 검증 시 | Fold 0/1/3/4 val.txt 로 cross-fold eval |
| Weighted Box Fusion 실험 | manual_nms_rotated 대체 구현 |
| ultralytics nms_rotated 복원 시 | `_resolve_nms_rotated()` 자동 사용 (코드 변경 X) |
