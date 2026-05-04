# `src/stage2_annotation.py`

> **Step 3** — YOLOv11-obb Annotation Localization (Measure / GDT / Roughness — Stage 2 클래스 변경 없음, D-028 확인)

## 1. 구현 요약

Stage 1 이 잘라낸 **PMI crop** (D-028, 이전 버전: View crop) 위에서 회전된 어노테이션(치수, GD&T, 표면거칠기)을 OBB(oriented bounding box)로 검출. perspective-warp de-rotation 으로 upright 패치 생성 (Stage 3-N 입력 준비).

> **D-028 입력 영역 변경**: Stage 1 새 클래스 `PMI` 가 가공정보 영역을 명시적으로 묶음. View 안에 PMI, PMI 안에 Measure/GDT/Roughness 의 계층이 명확해짐. 기존 코드는 View crop 입력 가능 (호환 유지) — 새 데이터에서는 PMI crop 권장.

**3개 CLI 서브커맨드**

```
train     ─ yolo11m-obb.pt 파인튜닝 (회전 augmentation ON)
predict   ─ JPG → §5.2 OBB 스키마 JSON
crop      ─ predict 후 perspective-warp 으로 de-rotation 패치 저장
```

**공개 함수**

```python
predict_one(image_path, weights, conf_thr, imgsz, device, parent_bbox) -> dict
crop_obb_regions(image_path, annotations, out_dir) -> list
warp_obb_crop(img, obb_pts) -> ndarray   # 단일 OBB → upright 패치
order_obb_points(pts) -> ndarray         # TL,TR,BR,BL 순서 정규화
```

**클래스**: `["Measure", "GDT", "Roughness"]` (논문 §3.2, D-028 확인 후 그대로 유지)

**근거 (변경 없음)**: ISO 1302 / ASME Y14 의 PMI 표준 3종 + Khan 2025 논문의 동일 클래스. Stage 1 새 클래스 `PMI` 가 입력 영역을 명시할 뿐, OBB 클래스 자체는 변동 없음.

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 베이스 모델 | `yolo11m-obb.pt` | OBB 변형 |
| 입력 해상도 | **1024** (1280 옵션) | OBB 는 axis-aligned 보다 무거움 |
| Epochs | **150** (axis-aligned 100) | OBB 수렴 느림 |
| Patience | 40 | epochs 늘어남 대응 |
| Rotation augmentation | **degrees=15.0** ★ | 회전 텍스트 학습 |
| Flip | OFF | 도면 좌우반전 의미 다름 |
| **OBB de-rotation** | **perspective-warp 적용** ★ | D-012 / Donut 정확도 결정적 |
| 4-point 순서 | TL, TR, BR, BL 자동 정규화 | 일관된 warp 결과 |
| 각도 정규화 | `[-90, 90)` | 표준 형식 |
| `parent_bbox` 인자 | 옵션 파라미터로 보존 | 글로벌 좌표 환산용 (pipeline.py) |
| 클래스 불균형 | Roughness oversampling 메모 (cfg) | 논문 152 instances 만 |

## 3. 사용법

### CLI

```bash
# 학습
python src/stage2_annotation.py train \
    --data configs/yolo_obb.yaml \
    --epochs 150 --imgsz 1024 --batch 8 --device 0

# View crop 위에서 OBB 추론
python src/stage2_annotation.py predict \
    --image outputs/crops/sample/View/sample__View_00.jpg \
    --weights checkpoints/yolo_obb.pt

# perspective-warp de-rotation 패치 저장
python src/stage2_annotation.py crop \
    --image outputs/crops/sample/View/sample__View_00.jpg \
    --weights checkpoints/yolo_obb.pt
```

### 공개 함수 사용 예

```python
from src.stage2_annotation import (
    predict_one, crop_obb_regions, warp_obb_crop,
)

rec = predict_one(
    Path("view.jpg"), Path("checkpoints/yolo_obb.pt"),
    parent_bbox=[100, 200, 1500, 1100],   # Stage 1 의 View bbox
)
crops = crop_obb_regions(Path("view.jpg"), rec["annotations"], Path("out/"))
# crops 의 각 항목은 de-rotated upright .jpg 파일 경로
```

## 4. 검증 결과

### 4.1 라벨 검증 (V3-A) — 학습 전

```bash
python -m src.validate.check_labels_obb \
    --labels-dir data/annotation/labels/train --cfg configs/yolo_obb.yaml
```

**6건 더미 데이터 검증 결과** (negative path 모두 정상 검출):
- 정상 OBB 4 + 자기교차 폴리곤 1 + 좌표 범위 초과 1 + 빈 라벨 1 + 파싱 에러 1
- empty 33% / parse 17% / OBB validity 80% 모두 정확 검출
- Roughness 2개만 → WARN 발생 (D-017 trigger 알림)
- view_002 의 회전 OBB 2개 → non_axis_aligned 25% PASS

### 4.2 ★ 모델 검증 (V3-B) — D-023 사용자 필수 임계값

```bash
python -m src.validate.check_stage2_model \
    --weights checkpoints/yolo_obb.pt --data configs/yolo_obb.yaml
```

| 항목 | 임계값 | Severity |
|---|---|---|
| **missing_rate[Measure]** | **< 0.08** | **critical** ★ |
| **missing_rate[GDT]** | **< 0.05** | **critical** ★ |
| missing_rate[Roughness] | < 0.30 | warning |
| **drawing_level_recall** | **≥ 0.85** | **critical** ★ |
| mAP@0.5 | ≥ 0.80 | critical |
| Measure / GDT / Roughness accuracy | 0.92 / 0.95 / 0.50 | critical / critical / warning |

## 5. 출력 형식

### 5.1 predict 출력 (HANDOFF §5.2)

```json
{
  "view_id": "sample__View_00",
  "image_path": "/abs/path/...",
  "image_size": [W, H],
  "parent_bbox": null,
  "annotations": [
    {
      "class": "Measure",
      "obb": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "angle": 12.5,
      "conf": 0.93
    }
  ]
}
```

### 5.2 crop 출력

```
outputs/crops/sample/annotations/
├── Measure/
│   ├── sample__View_00__Measure_00.jpg   ← upright (de-rotated)
│   ├── sample__View_00__Measure_01.jpg
│   └── ...
├── GDT/
│   └── sample__View_00__GDT_00.jpg
├── Roughness/
│   └── sample__View_00__Roughness_00.jpg
└── manifest.json
```

각 crop 은 `cv2.getPerspectiveTransform` + `cv2.warpPerspective` 로 회전 보정된 upright 직사각형.

## 6. 의존성

```
ultralytics>=8.3.0
opencv-python>=4.10.0
numpy>=1.26.0
torch (CUDA 12.8)
```

## 7. 관련 의사결정

- **D-001** 아키텍처 = 논문 (YOLOv11-obb)
- **D-009** 단일 모델 (언어 무관)
- **D-012** OBB crop 은 perspective-warp de-rotation 적용 (Donut 정확도 결정적)
- **D-015** FCF 컴파트�