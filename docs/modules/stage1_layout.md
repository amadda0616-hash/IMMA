# `src/stage1_layout.py`

> **Step 2** — YOLOv11-det Layout Segmentation (Isometric / PMI / Table=TB / Text=Notes / View, **5 클래스 D-028**)

## 1. 구현 요약

도면 1장에서 5개 영역 클래스를 axis-aligned BBox 로 검출하는 YOLOv11-det 모듈.

**3개 CLI 서브커맨드**

```
train     ─ yolo11m.pt 파인튜닝 (best.pt 자동 복사)
predict   ─ JPG → §5.1 스키마 JSON
crop      ─ predict 후 클래스별 폴더로 영역 자동 crop
```

**공개 함수** (pipeline.py / prepare_vlm_dataset.py 가 import)

```python
predict_one(image_path, weights, conf_thr, imgsz, device) -> dict
crop_regions(image_path, regions, out_dir, padding) -> list
```

**클래스 (D-028, 2026-04-28 갱신)**

Roboflow data.yaml 기준 (`IMMA.v1i.yolov11/data.yaml`):
- `CLASS_NAMES_RF = ["Isometric", "PMI", "Table", "Text", "View"]`

코드 내부 정규명 (D-029 매핑 후, 출력 JSON `regions[*].class`):
- `CLASS_NAMES = ["Isometric", "PMI", "TitleBlock", "Notes", "View"]`

| Roboflow | Internal | 역할 |
|---|---|---|
| `Isometric` | `Isometric` | 3D 등각투영도 (Stage 2 OBB skip) |
| `PMI` | `PMI` | ★ Stage 2 OBB 의 입력 영역 (Product Manufacturing Information) |
| `Table` | `TitleBlock` | 정보 블록 (도번/제목/재질/척도 등). 의미 보존 매핑 (D-029) |
| `Text` | `Notes` | 일반 노트, 주석 영역. 의미 보존 매핑 (D-029) |
| `View` | `View` | 2D 뷰 (정면/평면/단면/상세) |

매핑은 `_result_to_schema()` 1지점에서 적용 (`ROBOFLOW_TO_INTERNAL` dict).
하위 모듈 (`pipeline.py`, `prepare_vlm_dataset.py`, `stage3_alphabetical.py`) 은 변경 없이 내부명 그대로 사용.

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 베이스 모델 | `yolo11m.pt` | 정확도-속도 균형. 16GB VRAM 적합 |
| 입력 해상도 | **1280** | 작은 TitleBlock 셀 가독성 |
| Batch size | 8 (RTX 5080 기본) | 16GB VRAM 안전 |
| Epochs | 100 (CLI override 가능) | 논문 기본 |
| **Flip augmentation** | **fliplr=0.0, flipud=0.0** ★ | D-001 / 도면 방향성 보존 (Mirror 도면은 전혀 다른 도면) |
| HSV / scale / translate | 활성 | 일반 augmentation |
| Mosaic | 활성 | 작은 객체 학습 도움 |
| Mixup | 비활성 | 도면 의미 손상 |
| best.pt 처리 | `checkpoints/yolo_det.pt` 로 자동 복사 | 표준 경로 보장 |
| BBox clamp | 이미지 경계 초과 방지 | 안전 처리 |
| 유니코드 파일명 | `np.fromfile + cv2.imdecode/imencode` | 한글/일본어/러시아어 경로 |
| 출력 스키마 | HANDOFF §5.1 정확히 일치 | 다운스트림 호환 |

## 3. 사용법

### CLI

```bash
# 학습
python src/stage1_layout.py train \
    --data configs/yolo_det.yaml \
    --epochs 100 --imgsz 1280 --batch 8 --device 0

# 추론 → JSON
python src/stage1_layout.py predict \
    --image dataset/sample.jpg \
    --weights checkpoints/yolo_det.pt \
    --out outputs/sample.det.json

# 영역 자동 crop (Stage 2 / Stage 3-A 입력 준비)
python src/stage1_layout.py crop \
    --image dataset/sample.jpg \
    --weights checkpoints/yolo_det.pt \
    --out-dir outputs/crops/sample \
    --padding 5
```

### 공개 함수

```python
from src.stage1_layout import predict_one, crop_regions

rec = predict_one(Path("sample.jpg"), Path("checkpoints/yolo_det.pt"))
# rec = {"drawing_id": ..., "regions": [{"class": "View", "bbox": [...], ...}]}

crops = crop_regions(Path("sample.jpg"), rec["regions"], Path("outputs/crops/sample"))
# crops = [{"class": "View", "bbox": [...], "path": "outputs/crops/sample/View/sample__View_00.jpg"}]
```

## 4. 검증 결과

### 4.1 학습 후 검증 (V2-B)

```bash
python -m src.validate.check_stage1_model \
    --weights checkpoints/yolo_det.pt --data configs/yolo_det.yaml
```

**임계값 (논문 §4.1 / configs/validation_thresholds.yaml#stage1_model)**

| 항목 | 임계값 | Severity |
|---|---|---|
| mAP@0.5 (overall) | ≥ 0.85 | critical |
| View accuracy | ≥ 0.90 | critical |
| TitleBlock accuracy | ≥ 0.95 | critical |
| Notes accuracy | ≥ 0.90 | critical |
| FP rate (배경→TB) | < 0.10 | warning |

### 4.2 라벨 검증 (V2-A) — 학습 전

```bash
python -m src.validate.check_labels_yolo \
    --labels-dir data/layout/labels/train \
    --cfg configs/yolo_det.yaml
```

8 항목: 빈 라벨, 파싱 에러, BBox 유효성, 클래스 분포, 작은 BBox, 라벨-이미지 매칭, aspect ratio, 에러 샘플.

## 5. 출력 형식

### 5.1 predict 출력 (HANDOFF §5.1)

```json
{
  "drawing_id": "sample",
  "image_path": "/abs/path/sample.jpg",
  "image_size": [W, H],
  "regions": [
    {"class": "View",       "bbox": [x1,y1,x2,y2], "conf": 0.97},
    {"class": "TitleBlock", "bbox": [x1,y1,x2,y2], "conf": 0.99},
    {"class": "Notes",      "bbox": [x1,y1,x2,y2], "conf": 0.98}
  ]
}
```

### 5.2 crop 출력 디렉터리

```
outputs/crops/sample/
├── View/
│   ├── sample__View_00.jpg
│   └── sample__View_01.jpg
├