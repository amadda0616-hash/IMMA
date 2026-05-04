# `src/extract_pmi_crops.py`

> **Step 5.5 — Active Learning Stage 2 입력 준비** (D-024, D-034, D-037) — Stage 1 PMI 영역 crop 추출 + 적응형 패딩 (per-axis)

## 1. 구현 요약

Stage 1 Version A 모델 (`checkpoints/yolo_det.pt`, mAP@0.5 = 0.9364) 의 PMI 검출 결과로부터 각 PMI 영역을 crop 하여 Stage 2 OBB 라벨링의 입력 준비. **D-037 adaptive padding (per-axis)** 정책으로 화살표/리더선 캡처를 최적화하면서 인접 치수 침입을 최소화.

**워크플로 (Stage 2 입력 준비)**

```
outputs/stage2_input_drawings.txt (20개 도면 list)
        ↓
Stage 1 Version A 모델 inference
        ↓
★ extract_pmi_crops.py — PMI crop 추출 (per-axis adaptive padding, D-037)
        ↓
outputs/cvat_stage2_input_v2/ (844 crops + manifest.csv)
        ↓
CVAT 로컬 업로드 → Stage 2 OBB 라벨링 시작
```

**핵심 컴포넌트** (~440 lines)

| 함수 | 역할 |
|---|---|
| `load_drawings(args)` | `outputs/stage2_input_drawings.txt` 또는 직접 폴더에서 도면 목록 로드 |
| `imread_unicode(path)` | Windows 한글 파일명 호환 이미지 읽기 (`np.fromfile` + `cv2.imdecode`) |
| `imwrite_unicode(path, img)` | Windows 한글 파일명 호환 이미지 저장 |
| `main()` | argparse + YOLO inference + PMI bbox 필터 + crop + manifest CSV 저장 |

**1개 CLI 서브커맨드 (단일 실행)**

```bash
python src/extract_pmi_crops.py [--padding-mode adaptive|fixed] [--padding-ratio 0.4] ...
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 입력 모델 | **Stage 1 Version A** (mAP 0.9364) | auto_label_stage1.py 의 시드 모델. 즉시 사용 가능 |
| PMI 필터 | **cls=1 only** (Roboflow data.yaml) | D-028, D-034 — PMI = Stage 2 OBB 입력 영역 |
| Padding 방식 | **per-axis adaptive** (D-037, v2) | v1 fixed 10px (화살표 잘림) → v2 per-axis (축별 독립 비례) |
| Padding 수식 | `pad_x = clamp(bbox_w × ratio, [min, max])` | 가로 큰 텍스트는 pad_x ↑, 세로 큰 텍스트는 pad_y ↑ |
| 기본값 | ratio=0.4, min=30, max=80 | 화살표 캡처 (30~80px) + 인접 침입 회피 (max=80) |
| Manifest | pad_x, pad_y, padding_mode 컬럼 | 각 crop 의 적용된 padding 추적 (v3 비교용) |
| Group key | **D-024 보존** (manifest 기록) | 원본 도면 추적 가능 — 차후 통계/검증 용이 |
| 면적 필터 | `--min-pmi-area 100` (기본) | 너무 작은 PMI 노이즈 제외 (5×20px 미만) |
| 경계 clamp | 이미지 경계 내 강제 | crop 이 이미지 밖으로 나가지 않도록 정수 좌표로 clamp |

## 3. 사용법

### CLI

```bash
# 기본 (per-axis adaptive padding, 20개 도면, 844 crops)
python src/extract_pmi_crops.py

# adaptive 파라미터 튜닝
python src/extract_pmi_crops.py \
    --padding-mode adaptive \
    --padding-ratio 0.4 \
    --padding-min 30 \
    --padding-max 80

# 호환성 — fixed padding (v1 방식, 모든 방향 동일 — 사용 비권장)
python src/extract_pmi_crops.py --padding-mode fixed --padding 30

# 직접 폴더 + 처음 N장
python src/extract_pmi_crops.py --input dataset/ --limit 5

# conf 임계값 (작은 PMI 더 잡기)
python src/extract_pmi_crops.py --conf 0.15

# 최소 PMI 면적 (너무 작은 PMI 제외)
python src/extract_pmi_crops.py --min-pmi-area 200
```

### CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `checkpoints/yolo_det.pt` | Stage 1 가중치 |
| `--drawings` | `outputs/stage2_input_drawings.txt` | 도면 filename list |
| `--input` | None | 직접 입력 폴더 (--drawings 무시) |
| `--dataset-dir` | `dataset/` | 원본 도면 폴더 |
| `--output` | `outputs/cvat_stage2_input_v2/` | PMI crop 출력 폴더 |
| `--padding-mode` | `adaptive` | `adaptive` (per-axis) 또는 `fixed` |
| `--padding` | 30 | fixed 모드 padding px (기본 30, 구 v1=10) |
| `--padding-ratio` | 0.4 | adaptive 모드: bbox_w × ratio (x), bbox_h × ratio (y) |
| `--padding-min` | 30 | adaptive 모드 최소 padding px |
| `--padding-max` | 80 | adaptive 모드 최대 padding px |
| `--conf` | 0.25 | confidence 임계값 |
| `--imgsz` | 1280 | inference 해상도 (Stage 1 학습과 동일) |
| `--device` | None | GPU id (e.g. 0) 또는 cpu |
| `--limit` | 0 | 처리 도면 수 제한 (0 = 전체) |
| `--min-pmi-area` | 100 | 최소 PMI 박스 면적 px² (너무 작은 PMI 제외) |

## 4. 출력 형식

### 4.1 디렉터리 구조

```
outputs/cvat_stage2_input_v2/      ← v2 (adaptive padding, D-037)
├── DwgFoo__PMI_000.jpg            ← CVAT 업로드용
├── DwgFoo__PMI_001.jpg
├── DwgBar__PMI_000.jpg
├── ...                             ← 총 844 crops
└── manifest.csv                    ← 원본 좌표 + 적용 padding 기록
```

### 4.2 Crop 이미지 명명

```
{source_drawing_stem}__PMI_{pmi_idx:03d}.jpg
```

예) `11_jpeg.rf.8b46c563__PMI_000.jpg` (도면 11_jpeg 의 1번째 PMI crop)

### 4.3 manifest.csv 예시

```csv
crop_filename,source_drawing,source_group_key,pmi_idx,bbox_x1,bbox_y1,bbox_x2,bbox_y2,crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h,padding_mode,pad_x,pad_y,conf
11_jpeg.rf.8b46c563__PMI_000.jpg,11_jpeg.rf.8b46c563.jpg,11_jpeg,0,150,200,250,300,120,170,280,330,160,160,adaptive,30,30,0.9234
11_jpeg.rf.8b46c563__PMI_001.jpg,11_jpeg.rf.8b46c563.jpg,11_jpeg,1,400,450,500,550,370,420,530,580,160,160,adaptive,30,30,0.8912
...
```

**컬럼 설명**:

| 컬럼 | 의미 |
|---|---|
| `crop_filename` | crop 이미지 파일명 |
| `source_drawing` | 원본 도면 파일명 (full, .jpg 포함) |
| `source_group_key` | D-024 group key (stem.split('.rf.')[0]) |
| `pmi_idx` | 동일 도면 내 PMI index (0~N) |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | 원본 이미지의 PMI bbox 좌표 (Stage 1 inference 결과) |
| `crop_x1, crop_y1, crop_x2, crop_y2` | crop 범위 (padding 적용 후) |
| `crop_w, crop_h` | crop 크기 |
| `padding_mode` | `adaptive` 또는 `fixed` |
| `pad_x, pad_y` | 실제 적용된 padding (픽셀) |
| `conf` | Stage 1 모델의 confidence score (4자리) |

### 4.4 Console 출력 예시

```
13:45:00 INFO    Output: outputs/cvat_stage2_input_v2/
13:45:00 INFO    Padding mode: adaptive per-axis (ratio=0.40, min=30, max=80) [D-037]
13:45:00 INFO    Conf: 0.25 / imgsz: 1280 / min_pmi_area: 100

Extracting PMI: 100%|████| 20/20 [00:47<00:00, 2.35s/img, crops=844, small=18, no_pmi=0, err=0]

============================================================
PMI crop 추출 완료
  Drawings processed   : 20
  No PMI (skipped)     : 0
  Errors               : 0
  PMI crops saved      : 844
  Small PMI skipped    : 16 (area < 100 px²)
  Avg PMI/drawing      : 42.2
============================================================
Output: outputs/cvat_stage2_input_v2/

  pad_x (가로 padding) : min=30 / max=80 / mean=33.2 px
  pad_y (세로 padding) : min=30 / max=44 / mean=30.6 px
  Crop 형태 분포        : 가로형=90 / 세로형=61 / 정사각형=693

============================================================
[다음 단계 — Stage 2 OBB 라벨링]
  1. CVAT docker 설치 (필요 시)
  2. Project 생성: Stage2_Annotation_OBB
  3. Task 생성: Stage2_PMI_v2_844 (v1 task 는 백업으로 보존)
  4. Labels: Measure / GDT / Roughness (rectangle, rotation 가능)
  5. ZIP 업로드 (이미지만, manifest.csv 제외):
     cd outputs/cvat_stage2_input_v2/ && zip ../stage2_v2.zip *.jpg
  6. OBB 라벨링 → export YOLO format → Stage 2 학습
```

## 5. 의존성

```
ultralytics>=8.3.0       # Stage 1 모델 추론
opencv-python>=4.10.0    # ultralytics 내부
numpy>=1.26.0
tqdm>=4.66.0
torch (CUDA 12.8)        # D-030
```

CLI `--help` 시점에 ultralytics 미설치도 OK (lazy import).

## 6. 관련 의사결정

- **D-024** Group-aware split — group key 보존 (manifest 기록)
- **D-028** 5 클래스 (PMI = Roboflow cls 1)
- **D-029** 매핑 (PMI 는 매핑 X — Roboflow 이름 그대로)
- **D-034** Hierarchical (PMI = Stage 2 입력 영역)
- **D-036** 옵션 B 정책 (auto_pass + review priority 만 — 회전 증강 노이즈 제거)
- **D-037** Adaptive padding — v1 fixed 10px 문제 해결
  - v1 (fixed 10px): 화살표/리더선 잘림, 인접 치수 침입
  - **v2 (per-axis adaptive)**: pad_x = bbox_w × 0.4, pad_y = bbox_h × 0.4, clamp [30, 80]
  - v3 (aspect-aware): 정사각형 bbox 에 uniform pad 적용 (45° 회전 보강) — 별도 스크립트

## 7. 흔한 FAIL + 조치

| 증상 | 원인 / 해결 |
|---|---|
| `Weights not found` | Stage 1 학습이 아직 안 됨. `python src/stage1_layout.py train ...` 먼저 |
| `ultralytics not installed` | `uv pip install ultralytics` |
| OOM during inference | `--imgsz 1024` 로 축소 (Stage 1 학습 1280 과 다르나 추론은 OK) |
| PMI crops 없음 (0장) | 도면에 PMI 검출 안 됨. 1) conf 임계값 낮추기 (`--conf 0.15`) 2) 도면 확인 |
| 많은 crop 이 tiny (<5×5px) | `--min-pmi-area` 값 재조정 (기본 100) |
| 파일명 한글 깨짐 (Windows) | `imread_unicode` / `imwrite_unicode` 사용 — 자동 처리 |

## 8. 다음 단계

1. **실행** — `python src/extract_pmi_crops.py` (~1분, 20도면 → 844 crops)
2. **결과 확인** — `outputs/cvat_stage2_input_v2/manifest.csv` padding 통계
3. **CVAT 로컬 설치** (필요 시) — docker compose up -d
4. **CVAT Task 생성** — Project: `Stage2_Annotation_OBB`, Task: `Stage2_PMI_v2_844`
5. **이미지 ZIP 업로드** — `cd outputs/cvat_stage2_input_v2/ && zip ../stage2_v2.zip *.jpg`
6. **OBB 라벨링** — Measure / GDT / Roughness (rotated rectangle)
7. **Export + Stage 2 학습** — YOLO format → `src/stage2_annotation.py train`

## 9. 버전 이력 (v1 → v2 → v3)

| 버전 | Padding 방식 | 특징 | 사용 시기 |
|---|---|---|---|
| **v1** | fixed 10px | 모든 방향 동일 | ❌ 화살표 잘림 (deprecated) |
| **v2 (현재)** | per-axis adaptive | x, y 축 독립 계산. 비회전 90% / 회전 80% 만족 | ✅ Stage 2 라벨링 입력 (D-037) |
| **v3** | aspect-aware | 정사각형 (aspect<1.5) uniform pad / 비정사각형 per-axis | 회전 텍스트 보강 (별도 스크립트 `extract_pmi_crops_v3.py`) |

본 스크립트는 **v2 (per-axis adaptive)** 구현.
