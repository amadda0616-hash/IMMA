# `src/auto_label_stage1.py`

> **Step 5.5 — Active Learning Step 6** (D-024) — Stage 1 자동 라벨링 + Roboflow Pre-annotation Import 호환 출력 + Active Learning 우선순위 매니페스트

## 1. 구현 요약

Stage 1 Version A 모델 (`checkpoints/yolo_det.pt`, mAP@0.5 = 0.9364) 으로 `dataset/` 의 미라벨 도면 5,739장을 자동 라벨링. Roboflow Pre-annotation Import 형식으로 출력하여 검수 → 본격 학습으로 이어지는 Active Learning 흐름의 핵심.

**워크플로 (Active Learning 2단계 중 자동 라벨링 단계)**

```
Roboflow seed 100장 라벨링 (✅)
        ↓
Stage 1 Version A 학습 (✅, mAP 0.9364)
        ↓
★ auto_label_stage1.py — 5,739장 자동 라벨링 (~5분)
        ↓
Roboflow Pre-annotation Import → 사람 검수 (~16시간)
        ↓
Stage 1 전체 재학습 (Version B)
```

**핵심 컴포넌트** (~393 lines)

| 함수 | 역할 |
|---|---|
| `get_seed_stems(seed_dir)` | `IMMA.v1i.yolov11/{train,valid,test}/images/` 의 stem 추출 → seed 자동 제외 |
| `predict_to_yolo_txt(image, model, ...)` | YOLO predict → 정규화 BBox → YOLO txt 라인 + confidence 리스트 |
| `link_or_copy(src, dst)` | symlink (Linux) 또는 copy (Windows fallback) |
| `main()` | argparse + tqdm + manifest CSV 정렬/저장 |

**1개 CLI 서브커맨드 (단일 실행)**

```bash
python src/auto_label_stage1.py [--weights ...] [--input ...] [--output ...]
```

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 클래스 ID | **Roboflow 원본 순서** (Isometric=0, PMI=1, Table=2, Text=3, View=4) | Roboflow Pre-annotation Import 호환성 (D-029 매핑 안 함) |
| Seed 자동 제외 | `IMMA.v1i.yolov11/{train,valid,test}/images/` 스캔 | 이미 라벨링된 100장 중복 작업 방지 |
| Conf 임계값 | 0.25 (ultralytics 기본) | 너무 높으면 누락 ↑, 너무 낮으면 검수 부담 ↑ |
| 출력 형식 | YOLO det txt + 이미지 (symlink/copy) | Roboflow Pre-annotation 표준 |
| **Active Learning priority** | 4-tier (empty / low_conf / review / auto_pass) | 검수 우선순위 자동 결정 |
| Manifest 정렬 | priority 우선 + avg_conf 오름차순 | 가장 우선 검수할 도면이 맨 위 |
| 진행바 | tqdm + `set_postfix` 실시간 통계 | 5,739장 / ~5분 작업 가시화 |
| 이미지 처리 | symlink 기본 (Linux), `--copy-images` 옵션 (Windows / Roboflow zip 업로드) | 디스크 공간 절약 vs 호환성 |
| Empty 라벨 | 빈 txt 파일 생성 (실패 X) | Roboflow 호환성 (이미지-라벨 1:1) |
| 에러 처리 | per-image try/except + log + 계속 | 단일 도면 실패가 전체 중단 X |

## 3. Active Learning Priority 4-tier

| Priority | 조건 | 의미 | 검수 권장 |
|---|---|---|---|
| `empty` | 0개 박스 검출 | 모델이 아무것도 검출 못함 → 사람이 직접 라벨링 | **1순위** |
| `low_conf` | avg_conf < 0.5 | 모델이 불확실 → 우선 검수 (Active Learning 핵심) | **2순위** |
| `review` | 0.5 ≤ avg_conf < 0.85 | 보통 — 일반 검수 | 3순위 |
| `auto_pass` | avg_conf ≥ 0.85 | 모델이 매우 확신 → 자동 패스 가능 (스킵 옵션) | 4순위 (스킵) |

**임계값**: `LOW_CONF_THRESHOLD=0.5`, `HIGH_CONF_THRESHOLD=0.85` (모듈 상수, 차후 CLI 노출 가능)

## 4. 사용법

### CLI

```bash
# 기본 (전체 5,739장)
python src/auto_label_stage1.py

# 모든 인자 명시
python src/auto_label_stage1.py \
    --weights checkpoints/yolo_det.pt \
    --input dataset/ \
    --output outputs/auto_labels/ \
    --conf 0.25 \
    --imgsz 1280 \
    --device 0

# 디버깅 — 처음 50장만
python src/auto_label_stage1.py --limit 50

# Seed 도 포함 (보통 제외)
python src/auto_label_stage1.py --include-seed

# Windows / Roboflow zip 업로드 — symlink 대신 복사
python src/auto_label_stage1.py --copy-images
```

### CLI 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `checkpoints/yolo_det.pt` | Stage 1 가중치 |
| `--input` | `dataset/` | 입력 이미지 폴더 |
| `--output` | `outputs/auto_labels/` | 출력 폴더 |
| `--seed-dir` | `IMMA.v1i.yolov11/` | 제외할 seed 폴더 |
| `--conf` | 0.25 | confidence 임계값 |
| `--imgsz` | 1280 | 입력 해상도 (Stage 1 학습과 동일) |
| `--device` | None (auto) | GPU id (0) 또는 cpu |
| `--limit` | None | 처리 제한 (디버깅) |
| `--include-seed` | False | seed 도 포함 |
| `--copy-images` | False | symlink 대신 복사 |

## 5. 출력 형식

### 5.1 디렉터리 구조

```
outputs/auto_labels/
├── labels/                    ← YOLO txt (5,739개)
│   ├── DwgFoo.txt              ← class_id cx cy w h (정규화)
│   ├── DwgBar.txt
│   └── ...
├── images/                    ← 원본 이미지 symlink (Linux) / 복사 (Windows)
│   ├── DwgFoo.jpg
│   ├── DwgBar.jpg
│   └── ...
└── manifest.csv               ← UTF-8-SIG, priority + avg_conf 정렬
```

### 5.2 YOLO txt 형식

```
1 0.523456 0.612345 0.123456 0.087654
1 0.156789 0.234567 0.087654 0.054321
2 0.876543 0.987654 0.234567 0.123456
4 0.500000 0.500000 0.987654 0.876543
```

각 줄: `<class_id> <cx> <cy> <w> <h>` (모두 [0, 1] 정규화).

### 5.3 manifest.csv 예시

```csv
filename,n_boxes,avg_conf,min_conf,max_conf,priority,label_path
DwgFoo.jpg,0,0.0000,0.0000,0.0000,empty,labels/DwgFoo.txt
DwgBar.jpg,42,0.4231,0.2511,0.7892,low_conf,labels/DwgBar.txt
DwgBaz.jpg,38,0.7012,0.4521,0.9133,review,labels/DwgBaz.txt
DwgQux.jpg,45,0.9234,0.8521,0.9921,auto_pass,labels/DwgQux.txt
```

UTF-8-SIG 인코딩 (Excel 다국어 친화).

### 5.4 Console 출력 예시

```
20:10:00 INFO    Total images in dataset: 5839
20:10:00 INFO    Seed images to skip: 100 (IMMA.v1i.yolov11)
20:10:00 INFO    Images to auto-label: 5739
20:10:00 INFO    Loading model: checkpoints/yolo_det.pt
20:10:01 INFO    Model classes: {0: 'Isometric', 1: 'PMI', 2: 'Table', 3: 'Text', 4: 'View'}

Auto-labeling: 100%|████| 5739/5739 [04:54<00:00, 19.5img/s, empty=120, low=380, high=4521, err=0]

============================================================
Auto-label complete: 5739 images
  Empty (0 boxes):                 120  (2.1%) — 사람이 직접 라벨
  Low conf (avg < 0.50):           380  (6.6%) — ★ 우선 검수
  High conf (avg ≥ 0.85):         4521 (78.8%) — 자동 패스 가능
  Errors:                            0
Per-class total bboxes:
  0  Isometric    1850
  1  PMI        198400
  2  Table       17200
  3  Text         4830
  4  View        24300
============================================================
Next: Roboflow Pre-annotation Import
```

## 6. 검증 결과

### 6.1 단위 테스트 (작성 시점, ultralytics 미설치 환경)

```
$ python -c "import ast; ast.parse(open('src/auto_label_stage1.py').read())"
✓ 문법 정상 (393 lines, 13,667 bytes)

$ python src/auto_label_stage1.py --help
✓ argparse 정상 동작
```

### 6.2 실제 데이터셋 적용 (대기)

5,739장 처리 후 다음 통계 확인 권장:
- `priority` 분포 — `empty` / `low_conf` 합이 10% 이내면 모델 품질 양호
- `n_boxes` 평균 — 100장 seed 평균 35.7개 (D-031) 와 비슷한지
- 클래스 분포 — PMI 80% 압도적이어야 함 (D-031)

만약 분포가 크게 다르면 모델 일반화 문제 또는 데이터 품질 이슈 가능.

## 7. Roboflow Pre-annotation Import 가이드

### 7.1 zip 패키징

```bash
cd outputs/auto_labels/
zip -r auto_labels_stage1.zip labels/ images/
```

또는 두 폴더를 그대로 Roboflow 에 업로드 (방식은 Roboflow 인터페이스 따라).

### 7.2 Roboflow 측 작업

1. 프로젝트 → **Upload** → "Import from existing labels"
2. Format: **YOLO** 선택 (또는 YOLOv8/YOLOv11)
3. zip 또는 폴더 업로드
4. Class mapping 자동 인식 — `IMMA.v1i.yolov11` 의 5클래스와 동일해야 함
5. **Pre-annotation 으로 import** (직접 라벨링 시작 X — 검수 모드)

### 7.3 검수 우선순위

```bash
# manifest 의 priority 별 파일명 추출
python3 -c "
import csv
with open('outputs/auto_labels/manifest.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

for prio in ['empty', 'low_conf', 'review', 'auto_pass']:
    files = [r['filename'] for r in rows if r['priority'] == prio]
    print(f'=== {prio}: {len(files)} files ===')
    for f in files[:5]:
        print(f'  {f}')
"
```

Roboflow 에서 `priority='empty'` 와 `'low_conf'` 도면을 먼저 검수.

## 8. 의존성

```
ultralytics>=8.3.0       # Stage 1 모델 추론
opencv-python>=4.10.0    # ultralytics 내부
numpy>=1.26.0
tqdm>=4.66.0
torch (CUDA 12.8)        # D-030
```

CLI `--help` 시점에 ultralytics 미설치도 OK (lazy import).

## 9. 관련 의사결정

- **D-024** Group-aware split — seed key 와 일치하는 stem 자동 제외
- **D-026** 가공/조립 분류 (sort_by_drawing_type 와 병행 가능, 의존성 없음)
- **D-028** Stage 1 5 클래스 (출력 txt 의 class_id = Roboflow 순서)
- **D-029** Roboflow→내부 매핑 — 본 스크립트는 출력에 매핑 적용 안 함 (Roboflow Import 호환성)
- **D-030** PyTorch cu128 (RTX 5080 Blackwell)

## 10. 흔한 FAIL + 조치

| 증상 | 원인 / 해결 |
|---|---|
| `Weights not found` | Stage 1 학습이 아직 안 됨. `python src/stage1_layout.py train ...` 먼저 |
| `ultralytics not installed` | `uv pip install ultralytics` |
| OOM during inference | `--imgsz 1024` 로 축소 (학습 시 1280 와 다르나 추론은 OK) |
| 모든 도면이 `empty` | 모델 일반화 실패 — Stage 1 재학습 또는 `--conf 0.1` 시도 |
| symlink 실패 (Windows) | `--copy-images` 사용 |
| 처리 매우 느림 (>10분 / 5,739장) | GPU 미사용 — `--device 0` 명시 |

## 11. 다음 단계

1. **실행** — `python src/auto_label_stage1.py` (~5분)
2. **결과 확인** — `outputs/auto_labels/manifest.csv` priority 분포
3. **Roboflow Pre-annotation Import** — zip 업로드 또는 API
4. **사람 검수** — priority 순 (empty → low_conf → review → auto_pass)
5. **Stage 1 전체 재학습** — `IMMA.v2i.yolov11/` (검수 후 v2 export) → Version B 학습 (`history.md` §B 추가 기록)

## 12. 차후 개선 항목

- [ ] CLI 옵션으로 `LOW_CONF_THRESHOLD` / `HIGH_CONF_THRESHOLD` 노출
- [ ] `--upload-to-roboflow` 옵션 (Roboflow Python SDK 사용 시)
- [ ] 도면별 시각화 PNG 생성 옵션 (`--save-vis`) — 검수 보조용
- [ ] Top-N 가장 불확실한 도면 강조 표시 (entropy 기반)
- [ ] D-026 manifest 와 join — 가공도면 우선 검수 옵션
