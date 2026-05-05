# Google Drive 백업 자산 가이드

> **팀 Google Drive (IMMA)**: https://drive.google.com/drive/u/0/folders/1YweZCGEe8JbrRBaMSlSS7WIIx-yk_r8M
>
> GitHub repo 에 포함되지 않은 모든 외부 자산의 위치 / 크기 / 복원 절차 박제.

## 1. 자산 개요

| 자산 | 크기 | 압축 풀기 후 위치 (`<repo>/`) | 우선순위 | 비고 |
|---|---|---|---|---|
| **dataset.tar.gz** | ~1.1 GB | `dataset/` | ★★★ 필수 | 원본 도면 5,839 장 (★ 저작권 보호 — 외부 공유 금지) |
| **dataset_excluded.tar.gz** | ~5 MB | `dataset_excluded/` | ★ 옵션 | exclude_groups.py 로 제외된 46장 (참고용) |
| **IMMA.v1i.yolov11.tar.gz** | ~13 MB | `IMMA.v1i.yolov11/` | ★★★ 필수 | Roboflow seed 100장 (Stage 1 V.A 학습) |
| **checkpoints.tar.gz** | ~7.5 GB | `checkpoints/` | ★★★ 필수 | Stage 1/2 학습 weights 일체 |
| **data_annotation.tar.gz** | ~50 MB | `data/annotation/` | ★★★ 필수 | Stage 2 라벨 txt + 이미지 + train/val.txt (★ 저작권 종속) |
| **articles.tar.gz** | ~231 MB | `articles/` | ★ 옵션 | 참고 논문 PDFs (라이선스 확인 후 활용) |

총 약 **9 GB**.

---

## 2. 자산별 상세 명세

### 2.1 `dataset.tar.gz` (~1.1 GB) — ★★★ 필수

| 항목 | 값 |
|---|---|
| 위치 | `<repo>/dataset/` |
| 파일 수 | 5,839 JPG (Roboflow export 형식) |
| 클래스 | Stage 1 학습용 원본 도면 (5 클래스 D-028 라벨링은 별도 자산) |
| **저작권** | ★★★ **외부 공유 금지** — 팀 내부 사용만 가능 |
| 다운로드 시 |  Google Drive 에서 `dataset.tar.gz` 다운 → `tar -xzf dataset.tar.gz` → `dataset/` 폴더 생성 확인 |

**관련 코드**:
- `src/sort_by_yolo_pmi.py` (3-tier 분류)
- `src/auto_label_stage1.py` (Stage 1 자동 라벨링)
- `src/extract_pmi_crops_v3.py` (Stage 2 PMI crop 추출)

---

### 2.2 `IMMA.v1i.yolov11.tar.gz` (~13 MB) — ★★★ 필수

| 항목 | 값 |
|---|---|
| 위치 | `<repo>/IMMA.v1i.yolov11/` |
| 구조 | `train/` (80장) + `valid/` (20장) — 모두 라벨 포함 |
| 클래스 | 5개 (Isometric / PMI / Table / Text / View, D-028 매핑 D-029) |
| 용도 | Stage 1 Version A seed 학습 — `checkpoints/yolo_det.pt` 의 학습 데이터 |

**관련 코드**:
- `src/stage1_layout.py train` (학습 명령)
- `configs/yolo_det.yaml` (path: `IMMA.v1i.yolov11/`)

---

### 2.3 `checkpoints.tar.gz` (~7.5 GB) — ★★★ 필수

| 항목 | 값 |
|---|---|
| 위치 | `<repo>/checkpoints/` |
| 주요 파일 | `yolo_det.pt` (Stage 1) + `yolo_obb_runs/yolo_obb_v3_kfold_{0..4}/` (Stage 2 5-fold) |

#### 2.3.1 `checkpoints/yolo_det.pt` (~14 MB)

- Stage 1 (YOLOv11-det) seed 학습 결과
- mAP@0.5 = 0.935 (Version A)
- 사용 위치: `pipeline.py` Stage 1 inference

#### 2.3.2 `checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_{0..4}/` (~7 GB)

각 fold 별 디렉터리:
```
yolo_obb_v3_kfold_0/
├── weights/
│   ├── best.pt                  ← ★ 5-Fold Ensemble 사용
│   ├── last.pt                  ← resume 용
│   ├── epoch20.pt
│   ├── epoch40.pt
│   ├── ... (save_period=20, 10개)
│   └── epoch250.pt
├── results.csv                  ← K-fold aggregate 입력
├── args.yaml
└── ... (matplotlib plots)
```

**5-fold 통계** (`outputs/kfold_summary.json` 참조):
- mean mAP@0.5 = 0.932 ± 0.062
- Best fold = 2 (mAP 0.978)

**관련 코드**:
- `src/ensemble_predict.py` (5 fold 동시 추론)
- `src/pipeline.py` (`use_ensemble=True` default)
- `src/aggregate_kfold_results.py`

---

### 2.4 `data_annotation.tar.gz` (~50 MB) — ★★★ 필수

| 항목 | 값 |
|---|---|
| 위치 | `<repo>/data/annotation/` |
| 구조 | `images/{train,valid}/` + `labels/{train,valid}/` + `labels_backup_pre_clip/` + `data.yaml` + `train.txt` + `val.txt` |
| 파일 수 | 569 frames × {jpg + txt} ~ 1138 파일 + backup |
| **저작권** | ★ 종속 — 라벨 파일명에 도면 ID 노출 (예: `CAD_Drawing385_jpg.rf.<hash>__PMI_005.txt`) |

#### 2.4.1 K-fold 분할 (`data/annotation_kfold/`, ~150 KB)

```
data/annotation_kfold/
├── fold_0/
│   ├── data.yaml
│   ├── train.txt        ← 절대 WSL 경로 (portable X)
│   └── val.txt
├── fold_1/ ... fold_4/
```

**주의**: 절대 경로 사용 → 다른 사용자 환경에서 그대로 쓰려면 경로 갱신 필요. 또는 `prepare_kfold_dataset.py` 재실행으로 자동 생성.

---

### 2.5 `articles.tar.gz` (~231 MB) — ★ 옵션

| 항목 | 값 |
|---|---|
| 위치 | `<repo>/articles/` |
| 내용 | Khan et al. 2025 등 참고 논문 PDFs |
| 라이선스 | **다양 — 사용 전 개별 확인 필요** |

GitHub IMMA 원격 main 의 `논문_kr/` 폴더에도 일부 한글 논문 PDFs 가 있습니다 (이미 push 됨).

---

### 2.6 `dataset_excluded.tar.gz` (~5 MB) — ★ 옵션

| 항목 | 값 |
|---|---|
| 위치 | `<repo>/dataset_excluded/` |
| 내용 | `exclude_groups.py` 가 분류해서 제외한 46장 + 라벨 (group leak 방지, D-024) |
| 용도 | 검수 / 디버깅 참고용 |

---

## 3. 다운로드 + 복원 절차

### 3.1 권장 도구

| 도구 | 설치 | 사용법 |
|---|---|---|
| **rclone** (★ 권장) | `apt install rclone` + `rclone config` (Google Drive remote 설정) | `rclone copy gdrive:IMMA ./` |
| **gdown** | `pip install gdown` | `gdown --folder <Folder ID>` |
| 수동 다운로드 | 브라우저로 Google Drive 접속 | zip 다운 → 압축 해제 |

### 3.2 표준 복원 흐름

```bash
# 1. git clone (코드 + 문서 + 박제 — 4.2 MB)
cd ~/your/work/path
git clone https://github.com/amadda0616-hash/IMMA.git Drawing
cd Drawing

# 2. Google Drive 에서 자산 다운로드
#    (rclone 예시 — 환경 따라 도구 선택)
rclone copy gdrive:IMMA/dataset.tar.gz ./
rclone copy gdrive:IMMA/IMMA.v1i.yolov11.tar.gz ./
rclone copy gdrive:IMMA/checkpoints.tar.gz ./
rclone copy gdrive:IMMA/data_annotation.tar.gz ./
# (옵션)
rclone copy gdrive:IMMA/articles.tar.gz ./
rclone copy gdrive:IMMA/dataset_excluded.tar.gz ./

# 3. 압축 해제
tar -xzf dataset.tar.gz
tar -xzf IMMA.v1i.yolov11.tar.gz
tar -xzf checkpoints.tar.gz
tar -xzf data_annotation.tar.gz
tar -xzf articles.tar.gz
tar -xzf dataset_excluded.tar.gz

# 4. 검증
ls -la dataset/ IMMA.v1i.yolov11/ checkpoints/
ls -la data/annotation/
echo "---"
ls checkpoints/yolo_obb_runs/  # 5 fold 폴더 모두 보여야 함

# 5. K-fold 절대 경로 갱신 (필요 시)
python src/prepare_kfold_dataset.py --root . --n-folds 5
# → data/annotation_kfold/ 재생성

# 6. Python 환경 구축 (별도)
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# 7. smoke test
python src/pipeline.py run \
    --image IMMA.v1i.yolov11/valid/images/<sample>.jpg \
    --out outputs/smoke_test.json \
    --skip-numerical --skip-alphabetical \
    --device cuda:0
```

### 3.3 복원 검증 체크리스트

| 검증 항목 | 명령 | 기대 결과 |
|---|---|---|
| dataset 5,839 장 | `ls dataset/*.jpg \| wc -l` | 5839 |
| IMMA seed 100 장 | `ls IMMA.v1i.yolov11/{train,valid}/images/*.jpg \| wc -l` | 100 |
| Stage 1 weights | `ls -la checkpoints/yolo_det.pt` | ~14 MB |
| Stage 2 5 folds | `ls checkpoints/yolo_obb_runs/ \| grep kfold \| wc -l` | 5 |
| 각 fold best.pt | `ls checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_*/weights/best.pt \| wc -l` | 5 |
| Stage 2 라벨 (train) | `ls data/annotation/labels/train/*.txt \| wc -l` | ~445 |
| Stage 2 라벨 (valid) | `ls data/annotation/labels/valid/*.txt \| wc -l` | ~110 |

---

## 4. 백업 업데이트 정책 (★ 향후)

| 시점 | 업데이트 자산 | 작업자 |
|---|---|---|
| Stage 1 V.B 재학습 후 | `checkpoints.tar.gz` (yolo_det.pt 갱신) | ML 담당 |
| Stage 3-N Donut Numerical fine-tune 후 | `checkpoints.tar.gz` (donut_numerical/ 추가) | ML 담당 |
| 추가 라벨링 후 | `data_annotation.tar.gz` | 라벨링 담당 |
| 신규 도면 수집 시 | `dataset.tar.gz` | 데이터 담당 |

압축 형식 통일: `tar.gz` (gzip), 또는 `tar.zst` (zstd, 더 빠른 압축).

---

## 5. 트러블슈팅

### Q1. `dataset.tar.gz` 압축 해제 후 `dataset/` 폴더가 비어있다

→ 권한 문제 가능. `tar -xzvf dataset.tar.gz --no-same-owner` 시도. 또는 `sudo chown -R $USER:$USER dataset/`.

### Q2. checkpoints 다운 후 `pipeline.py` 가 fold 미발견 에러

→ 압축 풀기 위치 확인. `<repo>/checkpoints/yolo_obb_runs/yolo_obb_v3_kfold_{0..4}/weights/best.pt` 5개 모두 존재해야 함.

### Q3. K-fold 절대 경로가 다른 사용자 환경에서 깨진다

→ `python src/prepare_kfold_dataset.py --root . --n-folds 5` 재실행. WSL2 환경에서 절대 경로가 자동 갱신됨.

### Q4. WSL2 외 환경 (네이티브 Linux) 에서 사용

→ `data/annotation_kfold/*/train.txt`, `val.txt` 의 `/mnt/c/...` 경로를 실제 path 로 sed 일괄 변환:
```bash
find data/annotation_kfold -name "*.txt" -exec sed -i 's|/mnt/c/Users/user/github/Drawing|/your/actual/path|g' {} \;
```

---

## 6. 관련 문서

- [`MANUAL.md §0.1`](../MANUAL.md) — 외부 자산 다운로드 가이드 (요약)
- [`README.md §5.0`](../README.md) — 환경 구축 + Google Drive 안내
- [`PROJECT_HANDOFF.md §11 D-041`](../PROJECT_HANDOFF.md) — Git 워크플로 박제
- [`docs/PHASE15_CHECKLIST.md`](./PHASE15_CHECKLIST.md) — 다음 단계
- [`history.md §A.11.14.5`](../history.md) — Google Drive 백업 박제

---

**Last updated**: 2026-05-04 (Phase 14 완료, GitHub IMMA 첫 push)
