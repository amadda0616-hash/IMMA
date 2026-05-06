# 작업 매뉴얼 — Multi-Stage Hybrid Framework for Engineering Drawings

> 본 매뉴얼은 **처음부터 끝까지** 따라할 수 있는 단계별 작업 가이드입니다.
> 사양·아키텍처는 [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md), 빠른 참조는 [`README.md`](./README.md).

---

## 0. 사전 요구사항

| 항목 | 값 |
|---|---|
| OS | Windows 11 + **WSL2 (Ubuntu 22.04 LTS)** |
| GPU | NVIDIA **RTX 5080** (Blackwell, 16GB VRAM) |
| Driver | NVIDIA Windows driver 555+ |
| CUDA | 12.4+ (WSL2 자동 인식) |
| Python | 3.10+ |
| 데이터셋 | `dataset/` 폴더에 **5,839 JPG** 적재 완료 (Roboflow export 형식) |
| 도면 언어 | **EN / KO / JP / RU / CN / DE** (도면 1장 = 단일 언어 가정, D-025 — DE 2026-05-04 추가) |
| 도면 종류 | **가공도면 + 조립도면** 혼재 — Stage 2 학습은 가공도면만 (D-026) |
| TB 분포 | **약 95% 도면에 TB 존재**. material/quantity 핵심 필드는 자주 누락 (D-027) |
| IDE | Antigravity (VS Code 기반, WSL Remote) 권장 |

WSL2 안에서 `nvidia-smi` 가 RTX 5080 을 인식하면 GPU 패스스루는 정상입니다.

### 0.1 ★ 외부 자산 다운로드 (Google Drive, 팀 공유)

> 저작권 / 용량 문제로 GitHub repo 에는 코드 + 문서만 포함. 다음 자산은 **팀 Google Drive** 에서 별도 다운로드:

**📁 [팀 Google Drive (IMMA)](https://drive.google.com/drive/u/0/folders/1YweZCGEe8JbrRBaMSlSS7WIIx-yk_r8M)**

| 자산 | 압축 풀기 후 위치 | 크기 | 비고 |
|---|---|---|---|
| `dataset/` | `<repo>/dataset/` | ~1.1 GB | **★ 저작권 보호 — 외부 공유 금지**, 5,839 장 |
| `IMMA.v1i.yolov11/` | `<repo>/IMMA.v1i.yolov11/` | ~13 MB | Stage 1 seed 학습 100장 (Roboflow export) |
| `checkpoints/` | `<repo>/checkpoints/` | ~7.5 GB | yolo_det.pt + yolo_obb_runs/yolo_obb_v3_kfold_{0..4}/ |
| `articles/` | `<repo>/articles/` | ~231 MB | 논문 PDFs (라이선스 확인 후 활용) |

```bash
# git clone 후 다운로드 절차
cd ~/path/to/work
git clone https://github.com/amadda0616-hash/IMMA.git Drawing
cd Drawing

# (다운로드 도구 — 환경 따라 선택)
#   - rclone: rclone copy gdrive:IMMA ./
#   - gdown:  gdown --folder <Google Drive folder ID>
#   - 수동:    Google Drive 웹에서 zip 다운 후 압축 해제

# 검증
ls -la dataset/ IMMA.v1i.yolov11/ checkpoints/
```

> **중요**: `dataset/` 의 도면은 저작권 보호 자료입니다. **외부 공유 금지** — 팀 내부 사용만 가능.

---

## 0-1. ★ 현재 진행 위치 & 다음 액션 (2026-04-29 갱신)

### Phase A — 환경 + 코드 + Stage 1 학습 ✅ 완료

| # | 단계 | 상태 |
|---|---|---|
| 1 | 환경 세팅 (§1) — PyTorch cu128 (D-030 Blackwell sm_120) | ✅ |
| 2 | 데이터셋 적재 (5,839 JPG, `dataset/`) | ✅ |
| 4 | Roboflow seed **100장 라벨링** (`IMMA.v1i.yolov11/`, 5클래스 D-028) | ✅ |
| 5 | Stage 1 **Version A 학습** (28.5분, mAP@0.5 = **0.9364**) | ✅ |
| 5.5 | `auto_label_stage1.py` 작성 + 5,839장 자동 라벨링 (5분 45초) | ✅ |
| V2-A | 라벨 품질 검증 (PASS) | ✅ |
| V2-B | 모델 성능 검증 (mAP 0.9364 PASS, PMI/Text 만 미달) | ✅ |

### Phase B — 데이터 전처리 ✅ 완료 (2026-04-29)

| # | 단계 | 상태 |
|---|---|---|
| 1.6 (구) | `sort_by_drawing_type.py` 휴리스틱 — 5,839장 4h30m 결과 mfg=0/asm=5313 비현실적 → **폐기** | 🔴 |
| 1.6 (신) | **`sort_by_yolo_pmi.py`** 작성 + 실행 (~3분) → mfg 5,349 / asm 441 / review 49 | ✅ |
| 검증 | `manufacturing/` random 100장 sample → 조립 0% / 부품 10~20% / 가공 80~90% | ✅ |
| 사용자 검수 | assembly + manual_review 시각 검수 → **18 group_keys** 식별 | ✅ |
| 정리 | `exclude_groups.py` 실행 (~9초) → 46 files 이동 (D-024 group 단위) | ✅ |
| 결과 | `dataset/` **5,793장** (학습 잔여) / `dataset_excluded/` 46장 (보관) | ✅ |
| 라이선스 | 5,839장 로컬 only 사용 = 개인 학습용 안전 (D-035) | ✅ |
| Pre-annotation | 보류 (비용 + 시간) — Version A 그대로 사용 (D-035) | 🟡 차후 |

### Phase C — Day 1~3 Stage 2 ~ Step 8 ⏳ 진행 예정

| # | 일자 | 작업 | 상태 |
|---|---|---|---|
| Day 1 | 2026-04-30 | Stage 2 PMI crop 라벨링 (CVAT 로컬, **500 crops**) | **★ 다음** |
| Day 2 | 2026-05-01 | Stage 2 학습 + Stage 3-A | ⬜ |
| Day 3 | 2026-05-02 | Stage 3-N fine-tune + Step 7 + Step 8 | ⬜ |
| 차후 | — | Step 9 enrichment + Pre-annotation Phase 2 | ⬜ |

자세한 흐름은 본 문서 §16 (TL;DR) 참조 / 학습 이력은 [`history.md`](./history.md) §A.

---

## 1. 환경 설정

### 1.1 시스템 패키지 (apt)

```bash
sudo apt update
sudo apt install -y \
    python3-venv python3-pip curl \
    tesseract-ocr \
    tesseract-ocr-eng tesseract-ocr-kor tesseract-ocr-jpn tesseract-ocr-rus \
    tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
    libgl1 libglib2.0-0     # OpenCV 런타임 의존성

# 5개 언어팩 검증 (D-025)
tesseract --list-langs
# 기대: eng, jpn, kor, osd, rus, chi_sim, chi_tra, ...
```

### 1.2 `uv` 패키지 매니저 설치

`uv` 는 Rust 로 작성되어 `pip` / `conda` 대비 압도적으로 빠른 의존성 해결·설치 속도를 제공합니다.

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 셸 환경 변수 적용
source $HOME/.bashrc

# 설치 확인
uv --version
```

### 1.3 가상환경 생성 및 활성화

프로젝트 루트(`/mnt/c/Users/user/github/Drawing`)에서 작업합니다.

```bash
cd /mnt/c/Users/user/github/Drawing

# uv 로 Python 3.10+ 가상환경 생성 (.venv)
uv venv

# 가상환경 활성화
source .venv/bin/activate
```

### 1.4 PyTorch 설치 (CUDA 12.8 전용, ★ D-030)

> **★ RTX 50 시리즈 호환성 (Blackwell, sm_120)**: PyTorch **cu124 빌드는 sm_90 (Hopper) 까지만 지원**하여 RTX 5080 비호환. 반드시 **cu128 빌드** 사용.

Blackwell 아키텍처(RTX 5080) 지원을 위해 `requirements.txt` 설치 전에 CUDA 12.8 빌드를 명시적으로 먼저 설치해야 합니다.

```bash
uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
```

**Stable cu128 미출시 시 nightly 사용**:

```bash
uv pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

**호환성 검증 (필수)**:

```bash
python -c "import torch; print('compute_capability:', torch.cuda.get_device_capability())"
# 기대 출력: compute_capability: (12, 0)   ← sm_120 = Blackwell
```

`(12, 0)` 이 출력되지 않으면 GPU kernel 호출 시 런타임 실패하므로 학습 진행 불가.

### 1.5 프로젝트 의존성 설치

```bash
uv pip install -r requirements.txt
```

`uv` 는 병렬 처리로 수 초 내에 완료됩니다.

### 1.6 양자화 / Blackwell 성능 극대화 라이브러리

16GB VRAM 에서 Donut VLM 의 메모리 효율을 위해 추가 설치:

```bash
# 1) bitsandbytes — 8-bit / 4-bit 양자화
uv pip install bitsandbytes

# 2) flash-attn — Attention 연산 속도·메모리 극대화
uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation

# 3) optimum (선택) — HuggingFace 모델 하드웨어 가속
uv pip install optimum
```

### 1.7 설치 검증

다음 스크립트를 실행해 모든 가속 기능이 정상 동작하는지 확인합니다.

```python
# verify_env.py
import torch

# 1. CUDA 및 디바이스 확인
print(f"CUDA Available : {torch.cuda.is_available()}")
print(f"Device Name    : {torch.cuda.get_device_name(0)}")
print(f"CUDA Version   : {torch.version.cuda}")

# 2. Blackwell bfloat16 지원
print(f"BFloat16       : {torch.cuda.is_bf16_supported()}")

# 3. flash-attn
try:
    import flash_attn
    print(f"FlashAttn-2    : {flash_attn.__version__}")
except ImportError:
    print("FlashAttention-2 NOT installed.")

# 4. bitsandbytes
try:
    import bitsandbytes  # noqa: F401
    print("bitsandbytes   : OK")
except ImportError:
    print("bitsandbytes NOT installed.")
```

```bash
python verify_env.py
```

기대 출력:
```
CUDA Available : True
Device Name    : NVIDIA GeForce RTX 5080
CUDA Version   : 12.4
BFloat16       : True
FlashAttn-2    : 2.x.x
bitsandbytes   : OK
```

> 💡 **Stage 3-N fine-tuning 팁**: 학습 시 `torch.bfloat16` + bitsandbytes 8-bit 로딩 을 활용하면 RTX 5080 의 16GB VRAM 안에서 더 큰 배치 사이즈 확보 가능.

---

## 2. 데이터셋 확인

### 2.1 현재 상태

```bash
# 5,839 JPG 파일 적재 검증
find dataset -maxdepth 1 -type f -iname "*.jpg" | wc -l
# 기대: 5839
```

### 2.2 파일명 형식 (Roboflow export)

```
{original_stem}.rf.{augmentation_hash}.jpg
```

예시:
- `11_jpeg.rf.8b46c563d114f0dffc24fa4f4fe0f14e.jpg`
- `11_jpeg.rf.de99e140e1ba3ae014968fb7c209d5fb.jpg`
  → 두 파일은 **동일 원본** `11_jpeg` 의 두 증강 변형

**Group key 추출 규칙** (D-024):

```python
group_key = filename.split('.rf.')[0]
```

### 2.3 데이터 누수 방지 (★ 중요)

같은 group key 의 모든 변형이 **train 또는 val 한쪽에만** 들어가야 합니다. 그렇지 않으면 train 의 증강본이 val 에 섞여 가짜 성능이 나옵니다.

`sklearn.model_selection.GroupShuffleSplit` 사용:

```python
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

filenames = sorted(p.name for p in Path("dataset").glob("*.jpg"))
groups = [fn.split(".rf.")[0] for fn in filenames]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(filenames, groups=groups))

train_files = [filenames[i] for i in train_idx]
val_files   = [filenames[i] for i in val_idx]
```

이 방식은 **§3 라벨링 export 후** train/val 폴더에 파일을 분배할 때 사용합니다.

---

## 3. (선택) Step 1.5 — TitleBlock 분류기 실행

> **D-019:** 학습 흐름의 필수 단계가 아닙니다. **데이터 품질 점검 / 디버깅** 용도.
> **2026-04-28 라벨링 시작 후 발견 (D-027)**: 실제 데이터셋의 TB 분포가 **약 95:5** (TB 있음 압도). 따라서 sort_by_titleblock 의 sampling 가치는 낮음. **가공/조립 분류** (D-026) 가 더 중요한 분류 기준 — 별도 도구 `src/sort_by_drawing_type.py` 가 차후 작성 예정.

```bash
# 매니페스트만 미리 확인 (이동 X)
python src/sort_by_titleblock.py --dryrun

# 실행 결과 확인 후 실제 이동
python src/sort_by_titleblock.py
```

산출물:
- `data/stage1_titleblock/` — TB 키워드 ≥ 2 검출
- `data/stage2_no_titleblock/` — 키워드 0 + 라인 밀도 낮음
- `data/manual_review/` — 애매
- `outputs/sort_titleblock_manifest.csv` — 매니페스트 (UTF-8-SIG, Excel 친화)

### 3.1 분류기 결과 검증 (V1)

50장 정도 사람이 직접 TB 유무 판정해서 ground truth CSV 작성:

```csv
filename,has_titleblock_actual
drawing_001.jpg,1
drawing_002.jpg,0
...
```

`data/validation_gt/step1_5_titleblock_gt.csv` 로 저장 후:

```bash
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv \
    --gt data/validation_gt/step1_5_titleblock_gt.csv
```

리포트 확인: `reports/<date>_step1.5_sorter.html`

**임계값 (configs/validation_thresholds.yaml):**
- `classifier_accuracy ≥ 0.85` (critical)
- `per_language_min ≥ 0.80` (warning)

---

## 3.5 Step 1.6 — 가공/조립 도면 분류 (D-026)

> **🔴 2026-04-29 갱신**: 휴리스틱 분류기 (`sort_by_drawing_type.py`) 폐기. **`sort_by_yolo_pmi.py`** 로 대체.

### 3.5.1 휴리스틱 (sort_by_drawing_type.py) — 폐기

5,839장 4시간 30분 실행 → **mfg=0 / asm=5,313 / review=526** 비현실적 결과.
- 원인: OCR 치수 검출 실패 + BOM Hough Lines false positive
- 격리: Stage 2 이후 OCR 미사용으로 안전 (D-026 갱신 주석 참조)

### 3.5.2 ★ 신규 (sort_by_yolo_pmi.py) — 권장

**전제 조건**: Stage 1 Version A 학습 + auto_label_stage1.py 실행 완료.

```bash
# Stage 1 Version A 자동 라벨 기반 PMI 카운트 분류 (~3분)
python src/sort_by_yolo_pmi.py
```

**분류 규칙**:
```
PMI ≥ 5                            → manufacturing (가공도면)
PMI < 5 AND (Iso ≥ 1 OR Table ≥ 3) → assembly (★ 사람 검수 후보)
PMI < 5 AND signal 없음            → manual_review_type (★ 검수)
```

**산출물** (`outputs/sort_by_yolo_pmi/`):
- `manifest.csv` — per-class counts + decision (5 클래스 × 5,839 도면)
- `manufacturing/` — symlinks (~5,349장, 디스크 절약)
- `assembly/` — copies (~441장 → 검수 후 조립도면만 잔여)
- `manual_review_type/` — copies (~49장 → 검수)
- `README.md` — 검수 가이드 (자동 생성)

**WSL2 호환성**: 검수 폴더는 자동으로 copy 모드 (Windows Explorer 썸네일 호환).

상세: [`docs/modules/sort_by_yolo_pmi.md`](./docs/modules/sort_by_yolo_pmi.md) (작성 예정)

## 3.6 Step 1.7 — 사용자 시각 검수 + 학습 데이터 정리 (★ 신규 워크플로)

### 3.6.1 검수 폴더 시각 확인 (~40~50분)

```bash
# Windows Explorer 큰 아이콘으로 검수
explorer.exe "outputs\sort_by_yolo_pmi\assembly"
explorer.exe "outputs\sort_by_yolo_pmi\manual_review_type"
```

**검수 룰**:
| 사용자가 본 도면 | 처리 |
|---|---|
| 🔴 명백한 조립도면 (BOM + 부품번호 풍선 + 분해도) | 폴더에 남김 (제외 대상) |
| 🔴 학습 부적합 (잘림/손상/비-도면) | 폴더에 남김 (제외 대상) |
| 🟢 가공도면 / 부품도면 / 회전 증강 / 부분 확대 | 폴더에서 삭제 (학습 유지) |

> ⚠️ `dataset/` 의 원본 파일은 직접 삭제 X. 검수 폴더 (copy) 만 삭제.

### 3.6.2 자동 group_key 추출 (~1초)

검수 완료 후:

```bash
python3 << 'EOF'
from pathlib import Path
import datetime

groups = set()
for folder in ["assembly", "manual_review_type"]:
    p = Path(f"outputs/sort_by_yolo_pmi/{folder}")
    if not p.exists():
        continue
    for img in p.iterdir():
        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg"}:
            groups.add(img.stem.split(".rf.")[0])

out = Path("outputs/exclude_list.txt")
header = f"# 조립도면 + 학습부적합 group_key (사용자 검수)\n# 생성: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n"
out.write_text(header + "\n".join(sorted(groups)) + "\n", encoding="utf-8")
print(f"✓ {len(groups)} unique group_keys → {out}")
EOF
```

### 3.6.3 자동 group 단위 일괄 제외 (~5~30초)

```bash
# 미리보기
python src/exclude_groups.py --list outputs/exclude_list.txt --dryrun

# 실제 실행
python src/exclude_groups.py --list outputs/exclude_list.txt
```

**작업**:
- D-024 정합성 — 같은 `group_key` 의 모든 `.rf.<hash>` 변형 함께 이동
- `dataset/` → `dataset_excluded/`
- `outputs/auto_labels/labels/` → `labels_excluded/` (동기)

**산출물**:
- `dataset_excluded/` (보관 — 삭제 X)
- `outputs/exclude_list.txt`
- `outputs/exclude_groups_manifest.csv`

### 3.6.4 검증

```bash
ls dataset/ | wc -l          # 학습 잔여
ls dataset_excluded/ | wc -l # 제외된 도면

# D-024 정합성 검증
python3 -c "
from pathlib import Path
ds = set(p.stem.split('.rf.')[0] for p in Path('dataset').glob('*.jpg'))
ex = set(p.stem.split('.rf.')[0] for p in Path('dataset_excluded').glob('*.jpg'))
print(f'dataset/ groups: {len(ds)}, excluded/ groups: {len(ex)}, overlap: {len(ds&ex)}')
"
# 기대: overlap = 0
```

### 3.6.5 실측 결과 (2026-04-29 사용자 작업)

| 항목 | 값 |
|---|---|
| 사용자 식별 조립도면 group | **18** |
| 제외 image (.rf.<hash> 변형 포함) | 46 |
| 제외 label (동기) | 46 |
| **dataset/ 학습 잔여** | **5,793 images / 2,991 unique groups** |
| `dataset_excluded/` (보관) | 46 / 18 |
| D-024 group 정합성 | overlap 0 ✅ |
| Roboflow 사전 증강 비율 | ~1.94× / group |

상세: [`history.md`](./history.md) §A.6.7~A.6.9

---

## 4. Stage 1 — Layout Detection (Isometric / PMI / Table=TB / Text=Notes / View, D-028)

> **★ 라벨링 전 주의 (D-026)**: Roboflow 데이터셋에 **가공도면 + 조립도면 혼재**.
> Stage 1 라벨링 자체는 둘 다 가능하지만, **Stage 2 학습 데이터는 가공도면만** 사용해야 함.
> **Step 1.6 (`sort_by_drawing_type.py`)** 으로 자동 분류 후 `data/manufacturing/` 위주로 진행 권장.
> 또는 Roboflow **Tag** 기능으로 100장 seed 에 `manufacturing` / `assembly` 수동 분류 (10초/장 × 100 = 17분).

> **현재 상태 (2026-04-28)**: Roboflow seed **100장 라벨링 완료** (`IMMA.v1i.yolov11/train/`).
> 5 클래스 체계 (`Isometric/PMI/Table/Text/View`) 채택 — D-028 박제.

### 4.1 라벨링 (Roboflow 또는 CVAT) — **5 클래스 (D-028)**

**클래스 정의 (axis-aligned BBox, Roboflow data.yaml 순서)**

| ID | Roboflow 라벨 | 내부 (D-029) | 설명 |
|---|---|---|---|
| 0 | `Isometric` | Isometric | 3D 등각투영도. Stage 2 OBB skip (치수 없음) |
| 1 | `PMI` | PMI | **★ Product Manufacturing Information** — Stage 2 OBB 의 입력 영역 |
| 2 | `Table` | **TitleBlock** | 정보 블록 (도번/제목/재질/척도/날짜 등 표). 없는 도면은 skip |
| 3 | `Text` | **Notes** | 일반 노트, 주석 영역. 없는 도면은 skip |
| 4 | `View` | View | 2D 뷰 영역 (정면/평면/단면/상세) |

> **D-029 매핑**: 데이터셋의 `Table` / `Text` 는 코드 내부에서 `TitleBlock` / `Notes` 로 자동 매핑됨 (`src/stage1_layout.ROBOFLOW_TO_INTERNAL`). 라벨링 시에는 **Roboflow 이름 그대로 사용**.

**Roboflow 사용 시 (권장)**

1. <https://roboflow.com> 접속 → 새 프로젝트 (Object Detection, axis-aligned)
2. Classes 추가: `Isometric`, `PMI`, `Table`, `Text`, `View` (정확히 이 이름, 5종)
3. `dataset/` 의 5,839 JPG 업로드
4. 라벨링 (각 도면에 영역별 BBox 그리기)
5. **Generate New Version** — 추가 augmentation 없음 (이미 사전 증강됨, D-024)
6. **Export** → format **YOLOv11** 또는 **YOLOv8** → ZIP 다운로드
7. ZIP 내부 `train/` `valid/` 폴더 구조 확인

**CVAT 사용 시 (로컬, 무료)**

```bash
# 별도 디렉터리에서
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
# → http://localhost:8080
```

Project 생성 → Labels: `Isometric / PMI / Table / Text / View` (5종) → Task 생성, dataset/ 업로드 → 라벨링 → Export "YOLO 1.1" format.

### 4.2 Group-aware Train/Val Split (★ D-024)

Roboflow export 의 기본 random split 은 데이터 누수 위험. **수동으로 group-aware split 재구성** 권장.

```python
# split_groupwise.py
from pathlib import Path
import shutil
from sklearn.model_selection import GroupShuffleSplit

SRC = Path("path/to/roboflow_export")  # 라벨링 결과 폴더
DST = Path("data/layout")

# 모든 라벨 파일 (정상 라벨이 있는 이미지만)
labels = sorted((SRC / "train" / "labels").glob("*.txt")) + \
         sorted((SRC / "valid" / "labels").glob("*.txt"))

filenames = [p.stem for p in labels]
groups    = [fn.split(".rf.")[0] for fn in filenames]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(filenames, groups=groups))

for split, idx in [("train", train_idx), ("val", val_idx)]:
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in idx:
        stem = filenames[i]
        # 원본에서 train/valid 어디에 있는지 모르므로 양쪽 다 시도
        for sub in ("train", "valid"):
            img = SRC / sub / "images" / f"{stem}.jpg"
            lbl = SRC / sub / "labels" / f"{stem}.txt"
            if img.exists():
                shutil.copy2(img, DST / "images" / split / f"{stem}.jpg")
                shutil.copy2(lbl, DST / "labels" / split / f"{stem}.txt")
                break

print(f"train: {len(train_idx)}, val: {len(val_idx)}")
```

```bash
python split_groupwise.py
```

### 4.3 라벨 품질 검증 (V2-A)

```bash
python -m src.validate.check_labels_yolo \
    --labels-dir data/layout/labels/train \
    --images-dir data/layout/images/train \
    --cfg configs/yolo_det.yaml
```

**critical 임계값 (configs/validation_thresholds.yaml):**
- `empty_label_rate < 0.05` — 빈 라벨 5% 미만
- `parse_error_rate = 0` — 형식 오류 없음
- `bbox_validity_rate = 1.0` — 좌표 [0,1] 100% 유효

리포트: `reports/layout_train_yolo_labels.html` (시각 그리드 + 차트 포함)

**FAIL 발생 시**:
- 빈 라벨 많음 → 라벨링 누락 도면 다시 작업
- 클래스 분포 비대칭 (TB 30% 미달) → TB 있는 도면을 의식적으로 더 라벨링

### 4.4 학습 — **Active Learning 2단계** (D-028 갱신)

#### 4.4.1 ★ 1단계: Stage 1 빠른 학습 (seed 100장)

> **현재 상태**: Roboflow seed 100장 라벨링 완료 (`IMMA.v1i.yolov11/train/`).
> 이 100장으로 빠른 학습 → 나머지 5,739장 자동 라벨링에 사용.

```bash
# Roboflow export (IMMA.v1i.yolov11) 를 yolo_det.yaml 의 path 로 사용
python src/stage1_layout.py train \
    --data configs/yolo_det.yaml \
    --model yolo11m.pt \
    --epochs 50 --imgsz 1280 --batch 8 \
    --device 0 \
    --name yolo_det_seed
```

학습 시간: RTX 5080 16GB 기준 약 **20~30분** (100장 × 50 epoch).

체크포인트:
- 학습 결과: `checkpoints/yolo_det_runs/yolo_det_seed/`
- 자동 복사: `checkpoints/yolo_det.pt` (best.pt) — 임시. 1단계 완료 후 자동 라벨링용으로 사용.

> **목표**: 100장으로 mAP@0.5 ≥ **0.60** 만 도달하면 충분 (자동 라벨링 시드용).
> 정확도 부족 시: epochs 100 / imgsz 1536 / yolo11l-seg 시도.

#### 4.4.2 자동 라벨링 (5,739장)

`auto_label_stage1.py` (별도 작성) 가 `predict_one()` 을 일괄 호출해 YOLO txt 형식으로 저장.
대안: Roboflow 의 **Auto Label** 기능 사용 (모델 업로드 후 도면 100장 단위로 호출).

#### 4.4.3 사람 검수 (~16시간)

Roboflow **Pre-annotation Import** 으로 자동 라벨 결과를 불러온 뒤 검수.
**Active Learning 우선순위**: confidence 낮은 도면부터 (모델이 헷갈려 하는 부분).

#### 4.4.4 ★ 2단계: 전체 재학습 (5,839장)

```bash
python src/stage1_layout.py train \
    --data configs/yolo_det.yaml \
    --model yolo11m.pt \
    --epochs 100 --imgsz 1280 --batch 8 \
    --device 0 \
    --name yolo_det_full
```

학습 시간: RTX 5080 16GB 기준 약 **4~6시간** (이미지 5,839장 × 80% × 100 epoch).

체크포인트:
- 학습 결과: `checkpoints/yolo_det_runs/yolo_det_full/`
- 자동 복사: `checkpoints/yolo_det.pt` (best.pt) — 최종 모델

### 4.5 모델 성능 검증 (V2-B)

```bash
python -m src.validate.check_stage1_model \
    --weights checkpoints/yolo_det.pt \
    --data configs/yolo_det.yaml \
    --device 0
```

**critical 임계값 (논문 §4.1)**:
- `mAP@0.5 ≥ 0.85`
- per-class accuracy: View ≥ 0.90 / PMI ≥ 0.85 / TitleBlock ≥ 0.95 / Notes ≥ 0.90 / Isometric ≥ 0.80 (5클래스 D-028)

리포트: `reports/<date>_stage1_model.html` (Confusion matrix + per-class chart)

**FAIL 발생 시 조치**:
- 특정 클래스만 낮음 → 해당 클래스 라벨 재검수, 추가 라벨링
- 전반 미달 → epochs ↑, imgsz ↑ (1536), 또는 yolo11l 시도

### 4.6 추론 + 영역 자동 crop (Stage 2 / Stage 3-A 입력 준비)

```bash
# 단일 도면 추론
python src/stage1_layout.py predict \
    --image dataset/sample.jpg \
    --weights checkpoints/yolo_det.pt
# → outputs/sample.det.json

# 자동 crop
python src/stage1_layout.py crop \
    --image dataset/sample.jpg \
    --weights checkpoints/yolo_det.pt \
    --padding 5
# → outputs/crops/sample/
#       View/sample__View_00.jpg, sample__View_01.jpg, ...
#       TitleBlock/sample__TitleBlock_00.jpg, ...
#       Notes/sample__Notes_00.jpg, ...
#       manifest.json
```

**5,839 도면 일괄 처리**:

```bash
# 일괄 추론 스크립트 (batch_stage1_crop.py 작성 필요 — 간단한 for loop)
for img in dataset/*.jpg; do
  python src/stage1_layout.py crop --image "$img" --weights checkpoints/yolo_det.pt
done
```

또는 `python -c` 로 import 후 `predict_one()` + `crop_regions()` 직접 호출 (더 빠름, 모델 1회 로드).

### 4.7 ★ PMI crop 추출 (D-037 adaptive padding, Stage 2 입력 준비)

> **신규 (2026-04-30)**: Stage 1 Version A 모델로 PMI 영역 자동 crop 추출. **v2 (per-axis adaptive) + v3 (aspect-aware)** 두 버전 지원. D-037 의사결정에 따라 padding 방식 선택.

**배경 (D-037)**:
- v1 (fixed 10px): 화살표 잘림 + 큰 padding 시 인접 치수 침입
- v2 (per-axis adaptive): `pad_x = bbox_w × 0.4` / `pad_y = bbox_h × 0.4` → 비회전 90% / 회전 80%
- v3 (aspect-aware): 정사각형 bbox (aspect<1.5) 에 uniform pad 0.6 / 비정사각형은 per-axis → 회전 약 90% 예상

#### 4.7.1 v2 실행 (per-axis padding)

```bash
# 기본 (20도면 → 844 PMI crops, ~47초)
python src/extract_pmi_crops.py

# 파라미터 튜닝
python src/extract_pmi_crops.py \
    --padding-ratio 0.4 \
    --padding-min 30 \
    --padding-max 80
```

**산출물**:
```
outputs/cvat_stage2_input_v2/
├── DwgFoo__PMI_000.jpg
├── DwgFoo__PMI_001.jpg
├── ...  (844 crops)
└── manifest.csv  (crop_filename, bbox, padding 정보 포함)
```

**manifest 통계 확인**:
```bash
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("outputs/cvat_stage2_input_v2/manifest.csv")
print(df[["pad_x", "pad_y"]].describe())
print(f"Crop shapes: {df['crop_w'].min()}-{df['crop_w'].max()} × {df['crop_h'].min()}-{df['crop_h'].max()}")
EOF
```

#### 4.7.2 v3 실행 (aspect-aware padding, 권장)

```bash
# 기본 (aspect-aware, threshold 1.5, ratio_square 0.6)
python src/extract_pmi_crops_v3.py

# aspect threshold 조정 (더 엄격하게)
python src/extract_pmi_crops_v3.py --aspect-threshold 1.3
```

**산출물**:
```
outputs/cvat_stage2_input_v3/
├── DwgFoo__PMI_000.jpg
├── ...  (844 crops)
└── manifest.csv  (+ aspect_ratio, padding_strategy 컬럼)
```

**manifest 확인 (strategy 분포)**:
```bash
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("outputs/cvat_stage2_input_v3/manifest.csv")
print("Strategy 분포:")
print(df["padding_strategy"].value_counts())
print(f"\nSquare diagonal padding (평균): {df[df['padding_strategy']=='square_diagonal']['pad_x'].mean():.1f} px")
print(f"Per-axis padding (평균): {df[df['padding_strategy']=='per_axis']['pad_x'].mean():.1f} px")
EOF
```

#### 4.7.3 v2 vs v3 비교

| 항목 | v2 (per-axis) | v3 (aspect-aware) |
|---|---|---|
| 파일 | `extract_pmi_crops.py` | `extract_pmi_crops_v3.py` |
| 출력 폴더 | `outputs/cvat_stage2_input_v2/` | `outputs/cvat_stage2_input_v3/` |
| 정사각형 처리 | per-axis (축 정렬) | uniform 큰 pad (45° 회전 보강) |
| 회전 텍스트 만족도 | 80% | 약 90% 예상 |
| manifest 컬럼 | pad_x, pad_y, padding_mode | + aspect_ratio, padding_strategy |

> 💡 **권장**: 두 버전 모두 실행 후 manifest 비교. v3 가 정사각형 bbox 를 더 잘 처리 → v3 선택 권장.

---

## 5. Stage 2 — Annotation Localization (Measure / GDT / Roughness, OBB)

> **★ 입력은 PMI crop** (D-028): Stage 1 의 5 클래스 중 `PMI` 영역만 Stage 2 의 입력으로 사용. (이전엔 View crop 사용 — 새 클래스 체계에서는 PMI 가 정확한 입력)
>
> **★ 가공도면만 라벨링 (D-026)**: 조립도면 PMI 는 거의 없어 Stage 2 학습 데이터로 무용. Roboflow Tag `manufacturing` 인 도면의 PMI crop 만 CVAT 에 업로드.

### 5.1 입력: View crop 만 사용

Stage 2 는 **Stage 1 이 잘라낸 View crop 위에서만** 작동합니다 (논문 §4.2 — TitleBlock/Notes 노이즈 제거 효과).

```
data/annotation/images/train/
├── sample__View_00.jpg
├── sample__View_01.jpg
└── ...
```

### 5.2 라벨링 (OBB)

**클래스 정의 (Oriented BBox, 8-point 회전 사각형)**

| ID | 클래스 | 설명 |
|---|---|---|
| 0 | `Measure` | 치수 (Ø, R, M, 일반 dimension, 공차 포함) |
| 1 | `GDT` | Feature Control Frame 통째 |
| 2 | `Roughness` | 표면거칠기 심볼 |

**도구 권장**: CVAT 또는 Roboflow (둘 다 OBB 지원). labelImg ❌ (OBB 미지원).

**Roboflow 사용 시**:
- Project type: **Instance Segmentation** 또는 **Keypoint** 선택 후 OBB 모드. 또는
- Project type: **Object Detection** 으로 만들고 export 시 YOLO **OBB** 포맷으로 변환

**라벨링 가이드**:
- Measure: 숫자 + 화살표 영역 통째로. 공차 표기 포함.
- GDT: 프레임 전체 (심볼 + 공차값 + Datum).
- Roughness: 체크마크/꺽쇠 + 수치 통째로.

### 5.2.1 ★ SKIP 라벨 활용 (D-038, 2026-05-01 추가)

Stage 2 라벨링 중 다음 케이스는 **Measure/GDT/Roughness 가 아니므로** SKIP 라벨로 처리:

**SKIP 케이스 분류 (CVAT 라벨에 4번째 라벨 `SKIP` 추가, reason attribute 9개)**:

| reason | 의미 | 예시 |
|---|---|---|
| `unreadable` | 가독성 한계 | 픽셀 깨짐, 그래픽 노이즈 |
| `stage1_fp_section` | 단면도 기호 | A-A', SECTION A |
| `stage1_fp_detail` | 상세도 기호 | DETAIL X |
| `stage1_fp_projection` | 제3각법 기호 | ⊕ (두 원뿔 투영) |
| `stage1_fp_table` | 표제란/도장/BOM | 도번, 재질, 회사 도장 |
| **`stage1_fp_notes`** ★ | **일반 주석 (★ Rescue 대상)** | **재질 명세, 가공 지시, 일반 공차** |
| `stage1_fp_isometric` | 등각도 | 3D 등각 일부 |
| `stage1_fp_other` | 기타 (default) | 분류 모호한 케이스 |
| `other` | 그 외 | 매우 드문 케이스 |

**라벨링 단축키**:
- `4` → SKIP 라벨 선택
- 박스 그리기 (placeholder, 위치/크기 무관)
- 우측 패널 → `reason` 드롭다운 선택
- Esc → 다음 frame

**★ stage1_fp_notes 의 중요성 (D-038)**:

Stage 1 Version A 가 일반 주석을 PMI 로 잘못 검출하는 케이스 존재:
- 재질 명세 (`材料は鉄かSUS403`, `MATERIAL: SS400`)
- 가공 지시 (`機械加工のこと`, `MACHINING REQUIRED`)
- 일반 공차 (`UNLESS OTHERWISE SPECIFIED ±0.1`)

이 내용들은 **메타데이터 JSON 의 필수 항목** 이므로 단순 SKIP 시 정보 손실 발생.
별도 Rescue Path 로 보존:

```bash
# 1단계: SKIP 라벨 추출 → reason 별 분리
python src/extract_skip_list.py \
    --xml outputs/cvat_stage2_v3_final.xml \
    --output-dir outputs/skip_lists/

# 2단계: stage1_fp_notes 만 Donut OCR Rescue
python src/rescue_misclassified_notes.py \
    --skip-list outputs/skip_lists/stage1_fp_notes.txt \
    --crops-dir outputs/cvat_stage2_input_v3_upscaled \
    --output outputs/rescued_notes.json

# 3단계: pipeline.py / stage4 merger 에서
# 최종 JSON 의 "general_notes" 필드로 병합
```

**관련 문서**:
- `docs/modules/extract_skip_list.md` — SKIP 분류 도구
- `docs/modules/rescue_misclassified_notes.md` — Rescue 도구
- `label_manual.md §3.5 Rule O` — 라벨링 룰
- `PROJECT_HANDOFF.md §11.38` — D-038 박제

### 5.3 Group-aware Split (★)

§4.2 와 동일한 방식. View crop 의 group key 는 `{원본}__View_NN` 형식이므로:

```python
groups = [fn.split("__View_")[0].split(".rf.")[0] for fn in filenames]
```

이렇게 하면 같은 원본 도면에서 잘린 View crop 들이 한쪽 split 에 모입니다.

### 5.4 OBB 라벨 검증 (V3-A)

```bash
python -m src.validate.check_labels_obb \
    --labels-dir data/annotation/labels/train \
    --cfg configs/yolo_obb.yaml
```

**critical 임계값**:
- `obb_validity_rate = 1.0` — 자기교차 폴리곤 / 좌표 범위 검사 100% 통과
- `parse_error_rate = 0`

**warning 임계값**:
- `roughness_min_count ≥ 50` — Roughness 부족 시 D-017 synthetic_gen 검토
- `non_axis_aligned_ratio ≥ 0.20` — 회전 OBB 다양성

### 5.5 학습 (★ 2026-05-03 갱신 — Option C augmentation + Resume 기능)

#### 5.5.1 Option β 표준 학습 (★ 권장, 12~14h)

```bash
python src/stage2_annotation.py train \
    --data data/annotation/data.yaml \
    --model yolo11l-obb.pt \
    --epochs 200 --imgsz 1280 --batch 6 \
    --patience 60 --device 0 \
    --name yolo_obb_v3_l1280 \
    --save-period 20
```

**옵션 사양**:
- 모델: yolo11l-obb (~26M params, balanced)
- imgsz 1280: 작은 GD&T/Roughness 검출 향상
- batch 6: VRAM ~13GB (RTX 5080 16GB 안전)
- epochs 200: augmentation 강화 시 적정 수렴
- patience 60: early stopping 여유
- save_period 20: 20 epoch마다 체크포인트 (PC 중단 대비)

#### 5.5.2 Augmentation (★ Option C 강화)

`src/stage2_annotation.py` 내부 적용 (수정 불필요, 자동):
- `degrees=30` (회전 강화)
- `scale=0.5` (크기 변동)
- `mixup=0.15` (★ 부족 클래스 보완)
- `copy_paste=0.3` (★ Roughness/GDT 인스턴스 증강)
- `fliplr=0`, `flipud=0` (D-001 도면 비대칭)
- `mosaic=1.0`

#### 5.5.3 ★ PC 중단 시 Resume

학습 중 PC 종료 / 장애 발생 시:

**자동 재개 (last.pt 감지)**:
```bash
# --name 동일하게 + --resume 추가
python src/stage2_annotation.py train \
    --data data/annotation/data.yaml \
    --epochs 200 --imgsz 1280 --batch 6 \
    --patience 60 --device 0 \
    --name yolo_obb_v3_l1280 \
    --save-period 20 \
    --resume                          # ★ 추가
```

→ ultralytics 가 last.pt 의 Optimizer 상태 + LR scheduler + augmentation seed 복원하여 **이어서 학습** (deterministic).

**특정 체크포인트에서 재개**:
```bash
python src/stage2_annotation.py train \
    --data data/annotation/data.yaml \
    --resume-from checkpoints/yolo_obb_runs/yolo_obb_v3_l1280/weights/epoch80.pt \
    --epochs 200 --imgsz 1280 --batch 6 \
    --device 0 \
    --name yolo_obb_v3_l1280
```

#### 5.5.4 저장 파일

`checkpoints/yolo_obb_runs/yolo_obb_v3_l1280/weights/`:
- `last.pt` — 매 epoch 갱신 (~200MB, resume 자동 감지 대상)
- `best.pt` — val mAP 최고 epoch (~200MB, 최종 사용)
- `epoch20.pt`, `epoch40.pt`, ..., `epoch200.pt` — save_period=20 시 10개

**디스크 사용**: ~2GB (save-period 20 기준)

학습 완료 후 `checkpoints/yolo_obb.pt` 로 best.pt 자동 복사.

#### 5.5.5 모델 옵션 비교 (참고)

| 옵션 | 모델 | imgsz | 시간 | 예상 mAP@0.5 | 권장 |
|---|---|---|---|---|---|
| α (기본) | yolo11m-obb | 1024 | 5h | 0.78~0.82 | 빠른 검증 |
| **β (★ 권장)** | **yolo11l-obb** | **1280** | **12~14h** | **0.84~0.88** | **균형** |
| γ | yolo11x-obb | 1024 | 14~16h | 0.85~0.89 | overfit 위험 |
| δ | yolo11x-obb | 1280 | 18~22h | 0.87~0.91 | 이론적 최고 |

### 5.6 ★ 모델 성능 검증 (V3-B, D-023 사용자 필수 임계값)

```bash
python -m src.validate.check_stage2_model \
    --weights checkpoints/yolo_obb.pt \
    --data configs/yolo_obb.yaml \
    --device 0 --iou 0.5 --conf 0.25
```

**★ critical 임계값 (D-023)**:
- `missing_rate[Measure] < 0.08` — 치수 8% 이상 놓치면 안 됨
- `missing_rate[GDT] < 0.05` — GD&T 5% 이상 놓치면 안 됨
- `drawing_level_recall ≥ 0.85` — 도면 평균 회수율 85% 이상

**warning 임계값**:
- `missing_rate[Roughness] < 0.30` — 논문도 0.46 (불균형 클래스)
- per-class accuracy: Measure 0.92 / GDT 0.95 / Roughness 0.50

리포트: 도면별 누락 분포 히스토그램 + per-class TP/FP/FN 표.

**FAIL 발생 시 (Plan B 발동 조건)**:
- Measure 누락 > 8% → eDOCr2 ∅ 템플릿 매칭 후처리 도입 검토 (D-017)
- Roughness 부족 → 합성 데이터 보강 모듈 작성

### 5.6.1 ★ 5-Fold Ensemble 추론 (D-040, 2026-05-04)

**배경**: V3-B 단일 모델 (Best Fold = 2) Measure missing **0.101** (D-023 임계 0.08 초과 → FAIL). conf 튜닝 무효 (0.15/0.25 동일 recall) → 5-fold ensemble 채택으로 D-023 PASS 확보.

#### D-023 재평가 (Ensemble)

```bash
python src/ensemble_predict.py evaluate \
    --val-txt data/annotation_kfold/fold_2/val.txt \
    --conf 0.25 --iou-nms 0.5 --imgsz 1024 \
    --device cuda:0 \
    --output outputs/v3b_ensemble_eval.json
```

**측정 결과 (110장 val)**:

| 클래스 | P | R | missing | 임계 | 판정 |
|---|---|---|---|---|---|
| Measure | 0.683 | **1.000** | **0.000** | <0.08 | ✅ PASS |
| GDT | 0.848 | 1.000 | 0.000 | <0.05 | ✅ PASS |
| Roughness | 0.846 | 1.000 | 0.000 | <0.30 | ✅ PASS |

drawing_recall = **1.000** / D-023 overall = **★ PASS ★**

#### 단일 이미지 추론

```bash
python src/ensemble_predict.py predict \
    --image data/annotation/images/valid/sample.jpg \
    --conf 0.25 --iou-nms 0.5 \
    --output outputs/sample_pred.json
```

#### Pipeline 통합 (★ default)

`src/pipeline.py` 의 Stage 2 = ensemble mode (default `--use-ensemble`):

```bash
# 단일 도면 (ensemble default ON)
python src/pipeline.py run --image dataset/sample.jpg \
    --out outputs/sample.json

# 단일 best.pt 모드 (legacy 디버깅)
python src/pipeline.py run --image dataset/sample.jpg \
    --no-ensemble --obb-weights checkpoints/yolo_obb.pt
```

**Trade-off (참고)**: Recall +0.101 (Measure) 대신 Precision -0.266 (Measure FP +46). Stage 3-A FP 처리량 5~10% ↑ — Phase 15 통합 시 모니터링.

**상세 박제**: `history.md §A.11.13` / `outputs/v3b_summary.txt` / `docs/modules/ensemble_predict.md` / D-040.

### 5.7 OBB Crop (Stage 3-N 입력 준비)

```bash
python src/stage2_annotation.py crop \
    --image outputs/crops/sample/View/sample__View_00.jpg \
    --weights checkpoints/yolo_obb.pt
# → outputs/crops/sample/annotations/
#       Measure/*.jpg   (perspective-warp de-rotated upright patches)
#       GDT/*.jpg
#       Roughness/*.jpg
#       manifest.json
```

`crop_obb_regions()` 가 **perspective-warp de-rotation** 으로 회전된 OBB 를 upright 직사각형 패치로 변환 (D-012). Donut VLM 정확도 향상의 핵심.

---

## 6. Stage 3-A — PaddleOCR-VL-1.5 (★ D-039, 2026-05-03 갱신 + ★ D-042 환경 검증 2026-05-04)

> **D-039**: Stage 3-A → **PaddleOCR-VL-1.5** 채택 (D-018 Donut DocVQA 폐기).
> **D-042**: 환경 설치 시 `config.text_config = config.get_text_config()` monkey-patch 필수.
>
> **변경 이유**: D-038 1차 Rescue (Donut DocVQA zero-shot) 다국어 도면에서 4% 성공 (실질 실패).
> 다국어 SOTA 모델 검색 결과 PaddleOCR-VL-1.5 채택 (8가지 사유, history.md §A.11.9 참조).
>
> **★ Phase 15a (2026-05-04) 환경 검증 PASS**: 0.91B params / 39.7s load / 2.26~3.47s inference / 3.29 GB VRAM.

### 6.0 ★ Phase 15a 환경 설치 (★ 별도 venv 분리)

Phase 14 의 ultralytics venv (`.venv`) 와 분리하여 의존성 충돌 회피:

```bash
cd /mnt/c/Users/user/github/Drawing

# 별도 venv 생성
uv venv --python 3.10 .venv-paddleocr
source .venv-paddleocr/bin/activate

# 의존성 (★ transformers 5.0.0 — 5.6+ 는 ROPE 호환성 이슈)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers==5.0.0" accelerate sentencepiece protobuf einops pillow

# 환경 자동 검증 (★ monkey-patch 자동 적용)
python src/stage3_paddleocr_install_check.py
# 종료 코드: PASS = 0, FAIL = 1
```

### 6.0.1 ★ D-042 monkey-patch (★ 모든 후속 코드에 필수 적용)

```python
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
import torch

mid = 'PaddlePaddle/PaddleOCR-VL-1.5'

# 1. Config 로드
config = AutoConfig.from_pretrained(mid, trust_remote_code=True)

# 2. ★ Critical workaround (D-042)
if not hasattr(config, "text_config") and hasattr(config, "get_text_config"):
    config.text_config = config.get_text_config()

# 3. Processor + Model 로드
processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    mid, config=config, trust_remote_code=True, dtype=torch.float16,
).to('cuda:0')
```

이 패턴을 `stage3_alphabetical.py`, `pipeline.py` 등 모든 모델 로드 코드에 적용해야 함.

### 모델 정보

| 항목 | 값 |
|---|---|
| 모델 | `PaddlePaddle/PaddleOCR-VL-1.5` |
| 크기 | 0.9B params (~3GB VRAM) |
| 출시일 | 2026-01-29 |
| 라이센스 | Apache 2.0 |
| OmniDocBench v1.5 | 94.50% (SOTA) |
| Table TEDS | 92.76% |
| Formula CDM | 94.21% |
| 다국어 | 100+ (CJK industry-leading) |
| 작업 | OCR / Formula / Table / Chart / Seal / Text Spotting |
| 출력 | Markdown + JSON (cell 좌표 포함) |

### 채택 사유 (8가지)

1. OmniDocBench 94.50% (DeepSeek-OCR-2 91.09% 대비 +3.41%)
2. **0.9B → RTX 5080 16GB 에서 Stage 2 동시 로드 가능** (DeepSeek 3B 는 OOM 위험)
3. **Table TEDS 92.76% 명시** → Title Block 표 처리
4. **Formula CDM 94.21% 명시** → Notes 수식/공차 정확
5. **Seal Recognition (1.5 신규)** → 도장/검도 도장 처리
6. **CJK industry-leading** → 일/한/중 도면 처리
7. **JSON cell 좌표** → Stage 4 merge 위치 정보
8. **2026-03-06 update (llama.cpp)** → 활발한 개발

### Stage 3-A 입력 흐름 (★ 2026-05-03 명확화)

```
Stage 1 (YOLOv11-det) 검출 결과
   ├─ Table 클래스 영역 (정상 검출)  ──→ Stage 3-A 입력
   │     └─ TitleBlock + BOM + 회사 도장 등
   │
   ├─ Text  클래스 영역 (정상 검출)  ──→ Stage 3-A 입력
   │     └─ Notes (일반 주석, 가공 지시 등)
   │
   └─ PMI 클래스 영역 (Stage 2 OBB 입력)
         ├─ Measure / GDT / Roughness → Stage 2 학습
         └─ SKIP `stage1_fp_notes` 23개 (Stage 1 false positive)
               └→ ★ D-038 Rescue → Stage 3-A 추가 입력
                  (rescue_misclassified_notes.py + PaddleOCR-VL-1.5 백엔드)

Stage 3-A 통합 출력:
   final_json["title_block"] + final_json["general_notes"]
```

**Rescue 범위 결정 (Option α)**:
- ✅ **stage1_fp_notes (23개)**: Rescue 적용 (Day 3 Donut → PaddleOCR-VL-1.5)
- ❌ **stage1_fp_table (13개)**: Rescue 안 함 — Stage 1 의 정상 Table 클래스 영역 사용
- ❌ 그 외 SKIP (unreadable 43 / other 134 / detail 33 / section 29 / projection 2): 폐기

### 사용 (예정, Day 3 작성)

```bash
# 의존성 설치 (Day 3)
uv pip install paddlepaddle-gpu paddleocr

# 단일 패치 추론
python -c "
from paddleocr import PaddleOCRVL
model = PaddleOCRVL.from_pretrained('PaddlePaddle/PaddleOCR-VL-1.5')
result = model.predict('outputs/crops/sample/TitleBlock/sample__TitleBlock_00.jpg', save_format='json')
print(result)
"
```

> **참고**: 기존 `src/stage3_alphabetical.py` (Donut DocVQA) 는 보존되며,
> Day 3 진입 시 PaddleOCR-VL-1.5 백엔드로 교체 또는 신규 모듈 작성 예정.

---

## 6. (구) Stage 3-A — Donut Alphabetical (★ D-039 로 폐기, 참고용)

> **D-018 (★ D-039 로 대체)**: Donut zero-shot. Fine-tuning 없음. TitleBlock + Notes crop 처리.

### 6.1 단일 패치 추론

```bash
# TitleBlock
python src/stage3_alphabetical.py predict \
    --image outputs/crops/sample/TitleBlock/sample__TitleBlock_00.jpg \
    --region titleblock --mode docvqa --language en
# → outputs/sample__TitleBlock_00.alpha.json

# Notes
python src/stage3_alphabetical.py predict \
    --image outputs/crops/sample/Notes/sample__Notes_00.jpg \
    --region notes
```

### 6.2 배치 처리 (Stage 1 crop 폴더 통째)

```bash
python src/stage3_alphabetical.py batch \
    --input-dir outputs/crops/sample \
    --out-dir outputs/sample/alphabetical
# 모든 TitleBlock/Notes crop 자동 처리
```

### 6.3 출력 JSON 예시 (TitleBlock)

```json
{
  "type": "TitleBlock",
  "fields": {
    "drawing_no": "DWG-001-A",
    "material": "SS400",
    "scale": "1:2",
    "drawn_by": "Kim"
  },
  "raw": {"What is the drawing number?": "DWG-001-A", ...},
  "model": "donut-base-finetuned-docvqa",
  "language_hint": "en"
}
```

### 6.4 성능 기대치 (논문)

- TitleBlock F1: 0.533 (낮음, zero-shot 한계)
- Notes F1: 0.810
- Hallucination: 0.40 ~ 0.48

낮은 정확도는 **Step 9 (Metadata Enrichment)** 에서 LLM 으로 보정 예정.

---

## 6.5 Stage 3-N — Donut Numerical Fine-tune (★ Phase 16a/b, 2026-05-06 갱신)

> **D-049/D-050/D-051 박제** — Phase 16a/b 진행 시 핵심 의사결정 + 한계 검증.
> 상세: `history.md §A.12.10` + `docs/KNOWN_LIMITATIONS.md §4`.

### 6.5.1 ★ Phase 16a — VLM pair 학습 데이터 자동 생성

**venv 전환** (Phase 14 ultralytics 환경 사용):
```bash
deactivate
source .venv/bin/activate
```

**실행 명령** (★ 인자명 정정 — 가이드 작성 시 인자명 확인 누락 발견):
```bash
python src/prepare_vlm_dataset.py numerical \
    --dataset dataset/ \
    --det-weights checkpoints/yolo_det.pt \
    --obb-weights checkpoints/yolo_obb.pt \
    --device 0 \
    --ocr-prefill \
    --limit 500
```

| 잘못된 인자 (이전 가이드) | 실제 인자 |
|---|---|
| `--input dataset/` | `--dataset dataset/` |
| `--stage1-weights ...` | `--det-weights ...` |
| `--stage2-ensemble checkpoints/yolo_obb_runs/` | `--obb-weights checkpoints/yolo_obb.pt` (★ 단일 파일) |
| `--output data/vlm/numerical/` | (없음 — 코드 내 고정 경로) |

**참고**: K-fold ensemble (5 fold) 은 prepare_vlm_dataset 미지원 → Phase 17 e2e 에서 ensemble_predict.py 적용.

**Phase 16a 실측 결과** (2026-05-06, 500 도면):
- 처리 시간: 24분 04초
- 산출 region: **11,470** (도면당 평균 22.94)
- 클래스 분포: Measure 8,750 (76.3%) / GDT 531 (4.6%) / Roughness 2,189 (19.1%)
- manifest.csv 정상 작성 (group_key 포함, D-024)

**★ D-049 sys.path bootstrap** (Task #92 패턴 적용):
- `prepare_vlm_dataset.py` 가 `from src.stage1_layout import ...` 사용
- 직접 실행 시 `ModuleNotFoundError: No module named 'src'` → bootstrap 으로 해결
- ★ 절대 금지: `pip install src` / `uv pip install src` (PyPI 무관 패키지)

### 6.5.2 ★ Auto-fill Numerical GT (★ 신규 D-050/D-051 검증)

Phase 16a 가 생성한 JSON 의 GT field 는 모두 `null` (검수 시드 상태). 1차 baseline 학습을 위해 OCR hint regex 매핑으로 자동 채움:

```bash
python src/auto_fill_numerical_gt.py \
    --report outputs/auto_fill_numerical_report.md
cat outputs/auto_fill_numerical_report.md
```

**Auto-fill 결과** (실측, Phase 16a 직후):
| 클래스 | Total | Filled | Rate | 평가 |
|---|---|---|---|---|
| Measure | 8,750 | 5,381 | **61.5%** ✅ | nominal 채움 가능 (★ baseline 가능) |
| GDT | 531 | 1 | 0.2% ❌ | 학습 사실상 불가 (D-051) |
| Roughness | 2,189 | 402 | 18.4% ⚠️ | 제한적 |
| **Total** | **11,470** | **5,784** | **50.4%** | Donut DataModule 학습 입력 |

**★ D-050 박제 — Tesseract OCR 한계 (Critical)**:
- `--psm 6` + `kor+eng+rus+jpn` 사용
- 도면 patch 작은 글자 (10~14 px) + 한자/일본어/한글 혼재 → OCR 노이즈 매우 큼
- tolerance `±` 부호 인식 0% / GDT symbol (⌖/⏤/⊥) 인식 0% / Ra 키워드 인식 거의 0%
- regex 보강 효과 ≈ 0 (OCR 노이즈가 본질 원인)

**★ D-051 박제 — 1차 baseline = Measure-only**:
- Phase 16b Donut numerical fine-tune 의 학습 효과는 **Measure nominal extraction** 에 한정
- GDT / Roughness 는 noisy GT 로 포함되지만 학습 효과 기대 X
- Phase 17 e2e 평가의 자리만 채움 → 후속 개선 우선순위 정량화

### 6.5.3 ★ Phase 16b — Donut Numerical 학습

**CLI** (★ cfg 파일 통합):
```bash
nohup python src/stage3_numerical.py train \
    --cfg configs/donut_numerical.yaml \
    --device 0 \
    > outputs/stage3n_train.log 2>&1 &
echo $! > outputs/stage3n_train.pid

# 5분 모니터링 (NaN check)
sleep 300 && tail -50 outputs/stage3n_train.log
```

**configs/donut_numerical.yaml 핵심**:
- 데이터: `data/vlm/numerical/`, 70/20/10 split
- epochs 30 / batch 4 / lr 1e-6 / cosine decay
- precision fp16 + gradient_checkpointing (RTX 5080 16GB 대응)
- 출력: `checkpoints/donut_numerical/`

**예상 학습 시간**: ~6h (overnight)

**★ 후속 (Phase 18+)**:
- 검수 도구 작성 (Streamlit / CVAT) + 사람 검수 ~3일
- GDT crop ~500 추가 라벨링 (extract_gdt_crops.py + CVAT)
- Stage 3-N full GT 재학습

---


## 7. Step 4 — VLM 학습 데이터 시드 자동 생성 (`prepare_vlm_dataset.py`)

**Phase 16a 의 핵심 — 도면 → Stage 1 + Stage 2 추론 → de-rotation crop → JSON 템플릿 생성**.

### 7.1 CLI (★ 인자명 정정 — D-049 sys.path bootstrap 적용)

```bash
deactivate && source .venv/bin/activate

python src/prepare_vlm_dataset.py numerical \
    --dataset dataset/ \
    --det-weights checkpoints/yolo_det.pt \
    --obb-weights checkpoints/yolo_obb.pt \
    --device 0 \
    --ocr-prefill \
    --limit 500
```

**3가지 서브커맨드**:
- `numerical` (★ Phase 16a 사용) — Stage 2 OBB → Measure/GDT/Roughness patch
- `alphabetical` — Stage 1 → TitleBlock/Notes crop (Phase 15 후속)
- `all` — 위 2개 동시

**핵심 옵션**:
- `--ocr-prefill`: Pytesseract 으로 `_review.ocr_hint` 자동 채움 (★ D-050 한계 인지)
- `--limit N`: 처리 도면 수 제한 (디버깅/sample)
- `--device 0` (numeric str — D-038 동일 패턴, OK)

### 7.2 산출물 구조

```
data/vlm/numerical/
├── manifest.csv                                  # group_key + 통계
├── <drawing>__View_<i>__Measure_<j>.jpg          # 회전 정렬된 patch
├── <drawing>__View_<i>__Measure_<j>.json         # GT 템플릿 (★ null 다수)
└── ...
```

**JSON 템플릿 예시**:
```json
{
  "type": "Measure",
  "nominal": null,         ← ★ 사람 검수 또는 auto_fill 필요
  "tolerance": null,
  "unit": "mm",
  "_review": {
    "completed": false,
    "ocr_hint": "12.5±0.05",
    "ocr_numeric": 12.5
  }
}
```

### 7.3 후속 — Auto-fill (`auto_fill_numerical_gt.py`, D-050 한계 인지)

Phase 16a 의 null GT 를 1차 baseline 학습용으로 자동 채움:

```bash
python src/auto_fill_numerical_gt.py \
    --report outputs/auto_fill_numerical_report.md
```

**실측 결과** (11,470 region 기준):
- Overall fill rate 50.4% (5,784 completed)
- Measure 61.5% / GDT 0.2% / Roughness 18.4%
- D-050: Tesseract OCR 한계로 tolerance / GDT symbol / Ra 정확도 ↓

상세: `docs/KNOWN_LIMITATIONS.md §4.1, §4.2` (★ Critical, 검수 도구 필요).

---

## 8. Stage 4 — JSON Merger + 통합 JSON

`pipeline.py` 가 Stage 1 → Stage 2 → Stage 3-A → Stage 3-N 결과를 **단일 통합 JSON** 으로 병합 (HANDOFF §5.5 schema).

### 8.1 통합 JSON 구조

```json
{
  "drawing_id": "0301040003_SHAFT-...",
  "image_path": "dataset/<sample>.jpg",
  "image_size": [W, H],
  "title_block": { ... },              // Stage 3-A (PaddleOCR-VL, D-039)
  "notes": [ ... ],                    // Stage 3-A
  "views": [
    {
      "view_id": "view_0",
      "bbox": [x1, y1, x2, y2],
      "annotations": [
        {
          "class": "Measure",
          "obb_global": [[x,y]*4],     // 글로벌 좌표 (Stage 4 변환)
          "obb_local":  [[x,y]*4],     // view-crop 좌표
          "angle": 12.5,
          "conf": 0.93,
          "parsed": { ... }            // Stage 3-N JSON
        }
      ]
    }
  ],
  "meta": {
    "model_versions": { ... },
    "timing_seconds": { "stage1": 1.2, "stage2": 6.0, "stage3_alphabetical": 50.3, "stage3_numerical": 178.0, "total": 235.5 },
    "timestamp": "2026-05-06T13:14:00Z"
  }
}
```

### 8.2 OBB 글로벌 좌표 변환

각 View crop 안의 OBB 는 local 좌표 → Stage 4 가 view bbox + 회전 행렬 적용해 **원본 도면 좌표** 로 자동 변환 (`obb_global`). 통합 JSON 의 사용자 검수 / Step 9 enrichment 입력으로 활용.

---

## 9. Step 7 — End-to-end Pipeline (`pipeline.py`)

### 9.1 CLI (★ 2026-05-06 검증)

```bash
PYTHONPATH=. python src/pipeline.py run \
    --image dataset/<sample>.jpg \
    --donut-num checkpoints/donut_numerical/final \
    --device cuda:0 \
    --language en \
    --out outputs/pipeline_e2e_smoke \
    --keep-tmp
```

**핵심 인자**:
- `--image` (필수): 입력 도면 1장
- `--out`: 출력 JSON 파일 경로 (★ 디렉토리 X, 단일 파일)
- `--donut-num`: Stage 3-N 체크포인트 (`checkpoints/donut_numerical/final`)
- `--use-ensemble` (default): Stage 2 5-fold ensemble (D-040 PASS)
- `--no-ensemble`: 단일 fold (디버깅)
- `--ensemble-ckpt-root`: K-fold ckpt 루트 (default `checkpoints/yolo_obb_runs/`)
- `--language`: en / ko / ja / ru / zh / de (Stage 3-A hint)
- `--skip-numerical` / `--skip-alphabetical`: 단계 분리
- `--keep-tmp`: 디버깅용 임시 파일 보존

### 9.2 Phase 17 진입 — Smoke test (2026-05-06 진행)

**옵션 A — 안전 (Stage 3-A skip)**:
```bash
PYTHONPATH=. python src/pipeline.py run \
    --image dataset/<sample>.jpg \
    --donut-num checkpoints/donut_numerical/final \
    --device cuda:0 --skip-alphabetical \
    --out outputs/pipeline_e2e_smoke --keep-tmp
# ~3분 50초, Stage 1 + 2 + 3-N 만 검증
```

**옵션 B — 풀 e2e (Stage 3-A 통합, ★ Phase 15c 후 가능)**:
```bash
# Phase 15c (PaddleOCR-VL backend 통합) 완료 후 가능
PYTHONPATH=. python src/pipeline.py run \
    --image dataset/<sample>.jpg \
    --donut-num checkpoints/donut_numerical/final \
    --device cuda:0 --language en \
    --out outputs/pipeline_e2e_full --keep-tmp
# ~6~8분, subprocess wrapper 통해 .venv-paddleocr 호출
```

### 9.3 Batch 모드

```bash
PYTHONPATH=. python src/pipeline.py batch \
    --input-dir data/pipeline_e2e_eval/ \
    --donut-num checkpoints/donut_numerical/final \
    --device cuda:0 \
    --out outputs/pipeline_e2e_baseline_v1/
# (배치 인자는 batch --help 로 별도 확인)
```

---

## 10. Step 8 — Validation 프레임워크 (V0 ~ V9)

각 Stage 의 사후 검증 도구 — `src/validate/check_*.py`. ★ D-013 임계값 + V6 D-023 critical.

| Validator | 대상 | 임계값 | 상태 |
|---|---|---|---|
| V0 `common.py` | Helper | — | ✅ |
| V1 `check_step1_5_sorter.py` | Step 1.5 분류 정확도 | per-language | ✅ |
| V2-A `check_labels_yolo.py` | YOLO det 라벨 | bbox 유효성 | ✅ PASS |
| V2-B `check_stage1_model.py` | Stage 1 모델 | mAP@50 ≥ 0.85 | ✅ V.A 0.94 |
| V3-A `check_labels_obb.py` | OBB 라벨 | obb 유효성 | ✅ |
| V3-B `check_stage2_model.py` | Stage 2 모델 | D-023 critical | ✅ Ensemble PASS |
| V5 `check_stage3a_alphabetical.py` | Stage 3-A | char acc ≥ 0.85 | ⚠️ 부분 PASS (4차 0.69) |
| V6 `check_stage3n_numerical.py` | Stage 3-N | D-023 hallucination ≤ 0.10 | ❌ FAIL (D-055, V6 baseline 0.34/0.72) |
| V7 `check_pipeline_e2e.py` | e2e | field_f1 등 | ⏳ 미진행 |
| V9 `check_enrichment.py` | Step 9 | provenance / cost | ⏳ 미진행 |

### 10.1 V6 검증 흐름 (★ Phase 16c)

```bash
# 1. Test set 분리 (group-aware D-024, seed=42)
PYTHONPATH=. python src/extract_test_set_for_v6.py
# → outputs/test_set_v6/{Measure,GDT,Roughness}/ + gt/

# 2. Stage 3-N batch 추론 (~18분)
PYTHONPATH=. python src/stage3_numerical.py batch \
    --input-dir outputs/test_set_v6/ \
    --ckpt checkpoints/donut_numerical/final \
    --device cuda:0 \
    --out-dir outputs/stage3n_baseline_v1_predictions/

# 3. V6 검증 (~10초)
PYTHONPATH=. python src/validate/check_stage3n_numerical.py \
    --predictions outputs/stage3n_baseline_v1_predictions/ \
    --gt outputs/test_set_v6/gt/ \
    --reports-dir reports/
```

**실측 (D-055, 2026-05-06)**:
- field_f1[Measure] 0.379 / numerical_accuracy 0.034 / hallucination 0.720
- 1차 baseline 학습 결과 — D-051 가설 검증 완료 (검수 GT 필수성)

---

## 11. Step 9 — Stage 5 Enrichment (★ KNOWN_LIMITATIONS 해결 후 진행)

**4-tier cascade** — Stage 4 통합 JSON 의 누락/추정 필드 보강.

### 11.1 Tier 흐름

| # | Tier | 처리 | 비용 | 신뢰도 |
|---|---|---|---|---|
| 1 | Deterministic | KB lookup (material_catalog.json 등) | 0 | ★★★ |
| 2 | Heuristic | regex / 임계값 (Ra<0.4 → grinding) | 0 | ★★ |
| 3 | RAG-LLM | Mock / Gemini / Qwen | $$ | ★ |
| 4 | HITL gate | low conf flag (★ 사람 검수) | 사람 시간 | (검수 후 ★★★) |

각 필드에 ★ provenance (source / confidence / cost, D-022) 자동 박제.

### 11.2 CLI

```bash
PYTHONPATH=. python src/stage5_enrichment.py \
    --input outputs/pipeline_e2e_baseline_v1/<drawing>.json \
    --output outputs/enriched/<drawing>.enriched.json \
    --provider mock      # mock / gemini / qwen
```

### 11.3 ★ 정책 (D-051/D-055 후속)

★ Step 9 Enrichment 는 **KNOWN_LIMITATIONS Critical 항목 (D-050 OCR / D-055 baseline 정확도) 해결 후 진행**. 현재 Stage 4 통합 JSON 의 정확도가 낮아 Enrichment 효과 측정 어려움. Phase 18 검수 도구 + 재학습 후 V9 평가 진행.

---

## 12. Phase 15c — PaddleOCR-VL backend 통합 (★ 진행 중, 2026-05-06)

### 12.1 배경

`pipeline.py` 가 D-018 Donut DocVQA (폐기) 호출 → D-039 PaddleOCR-VL-1.5 backend 로 교체.

**문제**: `.venv` (transformers 4.49.0) 가 `.venv-paddleocr` (5.0.0) 의 `AutoModelForImageTextToText.PaddleOCRVLConfig` 미지원.

**해결**: ★ Subprocess wrapper (옵션 B) — long-running worker + JSON line protocol.

### 12.2 구조

```
[.venv]                                  [.venv-paddleocr]
src/stage3_alphabetical_paddleocr.py     src/stage3_alphabetical_paddleocr_worker.py
       (wrapper)                                 (worker)

pipeline.py
  ↓ from src.stage3_alphabetical_paddleocr import load_model, predict_one
  ↓ subprocess.Popen(.venv-paddleocr/bin/python, worker.py)
  worker stdout → JSON line: {"title_block": {...}}
  worker stderr → outputs/paddleocr_worker.stderr.log (★ 파일 redirect, deadlock 방지)
```

### 12.3 핵심 fix 박제

- D-042 monkey-patch (`config.text_config = config.get_text_config()`)
- D-046 호출 방식 (task keyword + bf16 + apply_chat_template)
- ★ stderr 파일 redirect (PIPE buffer deadlock 방지)
- ★ atexit hook + shutdown signal (좀비 worker 방지)

### 12.4 디버깅

```bash
# Worker stderr 실시간 watch
tail -f outputs/paddleocr_worker.stderr.log

# 좀비 worker 정리
pkill -f paddleocr_worker
```

---

## 13. 다음 단계 (Phase 17+)

### 13.1 Phase 17 e2e 정규 평가 (Phase 15c 완료 후)

1. 5장 sample batch → V7 검증
2. 부분 PASS 인정 (Stage 3-N D-051 baseline + Stage 3-A 부분)
3. 후속 우선순위 정량화

### 13.2 Phase 18 — 검수 도구 + 재학습 (★ 최우선)

★ KNOWN_LIMITATIONS Critical 1순위:
1. Streamlit/CVAT 검수 도구 작성 (~1주)
2. 사람 검수 ~3일 (Phase 16a JSON GT 채움)
3. Stage 3-N 재학습 (~6h overnight)
4. V6 재평가 (목표 numerical_accuracy ≥ 70%, hallucination ≤ 10%)

### 13.3 Phase 19+ — Step 9 Enrichment + 종합 평가

1. KB 강화 (material_catalog 다국어 보강)
2. Enrichment 4-tier cascade 평가 (V9)
3. 종합 V7 e2e 통과 → 운영 준비

---

## 14. 트러블슈팅 (★ 박제)

| 문제 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: 'src'` | sys.path 미적용 | `PYTHONPATH=. python ...` 또는 D-049 bootstrap 적용 |
| `Invalid device string: '0'` | numeric str 미지원 | `--device cuda:0` (D-038 패턴) |
| `pip install src` 시도 | PyPI 외부 패키지 | ★ 절대 금지 (D-049 박제) |
| Worker timeout / deadlock | stderr PIPE buffer 가득 | 파일 redirect (Phase 15c fix) |
| Donut data_collator ValueError | input_ids 키 누락 | `default_data_collator` (D-052) |
| `num_items_in_batch` TypeError | transformers 5.x 호환성 | DonutTrainer subclass (D-053) |
| stage3_numerical.py EOF SyntaxError | 중복 코드 추가 | head -748 + 정리 |
| transformers 5.6.2 vs 4.49.0 박제 불일치 | 실제 .venv = 4.49.0 | 박제 정정 필요 (후속) |

---

## 15. 참고 문서 (★ 단일 source of truth)

- [`README.md`](./README.md) — 프로젝트 진입 + 진행 현황 (Day 1~4 LIVE)
- [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md) — 전체 의사결정 (D-001 ~ D-055)
- [`history.md`](./history.md) — 시간순 학습 이력 (Version A, Day 1~4, Phase 14~16)
- [`docs/KNOWN_LIMITATIONS.md`](./docs/KNOWN_LIMITATIONS.md) ★ 한계 / 미해결 단일 source + 추천 해결 방법
- [`PIPELINE.md`](./PIPELINE.md) — Phase 1~17 흐름도
- [`docs/NEXT_SESSION_GUIDE.md`](./docs/NEXT_SESSION_GUIDE.md) — 다음 세션 진입 가이드
- [`docs/PHASE15_CHECKLIST.md`](./docs/PHASE15_CHECKLIST.md) — Phase 15 체크리스트
- [`outputs/workflow_diagram_v4.png`](./outputs/workflow_diagram_v4.png) — v4 한글 다이어그램
- [`outputs/IMMA_progress_report_v4.docx`](./outputs/IMMA_progress_report_v4.docx) — 동료 공유용 진행 보고서
