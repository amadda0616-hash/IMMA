# 라벨링 매뉴얼 (Label Manual)

> **목적**: 5,839장 데이터셋의 Stage 1 (axis-aligned BBox) + Stage 2 (Oriented BBox) 라벨링 전체 흐름을 정리.
> **방법**: 하이브리드 (방법 C — Roboflow + CVAT) + Active Learning + eDOCr2 휴리스틱 부트스트랩.

> 관련 문서: [`MANUAL.md`](./MANUAL.md) §4~5 / [`PROJECT_HANDOFF.md`](./PROJECT_HANDOFF.md) §11 / [`docs/modules/`](./docs/modules/)

## 0-0. ★ 현재 진행 위치 (2026-04-29)

### Stage 1 Active Learning ✅ 완료

| # | 단계 | 결과 |
|---|---|---|
| 1 | Roboflow seed 100장 라벨링 | ✅ IMMA.v1i.yolov11/ |
| 2 | Stage 1 Version A 학습 | ✅ mAP 0.9364 |
| 3 | auto_label_stage1.py 실행 | ✅ 5,839장 자동 라벨 |
| 4 | sort_by_yolo_pmi.py + exclude_groups.py | ✅ 5,793장 정리 (조립도면 18 group 제외) |

### Stage 2 Active Learning ⏳ 진행 중

| # | 단계 | 상태 |
|---|---|---|
| 1 | extract_pmi_crops.py 실행 (auto_pass + review priority 20장) | ✅ 844 PMI crops 추출 |
| 2 | **★ Day 1 — CVAT Stage 2 OBB 라벨링 500장** (★ §3 참조) | **⏳ 진행** |
| 3 | Stage 2 Version A 학습 (Day 2) | ⬜ |
| 4 | auto_label_stage2.py 작성 + 실행 (Day 2) | ⬜ |
| 5 | 사람 검수 + Stage 2 Version B (Day 3) | ⬜ |

상세 학습 이력: [`history.md`](./history.md) §A

---

## 0-1. 자동 분류 도구 (Step 1.6, D-026)

라벨링 전 **`src/sort_by_drawing_type.py`** 실행 권장:

```bash
# dryrun
python src/sort_by_drawing_type.py --dryrun
# 매니페스트 검토 후 실제 분류
python src/sort_by_drawing_type.py
```

→ `data/manufacturing/` (가공도면 — Stage 1+2 학습)
→ `data/assembly/` (조립도면 — Stage 1 만)
→ `data/manual_review_type/` (사람 검수)

이후 `data/manufacturing/` 에서 100장 seed 선택해서 Roboflow 업로드.

상세: [`docs/modules/sort_by_drawing_type.md`](./docs/modules/sort_by_drawing_type.md)

---

## 0. 라벨링 시작 시점 발견 사항 (2026-04-28)

| 발견 | 영향 | 의사결정 |
|---|---|---|
| 中文 (Chinese) 도면 추가 | 5개 언어 (EN/KO/JP/RU/CN) | **D-025** — Tesseract 패키지·키워드 사전·KB 갱신 필요 |
| 가공/조립 도면 혼재 | Stage 2 학습은 가공도면만 | **D-026** — 가공/조립 분류 필수 |
| TB 핵심 필드 (material/quantity) 누락 흔함 | Stage 3-A hallucination 위험 | **D-027** — Step 9 enrichment 가 보강 |
| TB 분포 95:5 (TB 있음 압도) | sort_by_titleblock sampling 가치 ↓ | sort_by_drawing_type 으로 대체 |
| Roboflow data.yaml 클래스 5종 (`Isometric/PMI/Table/Text/View`) 채택 | Stage 1 클래스 재정의, PMI=Stage 2 입력 | **D-028** — 5클래스 / **D-029** — Roboflow→내부 매핑 (`Table→TitleBlock`, `Text→Notes`) |

---

## 1. 워크플로 한눈에 (2026-04-29 갱신)

```
[Stage 1 — 원본 도면 5,839 장]                              ★ ✅ 완료
        ↓
   Roboflow seed 100 장 라벨링 (~10 시간)                    ★ ✅ Day 0
        ↓
   Stage 1 Version A 학습 (~30 분)                           ★ ✅ mAP 0.9364
        ↓
   auto_label_stage1.py 실행 → 5,839 장 자동 라벨            ★ ✅ 5분 45초
        ↓
   sort_by_yolo_pmi.py + exclude_groups.py 정리              ★ ✅ 5,793 학습 잔여
        ↓
   Pre-annotation 보류 (D-035) — Version B 차후

[Stage 2 — PMI crop (D-034)]                                  ★ ⏳ Day 1~3
        ↓
   extract_pmi_crops.py — Stage 1 으로 PMI crop 추출         ★ ✅ 844 PMI / 20 도면
        ↓
   ★ CVAT 수동 OBB 라벨링 (500 장 — Day 1)                   ⏳ 본 문서 §3
        ↓
   Stage 2 Version A 학습 (~5 시간 — Day 2)
        ↓
   auto_label_stage2.py 작성 + 실행 → 644 추가 자동 OBB
        ↓
   사람 검수 + Stage 2 Version B (Day 3, 시간 여유 시)
        ↓
   V3-B 검증 (★ D-023 누락률)
```

> Stage 1 Active Learning 패턴이 Stage 2 에 그대로 적용됨. 차이점: OBB (회전 박스) + 3 클래스 (Measure/GDT/Roughness).

---

## 2. Roboflow — Stage 1 (Object Detection)

### 2.1 프로젝트 설정

```
Project Name: Stage1_Layout_Detection
Project Type: Object Detection ← Bounding Boxes
Tool:         Traditional
Visibility:   Private
Classes (정확히 이 이름, 5종 D-028):
   0: Isometric
   1: PMI
   2: Table        ← 코드 내부에서 TitleBlock 으로 매핑 (D-029)
   3: Text         ← 코드 내부에서 Notes 로 매핑 (D-029)
   4: View
```

> **⚠️ 클래스 이름 정확히 일치 필수**
> `configs/yolo_det.yaml` + `IMMA.v1i.yolov11/data.yaml` 와 동일한 문자열 (Roboflow 라벨).
> 대소문자 / 띄어쓰기 다르면 학습 시 클래스 매핑 실패.
> 코드 내부 매핑 (D-029): `Table → TitleBlock`, `Text → Notes` (의미 보존, Stage 3-A 토큰 호환).

### 2.2 입력 이미지

| 단계 | 이미지 | 수량 |
|---|---|---|
| Seed | `dataset/` 에서 sampling 100장 (★ 완료, `IMMA.v1i.yolov11/train/`) | 100 ✅ |
| Auto + 검수 | 나머지 5,739장 | 5,739 |

**Seed 선택 가이드** (다양성 확보):
- ★ **가공도면 우선** (D-026): 조립도면은 Stage 2 학습 무가치. 가공도면 90장 + 조립도면 10장 권장 (Stage 1 모델이 두 종류 모두 학습)
- ★ **5개 언어** (D-025) 비례: EN / KO / JP / RU / CN 골고루
- TB 있는 도면 압도적 (95:5, D-027) — TB 있는 도면 위주로 OK
- 도면 종류 (단일 부품 / 단면 / 상세) 골고루
- ★ **Roboflow Tag 활용**: 라벨링 시 각 도면에 `manufacturing` / `assembly` tag + 언어 tag (`lang_en` / `lang_ko` 등) 부여 → 후속 자동 라벨링 단계에서 필터링 가능

### 2.3 클래스별 라벨링 기준

#### 클래스 0: View

> 도면의 **그림 영역** — 정면도 / 평면도 / 측면도 / 단면도 / 상세도 등.

**포함**:
- 그림 자체 (실선/점선)
- 치수선·보조선·화살표 (View 안에서 작업)
- 내부의 GD&T·Roughness·Measure 표기 (Stage 2 가 처리할 영역)

**제외**:
- TitleBlock 표 영역
- Notes 텍스트 블록
- 도면 외곽 frame (이중 frame 인 경우 안쪽까지만)

**한 도면 = 여러 View** 가능. 각 View 를 별도 BBox 로.

```
┌────────────────────────────────────────┐
│  ┌─View1──────┐    ┌──View2────────┐   │
│  │  정면도    │    │   단면도 A-A  │   │
│  │  ⌀25  ⌀30 │    │               │   │
│  └────────────┘    └───────────────┘   │
│  ┌──────────TitleBlock──────────────┐  │
│  │ DWG-001    SS400    1:2          │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

#### 클래스 1: TitleBlock

> 도면 정보 표 — 도번 / 제목 / 척도 / 재질 / 작성자 / 날짜 / 회사 등.

**위치**: 보통 우하단 또는 하단 전체.

**박스 기준**: 표의 **가장 바깥쪽 테두리**까지 (셀 단위로 분리하지 말 것).

**TitleBlock 없는 도면** (예: 단일 부품 patch):
- 그냥 라벨 안 함. 모델이 "TitleBlock 없을 수도 있다"를 학습.

#### 클래스 2: Notes

> 일반 노트 / 주석 텍스트 블록.

**예시**:
```
1. UNLESS OTHERWISE SPECIFIED
2. ALL DIMENSIONS IN MM
3. BREAK ALL SHARP EDGES
```

**박스 기준**: 텍스트 블록 외곽 (배경/공백은 최소화).

**Notes 없는 도면** 흔함 → 안 그려도 됨.

### 2.4 박스 경계 룰 (공통)

- BBox 는 영역 전체를 감싸는 **가장 작은** 직사각형
- 세 클래스가 **서로 겹치지 않아야** 함
- 박스 안에 **여백 최소** (영역 외곽선에 딱 붙임)
- 부분 가려짐: 가려지지 않은 영역만 박스

### 2.5 Roboflow Auto Label / Smart Polygon (선택)

100장 seed 라벨링 시 도구 가속 옵션:
- **Smart Polygon** (SAM 기반): 클릭 한 번으로 영역 자동 그리기 — TitleBlock 처럼 확실한 경계 영역에 효과적
- **Manual Bounding Box**: 일반 BBox 도구

> ⚠️ Roboflow Free tier 의 Auto Label 은 월 한도 있음. 100장 정도는 충분.

### 2.6 Export

- Format: **YOLOv11** (또는 YOLOv8)
- Train/Test split: **수동으로 group-aware split 권장** (D-024 — 같은 `.rf.<hash>` group 이 train/val 양쪽에 들어가지 않게)
  → Roboflow 의 random split 대신 export 후 자체 split 스크립트 적용

---

## 3. CVAT — Stage 2 (Oriented BBox)

> ★ **2026-04-29 갱신**: 입력은 **PMI crops** (D-034 — `extract_pmi_crops.py` 산출). View crop 아님.
> ★ Day 1 권장: **500 장 라벨링** (적극 페이스 ~4 시간 / 꼼꼼 ~12 시간). 자세한 성능 추정은 §3.0 참조.

### 3.0 ★ 라벨링 장수별 예상 성능 (Stage 2 Active Learning Phase 1)

#### 데이터 양 → 성능 추정 (논문 Khan 2025 + Stage 1 Version A 패턴 기반)

**전제**:
- 1 PMI crop ≈ 1 OBB (Stage 1 의 PMI bbox 가 OBB 단위로 잘림)
- 클래스 분포 (논문): Measure 75% / GDT 20% / Roughness 5%
- 논문 baseline (전체 1,406 장 학습 시): Measure 0.95 / GDT 0.97 / Roughness 0.54

| 라벨링 장수 | OBB 분포 (Measure / GDT / Roughness) | mAP@0.5 (전체) | Measure missing (★ D-023) | GDT missing | Roughness mAP | V3-B 평가 |
|---|---|---|---|---|---|---|
| **100 장** | 75 / 20 / 5 | **0.50~0.60** | 25~35% | 30~40% | 0.10~0.30 | 🔴 **critical 미달** |
| **200 장** | 150 / 40 / 10 | **0.65~0.75** | 12~20% | 15~25% | 0.30~0.50 | 🟡 부분 통과 (Measure/GDT critical 미달 가능) |
| **300 장** | 225 / 60 / 15 | **0.72~0.82** | 8~15% | 10~18% | 0.40~0.55 | 🟢 critical 통과 가능 |
| **500 장 (★ 권장)** | 375 / 100 / 25 | **0.78~0.85** | 6~10% | 7~12% | 0.50~0.55 | ✅ **안정** — Day 1 채택 |
| 844 장 (전체 추출) | 633 / 169 / 42 | 0.82~0.88 | 4~8% | 5~10% | 0.50~0.58 | ✅ 권장 임계값 만족 |
| 논문 1,406 (참고) | 1,055 / 281 / 152 | 0.82 (논문 평균) | — | — | 0.54 | — |

#### D-023 critical 임계값 vs 라벨링 장수

| 임계값 (D-023) | 100 장 | 200 장 | 300 장 | 500 장 | 844 장 |
|---|---|---|---|---|---|
| Measure missing < 8% | ❌ | ❌ | 🟡 borderline | ✅ | ✅ |
| GDT missing < 5% | ❌ | ❌ | ❌ | 🟡 | ✅ |
| Drawing-level recall ≥ 0.85 | ❌ | 🟡 | ✅ | ✅ | ✅ |
| Roughness count ≥ 50 (D-017 trigger) | ❌ (5) | ❌ (10) | ❌ (15) | 🟡 (25) | ❌ (42) |

#### 클래스별 약점 분석

**Measure** (75%, 가장 흔함):
- 100~200 장에서도 어느 정도 학습 가능
- 다만 D-023 의 missing < 8% 는 **300 장 이상** 필요

**GDT** (20%, 중간):
- 100 장에서는 데이터 거의 없음 (~20개)
- **300~500 장 권장**

**Roughness** (5%, 매우 적음):
- 200 장 → 10 개 — 학습 거의 무의미
- **500 장 → 25 개** 부족 → D-017 (synthetic_gen) 트리거 필요
- 논문도 152 개 → 0.54 mAP 만 달성 — 데이터 부족 한계

#### 권장 시나리오 (시간 vs 정확도 trade-off)

| 시나리오 | 라벨링 시간 (적극 / 꼼꼼) | 학습 후 V3-B |
|---|---|---|
| 200 장 | ~1.7h / ~5h | 🟡 부분 통과 — critical 미달 가능 |
| 300 장 | ~2.5h / ~7h | 🟢 critical 통과 가능 |
| **★ 500 장 (Day 1 채택)** | **~4h / ~12h** | **✅ 안정** — D-023 critical 통과 |
| 844 장 전체 | ~7h / ~25h | ✅ 권장 임계값 만족, 다만 시간 부담 |

→ **Day 1 plan 500 장** 라벨링 + Stage 2 Version A 학습 + auto_label_stage2.py 로 자동 라벨 + 검수 후 Version B = **600+ 장 효과** (Active Learning 의 핵심).

#### Day 1 시간 옵션 (500 장 기준)

| 페이스 | OBB/시간 | 시간 | 비고 |
|---|---|---|---|
| **적극 (★ 권장)** | ~125 OBB/h (~30초/OBB) | **~4 시간** | 깔끔한 PMI crop 우선 + 빠른 패스 |
| 표준 | ~70 OBB/h (~50초/OBB) | ~7 시간 | 일반 페이스 (Day 1 plan 정확) |
| 꼼꼼 | ~40 OBB/h (~90초/OBB) | ~12 시간 | 정밀 라벨링 (1.5일 소요) |

> 💡 **R&D 팁**: 500 장 라벨링 후 Stage 2 Version A 가 Roughness 검출 약하면 **Roughness 만 추가 50 장 라벨링** 권장 (전체 추가 라벨링보다 효율 ↑).

### 3.1 CVAT 설치 (로컬, ★ 권장)

```bash
# 별도 디렉터리에서
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
# → http://localhost:8080
```

또는 클라우드: <https://app.cvat.ai/> (free tier 가능)

### 3.2 입력 이미지 (View crop)

Roboflow 100장 라벨링 + Stage 1 학습 완료 후:

```bash
# 100장 도면 각각에서 View crop 추출
for img in $(head -100 dataset_seed_list.txt); do
    python src/stage1_layout.py crop \
        --image dataset/$img \
        --weights checkpoints/yolo_det.pt \
        --out-dir outputs/crops_for_stage2/
done

# View crop 들만 모아서 CVAT 업로드용 폴더 준비
mkdir -p outputs/cvat_stage2_input/
find outputs/crops_for_stage2/ -path '*/View/*.jpg' -exec cp {} outputs/cvat_stage2_input/ \;
```

이 폴더 (보통 200~300 View crop) 를 CVAT 에 업로드.

### ★ 2026-04-29 갱신 — PMI crop 사용 (D-034)

**View crop 대신 PMI crop 사용** — Stage 1 의 PMI 클래스 (D-028) 가 Stage 2 OBB 의 입력 영역으로 명확화 (D-034).

```bash
# extract_pmi_crops.py 사용 (이미 작성됨)
python src/extract_pmi_crops.py
# → outputs/cvat_stage2_input/ 에 PMI crops 직접 저장
# → 20장 도면에서 ~844 PMI crops 추출

# 500장만 분리 (Day 1 plan, conf 내림차순)
python3 << 'EOF'
import csv, shutil
from pathlib import Path

src = Path("outputs/cvat_stage2_input")
dst = Path("outputs/cvat_stage2_input_v1")
dst.mkdir(exist_ok=True)

with open(src / "manifest.csv", encoding="utf-8-sig") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: -float(r["conf"]))

for r in rows[:500]:
    src_f = src / r["crop_filename"]
    if src_f.exists():
        shutil.copy2(src_f, dst / r["crop_filename"])
print(f"✓ Top 500 (conf 내림차순) → {dst}")
EOF
```

→ `outputs/cvat_stage2_input_v1/` 의 500 PMI crops 를 CVAT 에 업로드.

### 3.3 CVAT Project 설정

```
Project Name: Stage2_Annotation_OBB
Labels:
   - Measure   (color: red,    shape: rectangle / rotbox)
   - GDT       (color: blue,   shape: rectangle / rotbox)
   - Roughness (color: green,  shape: rectangle / rotbox)
```

**라벨 추가 시**:
- Edit Labels → Add Label
- Label name: 정확히 `Measure` / `GDT` / `Roughness`
- Allow shapes: Rectangle (CVAT 의 rectangle 은 회전 가능)
- 또는 `Skeleton` 대신 `Rotated Box` 옵션이 있는 새 버전 사용

### 3.4 Task 생성

```
Name: Stage2_Seed_100
Source: outputs/cvat_stage2_input/ (zip 또는 폴더 업로드)
Subset: train (검수 후 자체 split)
```

### 3.5 클래스별 라벨링 기준 (★ Stage 2 OBB) — 2026-04-30 갱신

> **OBB (Oriented BBox)**: 회전된 사각형. 4점 좌표 (TL, TR, BR, BL).

#### 클래스 0: Measure (치수, 빨강)

**포함**:
- 일반 dimension: `25`, `100`, `Ø25`, `R5`, `M8`, `M8x1.25`
- 직경 / 반지름: `Ø25.4`, `⌀12`, `R0.5`, `2xR0.2`
- 공차 포함 (양/단방향): `25 ±0.05`, `25 +0.1/-0.05`, `25 +0.1/0`
- **공차 등급 포함**: `Ø25h6`, `Ø25H7`, `Ø12h6`, `M8 6H` ★
- 나사 / Tap: `M8x1.25`, `M12 thru`, `4-M5`, `M8 Tap thru`
- 곡률 / 경사: `R5`, `5°`, `15°`, `45°`
- 기타: `Ø60 PCD` (피치원), `25 BSC` (기본치수)
- **표준 참조 dimensions** ★: `KS A ISO 6411`, `A2/4.25 양끝` (중심구멍 사양)
- 화살표 + 보조선 + 숫자 + 단위 → **하나의 OBB 로 통째**

**라벨링 예시**:
```
회전 텍스트:                   axis-aligned 텍스트:
        ┌──────┐                  ┌──────────┐
        │ 25.4 │ ← 회전된 OBB     │   480    │ ← 0° OBB
        │      │                  └──────────┘
        └──────┘
        화살표 부분도
        함께 박스에
```

**박스 기준**:
- 화살표 끝부터 숫자 끝까지 OBB 길이 방향
- 텍스트 회전 각도 따라 OBB 회전
- 공차 (+0.05 / -0.05) 와 nominal 이 분리되어 있어도 **한 OBB**

#### 클래스 1: GDT (Feature Control Frame, 파랑)

**포함** — **사각 박스 (FCF) 가 있는 모든 형상/위치 공차**:

| 심볼 | 의미 | 영문 | 카테고리 |
|---|---|---|---|
| `⏤` | 직진도 | Straightness | 형상 |
| `⏥` (= ▱) | 평면도 | Flatness | 형상 |
| `○` | 원형도 | Roundness | 형상 |
| `⌭` | 원통도 | Cylindricity | 형상 |
| `⌒` | 선형도 | Profile of any line | 윤곽 |
| `⌓` | 면형도 | Profile of any surface | 윤곽 |
| `⫽` | 평행도 | Parallelism | 자세 |
| `⊥` | 직각도 | Perpendicularity | 자세 |
| `∠` | 경사도 | Angularity | 자세 |
| `⊕` | 위치도 | Position | 위치 |
| `◎` | 동심도 | Concentricity | 위치 |
| `⏤` (수평 두 줄) | 대칭도 | Symmetry | 위치 |
| `↗` | 원주 흔들림 | Circular runout | 흔들림 |
| `↗↗` | 전 흔들림 | Total runout | 흔들림 |

**라벨링**:
```
┌─────┬──────────┬───┬───┐
│  ⏤  │ ⌀0.05Ⓜ  │ A │ B │  ← 이 사각형 전체를 하나의 OBB로
└─────┴──────────┴───┴───┘

┌─────┬───────┬───┐
│  ⊥  │  0.3  │ X │  ← 직각도 0.3 (데이텀 X)
└─────┴───────┴───┘
```

**박스 기준**:
- FCF 사각 박스 통째 (심볼 + 값 + 데이텀 컴파트먼트 모두 포함)
- 컴파트먼트 분리하지 않음 (D-015)
- **스택된 FCF** (위/아래 결합, 같은 feature 의 다중 공차) → 한 OBB 통째 권장

#### 클래스 2: Roughness (표면거칠기, 초록)

**포함**:
- ▽ ▽▽ ▽▽▽ 심볼 (구식)
- ISO 1302 표면거칠기 표기 (신식)
- `Ra 1.6`, `Ra 3.2`, `Ra 6.3`, `Ra 12.5`
- `Rz 6.3`, `Rmax 25`
- 제거가공 표기 (`Ra 0.8 ✓`)

**예시**:
```
   ▽
  Ra 1.6   ← 심볼 + 수치 통째 OBB
```

**박스 기준**: 심볼 + 값 (단위 μm 포함) — 한 OBB.

---

### 3.5.1 ★ 모호 케이스 처리 룰 (2026-04-30 신규)

라벨링 시 자주 만나는 **PMI 가 아닌 / 모호한 케이스** 처리 가이드.

#### Rule A — FCF 박스 보이면 GDT, 안 보이면 skip

| 케이스 | 처리 |
|---|---|
| `⊥ 0.3 X` (사각 박스 명확) | **GDT** ✅ |
| `0.009 A` + 화살표 (FCF 박스 없음 / 잘림) | **skip** (불완전한 PMI crop) |

#### Rule B — 표준 참조 dimensions = Measure

| 케이스 | 처리 |
|---|---|
| `KS A ISO 6411 / A2/4.25 양끝` (중심구멍 사양) | **Measure** ✅ |
| 그 외 ISO/KS/ASME 표준 참조 + 숫자 조합 | **Measure** |

#### Rule C — View 라벨 / 단면 마커 / 상세도 = skip

| 케이스 | 처리 |
|---|---|
| 단면 마커 박스 (`A`, `B`, `C` 사각 박스) | **skip** (View 영역) |
| `A-A`, `B-B` 단면 라인 라벨 | **skip** |
| `상세도 B (2:1)` 텍스트 | **skip** |
| `정면도`, `측면도`, `평면도` | **skip** |

#### Rule D — 부품번호 풍선 = skip

| 케이스 | 처리 |
|---|---|
| ⓪ ① ② 풍선 (조립도면) | **skip** (assembly 영역) |

#### Rule E — Notes / 일반 텍스트 = skip

| 케이스 | 처리 |
|---|---|
| "1. UNLESS OTHERWISE SPECIFIED" 등 | **skip** (Stage 3-A) |
| "ALL DIMENSIONS IN MM" | **skip** |
| 도면 작성자 / 날짜 / 회사명 | **skip** (TitleBlock 영역) |

---

### 3.5.2 ★ 정밀도 / 공차 / 평행도 처리 매트릭스

| 표기 종류 | 예시 | 클래스 |
|---|---|---|
| dimension + 양방향 공차 | `25 ±0.05` | **Measure** |
| dimension + 단방향 공차 | `25 +0.1/0` | **Measure** |
| dimension + 공차 등급 | `Ø25h6`, `M8 6H`, `Ø12h6` | **Measure** |
| dimension + 표면거칠기 결합 | `Ø25 / Ra 1.6` | **Measure + Roughness** (2개 OBB) |
| **평면도** (FCF 박스 ⏥) | `⏥ 0.05` | **GDT** |
| **직진도** (FCF 박스 ⏤) | `⏤ 0.02` | **GDT** |
| **평행도** (FCF 박스 ⫽) | `⫽ 0.1 A` | **GDT** |
| **직각도** (FCF 박스 ⊥) | `⊥ 0.3 X` | **GDT** |
| **위치도** (FCF 박스 ⊕) | `⊕ Ø0.1 A B C` | **GDT** |
| **흔들림** (FCF 박스 ↗) | `↗ 0.05 A` | **GDT** |
| **원형도** (FCF 박스 ○) | `○ 0.02` | **GDT** |
| 표면거칠기 | `Ra 1.6`, `▽▽▽` | **Roughness** |
| 단순 ISO/KS 표준 참조 | `KS A ISO 6411` | **Measure** |
| FCF 박스 잘림 / 부분 만 | (예: 0.009 A 화살표만) | **skip** |
| View 라벨 | `A-A`, `상세도 B` | **skip** |

---

### 3.5.3 ★ 빠른 결정 트리 (라벨링 시 즉시 사용)

```
                ┌─────────────────────┐
                │ 새 PMI crop 보임    │
                └─────────────────────┘
                         │
            ┌────────────┴────────────┐
            │ 사각 박스 (FCF) 보임?   │
            └────────────┬────────────┘
                YES      │     NO
                ↓        │      ↓
        ┌──────────────┐ │  ┌──────────────────┐
        │ 박스 안 심볼? │ │  │ ▽ 또는 Ra/Rz?    │
        │ ⊥⫽⏥⏤⊕↗○⌭ │ │  └──────┬───────────┘
        └──────┬───────┘ │     YES│      NO
            YES│         │        ↓        ↓
              ↓          │   ┌─────────┐  ┌─────────────────────┐
        ┌──────────┐     │   │Roughness│  │ 숫자 + 단위 + 화살표? │
        │   GDT ★  │     │   └─────────┘  └─────┬───────────────┘
        └──────────┘     │                    YES│      NO
                         │                       ↓        ↓
                         │                  ┌─────────┐  ┌────────┐
                         │                  │Measure ★│  │  skip  │
                         │                  └─────────┘  └────────┘
```

---

### 3.5.4 자주 만나는 케이스 빠른 분류

| 케이스 (Stage 1 PMI 박스 안) | 클래스 |
|---|---|
| `Ø25` 직경 | Measure |
| `Ø25 ±0.05` 직경+공차 | Measure |
| `Ø25h6` 직경+공차등급 | Measure |
| `R5` 반지름 | Measure |
| `M8x1.25` 나사 | Measure |
| `25` 일반 치수 | Measure |
| `5°` 각도 | Measure |
| `KS A ISO 6411 / A2/4.25` 표준 참조 | Measure |
| `⊥ 0.3 X` 직각도 (FCF) | **GDT** |
| `⫽ 0.1 A` 평행도 (FCF) | **GDT** |
| `⏥ 0.05` 평면도 (FCF) | **GDT** |
| `⊕ Ø0.1 A B C` 위치도 (FCF) | **GDT** |
| `Ra 1.6` 표면거칠기 | **Roughness** |
| `▽▽▽` 거칠기 심볼 | **Roughness** |
| 단면 마커 (A, B, C) 박스 | skip |
| "상세도 B (2:1)" | skip |
| 부품번호 풍선 | skip |
| Notes 텍스트 | skip |
| FCF 박스 잘림 | skip |
| 빈 / 비-PMI crop | skip |

#### Rule O — stage1_fp_notes 의 중요성 (D-038, 신규 2026-05-02)

**일반 주석 (재질/가공/공차 등) 영역이 PMI 로 오검출된 경우 처리 규칙**

**상황**:
- Stage 1 Version A 가 일반 주석 텍스트 (Notes 클래스) 를 PMI 로 잘못 검출
- 예: `材料は鉄かSUS403` (재질 명세), `+0.1以下のものは機械加工のこと` (가공 지시)
- SKIP 하지 않으면 메타데이터 JSON 에서 정보 누락

**라벨링 규칙**:
- **reason attribute**: **`stage1_fp_notes`** (정확히 이 이름만, stage1_fp_other 금지)
- CVAT SKIP 라벨에 reason attribute 로 이 값을 선택
- "SKIP 라벨 추가" → reason 선택 → **stage1_fp_notes** 클릭

**절대 금지**:
- ❌ stage1_fp_other 로 처리하면 rescue 누락 위험
- ❌ 단순 skip 만 하고 attribute 없으면 자동 분류 안 됨

**라벨링 후 자동 흐름**:
```
1. extract_skip_list.py 실행
   → outputs/skip_lists/stage1_fp_notes.txt 생성 (47개)

2. rescue_misclassified_notes.py 실행
   → Donut zero-shot OCR 로 텍스트 추출
   → outputs/rescued_notes.json 생성

3. pipeline.py / stage4 merger
   → 최종 JSON 의 'general_notes' 필드로 병합
   → 메타데이터 완성 ✅
```

**적용 케이스 예시** (D-038 발견 케이스):
- **재질 명세**: `材料は鉄かSUS403`, `MATERIAL: SS400`
- **가공 지시**: `機械加工のこと`, `MACHINING REQUIRED`
- **일반 공차**: `UNLESS OTHERWISE SPECIFIED ±0.1`, `±0.05`
- **검사 기준**: `CHECK ALL DIMENSIONS`, `INSPECT BEFORE SHIPMENT`
- **표면 처리**: `PAINT`, `COATING SPEC`

**중요성**:
- 메타데이터 JSON 의 **필수 항목** (general_notes 필드)
- 누락 시 도면 정보 손실 → 다운스트림 enrichment 불가
- rescue 비용 (~5초/crop, Donut OCR) 가 정보 손실보다 훨씬 저렴

**★ 1차 시도 결과 (Donut DocVQA, 2026-05-03 — 실패)**:
- 표면 통계 23/23 success (에러 없음)
- 실질 품질 4% (1/23) — 단일 문자/환각/부분 추출 다수
- 원인: 다국어 미스매칭 (영어 모델 vs 일본어 노트) + DocVQA 모델 부적합
- **결과 폐기** (환각 텍스트 메타데이터 오염 방지)

**★ 2차 시도 (D-039, 2026-05-03 박제) — PaddleOCR-VL-1.5 채택**:
- Stage 3-A 모델 변경: Donut DocVQA → **PaddleOCR-VL-1.5**
- Day 3 적용 시 D-038 Notes Rescue 도 동일 백엔드 사용
- 채택 사유 8가지 (OmniDocBench 94.50%, 0.9B 경량, Table/Formula 명시 metric, Seal 인식, CJK SOTA 등)
- 사전 검증 (Day 2 학습 백그라운드): 사용자 샘플 한/일/중/영/러시아어 zero-shot 추출

**★ Rescue 범위 명확화 (사용자 결정, 2026-05-03)**:
- **stage1_fp_notes (23개) 만 Rescue** (Option α 채택)
- **stage1_fp_table (13개) 는 Rescue 안 함** — 정보 가치 없음 (PMI 로 false positive 한 표제란 일부)
- **정상 표제란/BOM/도장 등은 Stage 1 의 `Table` 클래스 자체 검출 결과** → Stage 3-A 직접 입력 (논문 §4.3 정합)

**★ Phase 8 SKIP-only frame 처리 (2026-05-03 결정)**:
- **Option B 채택**: SKIP-only frame 통째로 제외
- **적용 범위 = Stage 2 학습 데이터 (`data/annotation/`) 에서만**
- 이미지는 `outputs/cvat_stage2_input_v3_upscaled/` 에 그대로 보존
- stage1_fp_notes 23개는 별도 Rescue 경로 (Stage 3-A 입력)

**참고 문서**:
- `docs/modules/extract_skip_list.md` — SKIP 라벨 분석
- `docs/modules/rescue_misclassified_notes.md §8` — Donut 한계 + 차후 대안
- `PROJECT_HANDOFF.md §11.38` — D-038 전체 정책 (1차 결과 박제)
- **`PROJECT_HANDOFF.md §11.39` — D-039 Stage 3-A 모델 채택 (PaddleOCR-VL-1.5)**
- `history.md §A.11.8` — Day 2 Donut DocVQA 시도 + 실패 분석
- **`history.md §A.11.9` — D-039 모델 선정 이유 + 비교 + 하이브리드 아키텍처**

### 3.6 OBB 라벨링 룰

- **회전 각도**: 텍스트 baseline 따라 정확히
- **4점 순서**: CVAT 가 자동 처리 (TL → TR → BR → BL)
- **너무 작은 OBB**: 면적 < 50 px² 는 의심 (라벨링 실수 가능성)
- **겹침**: 같은 View 안의 다른 OBB 끼리 겹쳐도 OK (어노테이션 밀집 자연스러움)

### 3.7 Export

```
Menu > Export task dataset
Format: "YOLO Oriented Bounding Boxes 1.0"
또는: "Datumaro" / "CVAT XML"
```

→ ZIP 안에 `labels/<stem>.txt` (9 필드 형식) 또는 CVAT XML.

---

## 4. 자동 라벨링 — 옵션 2 + 옵션 3 하이브리드

### 4.1 설계 원칙

**옵션 2 (학습된 모델)** + **옵션 3 (eDOCr2 휴리스틱)** 을 합칩니다:

```
For each unlabeled image:
    1. 학습된 YOLO 로 1차 예측 (conf 낮춰서 recall ↑)
    2. eDOCr2 휴리스틱으로 보조 예측 (TitleBlock / Frame)
    3. NMS + IoU merge: ML 결과 우선, ML 못 잡은 영역만 휴리스틱 추가
    4. 신뢰도별 분류:
        - high conf (>0.7) → "auto_verified" 태그
        - mid (0.4~0.7) → "auto_review" 태그 (사람 우선 검수)
        - low (<0.4) → 제외 또는 "auto_low" 태그
    5. CVAT XML / Roboflow YOLO 형식으로 export
```

### 4.2 작성할 3개 스크립트

#### `src/auto_label_stage1.py` — Stage 1 자동 라벨

```
입력:
    - --weights checkpoints/yolo_det.pt   (학습된 모델, 옵션)
    - --use-heuristic                     (eDOCr2 fallback 사용)
    - --input  dataset/                    (라벨링할 이미지 폴더)
    - --output data/layout/labels/auto/    (출력 라벨)
    - --format yolo|cvat|roboflow         (export 형식)
    - --conf-high 0.70                    (auto_verified 임계값)
    - --conf-low 0.40                     (제외 임계값)
    - --skip-existing                      (이미 라벨된 이미지 건너뜀)

처리:
    1. weights 있으면 YOLO predict
    2. --use-heuristic 옵션 시 eDOCr2 휴리스틱 도 실행
    3. NMS merge
    4. 신뢰도 태깅
    5. format 별 export

출력:
    yolo:      data/layout/labels/auto/<stem>.txt
    cvat:      cvat_annotations.xml (CVAT pre-annotation import용)
    roboflow:  roboflow_predictions.json (Roboflow upload format)
```

#### `src/auto_label_stage2.py` — Stage 2 자동 OBB

```
입력:
    - --det-weights checkpoints/yolo_det.pt   (Stage 1, View crop 추출용)
    - --obb-weights checkpoints/yolo_obb.pt    (Stage 2, 학습된 OBB 모델)
    - --input dataset/                         (원본 도면 폴더)
    - --output data/annotation/labels/auto/    (출력)
    - --format cvat|yolo_obb
    - --conf-high 0.70
    - --conf-low 0.40

처리:
    1. Stage 1 으로 View 영역 검출 + crop
    2. View crop 위에서 Stage 2 OBB 예측
    3. (옵션) eDOCr2 CRAFT-style 텍스트 검출 보조
    4. OBB 좌표를 원본 도면 좌표로 환산 (obb_local_to_global)
    5. CVAT XML (rotated rectangle) 또는 YOLO obb 형식 export

출력:
    cvat:    cvat_obb_annotations.xml
    yolo:    data/annotation/labels/auto/<stem>.txt (9 필드)
```

#### `src/edocr2_heuristic_label.py` — eDOCr2 OpenCV 휴리스틱

```
입력:
    - --input <jpg path>
    - --output-format yolo|cvat
    - --target {stage1|stage2|both}

처리 (eDOCr2 §3.1 기반):
    1. Frame detection
        - Vertical/horizontal line detection (directional kernels)
        - Outlier peaks → frame candidates
        - 가장 안쪽 frame 선택 (이중 프레임 대응)
    2. Rectangle hierarchy detection (Stage 1)
        - Contour detection + polygon filter (4-side)
        - 면적 기반 클러스터링:
            - 큰 사각형 → TitleBlock 후보
            - 가장 큰 영역 → View
            - 텍스트 밀집 영역 → Notes
        - 클래스 추정 (heuristic):
            - 우하단 + 그리드 라인 많음 → TitleBlock
            - 가운데 + 큰 면적 → View
            - 직선형 텍스트 + 작은 영역 → Notes
    3. (Stage 2) Text region detection
        - CRAFT 모델 (ultralytics 와 별도)
        - 또는 OpenCV connected components + 형태학
        - 회전 각도 추정 (minAreaRect)
        - 클래스 추정:
            - GD&T 프레임 형상 (사각형 + 컴파트먼트) → GDT
            - 꺽쇠 심볼 + 숫자 → Roughness
            - 그 외 텍스트 → Measure

출력:
    yolo:  Stage 1 → axis-aligned BBox
           Stage 2 → 9-field OBB
    cvat:  CVAT XML
```

### 4.3 하이브리드 메인 스크립트

세 스크립트가 통합된 진입점:

```bash
# Stage 1 자동 라벨링 (학습된 모델 + 휴리스틱 둘 다)
python src/auto_label_stage1.py \
    --weights checkpoints/yolo_det.pt \
    --use-heuristic \
    --input dataset/ \
    --output data/layout/labels/auto/ \
    --format roboflow \
    --conf-high 0.70 --conf-low 0.40

# Stage 2 자동 라벨링
python src/auto_label_stage2.py \
    --det-weights checkpoints/yolo_det.pt \
    --obb-weights checkpoints/yolo_obb.pt \
    --use-heuristic \
    --input dataset/ \
    --output data/annotation/labels/auto/ \
    --format cvat
```

### 4.4 Pre-annotation Import

#### Roboflow (Stage 1)
1. Project 의 "Upload" → "Pre-annotated images" 옵션
2. ZIP 으로 묶어 업로드: 이미지 + `<stem>.txt` (YOLO format) + `data.yaml`
3. Roboflow 가 자동으로 BBox 시각화
4. 사람이 잘못된 박스 수정 → 다시 export

#### CVAT (Stage 2)
1. Task 생성 후 "Upload annotations" → CVAT XML
2. 자동 박스 표시됨
3. 사람이 OBB 회전 각도 / 위치 수정
4. Export 시 검수 완료된 라벨

### 4.5 신뢰도 태깅 활용

각 박스의 `confidence` 를 CVAT label attribute 로 export:

```xml
<box label="View" ... confidence="0.85" auto_verified="true">
```

CVAT 에서 필터링:
- `auto_verified=true` 박스만 표시 → 빠르게 OK / 거부 판단
- `auto_review=true` 박스만 따로 표시 → 정밀 검수 필요한 것만 확인

### 4.6 Active Learning 반복

```
Iter 1: 100장 수동 → 학습 → 4,487장 자동 → 사람 검수
Iter 2: (옵션) 검수된 1,000장 추가 학습 → 나머지 3,487장 재예측 → 검수
Iter 3: 전체 학습 → V2-B / V3-B 검증
```

각 iteration 마다 모델 정확도 향상 → 다음 iteration 검수 부담 감소.

---

## 5. eDOCr2 휴리스틱의 장점·한계

### 장점
- **학습 데이터 필요 없음** — 0번째 도면도 즉시 라벨 생성 가능
- **결정적** — 같은 입력에 항상 같은 출력
- **TitleBlock 검출에 강함** — 격자 라인 검출이 정확

### 한계
- **클래스 분류 부정확** — Frame 은 잘 잡지만 View vs Notes 구분 약함
- **회전 텍스트 약함** — Stage 2 의 회전 OBB 는 정확도 낮음
- **GD&T 심볼 인식 안 됨** — 일반 텍스트와 구분 못 함

→ Stage 1 의 TitleBlock 라벨에는 휴리스틱이 효과적, View / Notes / OBB 는 ML 모델 의존도 ↑.

---

## 6. 시간 추정 (2026-04-29 실측 기준 갱신)

### Stage 1 (✅ 완료)

| 단계 | 실제 소요 | 비고 |
|---|---|---|
| Roboflow seed 100장 라벨링 | ~10 시간 | 사용자 작업 |
| Stage 1 Version A 학습 | **28.5 분** | 50 epochs / imgsz 1280 / RTX 5080 cu128 |
| auto_label_stage1.py (5,839장) | **5분 45초** | 19 img/s |
| sort_by_yolo_pmi.py (분류) | ~3 분 | 5,839장 텍스트 파싱 |
| 사용자 검수 + exclude_groups | ~30분 + 9초 | 18 group / 46 files |
| **Stage 1 합계** | **약 11 시간** | (Pre-annotation 보류 = D-035) |

### Stage 2 (⏳ 진행 중)

| 단계 | 예상 / 실제 | 비고 |
|---|---|---|
| extract_pmi_crops.py (20 도면) | **14 초** | 844 PMI crops |
| **CVAT seed 라벨링 (500장)** | **~4~7 시간** | Day 1 plan ★ (적극 4h / 표준 7h) |
| Stage 2 Version A 학습 | ~5 시간 | yolo11m-obb / 100 epochs / Day 2 |
| V3-B 검증 | ~10 분 | ★ D-023 누락률 |
| auto_label_stage2.py 작성 + 실행 | ~25분 | 344 추가 PMI crops |
| 사람 검수 (Day 3) | ~2시간 | priority 순 |
| Stage 2 Version B 학습 (시간 여유 시) | ~5 시간 | 차후 |
| **Stage 2 Active Learning Phase 1+2 합계** | **~13~20 시간** | (Day 1~3) |

### 라벨링 장수별 vs 시간 (Stage 2 Phase 1)

| 라벨링 장수 | 평균 OBB | 라벨링 시간 (30초/OBB) | V3-B 통과 가능? |
|---|---|---|---|
| 100 장 | ~100 | ~50 분 | 🔴 critical 미달 |
| 200 장 | ~200 | ~1.7 시간 (적극) / ~5 시간 (꼼꼼) | 🟡 부분 |
| 300 장 | ~300 | ~2.5 시간 (적극) / ~7 시간 | 🟢 가능 |
| **500 장 (★ Day 1 채택)** | **~500** | **~4 시간 (적극) / ~12 시간 (꼼꼼)** | ✅ **안정** |
| 844 장 (전체) | ~844 | ~7 시간 (적극) / ~25 시간 | ✅ 권장 |

> ★ 3일 plan 기준 — **500 장 + auto_label_stage2 + 검수** = Active Learning 효과로 600+ 장 효과 달성. D-023 critical 통과 안정.
| **총합** | **470h** | **~65h** | 1인 8일 |

---

## 7. 다음 액션

이 manual 의 §4 (자동 라벨링 스크립트) 를 실제로 작성하려면:

| 스크립트 | 우선순위 | 시점 |
|---|---|---|
| `src/auto_label_stage1.py` | 높음 | Stage 1 seed 학습 후 |
| `src/auto_label_sta