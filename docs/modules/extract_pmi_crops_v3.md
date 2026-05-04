# `src/extract_pmi_crops_v3.py`

> **Stage 2 입력 준비 — aspect-aware adaptive padding 버전** (D-037 확장) — 45° 회전 텍스트 보강

## 1. 구현 요약

v2 (per-axis padding) 의 한계를 보완한 aspect-aware 적응형 패딩. **정사각형 bbox (aspect < 1.5)** 에는 uniform 큰 padding 을 적용하여 45° 회전 텍스트의 대각선 화살표/리더선을 더 잘 캡처. **비정사각형 bbox (aspect ≥ 1.5)** 는 v2 의 per-axis 로직을 그대로 유지하여 인접 치수 침입을 최소화.

**워크플로 (v2 → v3 비교 선택)**

```
outputs/stage2_input_drawings.txt (20개 도면)
        ↓
Stage 1 Version A 모델 inference
        ↓
v2: extract_pmi_crops.py → outputs/cvat_stage2_input_v2/ (844 crops)
        OR
v3: extract_pmi_crops_v3.py → outputs/cvat_stage2_input_v3/ (844 crops)
        ↓
CVAT 업로드 → 시각 비교 → 더 좋은 버전 선택
        ↓
Stage 2 OBB 라벨링
```

**핵심 컴포넌트** (~483 lines)

| 함수 | 역할 |
|---|---|
| `calc_padding_v3(bbox_w, bbox_h, args)` | aspect-aware padding 계산 (square_diagonal vs per_axis 분기) |
| `load_drawings(args)` | 도면 목록 로드 |
| `imread_unicode(path)` / `imwrite_unicode(path, img)` | Windows 한글 호환 I/O |
| `main()` | argparse + YOLO inference + v3 padding 적용 + manifest CSV |

**1개 CLI 서브커맨드**

```bash
python src/extract_pmi_crops_v3.py [--aspect-threshold 1.5] [--padding-ratio-square 0.6] ...
```

## 2. 핵심 설계 결정: v2 vs v3

### 문제점 (v2 한계)

v2 의 per-axis padding 은 axis-aligned 계산이므로 **45° 회전 텍스트의 대각선 화살표를 온전히 캡처하지 못함**.

```
정사각형 bbox (w ≈ h)에 per-axis padding 적용:
┌─────────────┐
│   ┌─────┐   │  ← padding 적용 후에도
│   │┌───┐│   │     대각선 화살표 끝이
│   │└─╱─┘│   │     bbox 경계 바깥에
│   └─────┘   │
└─────────────┘

반면 diagonal 화살표는:
┌─────────────┐
│     ╱───╲   │  ← 45° 화살표
│   ┌╱─────╲┐ │
│   │       │ │
│   └╲─────╱┘ │
│     ╲───╱   │
└─────────────┘
→ uniform (큰) padding 필요
```

### v3 해결책 (aspect-aware)

**Aspect ratio = long_side / short_side** 를 기준으로 분기:

1. **정사각형 (aspect < 1.5)** → `square_diagonal` 전략
   - uniform pad = long_side × ratio_square (기본 0.6)
   - 대각선 화살표의 모든 끝점 포함 가능

2. **가로/세로형 (aspect ≥ 1.5)** → `per_axis` 전략 (v2 동일)
   - pad_x = bbox_w × ratio (기본 0.4)
   - pad_y = bbox_h × ratio (기본 0.4)
   - 인접 치수 침입 최소화 유지

### 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `--aspect-threshold` | 1.5 | long/short < 1.5 이면 정사각형 판정 |
| `--padding-ratio` | 0.4 | 비정사각형 (per-axis) 비율 |
| `--padding-ratio-square` | 0.6 | 정사각형 (uniform) 비율 — 더 큼 (0.6 > 0.4) |
| `--padding-min` | 30 | 최소 padding px |
| `--padding-max` | 80 | 최대 padding px |

### 의사결정

| 항목 | v2 (per-axis) | v3 (aspect-aware) | 선택 |
|---|---|---|---|
| 정사각형 화살표 | 일부 잘림 (80% 만족) | uniform pad 로 개선 (45° 보강) | v3 ✅ |
| 가로/세로형 | 최적화됨 (90% 만족) | 동일 로직 유지 | v3 ✅ |
| 복잡도 | 단순 | 약간 복잡 (분기) | v3 OK |
| 라벨링 룰 필요 | 회전 잘림 20% | 회전 잘림 감소 | v3 이득 |

## 3. 사용법

### CLI

```bash
# 기본 (aspect-aware, default threshold 1.5, ratio_square 0.6)
python src/extract_pmi_crops_v3.py

# 정사각형 판정 기준 조정 (1.5 → 1.3 더 엄격)
python src/extract_pmi_crops_v3.py --aspect-threshold 1.3

# 정사각형 padding 비율 튜닝 (0.6 → 0.5 로 축소)
python src/extract_pmi_crops_v3.py --padding-ratio-square 0.5

# v2 와 다른 폴더로 출력 (비교용)
python src/extract_pmi_crops_v3.py --output outputs/cvat_stage2_input_v3_test

# v2 와 같은 세팅으로 (per-axis only, aspect-aware OFF — 거의 v2 와 동일)
python src/extract_pmi_crops_v3.py --aspect-threshold 999.0
```

### v2 vs v3 비교 실험

```bash
# v2 실행
python src/extract_pmi_crops.py
# → outputs/cvat_stage2_input_v2/ (844 crops)

# v3 실행
python src/extract_pmi_crops_v3.py
# → outputs/cvat_stage2_input_v3/ (844 crops)

# 결과 비교 (manifest 통계 확인)
# outputs/cvat_stage2_input_v2/manifest.csv → strategy=per_axis (모두 동일)
# outputs/cvat_stage2_input_v3/manifest.csv → strategy=square_diagonal|per_axis 혼재
```

## 4. 출력 형식

### 4.1 디렉터리 구조

```
outputs/cvat_stage2_input_v3/      ← v3 (aspect-aware, D-037 확장)
├── DwgFoo__PMI_000.jpg
├── DwgFoo__PMI_001.jpg
├── ...
└── manifest.csv                    ← + aspect_ratio, padding_strategy 컬럼
```

### 4.2 manifest.csv 예시 (v3 추가 컬럼)

```csv
crop_filename,source_drawing,source_group_key,pmi_idx,bbox_x1,bbox_y1,bbox_x2,bbox_y2,bbox_w,bbox_h,aspect_ratio,padding_strategy,pad_x,pad_y,crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h,padding_mode,conf
11_jpeg.rf.8b46c563__PMI_000.jpg,11_jpeg.rf.8b46c563.jpg,11_jpeg,0,150,200,200,250,50,50,1.00,square_diagonal,30,30,120,170,230,280,110,110,adaptive,0.9234
11_jpeg.rf.8b46c563__PMI_001.jpg,11_jpeg.rf.8b46c563.jpg,11_jpeg,1,400,450,520,550,120,100,1.20,square_diagonal,60,60,340,390,580,610,240,220,adaptive,0.8912
11_jpeg.rf.8b46c563__PMI_002.jpg,11_jpeg.rf.8b46c563.jpg,11_jpeg,2,600,400,800,450,200,50,4.00,per_axis,30,20,570,380,830,470,260,90,adaptive,0.7654
...
```

**v3 신규 컬럼**:

| 컬럼 | 의미 |
|---|---|
| `bbox_w`, `bbox_h` | PMI bbox 가로/세로 (픽셀) |
| `aspect_ratio` | long_side / short_side (예: 1.00=정사각형, 4.00=매우 긴 가로형) |
| `padding_strategy` | `square_diagonal` (정사각형, uniform pad) \| `per_axis` (비정사각형, v2 로직) \| `fixed` (고정) |

### 4.3 Console 출력 형식 (라벨링 완료 후 실측치 기록 예정)

```
INFO    Output: outputs/cvat_stage2_input_v3/
INFO    Padding mode: aspect-aware [v3, D-037]
INFO      - per-axis ratio       : 0.40 (가로/세로형)
INFO      - square ratio         : 0.60 (정사각형, 45° 회전 보강)
INFO      - aspect threshold     : 1.50 (long/short)
INFO      - min/max padding      : 30 / 80 px

Extracting PMI v3: 100%|████| 20/20 [00:XX<00:00, X.XXs/img, crops=844, ...]

============================================================
PMI crop 추출 완료 (v3 aspect-aware)
  Drawings processed   : 20
  PMI crops saved      : 844 (v2 와 동일 입력으로 동일 수량)
  Small PMI skipped    : 16 (area < 100 px²)
============================================================

  pad_x (가로 padding) : min=30 / max=80 / mean=?? px        ← 실측 후 기록
  pad_y (세로 padding) : min=30 / max=80 / mean=?? px        ← 실측 후 기록
  aspect ratio        : min=1.00 / max=??? / mean=??         ← 실측 후 기록
  Strategy 분포        : square_diagonal=??? / per_axis=??? / fixed=0  ← 실측 후 기록
  square_diagonal pad : min=30 / max=?? / mean=?? px         ← 실측 후 기록

[다음 단계 — Stage 2 OBB 라벨링]
  1. v2 와 시각 비교
  2. v3 가 더 나으면 → ZIP + CVAT 업로드
  3. CVAT Task: Stage2_PMI_v3_844
```

> **참고**: v2 (per-axis only) 실측치는 pad_x mean=33.2, pad_y mean=30.6, max=44 px,
> 형태 분포 = 가로형 90 / 세로형 61 / 정사각형 693 (전체 844 / Small skipped 16).
> v3 의 strategy 분포 통계는 사용자 라벨링 진행 중이므로 후속 갱신 예정.

## 5. 의존성

```
ultralytics>=8.3.0
opencv-python>=4.10.0
numpy>=1.26.0
tqdm>=4.66.0
torch (CUDA 12.8)
```

## 6. 관련 의사결정

- **D-024** Group-aware split (manifest 기록)
- **D-028** 5 클래스 (PMI = cls 1)
- **D-029** Roboflow 매핑
- **D-034** Hierarchical (PMI = Stage 2 입력)
- **D-036** 옵션 B (auto_pass + review priority 만)
- **D-037** Adaptive padding
  - v2 (per-axis only): `pad_x = bbox_w × 0.4`, `pad_y = bbox_h × 0.4`
  - **v3 (aspect-aware)**: 정사각형 → uniform `pad = long × 0.6`, 비정사각형 → per-axis v2 동일

## 7. 비교 표: v2 vs v3

| 항목 | v2 | v3 |
|---|---|---|
| **파일명** | `extract_pmi_crops.py` | `extract_pmi_crops_v3.py` |
| **출력 폴더** | `outputs/cvat_stage2_input_v2/` | `outputs/cvat_stage2_input_v3/` |
| **정사각형 처리** | per-axis (pad_x, pad_y 동일) | uniform 큰 pad (0.6 배) |
| **비정사각형 처리** | per-axis (v2 로직) | per-axis (v2 동일) |
| **manifest 컬럼** | pad_x, pad_y, padding_mode | + aspect_ratio, padding_strategy |
| **회전 텍스트 보강** | 80% 만족 | 개선 예상 (uniform pad) |
| **라벨링 필요** | §3.5 회전 잘림 룰 | 일부 감소 |

## 8. 차후 개선

- [ ] aspect-threshold 최적값 재측정 (1.5 대신 1.3?)
- [ ] 회전 텍스트 비율 측정 후 v3 효과 정량화
- [ ] v3 라벨링 완료 후 v2 vs v3 mAP@0.5 비교
- [ ] v2, v3 외 추가 버전 검토 (예: 4분면 padding)

## 9. 다음 단계

1. **v2 먼저 실행** — `python src/extract_pmi_crops.py`
2. **v3 실행** — `python src/extract_pmi_crops_v3.py`
3. **manifest 비교** — `cvat_stage2_input_v2/manifest.csv` vs `v3/manifest.csv`
   - strategy 분포 확인 (square_diagonal 몇 %)
   - pad 통계 비교 (평균, 최대값)
4. **CVAT 시각 비교** — 정사각형 bbox 의 회전 텍스트 캡처 확인
5. **더 좋은 버전 선택** → CVAT Task 생성 + 라벨링
6. **라벨링 진행** — Stage2_PMI_v3_844 task (또는 v2 선택)
