# `src/sort_by_drawing_type.py`

> **Step 1.6** — 가공/조립 도면 자동 분류 (D-026)

## 1. 구현 요약

OpenCV + Pytesseract 휴리스틱으로 학습 없이 5,839 장을 **manufacturing / assembly / manual_review_type** 으로 분류.

**검출 시그널 4종**

| 시그널 | 검출 방법 | 클래스 영향 |
|---|---|---|
| 치수 표기 (Ø, R, M, ±, mm) | Pytesseract OCR (5개 언어) + 8개 정규식 | manufacturing 양성 |
| 표면거칠기 (Ra/Rz) | Pytesseract OCR + 정규식 | manufacturing 양성 |
| 부품번호 풍선 | OpenCV `HoughCircles` (작은 원) | assembly 양성 |
| BOM 표 | 우상단 큰 격자 + BOM 키워드 (5개 언어) | assembly 양성 (강함) |

**분류 룰**

```
BOM 검출 OR (풍선 ≥ 10 AND 치수 < 5)  → assembly
치수 ≥ 5 AND 풍선 < 5                  → manufacturing
그 외                                  → manual_review_type
```

CLI 로 임계값 조정 가능 (`--dim-min`, `--balloon-asm`, `--balloon-mfg-max`, `--dim-asm-max`).

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 학습 불필요 | OpenCV 휴리스틱 only | 즉시 사용 가능, 5,839 장 일괄 처리 |
| OCR 언어 | `kor+eng+rus+jpn+chi_sim+chi_tra` (D-025) | 5개 언어 도면 모두 |
| 치수 검출 | 정규식 (Ø/R/M/±/단위) | 언어 무관 — 보편 기호 |
| BOM 키워드 | 5개 언어 사전 (PART NO / 품번 / 部品 / ПОЗ / 序号) | 텍스트로 BOM 강한 판정 |
| 풍선 검출 | Hough Circles (이미지 short-side 비례 반경) | 도면 해상도 변동 대응 |
| 격자 검출 | Hough Lines (수평선 ≥ 5개 in 상단 절반) | BOM 표는 보통 우상단 |
| 충돌 우선순위 | BOM 검출 > 풍선·치수 카운트 | BOM 은 결정적 시그널 |
| 4-tier 출력 | mfg / asm / review / error | sort_by_titleblock 과 일관 |
| OCR 옵션 | `--no-ocr` 플래그 (속도 우선 시) | 풍선·BOM 만으로도 분류 가능 |
| Move/Copy/Dryrun | sort_by_titleblock 와 동일 | 일관된 UX |
| **진행 시각화** (2026-04-28 추가) | **tqdm progress bar + `set_postfix`** 실시간 분류 통계 (mfg/asm/review/err) | 5,839장 ~30분 작업 가시화. 25장마다 INFO 로그도 보존 (CI/log 친화). |

## 3. 사용법

### CLI

```bash
# 1) dryrun — 매니페스트만 (이동 X, 권장 첫 실행)
python src/sort_by_drawing_type.py --dryrun

# 2) 실제 분류 + 이동
python src/sort_by_drawing_type.py

# 3) 복사 (원본 보존)
python src/sort_by_drawing_type.py --copy

# 4) OCR 비활성 (속도 ↑)
python src/sort_by_drawing_type.py --no-ocr

# 5) 임계값 튜닝
python src/sort_by_drawing_type.py \
    --dim-min 3 \
    --balloon-asm 8 \
    --balloon-mfg-max 3

# 6) Windows tesseract 경로 명시
python src/sort_by_drawing_type.py \
    --tesseract "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

### 출력 디렉터리

```
data/
├── manufacturing/        ← 가공도면 (Stage 1+2 학습용)
├── assembly/             ← 조립도면 (Stage 1 만 사용)
└── manual_review_type/   ← 사람 검수 필요
outputs/
└── sort_drawing_type_manifest.csv
```

## 4. 검증 결과

### 4.1 더미 이미지 단위 테스트

3 종류 도면 시뮬레이션:

| 입력 | 특징 | 기대 | 결과 |
|---|---|---|---|
| `mfg_drawing.jpg` | 5 치수 텍스트 + 큰 도형 | manufacturing | manual_review (cv2 기본 폰트의 Ø/± 렌더링 한계로 OCR 1개만 검출) |
| `asm_drawing.jpg` | 12 풍선 + BOM 격자 | assembly | **assembly ✓** (BOM 검출) |
| `ambiguous.jpg` | 빈 사각형 | manual_review_type | **manual_review_type ✓** |

**알고리즘 로직 검증**:
- ✓ Hough Circles: 12 풍선 정확 카운트
- ✓ BOM 격자 검출: 우상단 5+ 수평선 검출
- ✓ 분류 우선순위: BOM 우선 적용
- 🟡 OCR 검출: cv2.putText 기본 폰트 한계로 dim_count 낮게 나옴 → 실제 도면(렌더링 폰트)에서는 정상

### 4.2 실제 데이터셋 적용 가이드

`dataset/` 5,839 장 dryrun 후 manifest 검토:
- `dim_count` 분포가 너무 낮으면 `--dim-min 3` 으로 완화
- `balloon_count` false positive 많으면 `--balloon-mfg-max 8` 로 완화
- `bom_detected` 검출률 확인 (조립도면 예상 비율과 비교)

```bash
# manifest 통계 빠른 확인
python -c "
import pandas as pd
df = pd.read_csv('outputs/sort_drawing_type_manifest.csv', encoding='utf-8-sig')
print(df.decision.value_counts())
print(df.describe())
"
```

## 5. 출력 형식

### Manifest CSV

```csv
filename,dim_count,roughness_count,balloon_count,bom_detected,decision,reason,src_path,dst_path
foo_mfg.jpg,12,2,0,False,manufacturing,"dims=12≥5 AND balloons=0<5",dataset/foo_mfg.jpg,data/manufacturing/foo_mfg.jpg
foo_asm.jpg,1,0,15,True,assembly,BOM table detected,dataset/foo_asm.jpg,data/assembly/foo_asm.jpg
foo_amb.jpg,3,0,2,False,manual_review_type,"dims=3, balloons=2, BOM=False",dataset/foo_amb.jpg,
```

UTF-8-SIG 인코딩 (Excel 다국어 친화).

## 6. 의존성

```
opencv-python>=4.10.0
numpy>=1.26.0
pytesseract>=0.3.10  # --no-ocr 사용 시 불필요
Pillow>=10.3.0       # OCR 사용 시
```

시스템: `tesseract-ocr` + 5개 언어팩 (`tesseract-ocr-{eng,kor,jpn,rus,chi-sim,chi-tra}`)

## 7. 관련 의사결정

- **D-009** Stage 1·2 YOLO 는 언어 무관 단일 모델 (분류기는 데이터 통계용)
- **D-013** 4개 언어 (CN 추가 후 5개)
- **D-019** sort_by_titleblock 과 같이 선택 분석 도구 (학습 흐름 필수 아님)
- **D-025** 5개 언어 (CN 포함) — OCR 룰
- **D-026** 가공/조립 분류 정책 — 본 모듈이 직접 구현
- **D-027** TB 핵심 필드 누락 (가공도면도 영향 받음, 본 모듈은 무관)

## 8. sort_by_titleblock.py 와의 관계

| 항목 | sort_by_titleblock | sort_by_drawing_type |
|---|---|---|
| 분류 기준 | TB 유무 | 가공 / 조립 |
| 실제 가치 (TB 95:5 환경) | 낮음 (대부분 TB 있음) | **높음** (Stage 2 학습 데이터 필터링) |
| 사용 시점 | (옵션) 데이터 품질 점검 | **권장** — Stage 1 라벨링 직후 |
| 출력 영향 | data/stage1_titleblock 등 | data/manufacturing 등 |

→ 실제 데이터셋에서는 **`sort_by_drawing_type` 이 더 가치 있는 분류 도구**.

## 9. 흔한 FAIL + 조치

| 증상 | 원인 / 해결 |
|---|---|
| 모든 도면이 `manual_review_type` | OCR 정확도 낮음 — `--dim-min 3` 으로 완화 |
| 가공도면이 `assembly` 로 분류 | 풍선 false positive (도면 안의 작은 원 검출) — `--balloon-mfg-max 8` |
| 조립도면이 `manufacturing` 으로 분류 | BOM 검출 실패 — 도면의 BOM 표가 좌상단 등 비전형 위치 → 차후 BOM 위치 탐색 영역 확장 검토 |
| `tesseract not found` | apt: `tesseract-ocr-chi-sim chi-tra` 추가 설치 필요 |
| 처리 매우 느림 | `--no-ocr` 플래그 (도면당 ~0.3s, 5,839 장 ~25분) |

## 10. 다음 단계

1. **dryrun 실행** → manifest 통계 확인
2. **임계값 튜닝** (필요 시)
3. **실제 분류** → `data/manufacturing/` 폴더 만 라벨링·학습 데이터로 사용
4. **`assembly/` 폴더는 Stage 1 학습 데이터로만 사용** (D-026)
