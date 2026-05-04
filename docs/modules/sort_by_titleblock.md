# `src/sort_by_titleblock.py`

> **Step 1.5** — TitleBlock 유무 기준 데이터 사전 분류 (선택 도구)

## 1. 구현 요약

`dataset/` 의 JPG 도면을 다음 3개 폴더로 자동 분류한다.

```
data/stage1_titleblock/      ← TB 키워드 ≥ 2 검출 (TB 있음)
data/stage2_no_titleblock/   ← 키워드 0 + 라인 밀도 낮음 (단일 View)
data/manual_review/          ← 신호 모순 / 1개만 검출 (사람 검수 필요)
```

**판정 알고리즘**

1. 이미지 하단 35% crop
2. PyTesseract OCR (`lang='kor+eng+rus+jpn'` 단일 호출)
3. 4개 언어 키워드 사전과 fuzzy matching (대소문자·공백 무시)
4. 보조 신호: Canny edge + HoughLinesP 로 직선 밀도 계산
5. 임계값 기반 분류

**4개 언어 키워드 사전 (~70개)**

- EN: TITLE, DRAWING, DWG, SCALE, MATERIAL, REV, DATE, DRAWN, CHECKED, SHEET, PART NO, …
- KO: 도번, 도면, 척도, 재질, 개정, 날짜, 작성, 검도, 승인, 시트, …
- RU: ЧЕРТЕЖ, МАСШТАБ, МАТЕРИАЛ, ИЗМ, ДАТА, ЛИСТ, …
- JP: 図面, 図番, 縮尺, 材質, 改訂, 日付, 作成, 検図, …

## 2. 핵심 설계 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 모듈 위치 | **선택 분석 도구** (학습 흐름 필수 아님) | D-019 |
| OCR 엔진 | **PyTesseract** (`kor+eng+rus+jpn` 단일 호출) | EasyOCR 은 CJK·Cyrillic 혼합 단일 호출 불가 |
| 언어 | 4개 동시 처리 | D-013 도면 언어 정의 |
| 처리 영역 | 하단 35% crop | TitleBlock 표준 위치 |
| 키워드 임계값 | ≥ 2 → stage1, 0 → stage2 (CLI 변경 가능) | 1개는 false positive 위험 |
| 보조 신호 | 라인 밀도 (Canny + Hough) | TitleBlock 격자 검출 |
| 파일 처리 | **이동 (move)** — 원본 dataset/ 비워짐 | dryrun 옵션으로 리허설 |
| 매니페스트 | UTF-8-SIG CSV | Excel 한글/일본어/러시아어 호환 |
| Tesseract path | Linux: `/usr/bin/tesseract` (PATH) / Windows: `C:\Program Files\Tesseract-OCR\` fallback | OS 무관 |
| 유니코드 파일명 | `np.fromfile + cv2.imdecode` | Windows 한글/일본어 경로 |

## 3. 사용법

### CLI

```bash
# 1) 매니페스트만 미리 확인 (이동 X) — 권장 첫 실행
python src/sort_by_titleblock.py --dryrun

# 2) 실제 이동
python src/sort_by_titleblock.py

# 3) 임계값 튜닝
python src/sort_by_titleblock.py --keyword-threshold 1
python src/sort_by_titleblock.py --density-threshold 0.0012

# 4) Tesseract 경로 커스터마이즈
python src/sort_by_titleblock.py --tesseract /usr/bin/tesseract
```

### 공개 함수 (import 가능)

```python
from src.sort_by_titleblock import (
    process_one,         # 단일 이미지 분류
    count_keywords,      # OCR 텍스트 → 키워드 매칭 카운트
    compute_line_density,
)
```

## 4. 검증 결과

### 4.1 8건 더미 매니페스트 단위 테스트

`outputs/sort_titleblock_manifest.csv` 8행 (4 lang 혼합 + 빈 라벨 + 에러 케이스) 으로 V1 검증기 통과 확인:

```
manual_review_rate  0.2500  WARN ≤ 0.2000
error_rate          0.1250  WARN ≤ 0.0100
classifier_accuracy 1.0000  PASS ≥ 0.8500   (5건 GT 매칭, manual_review 제외)
per_language_acc[en] 1.0000 PASS ≥ 0.8000  (n=4)
per_language_acc[ja] 1.0000 PASS ≥ 0.8000  (n=1)
```

### 4.2 실제 데이터셋 검증 (대기)

`dataset/` 의 5,839 JPG 에 대한 실제 분류는 사용자 실행 필요 → 사후 V1 검증기로 정확도 측정.

## 5. 출력 형식

### 매니페스트 CSV

```csv
filename,keyword_hits,matched_keywords,line_density,decision,src_path,dst_path,note
drawing_001_KO.jpg,3,en:TITLE;ko:도번;ko:재질,0.0021,stage1_titleblock,...,...,
drawing_005_view.jpg,0,,0.0004,stage2_no_titleblock,...,...,
drawing_008_corrupt.jpg,0,,0.0,error,...,,imread_failed
```

### 폴더 구조 (실행 후)

```
data/
├── stage1_titleblock/      (TB 있음 분류)
├── stage2_no_titleblock/   (TB 없음 분류)
└── manual_review/          (애매)
outputs/
└── sort_titleblock_manifest.csv
```

## 6. 의존성

```
pytesseract>=0.3.10
Pillow>=10.3.0
opencv-python>=4.10.0
numpy>=1.26.0
```

시스템: `tesseract-ocr` + 4개 언어팩 (`tesseract-ocr-{eng,kor,jpn,rus}`)

## 7. 관련 의사결정

- **D-009** Stage 1·2 YOLO 는 언어 무관 단일 모델 (분류기는 데이터 통계용)
- **D-010** 도면 1장 = 단일 언어 (KO/EN/JP/RU)
- **D-011** 분류기는 `lang='kor+eng+rus+jpn'` 단일 호출 + 라인 밀도 보조
- **D-013** 4개 언어 정확히 정의
- **D-019** sort_by_titleblock 은 선택 분석 도구 (학습 흐름 필수 아님)

## 8. 검증 모듈

[`check_step1_5_sorter.py`](./check_step1_5_sorter.md) — V1 검증기

```bash
python -m src.validate.check_step1_5_sorter \
    --manifest outputs/sort_titleblock_manifest.csv \
    --gt data/validation_gt/step1_5_titleblock_gt.csv
```
