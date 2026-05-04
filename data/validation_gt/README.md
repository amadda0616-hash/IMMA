# Validation Ground Truth

검증 모듈(`src/validate/check_*.py`) 이 사용하는 사람 검수 GT 파일.

## 파일 명세

| 파일 | 형식 | 사용 검증기 | 비고 |
|---|---|---|---|
| `step1_5_titleblock_gt.csv` | CSV: `filename, has_titleblock_actual` | `check_step1_5_sorter.py` | 50장 샘플 권장 |
| `stage1_iou_check.json` | JSON | `check_labels_yolo.py` | 검증자 2명의 BBox 비교 |
| `stage1_human_review.csv` | CSV | `check_stage1_crops.py` | crop 시각 검수 결과 |
| `stage2_obb_iou_check.json` | JSON | `check_labels_obb.py` | OBB 인터-라벨러 비교 |
| `stage3a_titleblock_gt.json` | JSON | `check_stage3a_alphabetical.py` | 도면별 expected fields |
| `stage3a_notes_gt.json` | JSON | `check_stage3a_alphabetical.py` | 도면별 expected note items |
| `stage3n_numerical_gt.json` | JSON | `check_stage3n_numerical.py` | 패치별 expected schema JSON |

## CSV 형식: `step1_5_titleblock_gt.csv`

```csv
filename,has_titleblock_actual
drawing_001_KO.jpg,1
drawing_002_EN.jpg,1
drawing_005_view_only.jpg,0
```

- `filename` — `dataset/` 의 파일명 (확장자 포함)
- `has_titleblock_actual` — `1` (TitleBlock 있음) / `0` (없음)
- UTF-8 (BOM 허용). 한글/일본어/러시아어 파일명 정상 처리

## JSON 형식 (예: `stage3a_titleblock_gt.json`)

```json
{
  "drawing_001_KO": {
    "drawing_no": "EB-CTRL-001",
    "material": "SUS304",
    "scale": "1:2",
    "drawn_by": "김선영"
  },
  "drawing_002_EN": {
    "drawing_no": "PMP-BRK-204",
    "material": "Stainless Steel",
    "scale": "1:1"
  }
}
```

## 라벨링 가이드

1. **샘플 크기**: stage 별로 30~50건 권장 (통계적 신뢰 확보)
2. **언어 균형**: 4개 언어(EN/KO/JP/RU) 비율 가능한 균등하게
3. **2명 검수**: critical step (Stage 2 누락률, Stage 3-N) 은 검증자 2명 + 3차 분쟁 조정
4. **버전 관리**: `stage3a_titleblock_gt_v1.json` 처럼 버전 명시 권장

## 면책

이 폴더의 파일은 **사람이 작성한 ground truth**. 검증기 결과의 신뢰도는 GT 품질에 직접 의존. GT 자체에 오류가 있으면 검증 결과도 부정확.
