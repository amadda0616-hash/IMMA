# 2D 가공 도면 통합 크롤러

AI 에이전트 파이프라인 프로젝트용 도면 데이터 자동 수집 도구입니다.
부트캠프 프로젝트 및 공모전(비상업적 교육 목적)으로 활용됩니다.

---

## 1. 프로젝트 개요

### 1.1 목적

모델 학습용 2D 기계 가공 도면 데이터를 여러 공개 소스에서 자동 수집합니다.

### 1.2 대상 도면 유형

| 유형 | 설명 | 카테고리 폴더 |
|------|------|--------------|
| 용접 도면 | Welding, ISO 2553 | `welding/` |
| CNC 가공 도면 | Milling, Machining, Drilling | `cnc_machining/` |
| 선반/회전 도면 | Lathe, Turning | `lathe_turning/` |
| 기계설계 도면 | GD&T, Assembly, Detail | `design/` |
| 판금 도면 | Sheet Metal, Bending | `sheet_metal/` |

### 1.3 지원 소스

| 소스 | 상태 | 인증 | 비고 |
|------|------|------|------|
| Zenodo | ✅ 구현 완료 | 토큰 선택 | REST API, rate limit 60req/min |
| Mendeley Data | ✅ 구현 완료 | OAuth 선택 | Data API 인증 불요 |
| TraceParts | 🔜 예정 | API 키 필수 | API 키 신청 후 사용 |
| Kaggle | 🔜 예정 | kaggle.json | Kaggle CLI 활용 |
| Roboflow | 🔜 예정 | API 키 | 어노테이션 포함 데이터셋 |

### 1.4 필터링 체계

크롤링된 레코드는 2단계 필터를 거칩니다.

**1단계 — 제외 키워드 검사**: 건축(architectural, floor plan), 전기(circuit, PCB), 토목(civil engineering), 의료(medical, anatomical) 등 기계 가공과 무관한 분야를 자동 제외합니다.

**2단계 — 포함 점수 계산**: 용접, CNC, 선반, 설계 관련 키워드별로 가중치 점수(high:10, medium:5, low:2)를 매기고, 최소 점수(기본 5점) 이상인 레코드만 수집합니다.

---

## 2. 설치

### 2.1 의존성 설치

```bash
cd zenodo_crawler
pip install -r requirements.txt
```

`requirements.txt` 내용:

```
requests>=2.28.0
urllib3>=1.26.0
```

### 2.2 API 인증 설정 (선택)

#### Zenodo 토큰

토큰 없이도 동작하지만, 토큰 사용 시 rate limit이 완화됩니다(60 → 100 req/min).

```bash
export ZENODO_ACCESS_TOKEN="your_token_here"
```

토큰 발급: https://zenodo.org/account/settings/applications/tokens/new/

#### Mendeley OAuth

Mendeley Data API(`data.mendeley.com/api`)는 인증 없이 공개 데이터셋 검색/조회가 가능합니다. OAuth 설정은 추가 데이터 접근이 필요할 때만 하면 됩니다.

```bash
export MENDELEY_CLIENT_ID="your_client_id"
export MENDELEY_CLIENT_SECRET="your_client_secret"
```

앱 등록: https://dev.mendeley.com/myapps.html

---

## 3. 실행 방법

### 3.1 기본 실행 (전체 소스)

```bash
# 모든 소스에서 크롤링 (Zenodo + Mendeley)
python main.py

# dry-run: 다운로드 없이 검색 + 필터 결과만 확인
python main.py --dry-run
```

### 3.2 소스별 실행

```bash
# Zenodo만 실행
python main.py --source zenodo

# Mendeley만 실행
python main.py --source mendeley

# 모든 소스 (기본값)
python main.py --source all
```

### 3.3 수집 범위 제한

```bash
# 최대 100건만 수집
python main.py --max-records 100

# 쿼리당 최대 20건
python main.py --max-per-query 20

# 특정 쿼리만 실행
python main.py --query "welding drawing"
python main.py --query "CNC machining drawing"
```

### 3.4 로그 및 디버깅

```bash
# 상세 디버그 로그 출력
python main.py --verbose

# 또는 -v 축약형
python main.py -v --dry-run
```

로그 파일은 `output/logs/crawl_YYYYMMDD_HHMMSS.log`에 자동 저장됩니다.

### 3.5 통계 확인

```bash
# 기존 수집 데이터 기준 통계만 표시
python main.py --stats-only
```

### 3.6 출력 경로 변경

```bash
# 커스텀 출력 경로 지정
python main.py --output ./my_drawings
```

### 3.7 권장 실행 순서

```bash
# Step 1: dry-run으로 필터 결과 확인 (다운로드 X)
python main.py --dry-run --max-records 20 --verbose

# Step 2: 소량 테스트 (실제 다운로드)
python main.py --max-records 50

# Step 3: 전체 크롤링
python main.py
```

### 3.8 CLI 옵션 전체 목록

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--source` | `all` | 크롤링 소스 (`all`, `zenodo`, `mendeley`) |
| `--dry-run` | `false` | 다운로드 없이 검색+필터만 |
| `--query` | (전체) | 단일 검색 쿼리 지정 |
| `--max-records` | `5000` | 전체 최대 수집 수 |
| `--max-per-query` | (소스별) | 쿼리당 최대 수 |
| `--output` | `./output` | 출력 디렉토리 |
| `--verbose` / `-v` | `false` | 상세 디버그 로그 |
| `--stats-only` | `false` | 기존 통계만 표시 |

---

## 4. 출력 구조

```
output/
├── downloads/                    # 도면 파일 (카테고리별)
│   ├── welding/
│   │   └── {record_id}/
│   │       ├── drawing.dxf
│   │       └── detail.pdf
│   ├── cnc_machining/
│   ├── lathe_turning/
│   ├── design/
│   ├── sheet_metal/
│   └── uncategorized/
├── metadata/                     # 메타데이터
│   ├── records.json              # 전체 메타데이터 (JSON)
│   ├── records.csv               # 전체 메타데이터 (CSV, Excel 호환)
│   └── processed_ids.json        # 처리 완료 인덱스 (재시작용)
└── logs/                         # 실행 로그
    └── crawl_YYYYMMDD_HHMMSS.log
```

### 4.1 메타데이터 필드 (records.csv)

| 필드 | 설명 |
|------|------|
| `record_id` | 소스별 레코드 ID |
| `source` | 소스명 (`zenodo`, `mendeley`) |
| `title` | 데이터셋 제목 |
| `category` | 자동 분류 카테고리 |
| `include_score` | 포함 키워드 점수 |
| `license` | 라이선스 |
| `file_count` | 다운로드된 파일 수 |
| `url` | 원본 URL |
| `collected_at` | 수집 일시 (UTC) |
| `published_date` | 원본 게시일 |

---

## 5. 모듈 구조

```
zenodo_crawler/
├── config.py            # 키워드, 필터, 경로, API 설정 (소스 ON/OFF)
├── zenodo_client.py     # Zenodo REST API 호출, 페이지네이션, rate limit
├── mendeley_client.py   # Mendeley Data API, OAuth, Zenodo 형식 정규화
├── filters.py           # 2단계 필터 (제외 → 포함 점수 → 카테고리 분류)
├── downloader.py        # 파일 다운로드, 중복 건너뛰기, 재시도
├── metadata.py          # 메타데이터 JSON/CSV 저장, 처리 인덱스 관리
├── main.py              # CLI 진입점, 소스별 파이프라인 조립
├── requirements.txt     # Python 의존성
└── README.md            # 이 문서
```

### 5.1 설계 원칙

**소스 정규화**: 각 소스 클라이언트(`zenodo_client.py`, `mendeley_client.py`)가 API 응답을 공통 형식으로 변환합니다. 덕분에 `filters.py`, `downloader.py`, `metadata.py`는 소스에 관계없이 동일 코드로 동작합니다.

**모듈 독립성**: 새 소스 추가 시 `{source}_client.py`만 작성하고 `main.py`에 Phase를 추가하면 됩니다. 기존 모듈 수정이 불필요합니다.

---

## 6. 필터 설정 커스터마이즈

`config.py`에서 직접 수정 가능합니다.

### 6.1 검색 키워드 추가

```python
# config.py — Zenodo 검색 쿼리
SEARCH_QUERIES = [
    "mechanical drawing",
    "welding drawing",
    # ... 추가 키워드
]

# Mendeley 전용 쿼리
MENDELEY_SEARCH_QUERIES = [
    "mechanical engineering drawing",
    # ... 추가 키워드
]
```

### 6.2 제외 키워드 추가

```python
# config.py — 건축/전기/토목 등 비기계 분야
EXCLUDE_KEYWORDS = [
    "architectural", "floor plan",
    "circuit diagram", "PCB",
    # ... 추가 제외 키워드
]
```

### 6.3 포함 점수 조정

```python
# config.py
INCLUDE_WEIGHTS = {
    "high": 10,     # 용접, CNC, 선반 등 핵심 키워드
    "medium": 5,    # 도면 관련 일반 키워드
    "low": 2,       # 표준/기호 관련 키워드
}
MIN_INCLUDE_SCORE = 5  # 이 점수 이상만 수집 (낮추면 더 많이 수집)
```

### 6.4 소스 ON/OFF

```python
# config.py
ENABLED_SOURCES = {
    "zenodo": True,
    "mendeley": True,
}
```

---

## 7. 재실행 안전성

`processed_ids.json`으로 이미 처리된 레코드를 추적합니다. 크롤링을 중단(`Ctrl+C`)해도 진행 상태가 보존되며, 재실행 시 이전에 수집한 레코드는 자동으로 건너뜁니다.

```bash
# 이전 실행에서 200건 수집 후 중단된 경우
# 재실행 시 201번째부터 이어서 수집
python main.py
```

---

## 8. 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-04-16 | v1.0 — Zenodo 크롤러 초기 구현 |
| 2026-04-16 | v1.1 — Mendeley Data API 크롤러 추가, CLI `--source` 옵션 추가 |
