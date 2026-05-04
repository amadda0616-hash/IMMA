"""
config.py — 도면 크롤러 통합 설정 모듈

검색 키워드, 포함/제외 필터, 파일 형식, API 설정 등
모든 크롤링 파라미터를 한곳에서 관리합니다.

지원 소스: Zenodo, Mendeley Data, (향후 TraceParts 등)
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. API 설정
# ─────────────────────────────────────────────────────────────
# ── Zenodo ──
ZENODO_API_BASE = "https://zenodo.org/api"
ZENODO_ACCESS_TOKEN = os.getenv("ZENODO_ACCESS_TOKEN", "")  # 선택: 토큰 사용 시 rate limit 완화

# ── Mendeley Data ──
# OAuth 2.0 Client Credentials Flow (공개 데이터 읽기 전용)
# 발급: https://dev.mendeley.com/myapps.html
MENDELEY_API_BASE = "https://api.mendeley.com"
MENDELEY_AUTH_URL = "https://api.mendeley.com/oauth/token"
MENDELEY_CLIENT_ID = os.getenv("MENDELEY_CLIENT_ID", "")
MENDELEY_CLIENT_SECRET = os.getenv("MENDELEY_CLIENT_SECRET", "")

# Mendeley Data 웹 프론트엔드 (공개 데이터셋 검색 — 인증 없이 사용 가능)
MENDELEY_DATA_BASE = "https://data.mendeley.com/api"

# ── 공통 Rate Limit ──
RATE_LIMIT_DELAY = 1.2   # 요청 간 최소 대기 시간 (초)
MENDELEY_RATE_LIMIT_DELAY = 1.5  # Mendeley는 좀 더 보수적으로
MAX_RETRIES = 3           # 실패 시 재시도 횟수
RETRY_DELAY = 5           # 재시도 대기 시간 (초)
REQUEST_TIMEOUT = 30      # 요청 타임아웃 (초)

# ── 소스 활성화 ──
ENABLED_SOURCES = {
    "zenodo": True,
    "mendeley": True,
}

# ── Mendeley 전용 검색 쿼리 ──
# Mendeley는 검색 정확도가 낮아 더 구체적인 쿼리 사용
MENDELEY_SEARCH_QUERIES = [
    "mechanical engineering drawing",
    "welding drawing dataset",
    "CNC machining drawing",
    "lathe turning drawing",
    "2D CAD drawing",
    "engineering drawing annotation",
    "manufacturing technical drawing",
    "machine part drawing",
    "sheet metal fabrication drawing",
    "GD&T engineering",
]

MENDELEY_MAX_PER_QUERY = 100  # Mendeley 쿼리당 최대 (데이터량이 적음)
MENDELEY_PAGE_SIZE = 20        # Mendeley 페이지 크기

# ─────────────────────────────────────────────────────────────
# 2. 검색 키워드 — Zenodo 검색 쿼리에 사용
#    각 키워드 그룹별로 OR 검색을 수행합니다.
# ─────────────────────────────────────────────────────────────
SEARCH_QUERIES = [
    # 기계 도면 관련
    "mechanical drawing",
    "engineering drawing machining",
    "manufacturing drawing",
    "technical drawing mechanical",

    # 용접 도면
    "welding drawing",
    "weld joint drawing",
    "welding symbol",
    "ISO 2553 welding",

    # CNC / 밀링 / 선반
    "CNC machining drawing",
    "CNC milling drawing",
    "lathe turning drawing",
    "turning machining drawing",

    # 설계 도면
    "machine design drawing",
    "mechanical part drawing",
    "GD&T drawing",
    "tolerance dimensioning drawing",

    # CAD 데이터셋
    "2D CAD dataset",
    "DXF mechanical dataset",
    "engineering drawing dataset",

    # 제조/판금
    "sheet metal drawing",
    "fabrication drawing",
]

# ─────────────────────────────────────────────────────────────
# 3. 포함 키워드 — 검색 결과 내 2차 필터링
#    제목/설명/키워드에 아래 단어가 포함되면 가산점
# ─────────────────────────────────────────────────────────────
INCLUDE_KEYWORDS = {
    # 가공 유형 (가중치 높음)
    "high": [
        "welding", "weld", "weldment",
        "CNC", "milling", "machining",
        "lathe", "turning", "boring",
        "grinding", "drilling", "tapping",
        "sheet metal", "fabrication",
    ],
    # 도면 관련 (가중치 중간)
    "medium": [
        "mechanical drawing", "engineering drawing",
        "technical drawing", "manufacturing drawing",
        "machine design", "mechanical part",
        "assembly drawing", "detail drawing",
        "shop drawing", "production drawing",
    ],
    # 표준/기호 (가중치 낮음)
    "low": [
        "GD&T", "tolerance", "dimensioning",
        "surface roughness", "surface finish",
        "ISO 2553", "ISO 1101", "ASME Y14",
        "KS B", "JIS B",
        "DXF", "DWG", "CAD",
    ]
}

# 가중치 점수
INCLUDE_WEIGHTS = {
    "high": 10,
    "medium": 5,
    "low": 2,
}

# 포함 판정 최소 점수 (이 점수 이상이어야 다운로드)
MIN_INCLUDE_SCORE = 5

# ─────────────────────────────────────────────────────────────
# 4. 제외 키워드 — 이 단어가 포함되면 무조건 제외
#    건축, 전기, 토목, 의료 등 기계 가공과 무관한 분야
# ─────────────────────────────────────────────────────────────
EXCLUDE_KEYWORDS = [
    # 건축/토목
    "architectural", "architecture", "building design",
    "floor plan", "building plan", "house plan",
    "civil engineering", "structural engineering",
    "urban planning", "urban design",
    "landscape", "interior design",
    "construction management",

    # 전기/전자
    "electrical schematic", "circuit diagram",
    "circuit board", "PCB", "wiring diagram",
    "electronic schematic", "logic diagram",

    # 배관/설비
    "HVAC", "plumbing", "piping diagram",
    "P&ID", "fire protection",

    # 의료/화학/생물
    "medical", "anatomical", "anatomy",
    "chemical engineering", "biomedical",
    "pharmaceutical",

    # 기타 비관련
    "geological", "geographic", "cartography",
    "fashion design", "textile",
    "food processing",
    "software architecture", "UML",
    "flowchart", "network diagram",
]

# ─────────────────────────────────────────────────────────────
# 5. 파일 형식 필터
# ─────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {
    ".dxf", ".dwg",          # CAD 벡터
    ".pdf",                   # 도면 PDF
    ".png", ".jpg", ".jpeg",  # 도면 이미지
    ".tif", ".tiff",          # 스캔 도면
    ".svg",                   # 벡터 이미지
}

# 파일 크기 제한 (바이트)
MAX_FILE_SIZE = 500 * 1024 * 1024   # 500MB (대규모 ZIP 등 방지)
MIN_FILE_SIZE = 1024                 # 1KB 미만은 빈 파일로 판단

# ─────────────────────────────────────────────────────────────
# 6. 도면 카테고리 분류 키워드
#    다운로드된 도면을 자동으로 폴더 분류합니다.
# ─────────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "welding": [
        "welding", "weld", "weldment", "weld joint",
        "fillet weld", "butt weld", "groove weld",
        "ISO 2553", "welding symbol",
    ],
    "cnc_machining": [
        "CNC", "milling", "machining", "boring",
        "drilling", "tapping", "grinding",
        "5-axis", "3-axis", "machining center",
    ],
    "lathe_turning": [
        "lathe", "turning", "chuck", "spindle",
        "facing", "threading", "knurling",
        "rotational", "cylindrical",
    ],
    "design": [
        "machine design", "mechanical design",
        "assembly", "detail drawing",
        "GD&T", "tolerance", "dimensioning",
        "part drawing", "component",
    ],
    "sheet_metal": [
        "sheet metal", "bending", "stamping",
        "punching", "laser cutting", "forming",
        "flat pattern", "bend allowance",
    ],
}

# ─────────────────────────────────────────────────────────────
# 7. 출력 경로
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = Path("./output")
DOWNLOAD_DIR = BASE_OUTPUT_DIR / "downloads"
METADATA_DIR = BASE_OUTPUT_DIR / "metadata"
LOG_DIR = BASE_OUTPUT_DIR / "logs"

# 카테고리별 하위 폴더 자동 생성
CATEGORY_DIRS = {cat: DOWNLOAD_DIR / cat for cat in CATEGORY_KEYWORDS}
CATEGORY_DIRS["uncategorized"] = DOWNLOAD_DIR / "uncategorized"

# ─────────────────────────────────────────────────────────────
# 8. 크롤링 범위 설정
# ─────────────────────────────────────────────────────────────
MAX_RECORDS_PER_QUERY = 200   # 검색 쿼리당 최대 레코드 수
MAX_TOTAL_RECORDS = 5000       # 전체 최대 레코드 수
PAGE_SIZE = 50                 # 페이지당 결과 수 (Zenodo 최대 100)

# ─────────────────────────────────────────────────────────────
# 9. 라이선스 필터 (교육용 허용 라이선스만)
# ─────────────────────────────────────────────────────────────
ALLOWED_LICENSES = [
    "cc-by-4.0", "cc-by-sa-4.0", "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0", "cc-by-3.0", "cc-by-nc-3.0",
    "cc-zero", "cc0-1.0",
    "mit", "apache-2.0", "gpl-3.0",
    "open",  # Zenodo "open access"
]

# True이면 라이선스가 위 목록에 없으면 건너뜀
# False이면 라이선스 무관하게 수집
ENFORCE_LICENSE_FILTER = False  # 부트캠프 교육용이므로 완화 가능


def ensure_dirs():
    """모든 출력 디렉토리를 생성합니다."""
    for d in [DOWNLOAD_DIR, METADATA_DIR, LOG_DIR, *CATEGORY_DIRS.values()]:
        d.mkdir(parents=True, exist_ok=True)
