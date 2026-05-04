"""
mendeley_client.py — Mendeley Data API 클라이언트

Mendeley Data에서 공개 데이터셋을 검색하고 파일을 다운로드합니다.

접근 방식 (2가지):
  1. Mendeley Data 웹 API (data.mendeley.com/api) — 인증 없이 공개 데이터셋 검색/조회
  2. Mendeley OAuth API (api.mendeley.com)     — OAuth 2.0 Client Credentials로 상세 조회

부트캠프 프로젝트에서는 방식 1만으로도 충분합니다.
OAuth 설정 시 방식 2의 추가 데이터에 접근할 수 있습니다.
"""

import time
import logging
from typing import Generator, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    MENDELEY_API_BASE, MENDELEY_AUTH_URL,
    MENDELEY_CLIENT_ID, MENDELEY_CLIENT_SECRET,
    MENDELEY_DATA_BASE, MENDELEY_RATE_LIMIT_DELAY,
    MENDELEY_SEARCH_QUERIES, MENDELEY_MAX_PER_QUERY,
    MENDELEY_PAGE_SIZE,
    MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


class MendeleyClient:
    """Mendeley Data API 클라이언트"""

    def __init__(self):
        self.data_base = MENDELEY_DATA_BASE      # 인증 불요
        self.api_base = MENDELEY_API_BASE         # OAuth 필요
        self.session = self._create_session()
        self._last_request_time = 0.0
        self._oauth_token = None
        self._use_oauth = bool(MENDELEY_CLIENT_ID and MENDELEY_CLIENT_SECRET)

        if self._use_oauth:
            self._authenticate()

    # ─── 세션/인증 ─────────────────────────────────────────

    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 HTTP 세션을 생성합니다."""
        session = requests.Session()

        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_DELAY,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MendeleyCrawler/1.0 (educational-project; bootcamp)",
        })

        return session

    def _authenticate(self):
        """OAuth 2.0 Client Credentials로 액세스 토큰을 발급받습니다."""
        try:
            resp = self.session.post(
                MENDELEY_AUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "scope": "all",
                },
                auth=(MENDELEY_CLIENT_ID, MENDELEY_CLIENT_SECRET),
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()

            data = resp.json()
            self._oauth_token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)

            self.session.headers["Authorization"] = f"Bearer {self._oauth_token}"
            logger.info("Mendeley OAuth 인증 성공 (만료: %d초)", expires_in)

        except requests.exceptions.RequestException as e:
            logger.warning("Mendeley OAuth 인증 실패: %s — Data API로 대체합니다.", e)
            self._use_oauth = False
            self._oauth_token = None

    # ─── Rate Limit ────────────────────────────────────────

    def _rate_limit(self):
        """Rate limit 준수를 위한 대기."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MENDELEY_RATE_LIMIT_DELAY:
            time.sleep(MENDELEY_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """GET 요청을 수행합니다."""
        self._rate_limit()

        try:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("Mendeley Rate limit. %d초 대기...", retry_after)
                time.sleep(retry_after)
                return self._get(url, params)

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            logger.error("Mendeley API 요청 실패: %s — %s", url, e)
            return None

    # ─── 검색 (Data API — 인증 불요) ──────────────────────

    def search_data_api(
        self,
        query: str,
        max_records: int = MENDELEY_MAX_PER_QUERY,
    ) -> Generator[dict, None, None]:
        """
        Mendeley Data 웹 API로 공개 데이터셋을 검색합니다.
        인증 없이 사용 가능합니다.

        Args:
            query: 검색 키워드
            max_records: 최대 결과 수

        Yields:
            데이터셋 dict (Mendeley 레코드를 Zenodo 호환 형식으로 변환)
        """
        url = f"{self.data_base}/datasets"
        offset = 0
        fetched = 0

        while fetched < max_records:
            params = {
                "search": query,
                "limit": min(MENDELEY_PAGE_SIZE, max_records - fetched),
                "offset": offset,
            }

            logger.info(
                "Mendeley 검색: q='%s' offset=%d (누적 %d/%d)",
                query[:50], offset, fetched, max_records
            )

            data = self._get(url, params)
            if data is None:
                break

            # 응답 구조: 리스트 또는 {"results": [...]} 또는 {"data": [...]}
            results = self._extract_results(data)

            if not results:
                logger.info("Mendeley: 더 이상 결과 없음.")
                break

            for dataset in results:
                normalized = self._normalize_record(dataset)
                if normalized:
                    yield normalized
                    fetched += 1
                    if fetched >= max_records:
                        break

            offset += MENDELEY_PAGE_SIZE

            # 결과가 페이지 크기보다 작으면 마지막 페이지
            if len(results) < MENDELEY_PAGE_SIZE:
                break

        logger.info("Mendeley 검색 완료: '%s' → %d건", query[:50], fetched)

    # ─── 검색 (OAuth API) ─────────────────────────────────

    def search_oauth_api(
        self,
        query: str,
        max_records: int = MENDELEY_MAX_PER_QUERY,
    ) -> Generator[dict, None, None]:
        """
        Mendeley OAuth API로 데이터셋을 검색합니다.
        OAuth 인증이 필요합니다.

        Args:
            query: 검색 키워드
            max_records: 최대 결과 수

        Yields:
            정규화된 레코드 dict
        """
        if not self._use_oauth:
            logger.warning("OAuth 미설정. Data API로 대체합니다.")
            yield from self.search_data_api(query, max_records)
            return

        url = f"{self.api_base}/datasets"
        fetched = 0

        params = {
            "search": query,
            "fields": "id,version,name,description,data_licence",
        }

        logger.info("Mendeley OAuth 검색: q='%s'", query[:50])

        data = self._get(url, params)
        if data is None:
            return

        results = self._extract_results(data)

        for dataset in results:
            if fetched >= max_records:
                break
            normalized = self._normalize_record(dataset)
            if normalized:
                yield normalized
                fetched += 1

        logger.info("Mendeley OAuth 검색 완료: '%s' → %d건", query[:50], fetched)

    # ─── 통합 검색 메서드 ──────────────────────────────────

    def search(
        self,
        query: str,
        max_records: int = MENDELEY_MAX_PER_QUERY,
    ) -> Generator[dict, None, None]:
        """
        최적의 방법을 자동 선택하여 검색합니다.
        OAuth 설정 시 → OAuth API, 미설정 시 → Data API
        """
        if self._use_oauth:
            yield from self.search_oauth_api(query, max_records)
        else:
            yield from self.search_data_api(query, max_records)

    def search_all_queries(
        self,
        queries: list = None,
        max_per_query: int = MENDELEY_MAX_PER_QUERY,
    ) -> Generator[dict, None, None]:
        """
        여러 검색 쿼리를 순회하며 중복 제거된 레코드를 yield합니다.
        """
        if queries is None:
            queries = MENDELEY_SEARCH_QUERIES

        seen_ids = set()

        for i, query in enumerate(queries):
            logger.info("━━ Mendeley 쿼리 [%d/%d]: '%s' ━━", i + 1, len(queries), query)

            for record in self.search(query, max_records=max_per_query):
                record_id = record.get("id")
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                yield record

        logger.info("Mendeley 전체 검색 완료: 고유 레코드 %d건", len(seen_ids))

    # ─── 상세 조회 ─────────────────────────────────────────

    def get_dataset(self, dataset_id: str, version: int = None) -> Optional[dict]:
        """
        데이터셋 상세 정보 (파일 목록 포함)를 조회합니다.

        Args:
            dataset_id: Mendeley 데이터셋 ID (예: "r4xggzrjxm")
            version: 특정 버전 (None이면 최신)

        Returns:
            정규화된 레코드 dict 또는 None
        """
        # Data API로 상세 조회
        url = f"{self.data_base}/datasets/{dataset_id}"
        if version:
            url += f"/versions/{version}"

        data = self._get(url)
        if data is None:
            logger.error("Mendeley 상세 조회 실패: %s", dataset_id)
            return None

        return self._normalize_record(data, include_files=True)

    def get_dataset_files(self, dataset_id: str, version: int = None) -> list:
        """
        데이터셋의 파일 목록을 조회합니다.

        Args:
            dataset_id: 데이터셋 ID
            version: 특정 버전

        Returns:
            파일 정보 리스트
        """
        # 상세 조회에서 파일 목록 추출
        detail = self.get_dataset(dataset_id, version)
        if detail is None:
            return []

        return detail.get("files", [])

    # ─── 정규화 (Zenodo 호환 형식) ─────────────────────────

    def _normalize_record(self, dataset: dict, include_files: bool = False) -> Optional[dict]:
        """
        Mendeley 데이터셋을 Zenodo 호환 형식으로 변환합니다.
        filters.py의 DrawingFilter를 그대로 사용할 수 있게 합니다.

        Args:
            dataset: Mendeley API 응답의 데이터셋 dict
            include_files: True면 파일 목록도 포함

        Returns:
            Zenodo 호환 형식의 dict
        """
        # Mendeley Data API 응답 구조가 다양하므로 유연하게 처리
        dataset_id = (
            dataset.get("id")
            or dataset.get("doi")
            or dataset.get("version", {}).get("id")
            or "unknown"
        )

        # 제목
        title = (
            dataset.get("name")
            or dataset.get("title")
            or ""
        )

        # 설명
        description = (
            dataset.get("description")
            or dataset.get("abstract")
            or ""
        )

        # 키워드
        keywords = dataset.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]

        # 카테고리/분야
        categories = dataset.get("categories", [])
        if isinstance(categories, list):
            for cat in categories:
                if isinstance(cat, dict):
                    keywords.append(cat.get("label", ""))
                elif isinstance(cat, str):
                    keywords.append(cat)

        # 라이선스
        licence = dataset.get("data_licence", dataset.get("license", {}))
        if isinstance(licence, dict):
            licence_id = licence.get("id", licence.get("url", ""))
        else:
            licence_id = str(licence)

        # 파일 목록 정규화
        files = []
        raw_files = dataset.get("files", [])
        if isinstance(raw_files, list):
            for f in raw_files:
                normalized_file = self._normalize_file(f, dataset_id)
                if normalized_file:
                    files.append(normalized_file)

        # Zenodo 호환 형식으로 조합
        normalized = {
            "id": f"mendeley_{dataset_id}",
            "source": "mendeley",
            "doi": dataset.get("doi", ""),
            "metadata": {
                "title": title,
                "description": description,
                "keywords": keywords,
                "license": {"id": licence_id},
                "access_right": "open",
                "publication_date": dataset.get("publish_date", ""),
                "subjects": [],
                "notes": "",
            },
            "files": files,
        }

        return normalized

    def _normalize_file(self, file_info: dict, dataset_id: str) -> Optional[dict]:
        """Mendeley 파일 정보를 Zenodo 호환 형식으로 변환합니다."""
        filename = (
            file_info.get("filename")
            or file_info.get("name")
            or file_info.get("key")
            or ""
        )

        if not filename:
            return None

        size = file_info.get("size", file_info.get("content_length", 0))

        # 다운로드 URL 구성
        file_id = file_info.get("id", "")
        download_url = file_info.get("download_url", "")

        if not download_url and file_id:
            # Mendeley Data 파일 다운로드 URL 패턴
            download_url = (
                f"https://data.mendeley.com/public-files/datasets/{dataset_id}"
                f"/files/{file_id}/file_downloaded"
            )

        # content_details에서 URL 추출
        content_details = file_info.get("content_details", {})
        if isinstance(content_details, dict) and not download_url:
            download_url = content_details.get("download_url", "")

        return {
            "key": filename,
            "filename": filename,
            "size": size,
            "links": {
                "self": download_url,
            },
        }

    # ─── 유틸리티 ──────────────────────────────────────────

    @staticmethod
    def _extract_results(data) -> list:
        """API 응답에서 결과 리스트를 추출합니다."""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # 다양한 Mendeley 응답 형태 처리
            for key in ("results", "data", "datasets", "items", "hits"):
                if key in data:
                    val = data[key]
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict) and "hits" in val:
                        return val["hits"]
            # 단일 결과
            if "id" in data or "name" in data:
                return [data]
        return []

    def test_connection(self) -> bool:
        """API 연결 및 인증 상태를 테스트합니다."""
        # Data API 테스트 (인증 불요)
        url = f"{self.data_base}/datasets"
        params = {"search": "engineering", "limit": 1}

        data = self._get(url, params)
        if data is not None:
            results = self._extract_results(data)
            logger.info("Mendeley Data API 연결 성공 (%d건 확인)", len(results))
            auth = "OAuth 인증됨" if self._use_oauth else "인증 없음 (Data API만 사용)"
            logger.info("Mendeley 인증 상태: %s", auth)
            return True

        logger.error("Mendeley Data API 연결 실패")
        return False

    def close(self):
        """세션을 종료합니다."""
        self.session.close()
