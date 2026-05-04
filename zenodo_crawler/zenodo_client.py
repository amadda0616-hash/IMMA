"""
zenodo_client.py — Zenodo REST API 클라이언트

Zenodo 검색, 레코드 상세 조회, 페이지네이션을 처리합니다.
Rate limit 준수 및 재시도 로직을 포함합니다.
"""

import time
import logging
from typing import Generator, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    ZENODO_API_BASE, ZENODO_ACCESS_TOKEN,
    RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_DELAY,
    REQUEST_TIMEOUT, PAGE_SIZE, MAX_RECORDS_PER_QUERY,
)

logger = logging.getLogger(__name__)


class ZenodoClient:
    """Zenodo REST API 클라이언트"""

    def __init__(self, access_token: str = ""):
        self.base_url = ZENODO_API_BASE
        self.token = access_token or ZENODO_ACCESS_TOKEN
        self.session = self._create_session()
        self._last_request_time = 0.0

    def _create_session(self) -> requests.Session:
        """재시도 로직이 포함된 HTTP 세션을 생성합니다."""
        session = requests.Session()

        # 재시도 설정 (429 Too Many Requests, 500, 502, 503, 504)
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_DELAY,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # 공통 헤더
        headers = {
            "Accept": "application/json",
            "User-Agent": "ZenodoCrawler/1.0 (educational-project; bootcamp)",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        session.headers.update(headers)
        return session

    def _rate_limit(self):
        """Rate limit를 준수하기 위해 요청 간 대기합니다."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            wait = RATE_LIMIT_DELAY - elapsed
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """GET 요청을 수행합니다."""
        self._rate_limit()

        try:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            # Rate limit 초과 시 대기 후 재시도
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("Rate limit 초과. %d초 대기...", retry_after)
                time.sleep(retry_after)
                return self._get(url, params)

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            logger.error("API 요청 실패: %s — %s", url, e)
            return None

    # ─── 검색 메서드 ───────────────────────────────────────

    def search(
        self,
        query: str,
        max_records: int = MAX_RECORDS_PER_QUERY,
        resource_type: str = "dataset",
    ) -> Generator[dict, None, None]:
        """
        Zenodo에서 키워드로 검색합니다.
        페이지네이션을 자동 처리하며 레코드를 하나씩 yield합니다.

        Args:
            query: 검색 문자열
            max_records: 이 쿼리에서 가져올 최대 레코드 수
            resource_type: 리소스 유형 필터 ("dataset", "publication" 등)

        Yields:
            Zenodo 레코드 dict
        """
        url = f"{self.base_url}/records"
        page = 1
        fetched = 0

        while fetched < max_records:
            params = {
                "q": query,
                "size": min(PAGE_SIZE, max_records - fetched),
                "page": page,
                "sort": "bestmatch",
                "access_right": "open",  # 오픈 액세스만
            }

            # resource_type 필터 (Zenodo 쿼리 문법)
            if resource_type:
                params["q"] = f"{query} AND resource_type.type:{resource_type}"

            logger.info(
                "검색 중: q='%s' page=%d (누적 %d/%d)",
                query[:50], page, fetched, max_records
            )

            data = self._get(url, params)
            if data is None:
                logger.error("검색 응답 없음. 중단합니다.")
                break

            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", 0)

            if not hits:
                logger.info("더 이상 결과 없음. (total: %d)", total)
                break

            for record in hits:
                yield record
                fetched += 1
                if fetched >= max_records:
                    break

            page += 1

            # 전체 결과 수 초과 시 중단
            if fetched >= total:
                break

        logger.info("검색 완료: '%s' → %d건 수집", query[:50], fetched)

    def search_all_queries(
        self,
        queries: list,
        max_per_query: int = MAX_RECORDS_PER_QUERY,
    ) -> Generator[dict, None, None]:
        """
        여러 검색 쿼리를 순회하며 중복 제거된 레코드를 yield합니다.

        Args:
            queries: 검색 쿼리 리스트
            max_per_query: 쿼리당 최대 레코드 수

        Yields:
            중복 제거된 Zenodo 레코드 dict
        """
        seen_ids = set()

        for i, query in enumerate(queries):
            logger.info("━━ 쿼리 [%d/%d]: '%s' ━━", i + 1, len(queries), query)

            for record in self.search(query, max_records=max_per_query):
                record_id = record.get("id")
                if record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                yield record

        logger.info("전체 검색 완료: 고유 레코드 %d건", len(seen_ids))

    # ─── 상세 조회 메서드 ──────────────────────────────────

    def get_record(self, record_id: int) -> Optional[dict]:
        """
        레코드 ID로 상세 정보를 조회합니다.
        (파일 목록 포함)

        Args:
            record_id: Zenodo 레코드 ID

        Returns:
            레코드 dict 또는 None
        """
        url = f"{self.base_url}/records/{record_id}"
        data = self._get(url)

        if data is None:
            logger.error("레코드 조회 실패: %d", record_id)

        return data

    def get_record_files(self, record_id: int) -> list:
        """
        레코드의 파일 목록을 조회합니다.

        Args:
            record_id: Zenodo 레코드 ID

        Returns:
            파일 정보 리스트
        """
        url = f"{self.base_url}/records/{record_id}/files"
        data = self._get(url)

        if data is None:
            return []

        return data.get("entries", data.get("contents", []))

    # ─── 유틸리티 ──────────────────────────────────────────

    def test_connection(self) -> bool:
        """API 연결 및 인증 상태를 테스트합니다."""
        url = f"{self.base_url}/records"
        params = {"q": "test", "size": 1}

        data = self._get(url, params)
        if data is not None:
            total = data.get("hits", {}).get("total", 0)
            logger.info("Zenodo 연결 성공 (총 레코드: %d)", total)
            auth = "인증됨" if self.token else "미인증 (토큰 없음)"
            logger.info("인증 상태: %s", auth)
            return True

        logger.error("Zenodo 연결 실패")
        return False

    def close(self):
        """세션을 종료합니다."""
        self.session.close()
