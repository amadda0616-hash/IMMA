"""
downloader.py — 파일 다운로드 모듈

Zenodo 레코드의 파일을 카테고리별 폴더에 다운로드합니다.
재개(resume), 중복 건너뛰기, 진행률 표시를 지원합니다.
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests

from config import (
    CATEGORY_DIRS, RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    ZENODO_ACCESS_TOKEN,
)

logger = logging.getLogger(__name__)


class FileDownloader:
    """파일 다운로드 매니저"""

    def __init__(self, session: Optional[requests.Session] = None):
        """
        Args:
            session: 기존 requests.Session 재사용 (ZenodoClient의 세션)
        """
        self.session = session or requests.Session()
        self._downloaded_count = 0
        self._skipped_count = 0
        self._failed_count = 0

    @property
    def stats(self) -> dict:
        return {
            "downloaded": self._downloaded_count,
            "skipped": self._skipped_count,
            "failed": self._failed_count,
        }

    def download_files(
        self,
        record_id: int,
        files: list,
        category: str,
    ) -> list:
        """
        레코드의 파일들을 카테고리 폴더에 다운로드합니다.

        Args:
            record_id: Zenodo 레코드 ID (서브폴더 이름)
            files: 파일 정보 리스트 (filters.py에서 걸러진 eligible_files)
            category: 도면 카테고리 ("welding", "cnc_machining" 등)

        Returns:
            다운로드된 파일 경로 리스트
        """
        # 카테고리 폴더 / 레코드 ID 폴더
        target_dir = CATEGORY_DIRS.get(category, CATEGORY_DIRS["uncategorized"])
        record_dir = target_dir / str(record_id)
        record_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths = []

        for f in files:
            filename = f.get("key", f.get("filename", "unknown"))
            file_size = f.get("size", 0)
            download_url = self._get_download_url(f)

            if not download_url:
                logger.warning("다운로드 URL 없음: %s (record %d)", filename, record_id)
                self._failed_count += 1
                continue

            target_path = record_dir / filename

            # 중복 체크: 이미 같은 크기의 파일이 존재하면 건너뜀
            if target_path.exists():
                existing_size = target_path.stat().st_size
                if file_size == 0 or abs(existing_size - file_size) < 100:
                    logger.debug("건너뜀 (이미 존재): %s", target_path)
                    self._skipped_count += 1
                    downloaded_paths.append(str(target_path))
                    continue

            # 다운로드 수행
            success = self._download_single(download_url, target_path, file_size)
            if success:
                downloaded_paths.append(str(target_path))
                self._downloaded_count += 1
            else:
                self._failed_count += 1

        return downloaded_paths

    def _download_single(
        self,
        url: str,
        target_path: Path,
        expected_size: int = 0,
    ) -> bool:
        """
        단일 파일을 다운로드합니다.

        Args:
            url: 다운로드 URL
            target_path: 저장 경로
            expected_size: 예상 파일 크기 (검증용)

        Returns:
            성공 여부
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    "다운로드: %s → %s (시도 %d/%d)",
                    target_path.name, target_path.parent.name,
                    attempt, MAX_RETRIES
                )

                # 스트리밍 다운로드 (대용량 파일 메모리 절약)
                headers = {}
                if ZENODO_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {ZENODO_ACCESS_TOKEN}"

                resp = self.session.get(
                    url,
                    stream=True,
                    timeout=REQUEST_TIMEOUT * 3,  # 다운로드는 타임아웃 여유
                    headers=headers,
                )
                resp.raise_for_status()

                # 임시 파일에 쓰기 (다운로드 중단 시 불완전 파일 방지)
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

                total = int(resp.headers.get("Content-Length", 0)) or expected_size
                downloaded = 0

                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # 크기 검증
                if expected_size > 0 and abs(downloaded - expected_size) > 100:
                    logger.warning(
                        "크기 불일치: %s (기대 %d, 실제 %d)",
                        target_path.name, expected_size, downloaded
                    )

                # 임시 → 최종 파일 이동
                tmp_path.rename(target_path)

                size_mb = downloaded / (1024 * 1024)
                logger.info("완료: %s (%.2f MB)", target_path.name, size_mb)

                # Rate limit 대기
                time.sleep(RATE_LIMIT_DELAY)
                return True

            except requests.exceptions.RequestException as e:
                logger.warning(
                    "다운로드 실패 (시도 %d/%d): %s — %s",
                    attempt, MAX_RETRIES, target_path.name, e
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

            except OSError as e:
                logger.error("파일 저장 실패: %s — %s", target_path, e)
                return False

        logger.error("다운로드 최종 실패: %s", target_path.name)
        return False

    @staticmethod
    def _get_download_url(file_info: dict) -> Optional[str]:
        """파일 정보에서 다운로드 URL을 추출합니다."""
        # Zenodo API v2 구조
        links = file_info.get("links", {})
        if isinstance(links, dict):
            # 우선순위: self → content → download
            for key in ("self", "content", "download"):
                if key in links:
                    return links[key]

        # 직접 URL
        if "url" in file_info:
            return file_info["url"]

        # Zenodo legacy: bucket + key
        bucket = file_info.get("bucket", "")
        key = file_info.get("key", "")
        if bucket and key:
            return f"https://zenodo.org/api/files/{bucket}/{key}"

        return None

    def print_stats(self):
        """다운로드 통계를 출력합니다."""
        logger.info(
            "📊 다운로드 통계: 완료=%d | 건너뜀=%d | 실패=%d",
            self._downloaded_count, self._skipped_count, self._failed_count
        )
