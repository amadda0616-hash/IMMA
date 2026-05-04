"""
metadata.py — 메타데이터 저장/관리 모듈

수집된 레코드와 파일의 메타데이터를 JSON + CSV로 저장합니다.
재실행 시 이미 처리된 레코드를 건너뛰기 위한 인덱스도 관리합니다.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import METADATA_DIR

logger = logging.getLogger(__name__)


class MetadataStore:
    """메타데이터 저장소"""

    def __init__(self, output_dir: Path = METADATA_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 파일 경로
        self._records_json = self.output_dir / "records.json"
        self._records_csv = self.output_dir / "records.csv"
        self._index_file = self.output_dir / "processed_ids.json"

        # 메모리 인덱스: 이미 처리된 레코드 ID
        self._processed_ids: set = self._load_index()
        self._records: list = []

    # ─── 인덱스 관리 (중복 방지) ─────────────────────────

    def _load_index(self) -> set:
        """처리 완료된 레코드 ID 목록을 로드합니다."""
        if self._index_file.exists():
            try:
                data = json.loads(self._index_file.read_text(encoding="utf-8"))
                ids = set(data.get("processed_ids", []))
                logger.info("기존 처리 인덱스 로드: %d건", len(ids))
                return ids
            except (json.JSONDecodeError, KeyError):
                logger.warning("인덱스 파일 손상. 초기화합니다.")
        return set()

    def _save_index(self):
        """처리 완료된 레코드 ID 목록을 저장합니다."""
        data = {
            "processed_ids": sorted(self._processed_ids),
            "total_count": len(self._processed_ids),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self._index_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_processed(self, record_id) -> bool:
        """레코드가 이미 처리되었는지 확인합니다."""
        return str(record_id) in self._processed_ids

    def mark_processed(self, record_id):
        """레코드를 처리 완료로 표시합니다."""
        self._processed_ids.add(str(record_id))

    # ─── 레코드 저장 ─────────────────────────────────────

    def save_record(
        self,
        record: dict,
        filter_result: dict,
        downloaded_paths: list,
    ):
        """
        단일 레코드의 메타데이터를 저장합니다.

        Args:
            record: Zenodo API 원본 레코드
            filter_result: DrawingFilter.evaluate_record() 결과
            downloaded_paths: 다운로드된 파일 경로 리스트
        """
        metadata = record.get("metadata", record)
        record_id = record.get("id", record.get("record_id", "unknown"))

        entry = {
            # 기본 정보
            "record_id": str(record_id),
            "source": "zenodo",
            "title": metadata.get("title", ""),
            "description": _clean_html(metadata.get("description", ""))[:500],
            "doi": record.get("doi", ""),
            "url": f"https://zenodo.org/records/{record_id}",

            # 라이선스
            "license": _extract_license(metadata),
            "access_right": metadata.get("access_right", ""),

            # 키워드/분류
            "keywords": metadata.get("keywords", []),
            "category": filter_result.get("category", "uncategorized"),

            # 필터 결과
            "include_score": filter_result.get("include_score", 0),
            "matched_include": filter_result.get("matched_include", []),

            # 파일 정보
            "downloaded_files": downloaded_paths,
            "file_count": len(downloaded_paths),

            # 타임스탬프
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "published_date": metadata.get("publication_date", ""),
        }

        self._records.append(entry)
        self.mark_processed(record_id)

        # 매 레코드 저장 (안전)
        self._append_to_json(entry)
        self._append_to_csv(entry)
        self._save_index()

        logger.debug("메타데이터 저장: record %s", record_id)

    # ─── 파일 I/O ────────────────────────────────────────

    def _append_to_json(self, entry: dict):
        """JSON 파일에 레코드를 추가합니다."""
        # 기존 데이터 로드
        records = []
        if self._records_json.exists():
            try:
                records = json.loads(
                    self._records_json.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                records = []

        records.append(entry)

        self._records_json.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _append_to_csv(self, entry: dict):
        """CSV 파일에 레코드를 추가합니다."""
        fieldnames = [
            "record_id", "source", "title", "category",
            "include_score", "license", "file_count",
            "url", "collected_at", "published_date",
        ]

        file_exists = self._records_csv.exists()

        with open(self._records_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    # ─── 통계 ────────────────────────────────────────────

    def get_summary(self) -> dict:
        """수집 요약 통계를 반환합니다."""
        category_counts = {}
        total_files = 0

        for r in self._records:
            cat = r.get("category", "uncategorized")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_files += r.get("file_count", 0)

        return {
            "total_records": len(self._records),
            "total_files_downloaded": total_files,
            "total_processed_all_time": len(self._processed_ids),
            "category_distribution": category_counts,
        }

    def print_summary(self):
        """수집 요약을 로그에 출력합니다."""
        summary = self.get_summary()

        logger.info("=" * 60)
        logger.info("📋 수집 요약 리포트")
        logger.info("=" * 60)
        logger.info("  총 레코드:          %d건", summary["total_records"])
        logger.info("  총 다운로드 파일:    %d개", summary["total_files_downloaded"])
        logger.info("  누적 처리 레코드:    %d건", summary["total_processed_all_time"])
        logger.info("")
        logger.info("  카테고리별 분포:")
        for cat, count in sorted(summary["category_distribution"].items()):
            logger.info("    %-20s %d건", cat, count)
        logger.info("=" * 60)


# ─── 유틸리티 함수 ────────────────────────────────────────

def _clean_html(text: str) -> str:
    """HTML 태그를 제거합니다."""
    import re
    return re.sub(r"<[^>]+>", " ", text).strip()


def _extract_license(metadata: dict) -> str:
    """메타데이터에서 라이선스 정보를 추출합니다."""
    license_info = metadata.get("license", {})
    if isinstance(license_info, dict):
        return license_info.get("id", str(license_info))
    return str(license_info)
