"""
filters.py — 도면 필터링 모듈

크롤링된 Zenodo 레코드를 2단계로 필터링합니다.
  1단계: 제외 키워드 검사 (건축, 전기, 토목 등 무관 분야 제거)
  2단계: 포함 키워드 점수 계산 (용접, CNC, 선반, 설계 관련도 산정)
"""

import re
import logging
from typing import Optional

from config import (
    INCLUDE_KEYWORDS, INCLUDE_WEIGHTS, MIN_INCLUDE_SCORE,
    EXCLUDE_KEYWORDS, ALLOWED_EXTENSIONS, ALLOWED_LICENSES,
    MAX_FILE_SIZE, MIN_FILE_SIZE, ENFORCE_LICENSE_FILTER,
    CATEGORY_KEYWORDS,
)

logger = logging.getLogger(__name__)


class DrawingFilter:
    """Zenodo 레코드 필터링 클래스"""

    def __init__(self):
        # 제외 키워드를 정규식으로 사전 컴파일 (성능 최적화)
        self._exclude_patterns = [
            re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
            for kw in EXCLUDE_KEYWORDS
        ]

        # 포함 키워드를 레벨별로 정규식 컴파일
        self._include_patterns = {}
        for level, keywords in INCLUDE_KEYWORDS.items():
            self._include_patterns[level] = [
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                for kw in keywords
            ]

        # 카테고리 키워드 정규식 컴파일
        self._category_patterns = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            self._category_patterns[category] = [
                re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
                for kw in keywords
            ]

    # ─── 메인 필터 메서드 ──────────────────────────────────

    def evaluate_record(self, record: dict) -> dict:
        """
        Zenodo 레코드를 평가하여 필터 결과를 반환합니다.

        Args:
            record: Zenodo API 응답의 단일 레코드 (dict)

        Returns:
            {
                "accepted": bool,         # 다운로드 대상 여부
                "reason": str,            # 판정 사유
                "include_score": int,     # 포함 점수
                "matched_include": list,  # 매칭된 포함 키워드
                "matched_exclude": list,  # 매칭된 제외 키워드
                "category": str,          # 분류 카테고리
                "eligible_files": list,   # 다운로드 대상 파일 목록
            }
        """
        text = self._extract_searchable_text(record)

        result = {
            "accepted": False,
            "reason": "",
            "include_score": 0,
            "matched_include": [],
            "matched_exclude": [],
            "category": "uncategorized",
            "eligible_files": [],
        }

        # ── Step 1: 제외 키워드 검사 ─────────────────────────
        excluded = self._check_exclusion(text)
        if excluded:
            result["matched_exclude"] = excluded
            result["reason"] = f"제외 키워드 매칭: {', '.join(excluded[:3])}"
            logger.debug("제외됨 [%s]: %s", record.get("id"), result["reason"])
            return result

        # ── Step 2: 라이선스 검사 ─────────────────────────────
        if ENFORCE_LICENSE_FILTER:
            license_ok = self._check_license(record)
            if not license_ok:
                result["reason"] = "라이선스 미허용"
                logger.debug("라이선스 제외 [%s]", record.get("id"))
                return result

        # ── Step 3: 포함 점수 계산 ────────────────────────────
        score, matched = self._calculate_include_score(text)
        result["include_score"] = score
        result["matched_include"] = matched

        if score < MIN_INCLUDE_SCORE:
            result["reason"] = f"포함 점수 미달 ({score} < {MIN_INCLUDE_SCORE})"
            logger.debug("점수 미달 [%s]: %d점", record.get("id"), score)
            return result

        # ── Step 4: 파일 형식 필터 ────────────────────────────
        eligible = self._filter_files(record)
        if not eligible:
            result["reason"] = "적합한 파일 형식 없음"
            logger.debug("파일 없음 [%s]", record.get("id"))
            return result

        result["eligible_files"] = eligible

        # ── Step 5: 카테고리 분류 ─────────────────────────────
        result["category"] = self._classify_category(text)

        # ── 최종 판정: 통과 ───────────────────────────────────
        result["accepted"] = True
        result["reason"] = f"통과 (점수: {score}, 카테고리: {result['category']})"
        logger.info(
            "통과 [%s] 점수=%d 카테고리=%s 파일=%d개",
            record.get("id"), score, result["category"], len(eligible)
        )

        return result

    # ─── 내부 메서드 ───────────────────────────────────────

    def _extract_searchable_text(self, record: dict) -> str:
        """레코드에서 검색 대상 텍스트를 추출합니다."""
        parts = []

        # 메타데이터 필드 추출
        metadata = record.get("metadata", record)

        # 제목 (가중치를 위해 2번 반복)
        title = metadata.get("title", "")
        parts.extend([title, title])

        # 설명
        description = metadata.get("description", "")
        # HTML 태그 제거
        description = re.sub(r"<[^>]+>", " ", description)
        parts.append(description)

        # 키워드 목록
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, list):
            parts.extend(keywords)

        # 주제 (subjects)
        subjects = metadata.get("subjects", [])
        if isinstance(subjects, list):
            for s in subjects:
                if isinstance(s, dict):
                    parts.append(s.get("term", ""))
                else:
                    parts.append(str(s))

        # 노트
        notes = metadata.get("notes", "")
        parts.append(notes)

        return " ".join(parts)

    def _check_exclusion(self, text: str) -> list:
        """제외 키워드 매칭 검사. 매칭된 키워드 리스트를 반환합니다."""
        matched = []
        for i, pattern in enumerate(self._exclude_patterns):
            if pattern.search(text):
                matched.append(EXCLUDE_KEYWORDS[i])
        return matched

    def _calculate_include_score(self, text: str) -> tuple:
        """
        포함 키워드 점수를 계산합니다.

        Returns:
            (score: int, matched_keywords: list)
        """
        total_score = 0
        matched = []

        for level, patterns in self._include_patterns.items():
            weight = INCLUDE_WEIGHTS[level]
            for i, pattern in enumerate(patterns):
                if pattern.search(text):
                    kw = INCLUDE_KEYWORDS[level][i]
                    total_score += weight
                    matched.append(f"{kw}({level}:{weight})")

        return total_score, matched

    def _check_license(self, record: dict) -> bool:
        """라이선스가 허용 목록에 있는지 검사합니다."""
        metadata = record.get("metadata", record)

        # Zenodo 라이선스 구조: metadata.license.id 또는 metadata.access_right
        license_info = metadata.get("license", {})
        if isinstance(license_info, dict):
            license_id = license_info.get("id", "").lower()
        else:
            license_id = str(license_info).lower()

        access_right = metadata.get("access_right", "").lower()

        # 허용 라이선스 체크
        for allowed in ALLOWED_LICENSES:
            if allowed in license_id or allowed in access_right:
                return True

        # open access는 허용
        if access_right == "open":
            return True

        return False

    def _filter_files(self, record: dict) -> list:
        """
        레코드의 파일 목록에서 다운로드 대상 파일만 필터링합니다.

        Returns:
            적합한 파일 정보 리스트:
            [{"key": "filename.dxf", "size": 12345, "links": {"self": "url"}}]
        """
        eligible = []
        files = record.get("files", [])

        for f in files:
            filename = f.get("key", f.get("filename", ""))
            size = f.get("size", 0)
            ext = self._get_extension(filename)

            # 확장자 검사
            if ext not in ALLOWED_EXTENSIONS:
                continue

            # 파일 크기 검사
            if size > MAX_FILE_SIZE:
                logger.debug("파일 크기 초과: %s (%d bytes)", filename, size)
                continue
            if 0 < size < MIN_FILE_SIZE:
                logger.debug("파일 크기 미달: %s (%d bytes)", filename, size)
                continue

            eligible.append(f)

        return eligible

    def _classify_category(self, text: str) -> str:
        """
        텍스트를 분석하여 도면 카테고리를 결정합니다.
        가장 많이 매칭된 카테고리를 반환합니다.
        """
        scores = {}
        for category, patterns in self._category_patterns.items():
            count = sum(1 for p in patterns if p.search(text))
            if count > 0:
                scores[category] = count

        if not scores:
            return "uncategorized"

        # 최고 점수 카테고리 반환
        return max(scores, key=scores.get)

    @staticmethod
    def _get_extension(filename: str) -> str:
        """파일명에서 확장자를 추출합니다 (소문자 변환)."""
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ""


# ─── 유틸리티 함수 ────────────────────────────────────────

def create_filter() -> DrawingFilter:
    """DrawingFilter 싱글턴 생성 헬퍼"""
    return DrawingFilter()
