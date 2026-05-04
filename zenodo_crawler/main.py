#!/usr/bin/env python3
"""
main.py — 2D 가공 도면 통합 크롤러 CLI

지원 소스: Zenodo, Mendeley Data

사용법:
    # 전체 크롤링 (모든 소스, 검색 + 필터 + 다운로드)
    python main.py

    # dry-run 모드: 다운로드 없이 검색+필터 결과만 확인
    python main.py --dry-run

    # 특정 소스만 실행
    python main.py --source zenodo
    python main.py --source mendeley
    python main.py --source all           # (기본값)

    # 특정 쿼리만 실행
    python main.py --query "welding drawing"

    # 최대 수집 수 제한
    python main.py --max-records 100

    # 커스텀 출력 경로
    python main.py --output ./my_drawings

    # 통계만 표시 (기존 메타데이터 기준)
    python main.py --stats-only
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# 프로젝트 모듈 import
from config import (
    SEARCH_QUERIES, MENDELEY_SEARCH_QUERIES,
    MAX_RECORDS_PER_QUERY, MENDELEY_MAX_PER_QUERY,
    MAX_TOTAL_RECORDS, ENABLED_SOURCES,
    BASE_OUTPUT_DIR, LOG_DIR, ensure_dirs,
)
from zenodo_client import ZenodoClient
from mendeley_client import MendeleyClient
from filters import DrawingFilter
from downloader import FileDownloader
from metadata import MetadataStore


def setup_logging(log_dir: Path, verbose: bool = False):
    """로깅 설정: 콘솔 + 파일 동시 출력"""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"crawl_{timestamp}.log"

    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # 파일 핸들러 (항상 DEBUG)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    logging.info("로그 파일: %s", log_file)
    return log_file


def parse_args():
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="2D 가공 도면 통합 크롤러 (Zenodo + Mendeley)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --dry-run                  # 검색만 (다운로드 X)
  python main.py --source zenodo            # Zenodo만 수집
  python main.py --source mendeley          # Mendeley만 수집
  python main.py --source all               # 모든 소스 (기본)
  python main.py --max-records 50           # 50건만 수집
  python main.py --query "CNC drawing"      # 특정 쿼리만
  python main.py --verbose                  # 상세 로그
  python main.py --stats-only               # 기존 통계만 보기
        """,
    )

    parser.add_argument(
        "--source", type=str, default="all",
        choices=["all", "zenodo", "mendeley"],
        help="크롤링 소스를 선택합니다 (기본: all).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="다운로드 없이 검색 + 필터 결과만 확인합니다.",
    )
    parser.add_argument(
        "--query", type=str, default="",
        help="단일 검색 쿼리를 지정합니다 (기본: config.py의 전체 쿼리 사용).",
    )
    parser.add_argument(
        "--max-records", type=int, default=MAX_TOTAL_RECORDS,
        help=f"전체 최대 수집 레코드 수 (기본: {MAX_TOTAL_RECORDS}).",
    )
    parser.add_argument(
        "--max-per-query", type=int, default=0,
        help="쿼리당 최대 레코드 수 (기본: 소스별 설정값 사용).",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="출력 디렉토리 경로 (기본: ./output).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="상세 디버그 로그를 표시합니다.",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="기존 메타데이터 기준으로 통계만 표시합니다.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# 소스별 크롤링 함수
# ─────────────────────────────────────────────────────────────

def crawl_source(
    source_name: str,
    record_iterator,
    get_detail_fn,
    drawing_filter: DrawingFilter,
    downloader: FileDownloader,
    metadata_store: MetadataStore,
    args,
    max_records: int,
) -> dict:
    """
    단일 소스 크롤링 공통 로직.

    Args:
        source_name: 소스 이름 ("zenodo", "mendeley")
        record_iterator: 검색 결과 제너레이터
        get_detail_fn: 상세 조회 함수 (record_id → dict)
        drawing_filter: 필터 인스턴스
        downloader: 다운로더 인스턴스
        metadata_store: 메타데이터 저장소
        args: CLI 인자
        max_records: 이 소스에서 수집할 최대 레코드 수

    Returns:
        {"searched": int, "accepted": int, "rejected": int, "skipped": int}
    """
    logger = logging.getLogger(f"crawl.{source_name}")
    stats = {"searched": 0, "accepted": 0, "rejected": 0, "skipped": 0}

    for record in record_iterator:
        record_id = record.get("id", "unknown")
        stats["searched"] += 1

        # 최대 수집 수 체크
        if stats["accepted"] >= max_records:
            logger.info("[%s] 최대 수집 수 도달 (%d건).", source_name, max_records)
            break

        # 이미 처리된 레코드 건너뛰기
        if metadata_store.is_processed(record_id):
            stats["skipped"] += 1
            continue

        # 상세 조회 (소스별 방식)
        if get_detail_fn:
            detailed = get_detail_fn(record_id)
            if detailed is None:
                logger.warning("[%s] 상세 조회 실패: %s", source_name, record_id)
                continue
        else:
            # Mendeley는 검색 결과에 이미 파일 정보 포함 가능
            detailed = record

        # 필터 평가
        result = drawing_filter.evaluate_record(detailed)

        if not result["accepted"]:
            stats["rejected"] += 1
            if args.dry_run and args.verbose:
                title = detailed.get("metadata", {}).get("title", "")[:60]
                logger.debug(
                    "  ❌ [%s] %s — %s",
                    record_id, title, result["reason"]
                )
            continue

        # 통과
        stats["accepted"] += 1
        title = detailed.get("metadata", {}).get("title", "")[:60]
        logger.info(
            "  ✅ [%s] #%d [%s] %s (점수:%d 카테고리:%s 파일:%d개)",
            source_name, stats["accepted"], record_id, title,
            result["include_score"],
            result["category"],
            len(result["eligible_files"]),
        )

        # 다운로드
        downloaded_paths = []
        if not args.dry_run:
            downloaded_paths = downloader.download_files(
                record_id=record_id,
                files=result["eligible_files"],
                category=result["category"],
            )

        # 메타데이터 저장
        metadata_store.save_record(detailed, result, downloaded_paths)

        # 진행률 (30건마다)
        if stats["accepted"] % 30 == 0:
            logger.info(
                "─── [%s] 진행: 검색 %d → 통과 %d, 제외 %d, 건너뜀 %d ───",
                source_name, stats["searched"], stats["accepted"],
                stats["rejected"], stats["skipped"]
            )

    return stats


# ─────────────────────────────────────────────────────────────
# 메인 크롤링 실행
# ─────────────────────────────────────────────────────────────

def run_crawler(args):
    """크롤링 메인 루프"""
    logger = logging.getLogger("main")

    # ── 출력 디렉토리 설정 ────────────────────────────────
    if args.output:
        import config
        config.BASE_OUTPUT_DIR = Path(args.output)
        config.DOWNLOAD_DIR = config.BASE_OUTPUT_DIR / "downloads"
        config.METADATA_DIR = config.BASE_OUTPUT_DIR / "metadata"
        config.LOG_DIR = config.BASE_OUTPUT_DIR / "logs"
        for cat in config.CATEGORY_KEYWORDS:
            config.CATEGORY_DIRS[cat] = config.DOWNLOAD_DIR / cat
        config.CATEGORY_DIRS["uncategorized"] = config.DOWNLOAD_DIR / "uncategorized"

    ensure_dirs()

    # ── 공통 컴포넌트 ────────────────────────────────────
    drawing_filter = DrawingFilter()
    metadata_store = MetadataStore()
    downloader = FileDownloader()

    # ── 통계만 보기 ──────────────────────────────────────
    if args.stats_only:
        metadata_store.print_summary()
        return

    # ── 소스 결정 ────────────────────────────────────────
    run_zenodo = args.source in ("all", "zenodo") and ENABLED_SOURCES.get("zenodo", True)
    run_mendeley = args.source in ("all", "mendeley") and ENABLED_SOURCES.get("mendeley", True)

    logger.info("=" * 60)
    logger.info("🔧 2D 가공 도면 통합 크롤러 시작")
    logger.info("=" * 60)
    logger.info("  활성 소스: %s",
        ", ".join(s for s, on in [("Zenodo", run_zenodo), ("Mendeley", run_mendeley)] if on)
    )
    logger.info("  최대 수집: %d건", args.max_records)
    logger.info("  모드: %s", "DRY-RUN" if args.dry_run else "FULL (다운로드 포함)")
    logger.info("")

    all_stats = {}
    remaining = args.max_records

    try:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 1: Zenodo
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if run_zenodo and remaining > 0:
            logger.info("━" * 60)
            logger.info("📦 [Phase 1/2] Zenodo 크롤링 시작")
            logger.info("━" * 60)

            zenodo = ZenodoClient()
            if not zenodo.test_connection():
                logger.error("Zenodo 연결 실패. 건너뜁니다.")
            else:
                # 쿼리 결정
                queries = [args.query] if args.query else SEARCH_QUERIES
                max_pq = args.max_per_query or MAX_RECORDS_PER_QUERY

                record_iter = zenodo.search_all_queries(queries, max_pq)
                stats = crawl_source(
                    source_name="zenodo",
                    record_iterator=record_iter,
                    get_detail_fn=zenodo.get_record,
                    drawing_filter=drawing_filter,
                    downloader=downloader,
                    metadata_store=metadata_store,
                    args=args,
                    max_records=remaining,
                )
                all_stats["zenodo"] = stats
                remaining -= stats["accepted"]
                zenodo.close()

            logger.info("")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Phase 2: Mendeley Data
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if run_mendeley and remaining > 0:
            logger.info("━" * 60)
            logger.info("📦 [Phase 2/2] Mendeley Data 크롤링 시작")
            logger.info("━" * 60)

            mendeley = MendeleyClient()
            if not mendeley.test_connection():
                logger.error("Mendeley 연결 실패. 건너뜁니다.")
            else:
                # 쿼리 결정
                queries = [args.query] if args.query else MENDELEY_SEARCH_QUERIES
                max_pq = args.max_per_query or MENDELEY_MAX_PER_QUERY

                record_iter = mendeley.search_all_queries(queries, max_pq)

                # Mendeley는 검색 결과에 파일 포함 가능 → 상세 조회는 선택적
                def mendeley_get_detail(record_id):
                    """Mendeley 상세 조회: mendeley_ 접두사 제거"""
                    clean_id = str(record_id).replace("mendeley_", "")
                    return mendeley.get_dataset(clean_id)

                stats = crawl_source(
                    source_name="mendeley",
                    record_iterator=record_iter,
                    get_detail_fn=mendeley_get_detail,
                    drawing_filter=drawing_filter,
                    downloader=downloader,
                    metadata_store=metadata_store,
                    args=args,
                    max_records=remaining,
                )
                all_stats["mendeley"] = stats
                remaining -= stats["accepted"]
                mendeley.close()

    except KeyboardInterrupt:
        logger.warning("\n⚠️  사용자에 의해 중단되었습니다.")

    finally:
        # ── 최종 요약 ────────────────────────────────────
        logger.info("")
        logger.info("━" * 60)
        logger.info("📊 최종 결과 (소스별)")
        logger.info("━" * 60)

        grand = {"searched": 0, "accepted": 0, "rejected": 0, "skipped": 0}

        for source, stats in all_stats.items():
            logger.info(
                "  [%-10s] 검색: %4d → 통과: %4d, 제외: %4d, 건너뜀: %4d",
                source,
                stats["searched"], stats["accepted"],
                stats["rejected"], stats["skipped"],
            )
            for k in grand:
                grand[k] += stats[k]

        logger.info("  " + "─" * 54)
        logger.info(
            "  [%-10s] 검색: %4d → 통과: %4d, 제외: %4d, 건너뜀: %4d",
            "합계", grand["searched"], grand["accepted"],
            grand["rejected"], grand["skipped"],
        )
        logger.info("")

        if not args.dry_run:
            downloader.print_stats()

        metadata_store.print_summary()
        logger.info("크롤러 종료.")


def main():
    args = parse_args()
    log_file = setup_logging(LOG_DIR, verbose=args.verbose)
    run_crawler(args)


if __name__ == "__main__":
    main()
