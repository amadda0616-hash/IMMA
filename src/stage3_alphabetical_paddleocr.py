"""
src/stage3_alphabetical_paddleocr.py

★ Phase 15c — pipeline.py Stage 3-A backend wrapper (PaddleOCR-VL-1.5, D-039).

Long-running subprocess + JSON line protocol — `.venv-paddleocr` 에서 worker 실행.

★ Same interface as `src/stage3_alphabetical.py` (pipeline.py 호환):
    - load_model(model_name, device) -> (processor, model, device) 형태 반환
    - predict_one(image_path, region_type, mode, processor, model, device,
                  questions, language_hint) -> JSON dict

★ 다른 backend (Donut DocVQA, D-018 폐기) 와 교환 가능 — 단순 import 변경만.

CLI (디버깅용):
    PYTHONPATH=. python src/stage3_alphabetical_paddleocr.py \
        --image data/<crop>.jpg --region-type titleblock --device cuda:0
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root bootstrap (D-049 패턴)
_PROJECT_ROOT_BOOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_BOOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_BOOT))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV_PYTHON = PROJECT_ROOT / ".venv-paddleocr" / "bin" / "python"
WORKER_SCRIPT = PROJECT_ROOT / "src" / "stage3_alphabetical_paddleocr_worker.py"
# ★ Phase 15c (2026-05-06) — worker stderr 를 파일로 redirect.
# 이유: subprocess.PIPE 의 stderr buffer (~64KB) 가 transformers 5.0.0 warning
# 다수로 가득 차 worker write block → READY signal 못 보냄 → wrapper 무한 대기 (deadlock).
# 파일 redirect 시 buffer 무한 → deadlock 0.
DEFAULT_STDERR_LOG = PROJECT_ROOT / "outputs" / "paddleocr_worker.stderr.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stage3_alphabetical_paddleocr")


# ===========================================================================
# Subprocess worker manager
# ===========================================================================
class PaddleOCRWorker:
    """Long-running subprocess wrapper for PaddleOCR-VL-1.5 (D-039 backend).

    Communication: JSON line protocol via stdin/stdout.
    Errors / logs: subprocess stderr (printed to parent stderr on failure).
    """

    READY_TIMEOUT = 300   # 모델 로드 ~50초 + 여유 (cold start)
    REQ_TIMEOUT = 180     # per-request timeout (large patches)

    def __init__(self, device: str = "cuda:0",
                 venv_python: Optional[Path] = None,
                 worker_script: Optional[Path] = None,
                 stderr_log: Optional[Path] = None):
        self.device = device
        self.venv_python = venv_python or DEFAULT_VENV_PYTHON
        self.worker_script = worker_script or WORKER_SCRIPT
        self.stderr_log = stderr_log or DEFAULT_STDERR_LOG
        self.proc: Optional[subprocess.Popen] = None
        self._stderr_fp = None  # opened file handle (close 시 정리)

    # -----------------------------------------------------------------
    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return  # already running

        if not self.venv_python.exists():
            raise RuntimeError(
                f".venv-paddleocr python not found: {self.venv_python}\n"
                f"Phase 15a 환경 (별도 venv) 을 먼저 설정하세요."
            )
        if not self.worker_script.exists():
            raise RuntimeError(f"Worker script not found: {self.worker_script}")

        log.info("Launching PaddleOCR-VL worker subprocess (device=%s)", self.device)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        # ★ Phase 15c — stderr 를 파일로 redirect (PIPE deadlock 방지)
        self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_fp = open(self.stderr_log, "w", encoding="utf-8")
        log.info("Worker stderr → %s", self.stderr_log)

        self.proc = subprocess.Popen(
            [str(self.venv_python), str(self.worker_script),
             "--device", self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fp,    # ★ 파일 redirect (PIPE 아님)
            text=True,
            bufsize=1,
            env=env,
        )

        # READY 대기 (최대 300초)
        t0 = time.perf_counter()
        ready_msg = None
        while time.perf_counter() - t0 < self.READY_TIMEOUT:
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.poll()
                if rc is not None:
                    err_tail = self._read_stderr_tail()
                    raise RuntimeError(
                        f"Worker died during init (rc={rc}). stderr tail:\n{err_tail}"
                    )
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore non-JSON lines
            if msg.get("status") == "ready":
                ready_msg = msg
                break
            if msg.get("status") == "init_error":
                raise RuntimeError(f"Worker init_error: {msg.get('error')}")

        if ready_msg is None:
            self.shutdown()
            raise TimeoutError(
                f"Worker did not signal READY within {self.READY_TIMEOUT}s"
            )
        elapsed = time.perf_counter() - t0
        log.info("★ Worker READY  (load=%.1fs)", elapsed)

    # -----------------------------------------------------------------
    def predict(self, image_path: Path,
                region_type: str = "titleblock",
                language_hint: Optional[str] = None) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            self.start()

        req = {
            "image_path": str(Path(image_path).resolve()),
            "region_type": region_type,
        }
        if language_hint:
            req["language_hint"] = language_hint

        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            err_tail = self._read_stderr_tail()
            raise RuntimeError(f"Worker pipe broken. stderr tail:\n{err_tail}")

        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            err_tail = self._read_stderr_tail()
            raise RuntimeError(
                f"Worker no response (rc={rc}). stderr tail:\n{err_tail}"
            )

        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"error": f"json_decode_response", "raw": line[:500]}

    # -----------------------------------------------------------------
    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired, OSError):
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            self.proc = None
            # stderr file handle 정리
            if self._stderr_fp is not None:
                try:
                    self._stderr_fp.close()
                except OSError:
                    pass
                self._stderr_fp = None
            log.info("Worker shut down")

    # -----------------------------------------------------------------
    def _read_stderr_tail(self, max_chars: int = 4000) -> str:
        """Worker 가 죽었을 때 stderr 파일 마지막 부분 회수."""
        try:
            if not self.stderr_log.exists():
                return "(no stderr log)"
            data = self.stderr_log.read_text(encoding="utf-8", errors="replace")
            return data[-max_chars:] if len(data) > max_chars else data
        except OSError as e:
            return f"(stderr read failed: {e})"


# ===========================================================================
# Singleton instance + pipeline.py 호환 인터페이스
# ===========================================================================
_worker: Optional[PaddleOCRWorker] = None


def _get_worker(device: str = "cuda:0") -> PaddleOCRWorker:
    global _worker
    if _worker is None or _worker.proc is None or _worker.proc.poll() is not None:
        _worker = PaddleOCRWorker(device=device)
        _worker.start()
        atexit.register(_worker.shutdown)
    return _worker


# --------- pipeline.py 가 import 하는 이름 ---------
def load_model(model_name: str = "PaddlePaddle/PaddleOCR-VL-1.5",
               device: Optional[str] = None) -> Tuple[Any, Any, str]:
    """Compatibility wrapper.

    실제 모델은 subprocess 가 보유 — placeholder (None, None, device) 반환.
    pipeline.py 는 이 placeholder 를 그대로 predict_one 에 전달 (사용 X).
    """
    if device is None:
        device = "cuda:0"
    _ = _get_worker(device=device)  # warm start
    return None, None, device


def predict_one(image_path: Path,
                region_type: str = "titleblock",
                mode: str = "docvqa",  # ignored (PaddleOCR-VL 은 mode 없음)
                processor: Any = None,
                model: Any = None,
                device: Optional[str] = None,
                questions: Optional[List[Tuple[str, str]]] = None,  # ignored
                language_hint: Optional[str] = None) -> Dict[str, Any]:
    """End-to-end prediction (subprocess delegate to PaddleOCR-VL-1.5).

    Parameters mirror src/stage3_alphabetical.py for drop-in compatibility.
    """
    worker = _get_worker(device=device or "cuda:0")
    return worker.predict(image_path, region_type, language_hint)


# ===========================================================================
# CLI (디버깅용)
# ===========================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 3-A PaddleOCR-VL backend (subprocess wrapper)",
    )
    p.add_argument("--image", required=True, help="입력 도면 crop 이미지")
    p.add_argument("--region-type", default="titleblock",
                   choices=["titleblock", "title_block", "title", "notes", "note"])
    p.add_argument("--language", default=None, help="ko / en / ja / ru / zh / de")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        log.error("Image not found: %s", image_path)
        return 2

    result = predict_one(
        image_path=image_path,
        region_type=args.region_type,
        language_hint=args.language,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
