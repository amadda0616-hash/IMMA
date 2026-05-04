"""
src/validate/common.py

Shared infrastructure for the validation framework.

Provides
--------
- ``Severity`` / ``Status`` enums
- ``CheckResult`` : single-check record (name, value, threshold, status, ...)
- ``ValidationReport`` : aggregator producing console / JSON / HTML
- ``load_thresholds(path)`` : YAML loader
- ``threshold_lookup(cfg, dotted_path)`` : helper to fetch nested thresholds
- ``Plot`` helper : matplotlib → base64-embedded PNG for HTML reports

Decision references
-------------------
- D-020 : per-step validation is mandatory; outputs go to ``reports/``.
- D-021 : severity = critical (block) / warning (warn) / info (monitor).
- D-022 : console + JSON + HTML triple output.
- D-023 : Stage 2/3 user-required thresholds.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# matplotlib is loaded lazily inside Plot helpers so that headless
# environments don't have to import it for every check.

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "validation_thresholds.yaml"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

# ANSI colors (console output)
_C_RESET  = "\033[0m"
_C_BOLD   = "\033[1m"
_C_GREEN  = "\033[92m"
_C_YELLOW = "\033[93m"
_C_RED    = "\033[91m"
_C_GRAY   = "\033[90m"
_C_CYAN   = "\033[96m"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Severity(str, Enum):
    CRITICAL = "critical"   # 미달 시 다음 step 차단
    WARNING  = "warning"    # 경고만, 진행 가능
    INFO     = "info"       # 통계용


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    """Single validation check outcome."""
    name: str
    value: Union[float, int, str, None] = None
    threshold: Union[float, int, str, None] = None
    direction: str = "ge"   # ge / le / eq / between / none
    severity: Severity = Severity.WARNING
    status: Status = Status.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["status"] = self.status.value
        return d

    @classmethod
    def evaluate(cls,
                 name: str,
                 value: Union[float, int, None],
                 threshold: Union[float, int, None],
                 direction: str = "ge",
                 severity: Union[Severity, str] = Severity.WARNING,
                 message: str = "",
                 details: Optional[Dict[str, Any]] = None) -> "CheckResult":
        """Build a CheckResult and auto-set status based on direction.

        direction:
            ge  -> PASS if value >= threshold
            le  -> PASS if value <= threshold
            eq  -> PASS if value == threshold
            none -> always INFO
        """
        if isinstance(severity, str):
            severity = Severity(severity)

        status = Status.INFO
        if direction == "none" or threshold is None or value is None:
            status = Status.INFO
        elif direction == "ge":
            status = Status.PASS if value >= threshold else _fail_status(severity)
        elif direction == "le":
            status = Status.PASS if value <= threshold else _fail_status(severity)
        elif direction == "eq":
            status = Status.PASS if value == threshold else _fail_status(severity)

        return cls(
            name=name,
            value=value,
            threshold=threshold,
            direction=direction,
            severity=severity,
            status=status,
            message=message,
            details=details or {},
        )


def _fail_status(severity: Severity) -> Status:
    if severity == Severity.CRITICAL:
        return Status.FAIL
    if severity == Severity.WARNING:
        return Status.WARN
    return Status.INFO


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    """Aggregate multiple CheckResults + artifacts (plots, tables, ...).

    Produces console output + JSON + HTML.
    """
    title: str
    step: str                         # e.g. "step1.5", "stage1_model"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    checks: List[CheckResult] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- mutators ------------------------------------------------------
    def add(self, check: CheckResult) -> "ValidationReport":
        self.checks.append(check)
        return self

    def add_eval(self, *args, **kwargs) -> "ValidationReport":
        return self.add(CheckResult.evaluate(*args, **kwargs))

    def add_plot(self, title: str, png_bytes: bytes,
                 description: str = "") -> "ValidationReport":
        """Attach a base64 PNG plot for the HTML report."""
        b64 = base64.b64encode(png_bytes).decode("ascii")
        self.artifacts.append({
            "kind": "plot",
            "title": title,
            "description": description,
            "data_uri": f"data:image/png;base64,{b64}",
        })
        return self

    def add_table(self, title: str, rows: List[Dict[str, Any]],
                  columns: Optional[List[str]] = None,
                  description: str = "") -> "ValidationReport":
        self.artifacts.append({
            "kind": "table",
            "title": title,
            "description": description,
            "columns": columns or (list(rows[0].keys()) if rows else []),
            "rows": rows,
        })
        return self

    # --- queries -------------------------------------------------------
    @property
    def overall_status(self) -> Status:
        if any(c.status == Status.FAIL for c in self.checks):
            return Status.FAIL
        if any(c.status == Status.WARN for c in self.checks):
            return Status.WARN
        if any(c.status == Status.ERROR for c in self.checks):
            return Status.ERROR
        return Status.PASS

    @property
    def counts(self) -> Dict[str, int]:
        c = {s.value: 0 for s in Status}
        for chk in self.checks:
            c[chk.status.value] += 1
        return c

    # --- outputs -------------------------------------------------------
    def to_console(self, use_color: bool = True) -> str:
        """Pretty-print the report. Returns the printable string."""
        def col(text: str, color: str) -> str:
            return f"{color}{text}{_C_RESET}" if use_color else text

        lines: List[str] = []
        bar = "═" * 64
        lines.append(col(f"╔{bar}╗", _C_BOLD))
        lines.append(col(f"║  {self.title:<60}║", _C_BOLD))
        lines.append(col(f"║  step={self.step}  ts={self.timestamp:<41}║", _C_GRAY))
        lines.append(col(f"╚{bar}╝", _C_BOLD))
        lines.append("")

        for i, chk in enumerate(self.checks, 1):
            mark, color = _status_mark(chk.status, use_color)
            label = chk.name
            if len(label) > 38:
                label = label[:35] + "..."
            value_str = _fmt_value(chk.value)
            thr_str = _fmt_threshold(chk.threshold, chk.direction)
            line = (
                f" [{i:2d}/{len(self.checks)}] "
                f"{label:<40} {value_str:>10}  "
                f"{mark} {col(chk.status.value, color)}  {thr_str}"
            )
            lines.append(line)
            if chk.message:
                lines.append(f"        {col('· ' + chk.message, _C_GRAY)}")

        # Summary
        lines.append("")
        c = self.counts
        overall = self.overall_status
        ov_color = {
            Status.PASS: _C_GREEN, Status.WARN: _C_YELLOW,
            Status.FAIL: _C_RED, Status.ERROR: _C_RED,
            Status.INFO: _C_GRAY,
        }[overall]
        lines.append(
            f" Overall: {col(overall.value, ov_color)}   "
            f"PASS={c['PASS']}  WARN={c['WARN']}  FAIL={c['FAIL']}  "
            f"INFO={c['INFO']}  ERROR={c['ERROR']}"
        )
        return "\n".join(lines)

    def to_json(self, path: Optional[Path] = None) -> Dict[str, Any]:
        data = {
            "title": self.title,
            "step": self.step,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "counts": self.counts,
            "checks": [c.to_dict() for c in self.checks],
            "artifacts": [
                {k: v for k, v in a.items() if k != "data_uri"}
                for a in self.artifacts
            ],
            "metadata": self.metadata,
        }
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def to_html(self, path: Optional[Path] = None) -> str:
        html = _render_html(self)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def emit(self,
             reports_dir: Path = DEFAULT_REPORTS_DIR,
             stem: Optional[str] = None,
             use_color: bool = True) -> Dict[str, Path]:
        """Print to console + write JSON + HTML. Returns output paths."""
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        date = datetime.now().strftime("%Y-%m-%d")
        stem = stem or f"{date}_{self.step}"
        json_path = reports_dir / f"{stem}.json"
        html_path = reports_dir / f"{stem}.html"

        print(self.to_console(use_color=use_color))
        self.to_json(json_path)
        self.to_html(html_path)
        return {"json": json_path, "html": html_path}


# ---------------------------------------------------------------------------
# YAML thresholds
# ---------------------------------------------------------------------------
def load_thresholds(path: Union[Path, str] = DEFAULT_THRESHOLDS_PATH) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Thresholds YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def threshold_lookup(cfg: Dict[str, Any], dotted: str) -> Dict[str, Any]:
    """Fetch a nested threshold node by dotted path, e.g. ``stage2_model.missing_rate_max.Measure``."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Threshold path not found: {dotted} (missing '{key}')")
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Plot helpers (matplotlib lazy import)
# ---------------------------------------------------------------------------
def fig_to_png(fig) -> bytes:
    """Render a matplotlib Figure to PNG bytes (for HTML embedding)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    return buf.getvalue()


def make_bar_chart(labels: List[str], values: List[float],
                   title: str = "", ylabel: str = "",
                   ylim: Optional[tuple] = None,
                   horizontal: bool = False) -> bytes:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if horizontal:
        ax.barh(labels, values, color="#4a90e2")
        ax.set_xlabel(ylabel)
    else:
        ax.bar(labels, values, color="#4a90e2")
        ax.set_ylabel(ylabel)
    if ylim:
        (ax.set_ylim if not horizontal else ax.set_xlim)(*ylim)
    ax.set_title(title)
    ax.grid(axis=("x" if horizontal else "y"), alpha=0.25)
    fig.tight_layout()
    png = fig_to_png(fig)
    plt.close(fig)
    return png


def make_confusion_matrix(matrix: List[List[int]],
                          labels: List[str],
                          title: str = "Confusion Matrix") -> bytes:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{int(arr[i, j])}",
                    ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() * 0.5 else "black",
                    fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    png = fig_to_png(fig)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# HTML rendering (inline Jinja2 template)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }} — Validation Report</title>
<style>
 body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        max-width: 1180px; margin: 32px auto; padding: 0 20px; color: #222; }
 h1 { border-bottom: 3px solid #333; padding-bottom: 8px; }
 .meta { color: #666; font-size: 13px; }
 .summary { background: #f5f5f5; border-radius: 8px; padding: 14px 18px;
            margin: 18px 0; }
 .pill { display: inline-block; padding: 3px 10px; border-radius: 12px;
         font-weight: bold; font-size: 13px; color: #fff; }
 .pill.PASS  { background: #2e8b57; }
 .pill.WARN  { background: #d68910; }
 .pill.FAIL  { background: #c0392b; }
 .pill.INFO  { background: #888; }
 .pill.ERROR { background: #6c3483; }
 table.checks { width: 100%; border-collapse: collapse; margin: 14px 0; }
 table.checks th, table.checks td { padding: 8px 10px; text-align: left;
   border-bottom: 1px solid #eee; font-size: 14px; vertical-align: top; }
 table.checks th { background: #fafafa; font-size: 12px; color: #666;
   text-transform: uppercase; letter-spacing: 0.5px; }
 .num { font-variant-numeric: tabular-nums; text-align: right; }
 .sev-critical { color: #c0392b; font-weight: bold; }
 .sev-warning  { color: #d68910; }
 .sev-info     { color: #888; }
 .msg { color: #555; font-size: 12px; }
 .artifact { margin: 20px 0; border: 1px solid #eee;
             border-radius: 8px; padding: 14px 18px; background: #fbfbfb; }
 .artifact h3 { margin-top: 0; font-size: 15px; }
 .artifact img { max-width: 100%; height: auto; }
 .artifact .desc { color: #666; font-size: 13px; margin-bottom: 10px; }
 table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
 table.data-table th, table.data-table td { border-bottom: 1px solid #eee;
    padding: 6px 10px; }
 footer { margin-top: 40px; color: #888; font-size: 12px; text-align: center; }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<div class="meta">
  step = <code>{{ step }}</code> &nbsp; · &nbsp;
  generated = <code>{{ timestamp }}</code>
</div>

<div class="summary">
  <strong>Overall:</strong>
  <span class="pill {{ overall }}">{{ overall }}</span>
  &nbsp; PASS = {{ counts.PASS }} &nbsp;
  WARN = {{ counts.WARN }} &nbsp;
  FAIL = {{ counts.FAIL }} &nbsp;
  INFO = {{ counts.INFO }} &nbsp;
  ERROR = {{ counts.ERROR }}
</div>

<table class="checks">
 <thead>
  <tr><th>#</th><th>Check</th><th>Value</th><th>Threshold</th><th>Severity</th><th>Status</th></tr>
 </thead>
 <tbody>
  {% for c in checks %}
  <tr>
   <td>{{ loop.index }}</td>
   <td>
     <strong>{{ c.name }}</strong>
     {% if c.message %}<br><span class="msg">{{ c.message }}</span>{% endif %}
   </td>
   <td class="num">{{ c.value_fmt }}</td>
   <td class="num">{{ c.thr_fmt }}</td>
   <td class="sev-{{ c.severity }}">{{ c.severity }}</td>
   <td><span class="pill {{ c.status }}">{{ c.status }}</span></td>
  </tr>
  {% endfor %}
 </tbody>
</table>

{% for a in artifacts %}
 <div class="artifact">
  <h3>{{ a.title }}</h3>
  {% if a.description %}<div class="desc">{{ a.description }}</div>{% endif %}
  {% if a.kind == 'plot' %}
   <img src="{{ a.data_uri }}" alt="{{ a.title }}">
  {% elif a.kind == 'table' %}
   <table class="data-table">
    <thead><tr>
     {% for col in a.columns %}<th>{{ col }}</th>{% endfor %}
    </tr></thead>
    <tbody>
     {% for row in a.rows %}
     <tr>
      {% for col in a.columns %}<td>{{ row[col] }}</td>{% endfor %}
     </tr>
     {% endfor %}
    </tbody>
   </table>
  {% endif %}
 </div>
{% endfor %}

<footer>
 Generated by <code>src/validate/common.py</code> ·
 PROJECT_HANDOFF.md §14 · D-020 / D-021 / D-022
</footer>
</body>
</html>
"""


def _render_html(report: ValidationReport) -> str:
    """Render the report HTML using Jinja2 (lazy import)."""
    try:
        from jinja2 import Template  # noqa: PLC0415
    except ImportError as e:  # noqa: F841
        return _render_html_fallback(report)

    tpl = Template(_HTML_TEMPLATE, autoescape=True)
    checks_view = []
    for c in report.checks:
        checks_view.append({
            "name": c.name,
            "value_fmt": _fmt_value(c.value),
            "thr_fmt": _fmt_threshold(c.threshold, c.direction),
            "severity": c.severity.value,
            "status": c.status.value,
            "message": c.message,
        })
    return tpl.render(
        title=report.title,
        step=report.step,
        timestamp=report.timestamp,
        overall=report.overall_status.value,
        counts=report.counts,
        checks=checks_view,
        artifacts=report.artifacts,
    )


def _render_html_fallback(report: ValidationReport) -> str:
    """Tiny fallback HTML when jinja2 is missing (graceful degradation)."""
    rows = "".join(
        f"<tr><td>{c.name}</td><td>{_fmt_value(c.value)}</td>"
        f"<td>{_fmt_threshold(c.threshold, c.direction)}</td>"
        f"<td>{c.severity.value}</td>"
        f"<td><b>{c.status.value}</b></td></tr>"
        for c in report.checks
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{report.title}</title></head><body>"
        f"<h1>{report.title}</h1><p>step={report.step} · {report.timestamp}</p>"
        f"<p>Overall: <b>{report.overall_status.value}</b></p>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Check</th><th>Value</th><th>Threshold</th><th>Severity</th><th>Status</th></tr>"
        f"{rows}</table></body></html>"
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def _fmt_threshold(t: Any, direction: str) -> str:
    if t is None or direction in ("none", ""):
        return "—"
    sym = {"ge": "≥", "le": "≤", "eq": "=", "between": "∈"}.get(direction, "")
    if isinstance(t, float):
        t_str = f"{t:.4f}" if abs(t) < 1000 else f"{t:.1f}"
    else:
        t_str = str(t)
    return f"{sym} {t_str}".strip()


def _status_mark(status: Status, use_color: bool) -> tuple:
    table = {
        Status.PASS:  ("✓", _C_GREEN),
        Status.WARN:  ("!", _C_YELLOW),
        Status.FAIL:  ("✗", _C_RED),
        Status.INFO:  ("·", _C_GRAY),
        Status.ERROR: ("E", _C_RED),
    }
    mark, color = table[status]
    if not use_color:
        color = ""
    return mark, color


# ---------------------------------------------------------------------------
# Logging setup helper (used by check_*.py)
# ---------------------------------------------------------------------------
def setup_logging(name: str = "validate", level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return logging.getLogger(name)
