"""
pipeline/telemetry.py
---------------------
Telemetry primitives for The Relay:
  - RequestRecord   – one row in results.csv (12-column exact schema)
  - StepBreakdown   – per-step span detail written to traces.jsonl
  - CSVWriter       – appends RequestRecord rows to results.csv
  - TracesWriter    – appends full JSON lines to traces.jsonl
  - make_request_id – canonical request-id helper
"""

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from config import RESULTS_CSV, TRACES_JSONL

# ---------------------------------------------------------------------------
# Exact column order required by AGENTS.md (12 columns)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


# ---------------------------------------------------------------------------
# StepBreakdown – per-step span telemetry (goes to traces.jsonl only)
# ---------------------------------------------------------------------------
@dataclass
class StepBreakdown:
    """Detailed span telemetry for Section 4 ('Where the time goes')."""

    question_id: str = ""
    step_name: str = ""
    queue_wait_ms: float = 0.0
    step1_work_ms: float = 0.0
    step2_tool_ms: float = 0.0
    step3_work_ms: float = 0.0
    step4_work_ms: float = 0.0
    retry_pre_ttft_ms: float = 0.0
    retry_post_ttft_ms: float = 0.0
    first_token_ms: float = 0.0
    loop_count: int = 0
    degraded_mode: bool = False
    final_answer_preview: str = ""


# ---------------------------------------------------------------------------
# RequestRecord – one row in results.csv
# ---------------------------------------------------------------------------
@dataclass
class RequestRecord:
    request_id: str
    scenario: str
    source: Literal["real", "stand-in"]
    start_time: str                                                # ISO-8601 string
    first_token_ms: float                                          # ms
    total_ms: float                                                # ms
    status: Literal["ok", "rate_limited", "timeout", "error"]
    retries: int
    tokens_in: int
    tokens_out: int
    retrieval_setting: str
    answer_grade: Literal["correct", "wrong", "refused_correctly", "refused_wrongly", ""]

    # Not written to CSV; written to traces.jsonl only
    breakdown: Optional[StepBreakdown] = None
    steps: list = field(default_factory=list)                      # list[StepBreakdown]

    def to_csv_row(self) -> dict:
        """Return a dict with exactly the 12 CSV columns in order."""
        return {col: getattr(self, col) for col in CSV_COLUMNS}


# ---------------------------------------------------------------------------
# make_request_id
# ---------------------------------------------------------------------------
def make_request_id(scenario: str, source: str) -> str:
    """Generate a canonical request_id: '{scenario}_{source}_{8-hex}'."""
    return f"{scenario}_{source}_{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# CSVWriter
# ---------------------------------------------------------------------------
class CSVWriter:
    """
    Appends RequestRecord rows to results.csv.

    Auto-creates the file with the 12-column header if it is missing or empty.
    Thread-safety: each write() call opens/closes the file (safe for
    multi-process runners that each call write independently).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or RESULTS_CSV
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Write header if the file is missing or empty."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def write(self, record: RequestRecord) -> None:
        """Append one row for *record*. Creates header if file was deleted."""
        self._ensure_header()
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writerow(record.to_csv_row())


# ---------------------------------------------------------------------------
# TracesWriter
# ---------------------------------------------------------------------------
class TracesWriter:
    """Appends one JSON line per RequestRecord to traces.jsonl."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or TRACES_JSONL

    def write(self, record: RequestRecord) -> None:
        """Append the full record (including step breakdowns) as a JSON line."""
        payload = record.to_csv_row()                             # CSV fields
        if record.breakdown is not None:
            payload["breakdown"] = asdict(record.breakdown)
        if record.steps:
            payload["steps"] = [asdict(s) if hasattr(s, "__dataclass_fields__") else s for s in record.steps]
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")


# ---------------------------------------------------------------------------
# Convenience: single log call that writes to both sinks
# ---------------------------------------------------------------------------
_csv_writer: Optional[CSVWriter] = None
_traces_writer: Optional[TracesWriter] = None


def _get_writers() -> tuple[CSVWriter, TracesWriter]:
    global _csv_writer, _traces_writer
    if _csv_writer is None:
        _csv_writer = CSVWriter()
    if _traces_writer is None:
        _traces_writer = TracesWriter()
    return _csv_writer, _traces_writer


def log_request(record: RequestRecord) -> None:
    """Write *record* to both results.csv and traces.jsonl."""
    csv_w, traces_w = _get_writers()
    csv_w.write(record)
    traces_w.write(record)


# ---------------------------------------------------------------------------
# Legacy shim (keeps existing callers that use log_request_record working)
# ---------------------------------------------------------------------------
def log_request_record(record: RequestRecord) -> None:
    log_request(record)


def init_results_csv(overwrite: bool = False) -> None:
    """Re-create results.csv (and optionally traces.jsonl)."""
    if overwrite or not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
    if overwrite and TRACES_JSONL.exists():
        TRACES_JSONL.unlink()
    # Reset module-level writers so they re-open the fresh file
    global _csv_writer, _traces_writer
    _csv_writer = None
    _traces_writer = None
