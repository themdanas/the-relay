"""
Telemetry dataclass for a single question run across the pipeline.
Stored per-question and later serialized to traces.jsonl.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class QuestionTrace:
    qid: str
    question: str
    setting_name: str

    # Step 1 – Router
    router_status: str = ""
    router_paper_filter: Optional[str] = None
    router_queue_ms: float = 0.0
    router_work_ms: float = 0.0
    router_tokens_in: int = 0
    router_tokens_out: int = 0

    # Step 2 – Retrieval
    retrieval_ms: float = 0.0
    chunks_retrieved: int = 0
    total_context_tokens: int = 0

    # Step 3 – Answer
    answer_status: str = ""
    answer_text: str = ""
    answer_queue_ms: float = 0.0
    answer_work_ms: float = 0.0
    answer_tokens_in: int = 0
    answer_tokens_out: int = 0
    first_token_ms: Optional[float] = None
    retries_used: int = 0

    # Grading
    grade: str = ""

    # Wall-clock
    total_ms: float = 0.0
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

