"""
Pipeline steps: router → retrieval → answer generation.
Each step returns early on budget exhaustion or rate-limit.
"""
import asyncio
from typing import Optional, Tuple

from config import RETRIEVAL_SETTINGS, USER_PATIENCE_BUDGET_MS
from eval.questions import EvalQuestion
from eval.grader import grade_answer
from ingest.indexer import LibraryIndexer
from llm.client import UnifiedLLMClient, LLMCallResult
from pipeline.prompts import ROUTER_SYSTEM, ROUTER_USER, ANSWER_SYSTEM, ANSWER_USER
from pipeline.telemetry import QuestionTrace


async def run_pipeline_for_question(
    question: EvalQuestion,
    setting_name: str,
    client: UnifiedLLMClient,
    indexer: LibraryIndexer,
    budget_ms: float = USER_PATIENCE_BUDGET_MS,
) -> QuestionTrace:
    """
    Runs the full 3-step RAG pipeline for one question:
      Step 1 – LLM router  (identifies which paper to filter by)
      Step 2 – Hybrid retrieval (BM25 + dense + optional rerank)
      Step 3 – LLM answer generation (streamed SSE)

    Returns a fully-populated QuestionTrace.
    """
    import time
    t_total_start = time.perf_counter()

    trace = QuestionTrace(
        qid=question.qid,
        question=question.question,
        setting_name=setting_name,
    )

    degrade = client.governor.should_degrade
    remaining_ms = budget_ms

    # ── Step 1: Router ──────────────────────────────────────────────────────
    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user",   "content": ROUTER_USER.format(question=question.question)},
    ]

    router_result: LLMCallResult = await client.chat_completion(
        messages=router_messages,
        step_num=1,
        remaining_budget_ms=remaining_ms,
        max_tokens=20,
        temperature=0.0,
        stream=False,
    )

    trace.router_status    = router_result.status
    trace.router_queue_ms  = router_result.queue_wait_ms
    trace.router_work_ms   = router_result.work_ms
    trace.router_tokens_in = router_result.tokens_in
    trace.router_tokens_out= router_result.tokens_out

    paper_filter: Optional[str] = None
    if router_result.status == "ok":
        raw = router_result.content.strip().lower()
        for pid in RETRIEVAL_SETTINGS:  # paper IDs are in config
            pass
        # Match against known arXiv IDs
        for pid in ["1706.03762v7", "2410.00037v1", "2507.07955v2"]:
            if pid.split("v")[0] in raw or pid in raw:
                paper_filter = pid
                break

    trace.router_paper_filter = paper_filter
    elapsed_ms = (time.perf_counter() - t_total_start) * 1000.0
    remaining_ms = max(0.0, budget_ms - elapsed_ms)

    # ── Step 2: Retrieval ───────────────────────────────────────────────────
    chunks, retrieval_ms = indexer.search(
        query=question.question,
        setting_name=setting_name,
        paper_filter=paper_filter,
        degrade_for_rate_limit=degrade,
    )

    trace.retrieval_ms       = retrieval_ms
    trace.chunks_retrieved   = len(chunks)
    trace.total_context_tokens = sum(c.token_count for c in chunks)

    elapsed_ms = (time.perf_counter() - t_total_start) * 1000.0
    remaining_ms = max(0.0, budget_ms - elapsed_ms)

    # ── Step 3: Answer ──────────────────────────────────────────────────────
    context_text = "\n\n".join(c.text for c in chunks)
    answer_messages = [
        {"role": "system", "content": ANSWER_SYSTEM.format(context=context_text)},
        {"role": "user",   "content": ANSWER_USER.format(question=question.question)},
    ]

    answer_result: LLMCallResult = await client.chat_completion(
        messages=answer_messages,
        step_num=3,
        remaining_budget_ms=remaining_ms,
        max_tokens=400,
        temperature=0.0,
        stream=True,
    )

    trace.answer_status    = answer_result.status
    trace.answer_text      = answer_result.content
    trace.answer_queue_ms  = answer_result.queue_wait_ms
    trace.answer_work_ms   = answer_result.work_ms
    trace.answer_tokens_in = answer_result.tokens_in
    trace.answer_tokens_out= answer_result.tokens_out
    trace.first_token_ms   = answer_result.first_token_offset_ms
    trace.retries_used     = answer_result.retries_used

    # ── Grading ─────────────────────────────────────────────────────────────
    trace.grade = grade_answer(question, trace.answer_text, trace.answer_status)

    trace.total_ms = (time.perf_counter() - t_total_start) * 1000.0
    return trace

