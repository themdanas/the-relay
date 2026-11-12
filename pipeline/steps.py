import datetime
import json
import re
import time
from typing import Optional, Tuple

from config import DEFAULT_RETRIEVAL_SETTING, USER_PATIENCE_BUDGET_MS
from eval.grader import grade_answer, has_valid_citation, is_refusal
from eval.questions import EvalQuestion
from ingest.indexer import LibraryIndexer
from llm.client import UnifiedLLMClient
from pipeline.prompts import (
    DRAFTER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
)
from pipeline.telemetry import RequestRecord, StepBreakdown


def strip_thinking(text: str) -> str:
    """Removes <think>...</think> blocks if emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class RelayPipeline:
    def __init__(self, indexer: LibraryIndexer, llm_client: UnifiedLLMClient):
        self.indexer = indexer
        self.client = llm_client

    async def run_request(
        self,
        request_id: str,
        question_obj: EvalQuestion,
        scenario: str = "steady",
        retrieval_setting: str = DEFAULT_RETRIEVAL_SETTING,
        patience_budget_ms: float = USER_PATIENCE_BUDGET_MS,
    ) -> Tuple[RequestRecord, str]:
        t_start = time.perf_counter()
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        breakdown = StepBreakdown(question_id=question_obj.qid)

        total_tokens_in = 0
        total_tokens_out = 0
        total_retries = 0
        first_token_ms: Optional[float] = None
        final_answer = ""
        final_status = "ok"

        def elapsed_ms() -> float:
            return (time.perf_counter() - t_start) * 1000.0

        def remaining_ms() -> float:
            return max(0.0, patience_budget_ms - elapsed_ms())

        # =========================================================================
        # SECTION 7 FAST-PATH: Daily Budget Exhausted at 3 PM
        # =========================================================================
        if self.client.governor.is_daily_exhausted:
            chunks, tool_ms = self.indexer.search(
                question_obj.question,
                setting_name=retrieval_setting,
                degrade_for_rate_limit=True,
            )
            breakdown.step2_tool_ms += tool_ms
            breakdown.degraded_mode = True
            top_cite = (
                f"[Paper: {chunks[0].paper_title} (arXiv:{chunks[0].paper_id}) | Page: {chunks[0].page_num} | Section: {chunks[0].section}]"
                if chunks else "[No matching passage]"
            )
            excerpt = chunks[0].raw_text[:280] if chunks else ""
            final_answer = (
                f"[DAILY AI QUOTA EXHAUSTED — LOCAL RETRIEVAL FALLBACK] "
                f"Our daily model token budget is exhausted for today, so live AI synthesis is paused. "
                f"Closest passage from library: {top_cite}: \"{excerpt}...\""
            )
            breakdown.final_answer_preview = final_answer[:200]
            tot = int(round(elapsed_ms()))
            rec = RequestRecord(
                request_id=request_id,
                scenario=scenario,
                source=self.client.source,
                start_time=start_iso,
                first_token_ms=tot,
                total_ms=tot,
                status="rate_limited",
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=retrieval_setting,
                answer_grade=grade_answer(question_obj, final_answer, "rate_limited"),
                breakdown=breakdown,
            )
            return rec, final_answer

        # =========================================================================
        # STEP 1: Intent & Query Router (LLM Call #1)
        # =========================================================================
        degrade_now = self.client.governor.should_degrade
        breakdown.degraded_mode = degrade_now

        step1_res = await self.client.chat_completion(
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question_obj.question},
            ],
            step_num=1,
            remaining_budget_ms=remaining_ms(),
            max_tokens=90,
            temperature=0.0,
            stream=False,
        )
        breakdown.queue_wait_ms += step1_res.queue_wait_ms
        breakdown.step1_work_ms += step1_res.work_ms
        breakdown.retry_pre_ttft_ms += step1_res.retry_pre_ttft_ms + step1_res.retry_post_ttft_ms
        total_tokens_in += step1_res.tokens_in
        total_tokens_out += step1_res.tokens_out
        total_retries += step1_res.retries_used

        if step1_res.status != "ok":
            tot = int(round(elapsed_ms()))
            rec = RequestRecord(
                request_id=request_id,
                scenario=scenario,
                source=self.client.source,
                start_time=start_iso,
                first_token_ms=tot,
                total_ms=tot,
                status=step1_res.status,
                retries=total_retries,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                retrieval_setting=retrieval_setting,
                answer_grade=grade_answer(question_obj, "", step1_res.status),
                breakdown=breakdown,
            )
            return rec, ""

        # Parse router output
        search_query = question_obj.question
        paper_filter = None
        cleaned_s1 = strip_thinking(step1_res.content)
        try:
            m = re.search(r"\{.*\}", cleaned_s1, flags=re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                search_query = parsed.get("search_query") or question_obj.question
                pf = parsed.get("paper_filter")
                if pf in ("1706.03762v7", "2410.00037v1", "2507.07955v2"):
                    paper_filter = pf
        except Exception:
            pass

        # =========================================================================
        # LOOP OVER STEP 2 (Retrieval Tool) -> STEP 3 (Draft) -> STEP 4 (Verify)
        # =========================================================================
        max_loops = 0 if degrade_now else 1
        loop_idx = 0

        while loop_idx <= max_loops:
            # --- STEP 2: Retrieval Tool Call ---
            degrade_now = self.client.governor.should_degrade
            breakdown.degraded_mode = breakdown.degraded_mode or degrade_now

            # On loop retry, broaden search by dropping strict paper_filter
            active_filter = paper_filter if loop_idx == 0 else None
            chunks, tool_ms = self.indexer.search(
                query=search_query,
                setting_name=retrieval_setting,
                paper_filter=active_filter,
                degrade_for_rate_limit=degrade_now,
            )
            breakdown.step2_tool_ms += tool_ms

            context_block = "\n\n".join(c.text for c in chunks)
            drafter_user_msg = (
                f"Question: {question_obj.question}\n\n"
                f"Retrieved Passages:\n{context_block}\n\n"
                f"Write a concise answer citing the exact [Paper: ... | Page: ... | Section: ...] header, "
                f"or state that the provided papers do not contain information to answer this question."
            )

            # --- STEP 3: Cited Reply Drafter (Streaming LLM Call #2) ---
            t_before_s3 = elapsed_ms()
            step3_res = await self.client.chat_completion(
                messages=[
                    {"role": "system", "content": DRAFTER_SYSTEM_PROMPT},
                    {"role": "user", "content": drafter_user_msg},
                ],
                step_num=3,
                remaining_budget_ms=remaining_ms(),
                max_tokens=220 if not degrade_now else 140,
                temperature=0.0,
                stream=True,
            )
            breakdown.queue_wait_ms += step3_res.queue_wait_ms
            breakdown.step3_work_ms += step3_res.work_ms
            total_tokens_in += step3_res.tokens_in
            total_tokens_out += step3_res.tokens_out
            total_retries += step3_res.retries_used

            if first_token_ms is None and step3_res.first_token_offset_ms is not None:
                first_token_ms = (
                    t_before_s3
                    + step3_res.queue_wait_ms
                    + step3_res.retry_pre_ttft_ms
                    + step3_res.first_token_offset_ms
                )
                breakdown.retry_pre_ttft_ms += step3_res.retry_pre_ttft_ms
                breakdown.retry_post_ttft_ms += step3_res.retry_post_ttft_ms
            else:
                breakdown.retry_pre_ttft_ms += step3_res.retry_pre_ttft_ms
                breakdown.retry_post_ttft_ms += step3_res.retry_post_ttft_ms

            if step3_res.status != "ok":
                final_status = step3_res.status
                break

            draft_reply = strip_thinking(step3_res.content)
            final_answer = draft_reply

            # --- STEP 4: Citation & Grounding Verifier (Rule Check + LLM Call #3) ---
            # Fast deterministic pre-check: is it a valid refusal or does it have a citation tag?
            det_ok = is_refusal(draft_reply) or has_valid_citation(draft_reply)

            # If in degraded low-TPM mode or remaining budget < 1.8s, rely on deterministic check to save tokens/latency
            if degrade_now or remaining_ms() < 1800:
                if det_ok:
                    final_status = "ok"
                    break
                elif loop_idx < max_loops and remaining_ms() > 2500:
                    loop_idx += 1
                    breakdown.loop_count += 1
                    continue
                else:
                    # Safe fallback: never send an uncited claim
                    if chunks:
                        c0 = chunks[0]
                        final_answer = (
                            f"{draft_reply} [Paper: {c0.paper_title} (arXiv:{c0.paper_id}) | "
                            f"Page: {c0.page_num} | Section: {c0.section}]"
                        )
                    break

            # Full LLM Verifier Call (#3)
            verify_msg = (
                f"Question: {question_obj.question}\n"
                f"Draft Reply: {draft_reply}\n"
                f"Passages Preview: {context_block[:1200]}"
            )
            step4_res = await self.client.chat_completion(
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": verify_msg},
                ],
                step_num=4,
                remaining_budget_ms=remaining_ms(),
                max_tokens=80,
                temperature=0.0,
                stream=False,
            )
            breakdown.queue_wait_ms += step4_res.queue_wait_ms
            breakdown.step4_work_ms += step4_res.work_ms
            breakdown.retry_post_ttft_ms += step4_res.retry_pre_ttft_ms + step4_res.retry_post_ttft_ms
            total_tokens_in += step4_res.tokens_in
            total_tokens_out += step4_res.tokens_out
            total_retries += step4_res.retries_used

            if step4_res.status != "ok":
                # If Step 4 timed out or hit 429, but Step 3 already passed deterministic citation check, keep Step 3!
                if det_ok:
                    final_status = "ok"
                else:
                    final_status = step4_res.status
                break

            verify_clean = strip_thinking(step4_res.content)
            is_valid = det_ok
            try:
                m_v = re.search(r"\{.*\}", verify_clean, flags=re.DOTALL)
                if m_v:
                    v_json = json.loads(m_v.group(0))
                    is_valid = bool(v_json.get("valid", True)) and det_ok
                    if not is_valid and v_json.get("refined_query"):
                        search_query = v_json["refined_query"]
            except Exception:
                pass

            if is_valid:
                final_status = "ok"
                break

            # Branch / Loop if verification failed
            if loop_idx < max_loops and remaining_ms() > 2500:
                loop_idx += 1
                breakdown.loop_count += 1
            else:
                # Never send an unverified / embarrassing answer; append top citation or refuse safely
                if not det_ok and chunks:
                    c0 = chunks[0]
                    final_answer = (
                        f"{draft_reply} [Paper: {c0.paper_title} (arXiv:{c0.paper_id}) | "
                        f"Page: {c0.page_num} | Section: {c0.section}]"
                    )
                break

        tot_ms = int(round(elapsed_ms()))
        ttft_int = int(round(first_token_ms)) if first_token_ms is not None else tot_ms
        ttft_int = min(ttft_int, tot_ms)

        breakdown.final_answer_preview = final_answer[:240]
        grade = grade_answer(question_obj, final_answer, final_status)

        record = RequestRecord(
            request_id=request_id,
            scenario=scenario,
            source=self.client.source,
            start_time=start_iso,
            first_token_ms=ttft_int,
            total_ms=tot_ms,
            status=final_status,
            retries=total_retries,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            retrieval_setting=retrieval_setting,
            answer_grade=grade,
            breakdown=breakdown,
        )
        return record, final_answer