import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="The Relay Calibrated Stand-In Model Service")
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_toks(text: str) -> int:
    return len(tokenizer.encode(text, disallowed_special=()))


class StandInState:
    def __init__(self):
        # Load real Groq calibration if available
        cal_path = Path("data/calibration.json")
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            self.base_ttft_p50_ms = float(cal.get("ttft_p50_ms", 568.0))
            self.base_tps = float(cal.get("tps_p50", 450.0))
            if "headers" in cal:
                hdrs = cal["headers"]
                if "x-ratelimit-limit-tokens" in hdrs:
                    self.tpm_limit = int(hdrs["x-ratelimit-limit-tokens"])
                if "x-ratelimit-limit-requests" in hdrs:
                    self.rpd_limit = int(hdrs["x-ratelimit-limit-requests"])
                    self.rpd_remaining = self.rpd_limit
        else:
            self.base_ttft_p50_ms = 568.0
            self.base_tps = 450.0

        # Configurable rate limits matching discovered Groq headers
        self.rpm_limit: int = 30
        self.tpm_limit: int = 8000
        self.rpd_limit: int = 1000
        self.rpd_remaining: int = 1000

        # Fault & tail-latency injection
        self.slow_tail_prob: float = 0.06     # 6% chance of "succeeding slowly"
        self.slow_tail_add_ms: float = 2400.0
        self.random_503_prob: float = 0.02    # 2% transient error rate
        self.load_slowdown_factor: float = 0.28

        # Sliding window tracking
        self.in_flight: int = 0
        self.req_timestamps: List[float] = []
        self.token_ledger: List[tuple] = []  # (timestamp, tokens)

    def prune_window(self, now: float):
        cutoff = now - 60.0
        self.req_timestamps = [t for t in self.req_timestamps if t > cutoff]
        self.token_ledger = [(t, tok) for t, tok in self.token_ledger if t > cutoff]

    def current_rpm(self) -> int:
        return len(self.req_timestamps)

    def current_tpm(self) -> int:
        return sum(tok for _, tok in self.token_ledger)

    def headers(self, now: float) -> Dict[str, str]:
        self.prune_window(now)
        rem_tok = max(0, self.tpm_limit - self.current_tpm())
        reset_tok_s = 60.0 - (now - self.token_ledger[0][0]) if self.token_ledger else 0.0
        return {
            "x-ratelimit-limit-requests": str(self.rpd_limit),
            "x-ratelimit-limit-tokens": str(self.tpm_limit),
            "x-ratelimit-remaining-requests": str(max(0, self.rpd_remaining)),
            "x-ratelimit-remaining-tokens": str(rem_tok),
            "x-ratelimit-reset-requests": "7m12s",
            "x-ratelimit-reset-tokens": f"{max(0.5, reset_tok_s):.2f}s",
        }


STATE = StandInState()


class AdminConfigPayload(BaseModel):
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpd_remaining: Optional[int] = None
    slow_tail_prob: Optional[float] = None
    random_503_prob: Optional[float] = None
    reset_window: bool = True


@app.post("/admin/configure")
async def configure_standin(cfg: AdminConfigPayload):
    if cfg.rpm_limit is not None:
        STATE.rpm_limit = cfg.rpm_limit
    if cfg.tpm_limit is not None:
        STATE.tpm_limit = cfg.tpm_limit
    if cfg.rpd_remaining is not None:
        STATE.rpd_remaining = cfg.rpd_remaining
    if cfg.slow_tail_prob is not None:
        STATE.slow_tail_prob = cfg.slow_tail_prob
    if cfg.random_503_prob is not None:
        STATE.random_503_prob = cfg.random_503_prob
    if cfg.reset_window:
        STATE.req_timestamps.clear()
        STATE.token_ledger.clear()
        STATE.in_flight = 0
    return {"status": "configured", "rpm_limit": STATE.rpm_limit, "tpm_limit": STATE.tpm_limit, "rpd_remaining": STATE.rpd_remaining}


def synthesize_mock_content(messages: List[Dict[str, str]]) -> str:
    """
    Produces realistic, grounded responses based on the actual retrieved chunks passed in the prompt
    so that Retrieval Settings (S1 vs S2 vs S3) genuinely affect answer quality on the Stand-In!
    """
    sys_msg = messages[0].get("content", "") if messages else ""
    user_msg = messages[-1].get("content", "") if messages else ""

    # Step 1: Router
    if "Query Router" in sys_msg:
        q_low = user_msg.lower()
        pf = None
        if any(w in q_low for w in ["bleu", "warmup", "wmt", "attention is all you need", "table 3", "recurrent"]):
            pf = "1706.03762v7"
        elif any(w in q_low for w in ["moshi", "mimi", "monologue", "helium", "rq-transformer"]):
            pf = "2410.00037v1"
        elif any(w in q_low for w in ["dynamic chunking", "h-net", "routing module", "bpe"]):
            pf = "2507.07955v2"
        return json.dumps({"paper_filter": pf, "search_query": user_msg[:120]})

    # Step 4: Verifier
    if "Citation & Grounding Verifier" in sys_msg:
        has_cite = "[Paper:" in user_msg or "do not contain" in user_msg.lower()
        return json.dumps({"valid": has_cite, "reason": "Verified against chunks", "refined_query": ""})

    # Step 3: Cited Drafter — inspects the actual Retrieved Passages in user_msg!
    q_low = user_msg.lower()
    # Unanswerable questions check
    if any(
        unans in q_low
        for unans in ["qlora rank 16", "gpt-4 architecture", "whisper-large-v3", "imagenet-22k object detection"]
    ):
        return "The provided papers do not contain information to answer this question."

    # Extract all citation headers present in the retrieved passages sent by Step 2
    headers = re.findall(r"(\[Paper:[^\]]+\])", user_msg)
    primary_cite = headers[0] if headers else "[Paper: Unknown | Page: 1 | Section: Intro]"

    # Extract sentences from the retrieved passages that overlap with question keywords
    passages_part = user_msg.split("Retrieved Passages:")[-1] if "Retrieved Passages:" in user_msg else user_msg
    # Augment with deterministic ground-truth synthesis ONLY IF the relevant paper/facts were actually retrieved!
    snippets = []
    if "beta_1" in q_low and ("0.98" in passages_part or "4000" in passages_part):
        snippets.append("The Adam optimizer uses beta_1 = 0.9, beta_2 = 0.98, epsilon = 10^-9, with warmup_steps = 4000.")
    elif "28.4" in passages_part and "wmt" in q_low:
        snippets.append("The big Transformer achieves a BLEU score of 28.4 on WMT 2014 English-to-German after training for 3.5 days on 8 P100 GPUs.")
    elif "recurrent" in q_low and "1706.03762" in passages_part:
        snippets.append("Self-Attention has O(n^2 * d) complexity per layer with O(1) sequential operations, whereas a Recurrent layer requires O(n * d^2) complexity and O(n) sequential operations.")
    elif "mimi" in q_low and ("12.5" in passages_part or "80" in passages_part):
        snippets.append("The Mimi neural audio codec operates at a frame rate of 12.5 Hz (80 ms frame size) at a bandwidth of 1.1 kbps.")
    elif "monologue" in q_low and "2410.00037" in passages_part:
        snippets.append("Moshi's Inner Monologue predicts time-aligned text tokens as a prefix to audio tokens at each timestep, significantly improving linguistic quality.")
    elif ("rq-transformer" in q_low or "helium" in q_low) and "2410.00037" in passages_part:
        snippets.append("Moshi's RQ-Transformer combines a 7B-parameter temporal Transformer (Helium backbone) with a smaller depth Transformer modeling residual codebooks.")
    elif "routing module" in q_low and "2507.07955" in passages_part:
        snippets.append("In an H-Net, the routing module computes boundary probabilities between adjacent representations using cosine similarity projected keys and queries.")
    elif "differentiable" in q_low and "2507.07955" in passages_part:
        snippets.append("Dynamic Chunking makes discrete boundary selections differentiable during end-to-end training via an EMA smoothing module and a Straight-Through Estimator (STE) in the upsampler.")
    elif "compression ratio" in q_low and "2507.07955" in passages_part:
        snippets.append("Dynamic Chunking controls average chunk length using a target compression ratio loss (ratio regularization loss) applied to the routing boundary probabilities.")
    elif "streaming transformer" in q_low and ("160" in passages_part or "200" in passages_part):
        snippets.append("Moshi is the streaming dialogue Transformer model, achieving 160ms theoretical latency and ~200ms practical end-to-end latency.")
    elif "table 3" in q_low and ("25.4" in passages_part or "32" in passages_part):
        snippets.append("In Table 3 variation (A), increasing the number of attention heads h to 32 causes quality to drop to a development BLEU of 25.4.")
    elif "bpe" in q_low and "2507.07955" in passages_part:
        snippets.append("The Dynamic Chunking paper critiques heuristic BPE tokenization because fixed subword vocabularies degrade byte/character-level robustness and penalize non-Latin languages.")
    else:
        # Fallback: pull raw lines from whatever chunks were retrieved
        clean_lines = [
            line.strip() for line in passages_part.splitlines()
            if line.strip() and not line.startswith("[Paper:") and not line.startswith("Write a concise")
        ]
        snippets.append(" ".join(clean_lines[:2])[:280])

    return f"{' '.join(snippets)} {primary_cite}"


@app.post("/openai/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))

    prompt_str = " ".join(m.get("content", "") for m in messages)
    tokens_in = count_toks(prompt_str) + 10
    reply_text = synthesize_mock_content(messages)
    tokens_out = count_toks(reply_text) + 4
    total_req_tokens = tokens_in + tokens_out

    now = time.monotonic()
    STATE.prune_window(now)

    # 1. Check Daily Limit (RPD)
    if STATE.rpd_remaining <= 0:
        hdrs = STATE.headers(now)
        hdrs["retry-after"] = "432.0"
        return JSONResponse(
            status_code=429,
            headers=hdrs,
            content={"error": {"message": "Rate limit reached for requests per day (RPD).", "type": "requests_per_day"}},
        )

    # 2. Check Per-Minute Limits (RPM & TPM)
    if STATE.current_rpm() + 1 > STATE.rpm_limit or (STATE.current_tpm() + total_req_tokens) > STATE.tpm_limit:
        hdrs = STATE.headers(now)
        retry_s = max(1.2, min(8.0, (total_req_tokens / max(1, STATE.tpm_limit)) * 60.0))
        hdrs["retry-after"] = f"{retry_s:.2f}"
        return JSONResponse(
            status_code=429,
            headers=hdrs,
            content={"error": {"message": "Rate limit reached for tokens/requests per minute (TPM/RPM).", "type": "tokens"}},
        )

    # 3. Record usage
    STATE.rpd_remaining -= 1
    STATE.req_timestamps.append(now)
    STATE.token_ledger.append((now, total_req_tokens))

    # 4. Random 503 transient failure check
    if random.random() < STATE.random_503_prob:
        return JSONResponse(
            status_code=503,
            headers=STATE.headers(now),
            content={"error": {"message": "Simulated upstream LPU node timeout", "type": "server_error"}},
        )

    # 5. Compute Calibrated TTFT + Load Slowdown + Prompt Prefill Scaling + Tail Stall
    STATE.in_flight += 1
    try:
        load_mult = min(2.5, 1.0 + STATE.load_slowdown_factor * max(0, STATE.in_flight - 2))
        prefill_add_ms = (tokens_in / 1000.0) * 95.0  # Larger prompts (like S3 900-token chunks) take longer to prefill!
        jitter = random.lognormvariate(0.0, 0.18)
        ttft_ms = (STATE.base_ttft_p50_ms * jitter * load_mult) + prefill_add_ms

        # "Some calls succeed slowly instead of failing quickly"
        if random.random() < STATE.slow_tail_prob:
            ttft_ms += STATE.slow_tail_add_ms

        hdrs = STATE.headers(now)

        if not stream:
            gen_ms = (tokens_out / STATE.base_tps) * 1000.0 * load_mult
            await asyncio.sleep((ttft_ms + gen_ms) / 1000.0)
            STATE.in_flight = max(0, STATE.in_flight - 1)
            return JSONResponse(
                status_code=200,
                headers=hdrs,
                content={
                    "id": f"chatcmpl-mock-{int(now*1000)}",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": reply_text}}],
                    "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": total_req_tokens},
                },
            )

        # Streaming SSE piece-by-piece
        words = reply_text.split(" ")
        piece_size = 4
        pieces = [" ".join(words[i : i + piece_size]) + (" " if i + piece_size < len(words) else "") for i in range(0, len(words), piece_size)]
        inter_piece_sec = max(0.008, ((tokens_out / STATE.base_tps) * load_mult) / max(1, len(pieces)))

        async def sse_generator():
            try:
                await asyncio.sleep(ttft_ms / 1000.0)
                for idx, piece in enumerate(pieces):
                    chunk_pkt = {
                        "id": f"chatcmpl-mock-{int(now*1000)}",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_pkt)}\n\n"
                    if idx < len(pieces) - 1:
                        await asyncio.sleep(inter_piece_sec)

                final_pkt = {
                    "id": f"chatcmpl-mock-{int(now*1000)}",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": total_req_tokens},
                }
                yield f"data: {json.dumps(final_pkt)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                STATE.in_flight = max(0, STATE.in_flight - 1)

        # Note: in streaming mode, in_flight is decremented inside sse_generator's finally block
        return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=hdrs)
    except Exception:
        STATE.in_flight = max(0, STATE.in_flight - 1)
        raise