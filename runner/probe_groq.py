import asyncio
import json
from pathlib import Path
import numpy as np

from config import GROQ_API_KEY, MODEL_NAME
from llm.client import UnifiedLLMClient
from llm.governor import RateLimitGovernor


async def probe_real_groq():
    if not GROQ_API_KEY:
        print("[ERROR] Set GROQ_API_KEY in your .env file first!")
        return

    gov = RateLimitGovernor(max_concurrency=10)
    client = UnifiedLLMClient(source="real", governor=gov, max_retries=0)

    print(f"=== Phase 1: Probing Real Service Headers & Baseline Latency ({MODEL_NAME}) ===")
    ttfts = []
    totals = []
    sample_headers = {}

    for i in range(5):
        res = await client.chat_completion(
            messages=[
                {"role": "system", "content": "Answer concisely in 2 sentences."},
                {"role": "user", "content": f"Explain scaled dot-product attention (probe {i+1})."},
            ],
            step_num=1,
            remaining_budget_ms=10000,
            max_tokens=80,
            stream=True,
        )
        if res.status == "ok":
            ttfts.append(res.first_token_offset_ms or res.work_ms)
            totals.append(res.work_ms)
            sample_headers = {k: v for k, v in res.raw_headers.items() if "ratelimit" in k.lower() or "retry" in k.lower()}
            print(
                f"  Req {i+1}: status={res.status} | TTFT={res.first_token_offset_ms:.1f}ms | "
                f"Total={res.work_ms:.1f}ms | in={res.tokens_in}, out={res.tokens_out}"
            )
        else:
            print(f"  Req {i+1}: FAILED ({res.status}) -> {res.error_detail}")
        await asyncio.sleep(0.5)

    print("\n=== Discovered Rate-Limit Headers (For Table 1) ===")
    for k, v in sorted(sample_headers.items()):
        print(f"  {k}: {v}")

    if ttfts:
        calibration = {
            "model": MODEL_NAME,
            "ttft_p50_ms": float(np.percentile(ttfts, 50)),
            "ttft_p95_ms": float(np.percentile(ttfts, 95)),
            "total_p50_ms": float(np.percentile(totals, 50)),
            "headers": sample_headers,
        }
        Path("data/calibration.json").write_text(json.dumps(calibration, indent=2))
        print(f"\n[SAVED] Calibration saved to data/calibration.json (p50 TTFT = {calibration['ttft_p50_ms']:.1f} ms)")

    await client.close()

if __name__ == "__main__":
    asyncio.run(probe_real_groq())