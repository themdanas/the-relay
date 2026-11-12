"""
runner/load_runner.py
---------------------
Benchmark orchestrator for The Relay.
Runs all required load-testing scenarios:
  - steady: Baseline load (16 questions, concurrency=1, S2 setting)
  - burst: Sudden burst spike (16 questions, concurrency=8, S2 setting)
  - sustained_overload: Overload stress test (32 requests, concurrency=10, 429 backoff)
  - rpm_halved: Halved rate limits (16 questions, concurrency=6, priority admission)
  - daily_budget_exhausted: Daily quota exhausted (16 questions, instant local fallback)
  - retrieval_sweep: Compares S1, S2, and S3 across all 16 questions for Table 3

Usage:
  python -m runner.load_runner [--scenario all|steady|burst|sustained_overload|rpm_halved|daily_budget_exhausted|retrieval_sweep]
                               [--source stand-in|real]
                               [--concurrency N]
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx

from config import (
    DEFAULT_RETRIEVAL_SETTING,
    STANDIN_BASE_URL,
    RETRIEVAL_SETTINGS,
    USER_PATIENCE_BUDGET_MS,
)
from eval.questions import EVAL_QUESTIONS, EvalQuestion
from ingest.indexer import LibraryIndexer
from llm.client import UnifiedLLMClient
from llm.governor import RateLimitGovernor
from pipeline.steps import RelayPipeline
from pipeline.telemetry import (
    CSVWriter,
    TracesWriter,
    make_request_id,
    init_results_csv,
)


STANDIN_HOST = "127.0.0.1"
STANDIN_PORT = 8000
STANDIN_URL = f"http://{STANDIN_HOST}:{STANDIN_PORT}"


class StandInServerManager:
    """Manages the lifecycle of the local FastAPI stand-in server."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None

    def start(self):
        # Check if already running
        try:
            r = httpx.get(f"{STANDIN_URL}/docs", timeout=1.0)
            if r.status_code == 200:
                print("[STANDIN] Server already running on port 8000.")
                return
        except Exception:
            pass

        print("[STANDIN] Spawning standalone FastAPI server (llm.standin_server)...")
        env = os.environ.copy()
        python_exe = sys.executable
        self.process = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "llm.standin_server:app", "--host", STANDIN_HOST, "--port", str(STANDIN_PORT), "--log-level", "warning"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server ready
        for _ in range(40):
            try:
                r = httpx.get(f"{STANDIN_URL}/docs", timeout=0.5)
                if r.status_code == 200:
                    print("[STANDIN] Server ready on http://127.0.0.1:8000/openai/v1")
                    return
            except Exception:
                time.sleep(0.2)
        print("[WARN] Stand-in server did not respond in 8 seconds.")

    def configure(
        self,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpd_remaining: Optional[int] = None,
        slow_tail_prob: Optional[float] = None,
        random_503_prob: Optional[float] = None,
        reset_window: bool = True,
    ):
        try:
            payload = {
                "rpm_limit": rpm_limit,
                "tpm_limit": tpm_limit,
                "rpd_remaining": rpd_remaining,
                "slow_tail_prob": slow_tail_prob,
                "random_503_prob": random_503_prob,
                "reset_window": reset_window,
            }
            httpx.post(f"{STANDIN_URL}/admin/configure", json=payload, timeout=2.0)
        except Exception as e:
            print(f"[WARN] Failed to configure standin: {e}")

    def stop(self):
        if self.process:
            print("[STANDIN] Stopping stand-in server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


async def execute_scenario(
    scenario_name: str,
    questions: List[EvalQuestion],
    pipeline: RelayPipeline,
    source: str,
    concurrency: int,
    csv_writer: CSVWriter,
    traces_writer: TracesWriter,
    retrieval_setting: str = DEFAULT_RETRIEVAL_SETTING,
):
    print(f"\n{'='*70}")
    print(f"  RUNNING SCENARIO: {scenario_name}")
    print(f"  Requests: {len(questions)} | Concurrency: {concurrency} | Setting: {retrieval_setting} | Source: {source}")
    print(f"{'='*70}")

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(questions)

    async def _handle_one(q: EvalQuestion, idx: int):
        nonlocal completed
        req_id = make_request_id(scenario_name, source)
        async with sem:
            rec, ans = await pipeline.run_request(
                request_id=req_id,
                question_obj=q,
                scenario=scenario_name,
                retrieval_setting=retrieval_setting,
                patience_budget_ms=USER_PATIENCE_BUDGET_MS,
            )
            csv_writer.write(rec)
            traces_writer.write(rec)

            completed += 1
            status_symbol = "✓" if rec.answer_grade in ("correct", "refused_correctly") else ("⚠" if rec.status == "rate_limited" else "✗")
            print(
                f"  [{completed:02d}/{total:02d}] {status_symbol} {q.qid} | "
                f"status={rec.status:<12} | grade={rec.answer_grade:<17} | "
                f"TTFT={rec.first_token_ms:4d}ms | total={rec.total_ms:4d}ms"
            )

    # Launch all concurrently bounded by sem
    tasks = [_handle_one(q, i) for i, q in enumerate(questions)]
    await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser(description="The Relay benchmark load runner")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "steady", "burst", "sustained_overload", "rpm_halved", "daily_budget_exhausted", "retrieval_sweep"],
        help="Scenario to run (default: all)",
    )
    parser.add_argument(
        "--source",
        default="stand-in",
        choices=["stand-in", "real"],
        help="'stand-in' (calibrated local server) or 'real' (Groq API, default: stand-in)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override concurrency for the scenario",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing results.csv and traces.jsonl before running",
    )
    args = parser.parse_args()

    # If --clean is passed or running full suite, initialize clean files
    if args.clean or args.scenario == "all":
        init_results_csv(overwrite=True)
    csv_writer = CSVWriter()
    traces_writer = TracesWriter()

    # Load components
    print("[INIT] Loading indexer models...")
    indexer = LibraryIndexer(load_models=True)

    # Stand-in server manager
    standin_mgr = StandInServerManager()
    if args.source == "stand-in":
        standin_mgr.start()

    # Governor & unified client
    gov = RateLimitGovernor(max_concurrency=12)
    client = UnifiedLLMClient(source=args.source, governor=gov, max_retries=2)
    pipeline = RelayPipeline(indexer, client)

    try:
        scenarios_to_run = (
            ["steady", "burst", "sustained_overload", "rpm_halved", "daily_budget_exhausted", "retrieval_sweep"]
            if args.scenario == "all"
            else [args.scenario]
        )

        for sc in scenarios_to_run:
            client.governor.reset_stats()

            if sc == "steady":
                # Real Groq mode: run 5 questions to conserve quota
                q_list = EVAL_QUESTIONS if args.source == "stand-in" else EVAL_QUESTIONS[:5]
                standin_mgr.configure(rpm_limit=60, tpm_limit=40000, rpd_remaining=1000, slow_tail_prob=0.0, reset_window=True)
                gov.max_concurrency = args.concurrency or 2
                client.governor.daily_exhausted_flag = False
                await execute_scenario(
                    scenario_name="steady",
                    questions=q_list,
                    pipeline=pipeline,
                    source=args.source,
                    concurrency=args.concurrency or 2,
                    csv_writer=csv_writer,
                    traces_writer=traces_writer,
                )

            elif sc == "burst":
                standin_mgr.configure(rpm_limit=30, tpm_limit=14000, rpd_remaining=1000, slow_tail_prob=0.05)
                gov.max_concurrency = args.concurrency or 8
                client.governor.daily_exhausted_flag = False
                await execute_scenario(
                    scenario_name="burst",
                    questions=EVAL_QUESTIONS,
                    pipeline=pipeline,
                    source=args.source,
                    concurrency=args.concurrency or 8,
                    csv_writer=csv_writer,
                    traces_writer=traces_writer,
                )

            elif sc == "sustained_overload":
                # 32 requests (16 questions repeated twice) under tight limits and tail jitter
                q_list = EVAL_QUESTIONS + EVAL_QUESTIONS
                standin_mgr.configure(
                    rpm_limit=20,
                    tpm_limit=7000,
                    rpd_remaining=1000,
                    slow_tail_prob=0.12,
                    random_503_prob=0.03,
                )
                gov.max_concurrency = args.concurrency or 10
                client.governor.daily_exhausted_flag = False
                await execute_scenario(
                    scenario_name="sustained_overload",
                    questions=q_list,
                    pipeline=pipeline,
                    source=args.source,
                    concurrency=args.concurrency or 10,
                    csv_writer=csv_writer,
                    traces_writer=traces_writer,
                )

            elif sc == "rpm_halved":
                # Halved rate-limit capacity (15 RPM, 4000 TPM)
                standin_mgr.configure(rpm_limit=15, tpm_limit=4000, rpd_remaining=1000, slow_tail_prob=0.05)
                gov.max_concurrency = args.concurrency or 6
                client.governor.daily_exhausted_flag = False
                await execute_scenario(
                    scenario_name="rpm_halved",
                    questions=EVAL_QUESTIONS,
                    pipeline=pipeline,
                    source=args.source,
                    concurrency=args.concurrency or 6,
                    csv_writer=csv_writer,
                    traces_writer=traces_writer,
                )

            elif sc == "daily_budget_exhausted":
                # Daily RPD / TPD exhausted -> triggers fast local retrieval fallback
                standin_mgr.configure(rpd_remaining=0)
                client.governor.daily_exhausted_flag = True
                await execute_scenario(
                    scenario_name="daily_budget_exhausted",
                    questions=EVAL_QUESTIONS,
                    pipeline=pipeline,
                    source=args.source,
                    concurrency=args.concurrency or 4,
                    csv_writer=csv_writer,
                    traces_writer=traces_writer,
                )
                client.governor.daily_exhausted_flag = False

            elif sc == "retrieval_sweep":
                # Reset stand-in to generous limits for clean retrieval comparison
                standin_mgr.configure(rpm_limit=60, tpm_limit=45000, rpd_remaining=1000, slow_tail_prob=0.0, reset_window=True)
                gov.max_concurrency = args.concurrency or 2
                client.governor.daily_exhausted_flag = False

                for s_name in ["S1_180tok_k3_norerank", "S2_450tok_k4_rerank", "S3_900tok_k6_norerank"]:
                    client.governor.reset_stats()
                    standin_mgr.configure(rpm_limit=60, tpm_limit=45000, rpd_remaining=1000, slow_tail_prob=0.0, reset_window=True)
                    await execute_scenario(
                        scenario_name=f"retrieval_sweep_{s_name}",
                        questions=EVAL_QUESTIONS,
                        pipeline=pipeline,
                        source=args.source,
                        concurrency=args.concurrency or 2,
                        csv_writer=csv_writer,
                        traces_writer=traces_writer,
                        retrieval_setting=s_name,
                    )

        print("\n" + "=" * 70)
        print("  BENCHMARK SUITE COMPLETE!")
        print("  Verify results with: python -m runner.verify_tables")
        print("=" * 70 + "\n")

    finally:
        await client.close()
        standin_mgr.stop()


if __name__ == "__main__":
    asyncio.run(main())

