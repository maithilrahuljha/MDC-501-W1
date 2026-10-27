"""
WHAT: Times 10,000 rows through batch scoring and 10,000 sequential HTTP requests, then prints one comparison table.
HOW:  start online.py in another terminal, then: python benchmark.py    (add --rows 2000 for a quick run)
OK:   prints "RESULT" with a table showing batch seconds, online seconds, and average online latency in ms.
"""

# =============================================================================
# WHY A PLAIN asyncio/httpx SCRIPT AND NOT LOCUST?
# -----------------------------------------------------------------------------
# Locust is excellent when you want to *load test*: thousands of virtual users,
# a live web dashboard, percentile charts. That is a different question.
#
# This lab asks one small question: "what does the same 10,000 rows cost if they
# arrive one at a time instead of all at once?" For that, a 40-line script that
# prints one honest table beats a tool the student has to install, configure and
# learn to read. Fewer moving parts, clearer lesson.
#
# If you later need to know "how many users can this service survive?", install
# locust and point it at POST /predict. Same model, bigger question.
# =============================================================================

import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx
from batch import DEFAULT_INPUT, ensure_input_file, score_batch
from online import FEATURES, load_or_train_model

BASE_DIR = Path(__file__).resolve().parent


def human_row(label: str, rows: int, seconds: float, ms_per_row: float) -> str:
    return f"{label:<34}{rows:>8,}{seconds:>13.3f}{ms_per_row:>15.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 4 - batch vs online benchmark")
    parser.add_argument("--rows", type=int, default=10_000, help="rows to score each way (default 10000)")
    parser.add_argument("--port", type=int, default=8000, help="port that online.py is listening on")
    parser.add_argument("--url", type=str, default="", help="full base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=3, help="how many times to time the batch call")
    args = parser.parse_args()

    base_url = args.url or f"http://127.0.0.1:{args.port}"
    rows_wanted = args.rows

    print("=" * 72)
    print("LAB 4 - BATCH vs ONLINE, SAME MODEL, SAME ROWS")
    print("=" * 72)
    print(f"  rows per method : {rows_wanted:,}")
    print(f"  online service  : {base_url}")
    print()

    # ---------------------------------------------------------------- 0. check
    print("[1/4] Knocking on the online service ...")
    try:
        with httpx.Client(timeout=5.0) as client:
            health = client.get(base_url + "/health")
        print(f"      /health -> HTTP {health.status_code} {health.text[:110]}")
        if health.status_code != 200:
            print("      The service is up but unhappy. Check the online.py terminal.")
            return 1
    except httpx.HTTPError as exc:
        print(f"      [OFFLINE] {type(exc).__name__} while contacting {base_url}")
        print()
        print("      The benchmark needs the online service running in another window.")
        print("      Fix:")
        print("        1. open a second terminal")
        print("        2. cd into lab4_batch_vs_online")
        print("        3. run:  python online.py")
        print("        4. leave it open, come back here, run this script again")
        return 1

    # ---------------------------------------------------------------- 1. data
    print("\n[2/4] Preparing the rows (identical for both methods) ...")
    input_path = ensure_input_file(rows_wanted, DEFAULT_INPUT)
    import pandas as pd

    rows = pd.read_csv(input_path)
    payloads = rows[FEATURES].to_dict(orient="records")
    model, metadata = load_or_train_model()
    print(f"      {len(rows):,} rows ready. Model: {metadata.get('source', 'unknown')}")

    # ---------------------------------------------------------------- 2. batch
    print(f"\n[3/4] BATCH: scoring {len(rows):,} rows in one call ({args.repeat} timed runs) ...")
    batch_times = []
    for run in range(args.repeat):
        predictions, elapsed = score_batch(model, rows)
        batch_times.append(elapsed)
        print(f"      run {run + 1}: {elapsed:.3f} s")
    batch_best = min(batch_times)
    batch_median = statistics.median(batch_times)

    # -------------------------------------------------------------- 3. online
    print(f"\n[4/4] ONLINE: sending {len(payloads):,} HTTP requests, ONE AT A TIME ...")
    print("      (sequential on purpose -- we are measuring per-request cost,")
    print("       not how much the service can take in parallel)")

    errors = 0
    latencies_ms = []
    started = time.perf_counter()
    with httpx.Client(timeout=10.0) as client:
        # Warm-up, not timed: the first request pays import/socket costs.
        for payload in payloads[:20]:
            client.post(base_url + "/predict", json=payload)

        progress_every = max(len(payloads) // 5, 1)
        for index, payload in enumerate(payloads, start=1):
            request_started = time.perf_counter()
            response = client.post(base_url + "/predict", json=payload)
            latencies_ms.append((time.perf_counter() - request_started) * 1000)
            if response.status_code != 200:
                errors += 1
            if index % progress_every == 0:
                done_pct = index / len(payloads)
                print(f"      {index:>7,} requests  ({done_pct:>5.1%})  "
                      f"last latency {latencies_ms[-1]:.2f} ms")
    online_total = time.perf_counter() - started

    if errors:
        print(f"\n      WARNING: {errors:,} requests did not return 200. Averaged anyway.")

    avg_latency = statistics.mean(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1] if latencies_ms else float("nan")
    slowdown = online_total / batch_best if batch_best > 0 else float("inf")

    # ---------------------------------------------------------------- 4. table
    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"{'METHOD':<34}{'ROWS':>8}{'TOTAL (s)':>13}{'PER ROW (ms)':>15}")
    print("-" * 72)
    print(human_row("Batch (1 call, in-process)", len(rows), batch_best, batch_best / len(rows) * 1000))
    print(human_row("Online (sequential HTTP)", len(payloads), online_total, online_total / len(payloads) * 1000))
    print("-" * 72)
    print()
    print(f"  Average latency per online request : {avg_latency:.2f} ms")
    print(f"  Median  latency per online request : {p50:.2f} ms")
    print(f"  95th percentile (p95) latency      : {p95:.2f} ms")
    print(f"  Online requests per second         : {len(payloads) / online_total:,.0f}")
    print(f"  Batch runs (best of {args.repeat})              : {batch_best:.3f} s "
          f"(median {batch_median:.3f} s)")
    print(f"  ONLINE IS SLOWER BY A FACTOR OF    : {slowdown:,.1f}x")
    print()
    print("=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  The model did the same arithmetic both times. The difference is")
    print("  everything AROUND the arithmetic: a TCP connection, an HTTP request,")
    print("  JSON parsing, a validation step, a thread pool hop, an HTTP response.")
    print()
    print("  Batch wins on COST. Online wins on FRESHNESS.")
    print()
    print("  So: use batch for last night's customers (cheap, simple, fine).")
    print("  Use online only when the answer must exist in the next second --")
    print("  and then budget for latency, because your p95 is what users feel.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
