"""
tests/benchmark_latency.py — Day 10 baseline latency/throughput benchmark.

Measures p50/p95 latency and throughput against the live FastAPI backend
(Groq as the LLM). Run this BEFORE making any Day 10 optimization, so you
have real before-numbers to compare an after-optimization run against.

Usage:
    myenv\\Scripts\\python.exe tests\\benchmark_latency.py

Prerequisite: uvicorn src.api:app must be running on port 8000.

Notes:
- Uses distinct queries (not one repeated query) so the LLM cache from
  Day 8 doesn't artificially inflate the numbers — this measures real
  Groq round-trip latency, not cache-hit latency.
- Sends requests sequentially, not concurrently — this benchmark is about
  per-request latency, not the concurrency behavior already verified in
  tests/test_async.py on Day 8.
- Rate limiter is 10 req/60s (Day 9). Default N_RUNS below stays under
  that per batch; increase RATE_WINDOW pause if you raise N_RUNS.
"""

import time
import statistics
import requests

API_BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# Distinct queries, cycled through — avoids cache hits skewing the numbers.
QUERIES = [
    "What is an x-vector?",
    "What is statistics pooling in the x-vector system?",
    "What is the architecture of ECAPA-TDNN?",
    "What are the main components of ECAPA-TDNN?",
    "How is speaker verification evaluated in these papers?",
    "What is the Squeeze-and-Excitation block used for?",
    "What is Multi-layer Feature Aggregation?",
    "How does Res2Net differ from a standard ResNet block?",
]

N_RUNS = 8  # stays at/under RATE_LIMIT (10/60s) in one batch


def run_benchmark():
    latencies = []
    errors = 0

    print(f"Running {N_RUNS} sequential requests against {CHAT_ENDPOINT}...\n")

    for i in range(N_RUNS):
        query = QUERIES[i % len(QUERIES)]
        start = time.perf_counter()

        try:
            response = requests.post(
                CHAT_ENDPOINT,
                json={"query": query, "history": []},
                timeout=60,
            )
            elapsed = time.perf_counter() - start

            if response.status_code == 200:
                latencies.append(elapsed)
                print(f"[{i+1}/{N_RUNS}] {elapsed:.2f}s — {query[:50]}")
            else:
                errors += 1
                print(f"[{i+1}/{N_RUNS}] ERROR {response.status_code} — {query[:50]}")

        except requests.exceptions.RequestException as e:
            errors += 1
            print(f"[{i+1}/{N_RUNS}] EXCEPTION: {e}")

    if not latencies:
        print("\nNo successful requests — nothing to report.")
        return

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_index = min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)
    p95 = latencies_sorted[p95_index]
    mean = statistics.mean(latencies_sorted)
    total_time = sum(latencies)
    throughput = len(latencies) / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("BASELINE LATENCY / THROUGHPUT (Groq, no optimization)")
    print("=" * 60)
    print(f"Successful requests : {len(latencies)}/{N_RUNS}  (errors: {errors})")
    print(f"Mean latency        : {mean:.2f}s")
    print(f"p50 latency         : {p50:.2f}s")
    print(f"p95 latency         : {p95:.2f}s")
    print(f"Min / Max           : {min(latencies):.2f}s / {max(latencies):.2f}s")
    print(f"Throughput          : {throughput:.2f} req/s (sequential)")
    print("=" * 60)



if __name__ == "__main__":
    run_benchmark()