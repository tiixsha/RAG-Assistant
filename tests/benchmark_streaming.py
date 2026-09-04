"""
tests/benchmark_streaming.py — optimization benchmark: streaming vs.
non-streaming, measuring time-to-first-token.

Compares against the baseline from tests/benchmark_latency.py:
    Mean latency: 3.77s | p50: 3.65s | p95: 4.84s

Usage:
    myenv\\Scripts\\python.exe tests\\benchmark_streaming.py

Prerequisite: uvicorn src.api:app must be running on port 8000.
"""

import time
import json
import statistics
import requests

API_BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE_URL}/chat/stream"

QUERIES = [
    "What is an x-vector?",
    "What is statistics pooling in the x-vector system?",
    "What is the architecture of ECAPA-TDNN?",
    "What are the main components of ECAPA-TDNN?",
    "How is speaker verification evaluated in these papers?",
    "What is the Squeeze-and-Excitation block used for?",
]

N_RUNS = 6


def run_streaming_benchmark():
    first_token_times = []
    total_times = []

    print(f"Running {N_RUNS} streaming requests against {STREAM_ENDPOINT}...\n")

    for i, query in enumerate(QUERIES[:N_RUNS]):
        start = time.perf_counter()
        first_token_time = None

        try:
            with requests.post(
                STREAM_ENDPOINT,
                json={"query": query, "history": []},
                stream=True,
                timeout=60,
            ) as response:

                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue

                    data = json.loads(line[len("data: "):])

                    if "token" in data and first_token_time is None:
                        first_token_time = time.perf_counter() - start

                    if data.get("done"):
                        total_time = time.perf_counter() - start
                        first_token_times.append(first_token_time)
                        total_times.append(total_time)
                        print(
                            f"[{i+1}/{N_RUNS}] first token: {first_token_time:.2f}s, "
                            f"total: {total_time:.2f}s — {query[:50]}"
                        )
                        break

                    if "error" in data:
                        print(f"[{i+1}/{N_RUNS}] STREAM ERROR: {data['error']}")
                        break

        except requests.exceptions.RequestException as e:
            print(f"[{i+1}/{N_RUNS}] EXCEPTION: {e}")

    if not first_token_times:
        print("\nNo successful streams — nothing to report.")
        return

    print("\n" + "=" * 60)
    print("STREAMING BENCHMARK (time-to-first-token)")
    print("=" * 60)
    print(f"Successful streams   : {len(first_token_times)}/{N_RUNS}")
    print(f"Mean time-to-first-token : {statistics.mean(first_token_times):.2f}s")
    print(f"p50 time-to-first-token  : {statistics.median(first_token_times):.2f}s")
    print(f"Mean total stream time   : {statistics.mean(total_times):.2f}s")
    print("=" * 60)
    print("\nCompare 'time-to-first-token' above against the non-streaming")
    print("baseline (tests/benchmark_latency.py) — that number represents")
    print("how long a user previously waited before seeing ANY output.")
    print("=" * 60)


if __name__ == "__main__":
    run_streaming_benchmark()