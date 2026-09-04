import asyncio
import time

import httpx


API_URL = "http://127.0.0.1:8000"


async def send_query(client: httpx.AsyncClient, query: str):
    response = await client.post(
        f"{API_URL}/chat",
        json={"query": query},
        timeout=120.0,
    )

    response.raise_for_status()

    data = response.json()

    return {
        "query": query,
        "answer": data["answer"],
    }


async def main():

    queries = [
        "What is an x-vector?",
        "What is statistics pooling?",
        "What is the architecture of ECAPA-TDNN?",
        "What is speaker verification?",
        "What are the advantages of ECAPA-TDNN?",
    ]

    start = time.perf_counter()

    async with httpx.AsyncClient() as client:

        tasks = [
            send_query(client, query)
            for query in queries
        ]

        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 80)
    print(f"Completed {len(results)} requests")
    print(f"Total time: {elapsed:.2f} seconds")
    print("=" * 80)

    for result in results:

        print("\n" + "-" * 80)
        print("QUERY:")
        print(result["query"])

        print("\nANSWER:")
        print(result["answer"])


if __name__ == "__main__":
    asyncio.run(main())