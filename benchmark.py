
import time
import statistics

from semantic_cache import answer_question, cache, stats, save_cache


def run_benchmark():
    # Preserve the existing cache and statistics.
    original_cache = [
        {
            "question": entry["question"],
            "embedding": entry["embedding"].clone()
            if hasattr(entry["embedding"], "clone")
            else entry["embedding"].copy(),
            "answer": entry["answer"],
        }
        for entry in cache
    ]

    original_stats = stats.copy()

    try:
        # Start the experiment with an empty cache.
        cache.clear()
        save_cache()

        for key in stats:
            stats[key] = 0 if key != "llm_time_seconds" else 0.0

        test_questions = [
            ("Exact repeat", "What is Python?"),
            ("Exact repeat", "What is Python?"),
            ("Paraphrase", "Explain what Python is."),
            ("Different topic", "What is photosynthesis?"),
            ("Tricky similarity", "Is Python faster than C?"),
        ]

        results = []

        print("\n=== SmartRouteAI Benchmark ===")

        for test_type, question in test_questions:
            print(f"\nTest: {test_type}")
            print(f"Question: {question}")

            start = time.perf_counter()
            answer = answer_question(question)
            elapsed = time.perf_counter() - start

            results.append({
                "type": test_type,
                "question": question,
                "elapsed": elapsed,
            })

            print(f"Answer: {answer}")
            print(f"Total elapsed: {elapsed:.3f} seconds")

        total = len(results)
        hits = stats["cache_hits"]
        misses = stats["cache_misses"]

        hit_rate = hits / total * 100 if total else 0
        latencies = [result["elapsed"] for result in results]

        print("\n=== Benchmark Summary ===")
        print(f"Total requests: {total}")
        print(f"Cache hits: {hits}")
        print(f"Cache misses: {misses}")
        print(f"Cache hit rate: {hit_rate:.1f}%")
        print(
            f"Average latency: "
            f"{statistics.mean(latencies):.3f} seconds"
        )
        print(f"Fastest request: {min(latencies):.3f} seconds")
        print(f"Slowest request: {max(latencies):.3f} seconds")

        if stats["llm_calls"]:
            print(
                f"Average LLM latency: "
                f"{stats['llm_time_seconds'] / stats['llm_calls']:.3f} seconds"
            )

    finally:
        # Restore the cache and statistics after the experiment.
        cache.clear()
        cache.extend(original_cache)
        save_cache()

        stats.clear()
        stats.update(original_stats)

        print("\nOriginal cache and statistics restored.")


if __name__ == "__main__":
    run_benchmark()
