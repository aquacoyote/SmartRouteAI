from sentence_transformers import SentenceTransformer, util
from cache_eval import TEST_CASES
from safety import safety_check

THRESHOLD = 0.80


def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    correct = 0
    false_hits = 0
    false_misses = 0
    blocked_by_safety = 0

    print(f"\nSimilarity threshold: {THRESHOLD:.2f}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("-" * 90)

    for number, (category, cached_q, incoming_q, expected_hit) in enumerate(
        TEST_CASES, start=1
    ):
        embeddings = model.encode(
            [cached_q, incoming_q],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        similarity_match = similarity >= THRESHOLD
        safe, reason = safety_check(cached_q, incoming_q)
        predicted_hit = similarity_match and safe

        if similarity_match and not safe:
            blocked_by_safety += 1

        passed = predicted_hit == expected_hit
        correct += passed

        if predicted_hit and not expected_hit:
            false_hits += 1
        elif not predicted_hit and expected_hit:
            false_misses += 1

        print(f"\n#{number:02} [{category}] {'PASS' if passed else 'FAIL'}")
        print(f"  Cached:   {cached_q}")
        print(f"  Incoming: {incoming_q}")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Safety: {reason}")
        print(f"  Expected: {'HIT' if expected_hit else 'MISS'}")
        print(f"  Predicted: {'HIT' if predicted_hit else 'MISS'}")

    total = len(TEST_CASES)

    print("\n" + "=" * 90)
    print("SAFETY LAYER EVALUATION")
    print("=" * 90)
    print(f"Correct decisions: {correct}/{total}")
    print(f"Decision accuracy: {correct / total * 100:.1f}%")
    print(f"False hits:        {false_hits}")
    print(f"False misses:      {false_misses}")
    print(f"Potential hits blocked by safety rules: {blocked_by_safety}")
    print("\nThese results apply only to this small labelled test set.")


if __name__ == "__main__":
    main()
