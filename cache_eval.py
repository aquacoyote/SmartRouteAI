
from sentence_transformers import SentenceTransformer, util

# Keep this aligned with semantic_cache.py
THRESHOLD = 0.80

# Each case: (category, cached question, incoming question, expected hit?)
TEST_CASES = [
    # 1. Clear paraphrases: should hit
    ("Paraphrase", "What is Python?", "Explain what Python is.", True),
    ("Paraphrase", "What is a list in Python?", "Explain Python lists.", True),
    ("Paraphrase", "What is a function?", "Explain what functions are.", True),
    ("Paraphrase", "What is machine learning?", "Explain machine learning.", True),
    ("Paraphrase", "What is a stack in data structures?",
     "Explain the stack data structure.", True),

    # 2. Exact repeats: should hit
    ("Exact repeat", "What is recursion?", "What is recursion?", True),
    ("Exact repeat", "What is an operating system?",
     "What is an operating system?", True),

    # 3. Related, but different questions: should miss
    ("Related topics", "What is Python?", "What is Python used for?", False),
    ("Related topics", "What is a list in Python?",
     "How do Python tuples differ from lists?", False),
    ("Related topics", "What is machine learning?",
     "What is deep learning?", False),
    ("Related topics", "What is a stack?",
     "What is a queue?", False),
    ("Related topics", "What is TCP?",
     "What is UDP?", False),

    # 4. Negation and reversed meaning: should miss
    ("Negation", "Is Python faster than C?",
     "Is Python not faster than C?", False),
    ("Reversed comparison", "Is Python faster than C?",
     "Is C faster than Python?", False),
    ("Opposite meaning", "Should passwords be stored as plain text?",
     "Should passwords never be stored as plain text?", False),

    # 5. Different topics: should miss
    ("Different topic", "What is photosynthesis?",
     "How does a CPU execute instructions?", False),
    ("Different topic", "Explain binary search.",
     "What causes earthquakes?", False),
    ("Different topic", "What is an eigenvalue?",
     "How do I make a linked list in C?", False),

    # 6. Tricky cases: investigate the result
    ("Tricky", "How do I sort a list in Python?",
     "How do I reverse a list in Python?", False),
    ("Tricky", "What is the time complexity of binary search?",
     "What is the space complexity of binary search?", False),
]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    expected_hits = sum(case[3] for case in TEST_CASES)
    expected_misses = len(TEST_CASES) - expected_hits

    correct = 0
    false_hits = 0
    false_misses = 0

    print(f"\nSimilarity threshold: {THRESHOLD:.2f}")
    print(f"Test cases: {len(TEST_CASES)}")
    print("-" * 100)

    for number, (category, cached_q, incoming_q, expected_hit) in enumerate(
        TEST_CASES, start=1
    ):
        embeddings = model.encode(
            [cached_q, incoming_q],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        predicted_hit = similarity >= THRESHOLD
        passed = predicted_hit == expected_hit

        if passed:
            correct += 1
        elif predicted_hit:
            false_hits += 1
        else:
            false_misses += 1

        prediction = "HIT" if predicted_hit else "MISS"
        expectation = "HIT" if expected_hit else "MISS"
        result = "PASS" if passed else "FAIL"

        print(f"\n#{number:02} [{category}] {result}")
        print(f"  Cached:   {cached_q}")
        print(f"  Incoming: {incoming_q}")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Expected: {expectation} | Predicted: {prediction}")

    total = len(TEST_CASES)
    accuracy = correct / total * 100

    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    print(f"Total cases:       {total}")
    print(f"Expected hits:     {expected_hits}")
    print(f"Expected misses:   {expected_misses}")
    print(f"Correct decisions: {correct}/{total}")
    print(f"Decision accuracy: {accuracy:.1f}%")
    print(f"False hits:        {false_hits}")
    print(f"False misses:      {false_misses}")

    if false_hits:
        print(
            "\nWARNING: False hits are the highest-priority failures. "
            "The cache would reuse an answer for a question that needs "
            "a different answer."
        )

    if false_misses:
        print(
            "\nNote: False misses reduce cache efficiency. "
            "The model failed to recognize an expected paraphrase."
        )

    print(
        "\nImportant: This tests similarity-based decisions, not whether "
        "the generated answers are factually correct."
    )


if __name__ == "__main__":
    main()
