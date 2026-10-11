
from sentence_transformers import SentenceTransformer, util
from safety_eval import THRESHOLD, safety_check

# New test cases, separate from the original 20.
# Format: (category, cached question, incoming question, expected hit)

TEST_CASES = [
    # Valid paraphrases
    ("Paraphrase", "What is recursion?", "Define recursion.", True),
    ("Paraphrase", "What is a queue?", "Explain what a queue is.", True),
    ("Paraphrase", "What is an operating system?",
     "Explain the meaning of an operating system.", True),
    ("Paraphrase", "What are the applications of machine learning?",
     "What is machine learning used for?", True),
    ("Paraphrase", "What is binary search?",
     "Explain the binary search algorithm.", True),

    # Same negation meaning: should not be blocked just because of negation
    ("Negation", "Should passwords never be stored in plain text?",
     "Passwords should not be stored in plain text.", True),

    # Similar topics, different information requested
    ("Question intent", "What is Python?",
     "What is Python used for?", False),
    ("Question intent", "Explain recursion.",
     "What are the disadvantages of recursion?", False),
    ("Complexity", "What is the time complexity of merge sort?",
     "What is the space complexity of merge sort?", False),
    ("Complexity", "What is the time complexity of binary search?",
     "What is the time complexity of linear search?", False),

    # Negation changes meaning
    ("Negation", "Is Java faster than C++?",
     "Is Java not faster than C++?", False),
    ("Negation", "Should passwords be stored in plain text?",
     "Should passwords never be stored in plain text?", False),

    # Comparison changes
    ("Comparison", "Is Java faster than C++?",
     "Is C++ faster than Java?", False),
    ("Comparison", "Is Python better than C?",
     "Is Python worse than C?", False),
    ("Comparison", "Is Java faster than Python?",
     "Is Java slower than Python?", False),

    # Related but different
    ("Related topics", "What is a database index?",
     "What is a database transaction?", False),
    ("Related topics", "How does encryption work?",
     "How does compression work?", False),
    ("Related topics", "What is supervised learning?",
     "What is unsupervised learning?", False),
    ("Related topics", "How do I sort a list in Python?",
     "How do I search a list in Python?", False),
    ("Related topics", "What is an IP address?",
     "What is a MAC address?", False),
]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    correct = 0
    false_hits = 0
    false_misses = 0
    blocked_hits = 0

    print(f"\nThreshold: {THRESHOLD:.3f}")
    print(f"New test cases: {len(TEST_CASES)}")
    print("-" * 90)

    for i, (category, cached_q, incoming_q, expected_hit) in enumerate(
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
            blocked_hits += 1

        passed = predicted_hit == expected_hit
        correct += int(passed)

        if predicted_hit and not expected_hit:
            false_hits += 1
        elif not predicted_hit and expected_hit:
            false_misses += 1

        print(
            f"\n#{i:02} [{category}] "
            f"{'PASS' if passed else 'FAIL'}"
        )
        print(f"  Cached:    {cached_q}")
        print(f"  Incoming:  {incoming_q}")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Safety:     {reason}")
        print(f"  Expected:   {'HIT' if expected_hit else 'MISS'}")
        print(f"  Predicted:  {'HIT' if predicted_hit else 'MISS'}")

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print("GENERALIZATION RESULTS")
    print("=" * 70)
    print(f"Correct decisions: {correct}/{total}")
    print(f"Accuracy:          {correct / total * 100:.1f}%")
    print(f"False hits:        {false_hits}")
    print(f"False misses:      {false_misses}")
    print(f"Potential hits blocked by safety rules: {blocked_hits}")
    print("\nReview every FAIL before changing the safety rules.")
    print("Results apply only to this test set.")


if __name__ == "__main__":
    main()
