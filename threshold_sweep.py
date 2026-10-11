
from sentence_transformers import SentenceTransformer, util
from cache_eval import TEST_CASES

THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    similarities = []

    print("Calculating similarities...")
    for category, cached_q, incoming_q, expected_hit in TEST_CASES:
        embeddings = model.encode(
            [cached_q, incoming_q],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        similarities.append((similarity, expected_hit))

    print("\nTHRESHOLD COMPARISON")
    print("-" * 76)
    print(
        f"{'Threshold':<12}{'Accuracy':<12}"
        f"{'False hits':<14}{'False misses':<16}{'Total hits':<12}"
    )

    for threshold in THRESHOLDS:
        correct = 0
        false_hits = 0
        false_misses = 0
        total_hits = 0

        for similarity, expected_hit in similarities:
            predicted_hit = similarity >= threshold
            total_hits += predicted_hit

            if predicted_hit == expected_hit:
                correct += 1
            elif predicted_hit:
                false_hits += 1
            else:
                false_misses += 1

        accuracy = correct / len(TEST_CASES) * 100

        print(
            f"{threshold:<12.3f}{accuracy:<12.1f}"
            f"{false_hits:<14}{false_misses:<16}{total_hits:<12}"
        )

    print("\nLower false hits are safer; lower false misses improve cache reuse.")
    print("This evaluates only our current labelled test set.")


if __name__ == "__main__":
    main()
