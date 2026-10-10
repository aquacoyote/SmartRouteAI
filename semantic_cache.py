
import time
import requests
import json
import torch 

from pathlib import Path 
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

CACHE_FILE = Path(__file__).parent / "cache.json"

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
THRESHOLD = 0.80

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready!")

# Cache entries contain question, embedding, and answer

def load_cache():
    if not CACHE_FILE.exists():
        return []

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        entries = data.get("entries", [])

        for entry in entries:
            entry["embedding"] = torch.tensor(
                entry["embedding"], dtype=torch.float32
            )
        print(f"Loaded {len(entries)} cached entries from disk.")
        return entries

    except (json.JSONDecodeError, OSError,KeyError,TypeError) as error:
        print(f"Could not load cache: {error}")
        print("Starting with an empty cache.")
        return []


cache = load_cache()

def save_cache():
    data ={"entries": []}

    for entry in cache:
        data["entries"].append({
            "question": entry["question"],
            "answer": entry ["answer"],
            "embedding": entry["embedding"].tolist()
        })
    with open(CACHE_FILE,"w",encoding="utf-8") as file:
        json.dump(data,file,indent=2)

# Metrics for our experiment
stats = {
    "requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "llm_calls": 0,
    "llm_time_seconds": 0.0,
}


def generate_answer(question):
    """Generate a real answer using our local LLM."""
    payload = {
        "model": LLM_MODEL,
        "prompt": (
            "Answer the user's question clearly and accurately. "
            "Be concise but explain important concepts.\n\n"
            f"Question: {question}"
        ),
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 200,
        },
    }

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()
    elapsed = time.perf_counter() - start

    stats["llm_calls"] += 1
    stats["llm_time_seconds"] += elapsed

    return data["response"].strip(), elapsed


def answer_question(question):
    stats["requests"] += 1

    # Convert the new question into an embedding
    embedding = embedding_model.encode(question)

    best_score = -1.0
    best_match = None

    # Find the most similar cached question
    for entry in cache:
        similarity = cos_sim(
            embedding,
            entry["embedding"],
        ).item()

        if similarity > best_score:
            best_score = similarity
            best_match = entry

    # Reuse the answer if similarity is high enough
    if best_match is not None and best_score >= THRESHOLD:
        stats["cache_hits"] += 1

        print(f"\n[CACHE HIT] Similarity: {best_score:.3f}")
        print(f"[MATCH] {best_match['question']}")

        return best_match["answer"]

    # No sufficiently similar cached question: call the LLM
    stats["cache_misses"] += 1

    print(f"\n[CACHE MISS] Best similarity: {best_score:.3f}")
    print(f"[LOCAL LLM] Calling {LLM_MODEL}...")

    answer, elapsed = generate_answer(question)

    # Save the question, embedding, and real AI-generated answer
    cache.append({
        "question": question,
        "embedding": embedding,
        "answer": answer,
    })

    save_cache()

    print(f"[GENERATION TIME] {elapsed:.2f} seconds")

    return answer


def show_stats():
    requests_count = stats["requests"]
    hits = stats["cache_hits"]
    calls = stats["llm_calls"]

    hit_rate = (
        hits / requests_count * 100
        if requests_count else 0
    )

    print("\n========== SMARTROUTE AI STATS ==========")
    print(f"Total requests:       {requests_count}")
    print(f"Cache hits:           {hits}")
    print(f"Cache misses:         {stats['cache_misses']}")
    print(f"Actual LLM calls:     {calls}")
    print(f"Cache hit rate:       {hit_rate:.1f}%")
    print(f"Total LLM time:       {stats['llm_time_seconds']:.2f}s")

    if calls:
        average = stats["llm_time_seconds"] / calls
        estimated_saved = hits * average

        print(f"Average LLM latency:  {average:.2f}s")
        print(
            "Estimated generation time avoided: "
            f"{estimated_saved:.2f}s"
        )

    print("API charges:          Rs 0 (local inference)")
    print("==========================================")


while True:
    try:
        question = input(
            "\nAsk a question (stats / exit): "
        ).strip()

        if question.lower() == "exit":
            show_stats()
            print("Side quest paused. Keep building!")
            break

        if question.lower() == "stats":
            show_stats()
            continue

        if not question:
            continue

        start = time.perf_counter()
        answer = answer_question(question)
        total_time = time.perf_counter() - start

        print(f"\nAnswer:\n{answer}")
        print(f"\n[TOTAL REQUEST TIME] {total_time:.2f}s")

    except requests.exceptions.ConnectionError:
        print(
            "Could not connect to Ollama. "
            "Make sure Ollama is installed and running."
        )

    except requests.exceptions.RequestException as error:
        print(f"LLM request failed: {error}")

    except KeyboardInterrupt:
        print("\nStopping SmartRoute AI...")
        show_stats()
        break