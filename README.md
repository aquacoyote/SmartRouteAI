# SmartRouteAI 🚀

A local LLM semantic caching prototype that reuses answers to similar questions to reduce response latency and unnecessary model inference.

## 💡 The Problem

Large language model (LLM) applications can repeatedly process questions that mean almost the same thing. Even when two questions are semantically similar, a basic exact-match cache may treat them as completely different requests.

SmartRouteAI explores **semantic caching**: using sentence embeddings to identify similar questions and reuse a previous answer when the similarity is high enough.

## ⚙️ How It Works

1. **Embed the question:** Convert the incoming question into a numerical vector using `all-MiniLM-L6-v2`.
2. **Search the cache:** Compare the new embedding with previously stored question embeddings using cosine similarity.
3. **Check the threshold:** If the closest match meets the similarity threshold (`0.80`), return its cached answer.
4. **Generate when needed:** Otherwise, send the question to a local LLM through Ollama.
5. **Save the result:** Store the question, answer, and embedding in `cache.json` for future requests.

## 🧰 Tech Stack

- Python
- Sentence Transformers
- PyTorch
- Ollama
- Llama 3.2 3B
- JSON-based persistent storage
- Git and GitHub

## 🖥️ Getting Started

### Prerequisites

Install:

- Python
- [Ollama](https://ollama.com/)
- Git

Pull the model used by the prototype:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before sending requests.

### Installation

Clone the repository:

```bash
git clone https://github.com/aquacoyote/SmartRouteAI.git
cd SmartRouteAI
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the semantic cache prototype:

```bash
python semantic_cache.py
```

Run the embedding demonstration:

```bash
python embeddings_demo.py
```

## 📊 Current Features

- Sentence embedding generation
- Cosine-similarity-based cache lookup
- Configurable semantic similarity threshold
- Persistent cache storage across runs
- Local LLM inference through Ollama
- Basic cache-hit and latency metrics

## ⚠️ Limitations

- Similarity does not guarantee that two questions need the same answer.
- The threshold has not yet been rigorously evaluated.
- Performance measurements need systematic benchmarking across varied questions.
- The current prototype uses a simple JSON cache rather than a vector database.
- Local inference speed depends on the hardware and model configuration.

## 🛣️ Roadmap

- [ ] Benchmark cache hits versus model-generated responses.
- [ ] Evaluate different similarity thresholds.
- [ ] Test semantic matches for correctness, not just similarity.
- [ ] Improve metrics and experiment reporting.
- [ ] Explore cost-aware routing across models of different sizes.

## 🎯 Project Goal

Build and evaluate a practical, locally runnable semantic caching system, measuring when it improves latency and identifying when reusing a cached answer is unsafe or inaccurate.

---

**Status:** Experimental learning project in active development.
