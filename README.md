# SmartRouteAI 🚀

SmartRouteAI is a prototype exploring **semantic caching and cost-aware routing for Large Language Model (LLM) applications**.

LLM applications can receive repeated or similarly worded questions, potentially resulting in redundant model calls. SmartRouteAI explores whether semantic caching can reuse previous responses for sufficiently similar questions while using safety checks to reduce incorrect cache matches.

The project focuses on three areas:

- **Semantic caching:** Identify potentially reusable responses even when questions are worded differently.
- **Cache safety:** Detect important differences between similar questions before reusing a cached response.
- **Evaluation:** Measure cache behavior and investigate the trade-off between cache reuse and answer correctness.

> **Project status:** Early-stage prototype. Initial cache benchmarking and rule-based safety evaluation are implemented. Generalization to unseen questions and real-world cost savings remain to be validated.

## ✨ Features

- Semantic similarity-based question matching using sentence embeddings.
- Configurable similarity threshold for cache matching.
- Persistent cache storage.
- Rule-based safety validation before cached responses are reused.
- Evaluation scripts for cache matching and safety checks.
- Benchmarking of cache hits, misses, and request latency.
- Local LLM integration using Ollama.

## 🛡️ Cache Safety Validation

Semantic similarity alone is not enough to determine whether two questions can safely share an answer. Questions may have highly similar wording while asking for different information.

For example:

- "What is the time complexity of binary search?"
- "What is the space complexity of binary search?"

These questions are similar in wording but ask about different complexity measures. Reusing the answer to one for the other could produce an incorrect response.

SmartRouteAI includes a rule-based safety layer in `safety.py` that checks for several potentially unsafe matches.

| Safety check | Example distinction |
|---|---|
| Negation mismatch | "Is this algorithm safe?" vs. "Is this algorithm not safe?" |
| Reversed comparisons | "Does Python outperform C?" vs. "Does C outperform Python?" |
| Different named algorithms | Binary search vs. linear search |
| Time vs. space complexity | Execution time vs. memory usage |
| Question intent | Asking what something is vs. asking what it is used for |

These checks are intended to prevent unsafe cache reuse. They use handcrafted rules and pattern matching; they are not a general-purpose natural-language reasoning system.

### Safety Evaluation

The `safety_eval.py` script evaluates semantic matching alongside the safety layer using a hand-written test suite of 26 question pairs.

**Current result: 26/26 test cases passed.**

This is an initial regression-testing result. It demonstrates that the current implementation passes the selected test cases, but it does not establish 100% safety or reliable performance on unseen questions.

The next evaluation stage will test generalization using new question pairs that were not used to develop the rules.

## 🖥️ Getting Started

### Prerequisites

- Python 3.10 or newer
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

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Run SmartRouteAI

Start the interactive semantic cache:

```powershell
python semantic_cache.py
```

Run the embedding demonstration:

```powershell
python embeddings_demo.py
```

Run the benchmark:

```powershell
python benchmark.py
```

Run the cache-matching evaluation:

```powershell
python cache_eval.py
```

Run the safety evaluation:

```powershell
python safety_eval.py
```

**Note:** The benchmark temporarily clears the cache for its experiment and restores the previous cache afterward.

The embedding model may be downloaded on its first run. Hugging Face Hub may display a warning about unauthenticated access; this does not necessarily indicate a failure.

## 📈 Initial Benchmark

A preliminary experiment was run locally using five requests: an exact repeat, a paraphrase, an unrelated question, and a semantically related but distinct question.

| Metric | Result |
|---|---:|
| Requests tested | 5 |
| Cache hits | 2 |
| Cache misses | 3 |
| Observed cache hit rate | 40% |
| Average end-to-end request latency | 5.060 s |
| Average LLM-backed request latency | 8.425 s |
| Average cache-hit latency | 0.014 s |
| Average LLM generation time, when called | 8.397 s |

The cache reused the answer for an exact repeat and a paraphrase. It rejected an unrelated question and a related question that required a different answer.

### Benchmark Limitations

These results come from a single five-request run. They are preliminary and do not establish general performance or answer correctness.

Further testing should include:

- Larger and repeatable question sets.
- Controlled comparisons against uncached requests.
- Repeated trials under comparable conditions.
- Explicit checks of whether reused answers are correct.
- Measurements of latency, false cache hits, false misses, and API-cost savings.

The initial benchmark and the safety evaluation measure different aspects of the system. The five-request experiment measures observed cache behavior and latency, while the safety evaluation measures performance on selected question pairs.

## 🧪 Evaluation Strategy

SmartRouteAI currently includes two evaluation scripts:

- **`cache_eval.py`** — Evaluates semantic cache matching against expected hit and miss outcomes.
- **`safety_eval.py`** — Evaluates cache matching with the additional rule-based safety checks.

The current safety evaluation passes all 26 hand-written test cases. This provides an initial regression baseline for future changes.

A key priority is to evaluate the system on unseen questions to determine whether the safety rules generalize beyond the examples used during development.

Important metrics for future experiments include:

| Metric | Purpose |
|---|---|
| Cache hit rate | Measures how often requests reuse cached responses |
| False cache hit rate | Measures how often an incorrect reuse decision occurs |
| False miss rate | Measures how often a potentially reusable response is rejected |
| Cache-hit latency | Measures the time required to return a cached response |
| End-to-end latency | Measures total request-response time |
| Estimated API-cost savings | Estimates the reduction in model expenditure from avoided calls |

A high cache hit rate alone is not sufficient: the goal is to reuse responses when appropriate without compromising answer correctness.

## 🧰 Tech Stack

- **Python** — Core implementation and evaluation scripts.
- **Sentence Transformers** — Generates semantic embeddings for comparing questions.
- **Ollama** — Runs the local LLM used by the prototype.
- **Requests** — HTTP requests where needed by the implementation.
- **Git and GitHub** — Version control and project hosting.

### Dependencies

Dependencies are listed in `requirements.txt`:

```text
sentence-transformers
requests
```
## 🛣️ Roadmap

- [x] Implement semantic similarity-based cache matching.
- [x] Add persistent cache support.
- [x] Integrate local LLM inference through Ollama.
- [x] Add rule-based safety checks for potentially unsafe cache matches.
- [x] Create an initial safety evaluation suite with 26 test cases.
- [ ] Evaluate safety checks on unseen question pairs.
- [ ] Expand the benchmark with a larger, repeatable question set.
- [ ] Compare semantic caching against exact-match caching and uncached requests.
- [ ] Evaluate different similarity thresholds.
- [ ] Measure cache-answer correctness, false cache hits, and false misses.
- [ ] Improve benchmark reproducibility and experiment reporting.
- [ ] Explore cost-aware routing across models of different sizes.

## ⚠️ Limitations

SmartRouteAI is an experimental prototype and has several limitations:

- The safety layer relies on handcrafted rules and cannot detect every semantic mismatch.
- The current evaluation suite is small and manually designed.
- Passing the existing tests does not guarantee correctness on unseen questions.
- The initial latency benchmark is based on only five requests.
- Generalization, production-scale behavior, and actual cost savings have not yet been established.

These limitations will guide the next stages of development and evaluation.

## 📄 License

No license has been specified yet. Until a license is added to the repository, reuse and redistribution are subject to the applicable default copyright rules.