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

Run the interactive semantic cache:

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

**Note:** The benchmark temporarily clears the cache for its experiment and restores the previous cache afterward.

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

**Limitations:** These results come from a single five-request run. They are preliminary and do not establish general performance or answer correctness. Further testing should use larger question sets, repeated trials, controlled uncached baselines, and explicit cache-answer correctness checks.

## 🛣️ Roadmap

- [ ] Expand the benchmark with a larger, repeatable question set.
- [ ] Compare cached requests against controlled uncached baselines.
- [ ] Evaluate different similarity thresholds.
- [ ] Test semantic matches for correctness, not just similarity.
- [ ] Improve metrics and experiment reporting.
- [ ] Explore cost-aware routing across models of different sizes.