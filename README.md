# RAG Evaluation Harness

A Retrieval-Augmented Generation (RAG) pipeline with a built-in evaluation harness that scores retrieval accuracy, answer faithfulness, and correct abstention on unanswerable questions — tested against a corpus of real documentation from FastAPI, Pydantic, and Starlette.

## What this is

Most RAG demos stop at "ask a question, get an answer." This project goes further: it automatically tests whether a RAG system is actually retrieving the right information and generating answers that are genuinely grounded in that information — not just plausible-sounding text.

The corpus (25 real documentation files from three overlapping Python libraries) was chosen deliberately: FastAPI, Pydantic, and Starlette share similar concepts (middleware, background tasks, exception handling) implemented differently, which creates realistic opportunities to test whether retrieval can distinguish similar-but-different information rather than just finding "something relevant."

## Why build this when RAGAS, TruLens, and DeepEval already exist

They do, and they're mature, industry-standard tools. This project isn't a claim to have invented a new category of tool — it's a from scratch implementation of the same evaluation methodology those tools formalize, built to demonstrate a working understanding of *why* RAG systems fail and how to measure it, rather than treating evaluation as a black box.

## Architecture

```
corpus/                 25 real doc files (FastAPI, Pydantic, Starlette)
test_set.json           25 ground-truth Q&A pairs across 6 difficulty categories
app/
  chunking.py           Splits docs into focused, header-aware chunks
  embedding.py          Embeds chunks (Gemini) and stores them in Chroma
  retrieval.py          Retrieves relevant chunks, generates grounded answers
  evaluation.py         Runs the ground-truth set through the pipeline,
                        scores retrieval accuracy, faithfulness (LLM-as-judge),
                        and abstention correctness
  api_utils.py           Shared retry/pacing logic for external API calls
  main.py               FastAPI app exposing /health, /query, /ingest, /evaluate
```

**Pipeline flow:** documents → chunked → embedded → stored in a vector database (Chroma) → a question is embedded and matched against stored chunks → the closest chunks are handed to an LLM, explicitly instructed to answer only from that context and say so when it can't. The evaluation harness runs this whole pipeline against known-answer questions and reports where it succeeds and fails.

## Test set design

25 questions across six categories, testing different failure modes:

| Category            | Count   | Tests                                    
|-------------------------------|---------|-----------------------                                   
| Straightforward     | 12      | Basic retrieval + generation correctness 
| Exact fact          | 4       | Faithfulness to specific details (e.g. exact algorithm names)                                   
| Distractor pair     | 3 pairs | Retrieval precision between similar libraries (.e.g. FastAPI vs. Starlette background tasks)   
| No answer in corpus | 2       | Correct abstention instead of hallucination 
| Ambiguous phrasing  | 2       | Robustness to vague or confusing questions 
| Multi-hop           | 2       | Combining information across two source documents 
                                 

## Results

```json
{
  "total_questions": 25,
  "retrieval_accuracy": 1.0,
  "faithfulness_rate": 0.96,
  "abstention_accuracy": 1.0
}
```

**One genuine faithfulness failure was found and is documented, not hidden:** on the question *"How do you declare a required path parameter with a type in FastAPI?"*, the system incorrectly answered "not found in the provided context" despite the answer being present in the retrieved chunks. This is a real, specific limitation surfaced by the evaluation harness working as intended — a system that reports 100% across the board on every run is a stronger signal of a measurement gap than of a flawless pipeline.

Full per-question results, including retrieved sources and judge reasoning for every question, are in `sample_evaluation_report.json`.

## Notable engineering findings along the way

- **A retrieval ranking bug, found and fixed with measurement, not guessing:** a known-correct chunk initially ranked 7th out of 436 for a valid question, because its instructional phrasing didn't match natural question phrasing. Embedding each chunk's heading alongside its body text moved the same chunk to rank 2-3 — verified before and after with a dedicated diagnostic script.
- **A hidden scoring bug:** an initial fragile JSON-parsing approach for the faithfulness judge silently dropped 18 of 25 real verdicts as "unparseable," inflating the reported faithfulness rate. Rebuilding the parser to extract JSON more robustly recovered all 25 real judgments — including the one genuine failure above, which had been masked by the parsing bug.
- **A silent duplicate-call bug:** an incremental edit left two calls to the same retry-wrapped function in place, doubling real API usage without any visible symptom other than exhausting rate limits unexpectedly quickly.
- **Free-tier API constraints:** the generation model's free tier allows a very small number of requests per day, discovered through direct testing rather than documentation alone. The evaluation harness is built to be resumable across multiple runs/days as a result — progress is saved after every question, so a quota limit costs only the remaining questions, not a full restart.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your real GEMINI_API_KEY
```

## Running

```bash
cd app
python embedding.py       # builds the vector index (only needed once, or if the corpus changes)
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API.

Endpoints:
- `GET /health` — liveness check
- `POST /query` — ask a question, get a grounded answer with sources
- `POST /ingest` — rebuild the vector index from `corpus/`
- `POST /evaluate` — run the full evaluation harness against `test_set.json`

## Known limitations

- The embedding model used for chunk-size calculations is approximated by word count, not an exact tokenizer, since this avoids an external dependency for a purely sizing-related decision.
- The ground-truth test set is hand-curated at a small scale (25 questions) — it demonstrates the evaluation methodology rather than constituting a statistically comprehensive benchmark.
- Free-tier API rate limits meant the evaluation harness had to be built with resumability and retry/pacing logic as first-class features, not an afterthought — this is reflected in `api_utils.py`.

## Deployment

Deployed on Render (free tier). The pre-built `chroma_store/` vector index is committed to the repository so a fresh deployment doesn't need to re-run ingestion (and re-spend API quota) on every restart.

Environment variable required: `LLM_API_KEY`.  (i used GEMINI API KEY)

Note: free-tier hosting spins down after periods of inactivity; the first request after idle time may take 30-60 seconds to respond while the service restarts.
