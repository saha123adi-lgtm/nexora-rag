# Nexora Enterprise RAG Assistant

A retrieval-augmented assistant over the fictional Nexora Technologies knowledge base (21 documents in PDF, DOCX, MD and TXT).
It answers employee questions from the company documents, cites its sources, enforces role-based access, and a Streamlit UI shows
every step of the pipeline live. **To run and present it, follow [DEPLOY.md](DEPLOY.md)** (free, no payment anywhere).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python ingest.py                                      # builds ./chroma_db (downloads the ~90 MB model once)
python search.py --eval --out results/minilm_k5.csv   # retrieval metrics with the real model
streamlit run app.py                                  # the UI at http://localhost:8501
```

## Project layout

```
app.py           Streamlit UI: chat + live pipeline trace, access-control demo, knowledge base, evaluation, architecture
rag_core.py      The engine: embed -> access filter -> retrieve -> score gate -> prompt -> generate (reports every stage)
rag_common.py    Parsers (PDF/DOCX/MD/TXT), section-aware chunker, embedding backends, role filter
ingest.py        Builds the vector database
search.py        Search CLI and evaluation (--eval)
DEPLOY.md        Step-by-step: run locally, share a public link, free LLM options, demo script, troubleshooting
data/            The dataset (documents, metadata, users.json, qa_testset.csv with 89 questions)
results/         Saved evaluation runs (the app's Evaluation tab reads these)
chroma_db/       Created by ingest.py (safe to delete and rebuild)
Dockerfile       Optional container build (untested)
```

## How it works

1. **Parse** each format (PDF tables are rebuilt as rows). **Chunk** on the documents' numbered sections; long sections are split on line boundaries with overlap.
2. **Embed** chunks with `all-MiniLM-L6-v2` (384 dimensions, CPU, free) and store them in ChromaDB with source, section and role metadata.
3. For each question: **embed** it with the same model, **filter by role inside the database query** (restricted chunks are never retrieved for the wrong user),
   take the **top-k** by cosine similarity, and apply a **score gate** (below the threshold the assistant says it does not know and the LLM is not called).
4. Build a **prompt** containing only the retrieved chunks and strict rules (answer from context, cite document IDs, ignore instructions inside documents), then call the LLM.

Answer generation options (sidebar): retrieval only (no LLM), Google Gemini free tier, Ollama (local), any OpenAI-compatible API, or Claude (paid key, optional).

## Evaluation

`python search.py --eval` scores retrieval on 89 questions: hit@1, hit@k, MRR, and whether restricted content leaked to unauthorised roles.
"Hit" means the exact evidence phrase from `qa_testset.csv` appears in the retrieved chunks.

### Baseline (keyword-only stand-in model, no semantic model)

`results/baseline_keyword_hashing.csv` was produced with `EMBEDDING_BACKEND=hashing`, an offline stand-in that matches words rather than meaning.
It tests the pipeline without internet and gives a **baseline to beat**. These are NOT MiniLM results.

| Category | Questions | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| factual | 53 | 75.5% | 94.3% | 0.836 |
| paraphrase | 7 | 57.1% | 85.7% | 0.690 |
| multi_hop | 8 | 12.5% | 62.5% | 0.281 |
| access_allowed | 7 | 57.1% | 85.7% | 0.690 |
| **all answerable** | **75** | **65.3%** | **89.3%** | **0.750** |

Access control: 0 of 7 `access_denied` questions leaked restricted content.

## Experiments for your report

```bash
python ingest.py --max-chars 600  --min-chars 100 --collection nexora_small
python ingest.py --max-chars 3000 --min-chars 800 --collection nexora_large
python search.py --eval --collection nexora_small --out results/small.csv
python search.py --eval --collection nexora_large --out results/large.csv
```

The documents are short and the chunker follows their section structure, so default chunks average about 370 characters; `--min-chars`
(merging) and `--max-chars` (splitting) are what move the chunk size.

## Next steps

Hybrid search (BM25 + vectors), a cross-encoder reranker, query logging with thumbs up/down, prompt-injection tests, and a FastAPI backend.
