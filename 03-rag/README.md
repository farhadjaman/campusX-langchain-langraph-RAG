# 03 — RAG

Retrieval-Augmented Generation. Five stages: **load → split → embed/store → retrieve → generate.**

Most bad RAG output is a *retrieval* failure, not a generation failure. Always look at the retrieved chunks before blaming the prompt.

## Order to work through

1. **Loaders** — `PyPDFLoader`, `WebBaseLoader`, `DirectoryLoader`. Garbage in, garbage out; check what you actually extracted.
2. **Splitting** — `RecursiveCharacterTextSplitter`. `chunk_size` and `chunk_overlap` are your main knobs. Too small loses context, too big dilutes the embedding.
3. **Embeddings** — `text-embedding-3-small` is cheap and fine for learning. Understand that similarity is cosine distance in vector space, nothing more.
4. **Vector stores** — Chroma locally. Later: pgvector, Qdrant, Pinecone.
5. **Retrievers** — `.as_retriever(search_kwargs={"k": 3})`. Then MMR for diversity, and metadata filtering.
6. **The generation prompt** — "answer ONLY from context" plus an explicit "say you don't know". Without that second clause the model will happily invent.
7. **Evaluation** — before tuning anything, write 10 question/answer pairs you care about. Otherwise you're guessing.

## Then, the improvements that actually matter

- **Hybrid search** — BM25 keyword + vector. Fixes the "exact product code" class of failure that pure embeddings miss.
- **Re-ranking** — retrieve 20, re-rank to 5 with a cross-encoder.
- **Query rewriting** — expand or decompose the question before retrieving.
- **Contextual retrieval** — prepend a short document-level summary to each chunk before embedding.
- **Agentic RAG** — this is where LangGraph comes back: let the graph decide whether to retrieve, retrieve again, or answer directly.

## Checkpoints

- [ ] Answer a question from your own PDF
- [ ] Show the retrieved chunks alongside the answer
- [ ] Make it correctly say "I don't know" for something not in the docs
- [ ] Find a query where retrieval fails, and explain *why* it failed
- [ ] Improve that query's result by changing chunking or adding hybrid search
- [ ] Rebuild the pipeline as a LangGraph graph that can retry retrieval

## Traps

- Re-embedding the whole corpus on every run. Persist the store.
- Chunks so small the answer is split across two of them, and `k=3` only grabs one.
- No "say you don't know" instruction, so the model confabulates confidently.
- Tuning chunk size by vibes with no eval set.

## Run

```bash
uv run python 03-rag/hello.py
```

Put your own PDFs/text in `../data/` — it's gitignored, so nothing sensitive gets committed.
