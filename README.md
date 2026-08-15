# GenAI learning workspace

LangChain → LangGraph → RAG. Do them in order; each builds on the last.

## Setup (once)

```bash
cd learning/genai
uv sync                      # creates .venv from pyproject.toml
cp .env.example .env         # then paste your API keys in
uv run python 01-langchain/hello.py
```


No `uv`? `brew install uv`. Or use plain venv: `python -m venv .venv && source .venv/bin/activate && pip install -e .`


For notebooks: `uv run jupyter lab`

## Layout

```
genai/
├── 01-langchain/     models, prompts, LCEL, tools, agents
├── 02-langgraph/     state machines, branching, memory, human-in-the-loop
├── 03-rag/           load → split → embed → retrieve → generate
├── shared/           config.py — env loading, model choice
├── data/             your PDFs and text files (gitignored)
└── rag-udemy-course/ the course you already had, moved here
```

Each module has `notes/` (what you learned), `exercises/` (things you build), `scratch/` (throwaway).

## The path

| # | Module | You're done when you can... |
|---|---|---|
| 1 | LangChain | build a chain with `\|`, get typed output back, and give a model a tool it actually calls |
| 2 | LangGraph | draw your agent as a graph, branch on state, and resume a run from a checkpoint |
| 3 | RAG | explain why your retrieval returned the wrong chunk, and fix it |

## Why this order

LangChain gives you the pieces — model wrappers, prompts, output parsers, tools. LangGraph is what you reach for when a chain isn't enough: loops, branches, retries, pausing for a human. RAG is an *application* built from both, and debugging it means understanding retrieval separately from generation. Learning RAG first is the common mistake — you end up unable to tell whether a bad answer came from bad retrieval or a bad prompt.

## Ground rules that save time

- **Turn on tracing from day one.** Set `LANGSMITH_TRACING=true` in `.env`. Being able to see the actual prompt sent to the model is the single biggest debugging win.
- **Pin your reading to the version.** LangChain 1.0 (Oct 2025) changed a lot. Pre-1.0 tutorials use `initialize_agent`, `LLMChain`, `ConversationChain` — all superseded. If a tutorial shows those, it's stale.
- **`create_agent` is the current agent entry point**, from `langchain.agents`. Models are named `"provider:model"`.
- **Start cheap.** `gpt-4o-mini` for everything while learning. Change `DEFAULT_MODEL` in `.env` when you need better.

## Docs

- LangChain: https://docs.langchain.com/oss/python/langchain/
- LangGraph: https://docs.langchain.com/oss/python/langgraph/
- LangSmith (tracing): https://docs.smith.langchain.com/
