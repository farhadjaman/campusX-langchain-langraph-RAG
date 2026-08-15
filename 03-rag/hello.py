"""RAG hello world: load -> split -> embed -> retrieve -> generate.

Drop a PDF or .txt into ../data/ first, or it will use the built-in sample.
Run:  uv run python 03-rag/hello.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

from shared.config import DATA, DEFAULT_MODEL, require

require("OPENAI_API_KEY")

# 1. LOAD
files = list(DATA.glob("*.txt")) + list(DATA.glob("*.md"))
if files:
    docs = [Document(page_content=f.read_text(), metadata={"source": f.name}) for f in files]
    print(f"Loaded {len(docs)} file(s) from data/")
else:
    docs = [Document(
        page_content=(
            "The Bengal tiger is the national animal of Bangladesh. "
            "The Sundarbans, a mangrove forest shared with India, is its main habitat. "
            "Estimates put the Bangladeshi population near 100 individuals."
        ),
        metadata={"source": "sample"},
    )]
    print("No files in data/ — using the built-in sample paragraph.")

# 2. SPLIT — chunk_size and overlap are the two knobs that matter most.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# 3. EMBED + STORE
store = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = store.as_retriever(search_kwargs={"k": 3})

# 4. GENERATE
prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. If the context does not contain the answer, "
    "say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}"
)


def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


rag = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | init_chat_model(DEFAULT_MODEL)
    | StrOutputParser()
)

q = "What is the national animal of Bangladesh and where does it live?"
print(f"\nQ: {q}\nA: {rag.invoke(q)}")

print("\n--- retrieved chunks ---")
for i, d in enumerate(retriever.invoke(q), 1):
    print(f"{i}. [{d.metadata.get('source')}] {d.page_content[:90]}...")
