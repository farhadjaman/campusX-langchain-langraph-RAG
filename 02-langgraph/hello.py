"""LangGraph 1.x hello world: state, nodes, conditional edges, persistence.

Run:  uv run python 02-langgraph/hello.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Annotated
from typing_extensions import TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# State is the single object every node reads and writes.
# Annotated[..., add] means "append", not "overwrite" — this is the key idea.
class State(TypedDict):
    topic: str
    steps: Annotated[list[str], add]
    score: int


def research(state: State) -> dict:
    return {"steps": ["researched"], "score": len(state["topic"])}


def short_path(state: State) -> dict:
    return {"steps": ["took short path"]}


def long_path(state: State) -> dict:
    return {"steps": ["took long path"]}


def decide(state: State) -> str:
    """A conditional edge returns the NAME of the next node."""
    return "long" if state["score"] > 10 else "short"


builder = StateGraph(State)
builder.add_node("research", research)
builder.add_node("short", short_path)
builder.add_node("long", long_path)

builder.add_edge(START, "research")
builder.add_conditional_edges("research", decide, {"short": "short", "long": "long"})
builder.add_edge("short", END)
builder.add_edge("long", END)

# A checkpointer makes the graph resumable and gives you conversation memory.
graph = builder.compile(checkpointer=InMemorySaver())

for topic in ["cats", "distributed systems"]:
    cfg = {"configurable": {"thread_id": topic}}
    out = graph.invoke({"topic": topic, "steps": [], "score": 0}, cfg)
    print(f"{topic:22} score={out['score']:<3} steps={out['steps']}")

print("\nMermaid diagram of the graph:\n")
print(graph.get_graph().draw_mermaid())
