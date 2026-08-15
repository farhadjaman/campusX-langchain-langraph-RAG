# 02 — LangGraph

When a chain isn't enough. LangGraph is a state machine: **state** + **nodes** that update it + **edges** that decide what runs next.

## Order to work through

1. **State** — a `TypedDict`. Understand `Annotated[list, add]` (append) vs a plain field (overwrite). This trips up everyone once.
2. **Nodes** — plain functions: take state, return a dict of updates. Return only what changed.
3. **Edges** — `add_edge` for fixed, `add_conditional_edges` for branching. A conditional edge returns the *name* of the next node.
4. **Compile** — `builder.compile()`. Nothing runs before this.
5. **Checkpointers** — `InMemorySaver` to start, then `SqliteSaver` to persist. This is what gives you memory and resumability.
6. **Threads** — `{"configurable": {"thread_id": "..."}}`. One thread = one conversation.
7. **Human-in-the-loop** — `interrupt()` to pause mid-graph, get approval, resume.
8. **Streaming** — `stream_mode="values"` vs `"updates"` vs `"messages"`.

## Checkpoints

- [ ] Draw your graph with `graph.get_graph().draw_mermaid()` and have it match your mental model
- [ ] Build a graph that loops back on itself until a condition is met
- [ ] Resume a conversation after restarting the process (SqliteSaver)
- [ ] Pause for human approval before a destructive step, then continue
- [ ] Explain when you'd reach for LangGraph over a plain LCEL chain

## Traps

- Forgetting `.compile()`.
- Returning full state from a node instead of just the changed keys.
- Using a plain `list` field and wondering why history keeps getting wiped — you wanted `Annotated[list, add]`.
- Reusing one `thread_id` for everything, so all conversations bleed together.

## Run

```bash
uv run python 02-langgraph/hello.py
```
