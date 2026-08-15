# 01 — LangChain

The vocabulary layer. Almost everything here is a **Runnable**: it has `.invoke()`, `.stream()`, `.batch()`, and composes with `|`.

## Order to work through

1. **Chat models** — `init_chat_model("openai:gpt-4o-mini")`. Messages in, message out.
2. **Prompt templates** — `ChatPromptTemplate.from_messages([...])` with `{variables}`.
3. **LCEL** — `prompt | model | parser`. Understand that `|` builds an object, it doesn't run anything.
4. **Output parsers** — `StrOutputParser`, then `.with_structured_output(PydanticModel)`. Structured output is what makes an LLM usable as a component instead of a chatbot.
5. **Tools** — the `@tool` decorator. The docstring *is* the spec the model reads. Write it carefully.
6. **Agents** — `create_agent(model=..., tools=[...])`. A loop: model picks a tool, tool runs, result goes back, repeat until it answers.
7. **Streaming** — `.stream()` and why token streaming matters for UX.

## Checkpoints

- [ ] Explain what `prompt | model | parser` actually constructs
- [ ] Get a Pydantic object back from a model, not a string you have to parse
- [ ] Write a tool whose docstring makes the model call it correctly on the first try
- [ ] Build an agent that chains two tool calls to answer one question
- [ ] Stream a response token by token

## Traps

- `.invoke()` takes a **dict** when the chain starts with a prompt template, a plain string when it starts with a model. Mixing these up is the most common early error.
- A tool's docstring and type hints are sent to the model. Vague docstring → tool never gets called.
- Old tutorials use `LLMChain`, `initialize_agent`, `ConversationChain`. All pre-1.0. Skip them.

## Run

```bash
uv run python 01-langchain/hello.py
```
