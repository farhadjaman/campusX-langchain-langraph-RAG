"""LangChain 1.x hello world: model -> prompt -> structured output -> agent.

Run:  uv run python 01-langchain/hello.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from shared.config import DEFAULT_MODEL, require

require("OPENAI_API_KEY")
model = init_chat_model(DEFAULT_MODEL)

# 1. Plain call
print("--- 1. plain ---")
print(model.invoke("Say hello in exactly five words.").content)

# 2. Prompt + model piped together (LCEL). The | operator builds a Runnable.
print("\n--- 2. LCEL chain ---")
prompt = ChatPromptTemplate.from_messages(
    [("system", "You explain things to a working developer. Be terse."),
     ("human", "What is {topic}?")]
)
chain = prompt | model
print(chain.invoke({"topic": "a vector embedding"}).content)


# 3. Structured output — the model returns a typed object, not a string.
class Summary(BaseModel):
    title: str = Field(description="under 8 words")
    key_points: list[str] = Field(description="exactly 3 points")


print("\n--- 3. structured output ---")
structured = model.with_structured_output(Summary)
result = structured.invoke("Summarize what a REST API is.")
print(result.title)
for p in result.key_points:
    print(" -", p)


# 4. An agent: model + tools, looping until done.
@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


print("\n--- 4. agent with a tool ---")
agent = create_agent(model=DEFAULT_MODEL, tools=[word_count])
out = agent.invoke({"messages": [{"role": "user",
                                 "content": "How many words are in 'the quick brown fox jumps'?"}]})
print(out["messages"][-1].content)
