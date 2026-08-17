from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import json
import os

# Where downloaded weights land. Must be set before the model loads.
# expanduser keeps this cross-platform:
#   macOS/Linux -> /Users/<you>/.cache/huggingface
#   Windows     -> C:\Users\<you>\.cache\huggingface
# On Windows, point it at another drive if C: is short on space, e.g.
#   os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# no dotenv needed — the model runs locally, so there is no API token

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=1024,
    ),
)

model = ChatHuggingFace(llm=llm)



response = model.invoke("Write me a ballad about LangChain")

print(json.dumps(response.model_dump(), indent=2, default=str))