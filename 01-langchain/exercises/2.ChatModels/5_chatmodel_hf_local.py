from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import json

# no dotenv needed — the model runs locally, so there is no API token

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=1024,
    ),
)

model = ChatHuggingFace(llm=llm)



response = model.invoke("Write me a ballad about LangChain")

print(json.dumps(response.model_dump(), indent=2, default=str))