from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import find_dotenv, load_dotenv
import json
load_dotenv(find_dotenv(".env.dev"))


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B:featherless-ai",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=1024,
)

model = ChatHuggingFace(llm= llm)



response = model.invoke("Write me a ballad about LangChain")

print(json.dumps(response.model_dump(), indent=2, default=str))