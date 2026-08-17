from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv
import json
load_dotenv(find_dotenv(".env.dev"))


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature="1.5")


response = llm.invoke("Write me a ballad about LangChain")

print(json.dumps(response.model_dump(), indent=2, default=str))