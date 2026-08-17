from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(".env.dev"))


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)


