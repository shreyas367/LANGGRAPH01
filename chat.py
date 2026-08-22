# from langchain_groq import ChatGroq

# import os
# from dotenv import load_dotenv

# load_dotenv()

# llmgroq=ChatGroq(
# model = "openai/gpt-oss-120b",    temperature=0.3,
#     api_key=os.getenv("GROQ_API_KEY")
# )


# response=llmgroq.invoke("where is capital of india")
# print(response.content.strip())



import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

response = llm.invoke("Explain RAG in simple terms.")

print(response.content)