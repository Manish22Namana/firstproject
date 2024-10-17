from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from langchain.chains import LLMChain
import os

from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_AI_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

app = FastAPI(
    title="Langchain LLM Summarize API",
    version="1.0",
    description="A Multi-LLM Summarize API with Langchain"
)

# Define Prompts
prompt2 = ChatPromptTemplate.from_template("Summarize the text {input} in {input1} words. Just give me the summarized text without any explanation and without changing the sense")



# Create API Routes with Langchain Pipelines

add_routes(app, prompt2 | llm, path="/summarize")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
