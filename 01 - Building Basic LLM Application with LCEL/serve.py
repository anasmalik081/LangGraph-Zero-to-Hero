from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# from langserve import add_routes
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables
load_dotenv()

# Load the Model
model = init_chat_model("llama-3.1-8b-instant", model_provider="groq")

# Prompt
system_template = "Translate the following text into {language}:"
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{text}")
])

# Output Parser
parser = StrOutputParser()

# Chain
chain = prompt | model | parser

# APP DEFINITION
# ---------------------------------------------------------------------
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A Simple API server using Langchain runnable interfaces"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class Input(BaseModel):
    language: str
    text: str

@app.post("/translate")
def translate(input: Input):
    result = chain.invoke({
        "language": input.language,
        "text": input.text
    })
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)