# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .langchain_logic import explain_topic


app = FastAPI()

# Enable CORS so React (running on port 3000) can talk to Python backend (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (or restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str

@app.post("/analyze")
def analyze_topic(data: TopicRequest):
    response = explain_topic(data.topic)
    return response
