from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import requests
import json
import os
from datetime import datetime

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

class PromptRequest(BaseModel):
    prompt: str
    mode: str
    language: str = "Python"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_connection():
    conn = sqlite3.connect("oshimiri.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        mode TEXT NOT NULL,
        language TEXT NOT NULL,
        prompt TEXT NOT NULL,
        response TEXT NOT NULL
    );""")
    conn.commit()
    return conn

def save_conversation(conn, mode, language, prompt, response):
    conn.execute(
        "INSERT INTO conversations (timestamp, mode, language, prompt, response) VALUES (?, ?, ?, ?, ?);",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode, language, prompt, response)
    )
    conn.commit()

def build_messages(prompt, mode, language):
    if mode == "generate":
        system = f"""You are Oshi, an expert software engineer and coding assistant for Oshimiri. 
Generate clean, well-commented, production-ready {language} code.
If the request is ambiguous or could be interpreted in multiple ways, ask for clarification before generating code.
Always explain what the code does briefly after the code block.
If the user mentions a specific library or framework, use that exact library with its real API - never invent fictional implementations."""
        user = f"Write {language} code for the following: {prompt}"
    else:
        system = f"""You are Oshi, an expert debugger and coding assistant for Oshimiri.
Analyze the provided code, identify all bugs and issues, explain what's wrong clearly, then provide the fixed code with comments explaining each fix.
If the code is unclear or you need more context, ask for clarification first."""
        user = f"Debug this {language} code:\n\n{prompt}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

@app.post("/oshimiri/")
async def handle_prompt(request: PromptRequest):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": build_messages(request.prompt, request.mode, request.language),
        "max_tokens": 2048,
        "temperature": 0.2
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    data = response.json()

    if response.status_code != 200:
        return {"error": data.get("error", {}).get("message", "Unknown error")}

    result = data["choices"][0]["message"]["content"]

    conn = create_connection()
    save_conversation(conn, request.mode, request.language, request.prompt, result)
    conn.close()

    return {
        "response": result,
        "usage": data.get("usage", {})
    }

@app.get("/conversations/")
async def get_conversations(limit: int = 50):
    conn = create_connection()
    cursor = conn.execute(
        "SELECT id, timestamp, mode, language, prompt, response FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {"conversations": [
        {"id": r[0], "timestamp": r[1], "mode": r[2], "language": r[3], "prompt": r[4], "response": r[5]}
        for r in rows
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
