# PolyMind — Multi-Agent AI Backend
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("❌ Missing API keys in .env file.")

# Configure clients
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="PolyMind API")

# Allow all origins (frontend dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class Query(BaseModel):
    topic: str
    preferred_style: str | None = None

@app.get("/")
async def root():
    return {"message": "PolyMind backend is live ✅"}

@app.post("/chat")
async def chat_endpoint(data: Query, request: Request):
    topic = data.topic
    style = data.preferred_style or "neutral"

    # Agent 1: OpenAI GPT-5-nano (example)
    try:
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-5-nano if available in your account
            messages=[
                {"role": "system", "content": f"You are a {style} AI thinker."},
                {"role": "user", "content": topic},
            ],
        )
        agent_alpha = openai_response.choices[0].message.content
    except Exception as e:
        agent_alpha = f"⚠️ OpenAI error: {e}"

    # Agent 2: Gemini Flash
    try:
        model_flash = genai.GenerativeModel("gemini-2.5-flash")
        agent_beta = model_flash.generate_content(f"{topic} (Style: {style})").text
    except Exception as e:
        agent_beta = f"⚠️ Gemini Flash error: {e}"

    # Agent 3: Gemini Pro
    try:
        model_pro = genai.GenerativeModel("gemini-2.5-pro")
        agent_gamma = model_pro.generate_content(f"{topic} (Style: {style})").text
    except Exception as e:
        agent_gamma = f"⚠️ Gemini Pro error: {e}"

    # === Improved Final Verdict ===
    responses = [agent_alpha, agent_beta, agent_gamma]

    # Filter out error responses
    valid_responses = [r for r in responses if not r.startswith("⚠️")]
    best_response = max(valid_responses, key=len, default="No valid responses.")

    # Add short previews of other agents for context
    others_summary = "\n".join(
        f"- {name}: {resp[:150]}..." 
        for name, resp in zip(["OpenAI", "Flash", "Pro"], responses) 
        if resp != best_response
    )

    final = f"{best_response}\n\nOther insights:\n{others_summary}"

    return {
        "agents": {
            "alpha": agent_alpha,
            "beta": agent_beta,
            "gamma": agent_gamma,
        },
        "final": final,
    }
