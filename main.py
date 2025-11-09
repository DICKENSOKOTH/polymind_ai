# PolyMind — Multi-Agent AI Backend (Optimized with Async)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

# Load keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("❌ Missing API keys in .env file.")

# Configure clients
genai.configure(api_key=GOOGLE_API_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # Changed to AsyncOpenAI

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


# Async function for OpenAI
async def get_openai_response(topic: str, style: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a {style} AI thinker. Be concise and insightful."},
                {"role": "user", "content": topic},
            ],
            max_tokens=800,  # Reduced for faster response
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"


# Async function for Gemini Flash
async def get_gemini_flash_response(topic: str, style: str) -> str:
    try:
        # Run in thread pool to avoid blocking
        def _generate():
            model_flash = genai.GenerativeModel("gemini-2.0-flash-exp")
            return model_flash.generate_content(
                f"{topic} (Style: {style}. Be concise.)",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.7,
                )
            ).text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate)
    except Exception as e:
        return f"⚠️ Gemini Flash error: {e}"


# Async function for Gemini Pro
async def get_gemini_pro_response(topic: str, style: str) -> str:
    try:
        # Run in thread pool to avoid blocking
        def _generate():
            model_pro = genai.GenerativeModel("gemini-2.0-flash-exp")
            return model_pro.generate_content(
                f"{topic} (Style: {style}. Be thoughtful and balanced.)",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=800,
                    temperature=0.7,
                )
            ).text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate)
    except Exception as e:
        return f"⚠️ Gemini Pro error: {e}"


@app.post("/chat")
async def chat_endpoint(data: Query, request: Request):
    topic = data.topic
    style = data.preferred_style or "neutral"
    
    # 🚀 Call all 3 agents in PARALLEL (this is the key optimization!)
    agent_alpha, agent_beta, agent_gamma = await asyncio.gather(
        get_openai_response(topic, style),
        get_gemini_flash_response(topic, style),
        get_gemini_pro_response(topic, style)
    )
    
    # === Improved Final Verdict ===
    responses = [agent_alpha, agent_beta, agent_gamma]
    
    # Filter out error responses
    valid_responses = [r for r in responses if not r.startswith("⚠️")]
    
    if not valid_responses:
        final = "All agents encountered errors. Please try again."
    else:
        # Choose the longest valid response as "best"
        best_response = max(valid_responses, key=len)
        
        # Add short previews of other agents for context
        others_summary = "\n".join(
            f"- {name}: {resp[:150]}..." 
            for name, resp in zip(["OpenAI", "Gemini Flash", "Gemini Pro"], responses) 
            if resp != best_response and not resp.startswith("⚠️")
        )
        
        if others_summary:
            final = f"{best_response}\n\n📋 Other perspectives:\n{others_summary}"
        else:
            final = best_response
    
    return {
        "agents": {
            "alpha": agent_alpha,
            "beta": agent_beta,
            "gamma": agent_gamma,
        },
        "final": final,
    }
