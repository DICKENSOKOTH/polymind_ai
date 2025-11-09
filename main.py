# PolyMind — Multi-Agent AI Backend (Ultra-Fast with Smart Final Verdict)
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
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

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
            max_tokens=600,  # Reduced further for speed
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"


# Async function for Gemini Flash
async def get_gemini_flash_response(topic: str, style: str) -> str:
    try:
        def _generate():
            model_flash = genai.GenerativeModel("gemini-2.5-flash")
            return model_flash.generate_content(
                f"{topic} (Style: {style}. Be concise.)",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=600,
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
        def _generate():
            model_pro = genai.GenerativeModel("gemini-2.5-pro")
            return model_pro.generate_content(
                f"{topic} (Style: {style}. Be thoughtful and balanced.)",
                generation_config=genai.GenerationConfig(
                    max_output_tokens=600,
                    temperature=0.7,
                )
            ).text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate)
    except Exception as e:
        return f"⚠️ Gemini Pro error: {e}"


# 🧠 Smart Final Verdict Generator - Uses AI to synthesize all responses
async def generate_final_verdict(topic: str, agent_alpha: str, agent_beta: str, agent_gamma: str, style: str) -> str:
    """
    Uses GPT-4o-mini to analyze all three agent responses and create a synthesized final verdict.
    This is much faster than using a heavy model and produces intelligent comparisons.
    """
    try:
        # Create a prompt for the AI judge
        judge_prompt = f"""You are an expert AI judge analyzing multiple AI responses to synthesize the best answer.

**Original Question:** {topic}

**Agent Alpha (OpenAI) Response:**
{agent_alpha}

**Agent Beta (Gemini Flash) Response:**
{agent_beta}

**Agent Gamma (Gemini Pro) Response:**
{agent_gamma}

**Your Task:**
1. Analyze the strengths and weaknesses of each response
2. Identify which agent(s) provided the most accurate, helpful, or comprehensive answer
3. Synthesize the best elements from all three responses into one superior answer
4. Be concise but comprehensive
5. Mention which agent had the best approach if relevant

Provide your final verdict below:"""

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and smart enough for synthesis
            messages=[
                {"role": "system", "content": "You are an expert AI judge that synthesizes multiple perspectives into clear, actionable insights."},
                {"role": "user", "content": judge_prompt},
            ],
            max_tokens=800,  # Slightly longer for synthesis
            temperature=0.5,  # Lower temp for more consistent judgments
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to simple logic if API fails
        valid_responses = [r for r in [agent_alpha, agent_beta, agent_gamma] if not r.startswith("⚠️")]
        if valid_responses:
            return f"⚠️ Could not generate AI synthesis. Here's the longest response:\n\n{max(valid_responses, key=len)}"
        return "⚠️ All agents encountered errors."


@app.post("/chat")
async def chat_endpoint(data: Query, request: Request):
    topic = data.topic
    style = data.preferred_style or "neutral"
    
    # 🚀 Call all 3 agents in PARALLEL
    agent_alpha, agent_beta, agent_gamma = await asyncio.gather(
        get_openai_response(topic, style),
        get_gemini_flash_response(topic, style),
        get_gemini_pro_response(topic, style)
    )
    
    # ⚡ Generate smart final verdict in parallel with returning agent responses
    # This happens while the frontend is already displaying agent cards!
    final_verdict = await generate_final_verdict(topic, agent_alpha, agent_beta, agent_gamma, style)
    
    return {
        "agents": {
            "alpha": agent_alpha,
            "beta": agent_beta,
            "gamma": agent_gamma,
        },
        "final": final_verdict,
    }
