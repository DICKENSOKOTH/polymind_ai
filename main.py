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
            max_tokens=600,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI error: {str(e)}"


# Async function for Gemini Flash
async def get_gemini_flash_response(topic: str, style: str) -> str:
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            def _generate():
                # Create model with safety settings disabled
                model_flash = genai.GenerativeModel(
                    "gemini-2.0-flash-exp",
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                # Try different prompt formulations on retries
                if attempt == 0:
                    prompt = f"{topic} (Style: {style}. Be concise.)"
                elif attempt == 1:
                    prompt = f"Please provide information about: {topic}. Response style: {style}"
                else:
                    prompt = f"Explain the following topic in a {style} way: {topic}"
                
                response = model_flash.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=600,
                        temperature=0.7,
                    )
                )
                
                # Check if response has valid content
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                        return response.text
                
                # If no content, raise exception to trigger retry
                raise Exception("No valid content in response")
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _generate)
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error
            if "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return "⚠️ Gemini Flash: Rate limit reached. Try again in a few moments."
            
            # For other errors, retry with different prompt
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
                
            return f"⚠️ Gemini Flash: Unable to generate response after {max_retries} attempts. The content may have triggered safety filters."
    
    return "⚠️ Gemini Flash: Max retries reached."


# Async function for Gemini Pro
async def get_gemini_pro_response(topic: str, style: str) -> str:
    max_retries = 2  # Reduced retries since we're trying a more stable model
    
    for attempt in range(max_retries):
        try:
            def _generate():
                # Use gemini-1.5-pro or gemini-1.5-flash (more stable than experimental)
                model_pro = genai.GenerativeModel(
                    "gemini-2.5-pro",  # Changed to stable Pro model
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                
                # Simpler prompt - less likely to trigger filters
                prompt = f"Provide a thoughtful response about: {topic}"
                
                response = model_pro.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=600,
                        temperature=0.7,
                    )
                )
                
                # Check if response has valid content
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts and len(candidate.content.parts) > 0:
                        return response.text
                
                # If no content, raise exception to trigger retry
                raise Exception("No valid content in response")
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _generate)
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error
            if "429" in error_msg or "quota" in error_msg.lower() or "exhausted" in error_msg.lower():
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return "⚠️ Gemini Pro: Rate limit reached. Try again in a few moments."
            
            # For other errors, retry once
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
                
            return f"⚠️ Gemini Pro temporarily unavailable. Using other agents' responses."
    
    return "⚠️ Gemini Pro: Unable to respond at this time."


# 🧠 Smart Final Verdict Generator
async def generate_final_verdict(topic: str, agent_alpha: str, agent_beta: str, agent_gamma: str, style: str) -> str:
    """
    Uses GPT-4o-mini to analyze all three agent responses and create a synthesized final verdict.
    Gives full reasoning and comprehensive synthesis.
    """
    try:
        judge_prompt = f"""You are an expert AI judge analyzing multiple AI responses to synthesize the best answer.

**Original Question:** {topic}

**Agent Alpha (OpenAI) Response:**
{agent_alpha}

**Agent Beta (Gemini Flash) Response:**
{agent_beta}

**Agent Gamma (Gemini Pro) Response:**
{agent_gamma}

**Your Task:**
1. Compare and contrast the three responses
2. Identify which agent(s) provided the most accurate, helpful, or insightful answer
3. Explain the strengths and weaknesses you noticed
4. Synthesize the best elements from all three into one comprehensive answer
5. Make your reasoning clear - explain WHY certain approaches worked better

Provide your detailed analysis and final verdict:"""

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI judge that analyzes multiple perspectives and synthesizes them into clear, comprehensive insights with detailed reasoning."},
                {"role": "user", "content": judge_prompt},
            ],
            max_tokens=1200,  # Increased from 400 for fuller responses
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to simple logic if API fails
        valid_responses = [r for r in [agent_alpha, agent_beta, agent_gamma] if not r.startswith("⚠️")]
        if valid_responses:
            return max(valid_responses, key=len)
        return "⚠️ All agents encountered errors."


@app.post("/chat")
async def chat_endpoint(data: Query, request: Request):
    topic = data.topic
    style = data.preferred_style or "neutral"
    
    # 🚀 Call ALL 4 agents in PARALLEL (including the judge!)
    agent_alpha_task = get_openai_response(topic, style)
    agent_beta_task = get_gemini_flash_response(topic, style)
    agent_gamma_task = get_gemini_pro_response(topic, style)
    
    # Start all 3 agents first
    agent_alpha, agent_beta, agent_gamma = await asyncio.gather(
        agent_alpha_task,
        agent_beta_task,
        agent_gamma_task
    )
    
    # Generate final verdict (happens quickly after agents finish)
    final_verdict = await generate_final_verdict(topic, agent_alpha, agent_beta, agent_gamma, style)
    
    return {
        "agents": {
            "alpha": agent_alpha,
            "beta": agent_beta,
            "gamma": agent_gamma,
        },
        "final": final_verdict,
    }
