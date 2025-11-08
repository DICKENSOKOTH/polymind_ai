# 🧠 PolyMind — Multi-Agent AI Backend

**PolyMind** is a multi-agent reasoning API that blends different AI models (OpenAI GPT + Google Gemini) to generate diverse perspectives and a unified final verdict.  
It’s designed for developers, researchers, and engineers who want to experiment with **AI collaboration, model comparison, or meta-reasoning** across systems.

---

## 🚀 Features

- 🤖 **3-Agent Thinking:** OpenAI GPT + Gemini Flash + Gemini Pro  
- 🔗 **Unified Verdict:** Synthesizes all responses into one balanced answer  
- ⚡ **FastAPI Backend:** Lightweight, async, production-ready  
- 🔒 **Secure Secrets:** Keeps API keys private using Google Cloud Secret Manager  
- ☁️ **Cloud Run-Ready:** Containerized and deployable with one command  
- 🧩 **Frontend-Friendly:** Works with any HTML/JS frontend (e.g. index.html included)

---

## 🧩 Project Structure

polymind/
├── main.py               # FastAPI multi-agent backend (loads keys from .env)
├── index.html            # Frontend UI (3 columns + vote + final verdict)
├── requirements.txt      # Python deps
├── Dockerfile            # For Cloud Run / container deploy
├── .gitignore            # Ignore .env + local files
├── README.md             # (this file)
└── .env                  # LOCAL ONLY (not committed) with your API keys
