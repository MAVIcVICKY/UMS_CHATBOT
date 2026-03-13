# =====================================================
# IMPORT LIBRARIES AND PROJECT MODULES
# =====================================================

import os
from dotenv import load_dotenv
import google.generativeai as genai

from .intent import detect_intent
from .retrieval import retrieve_docs


# =====================================================
# LOAD ENVIRONMENT VARIABLES AND INITIALIZE GEMINI MODEL
# This section loads the API key and prepares the Gemini LLM.
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.0-flash")


# =====================================================
# FUNCTION: generate_answer
# PURPOSE:
# This is the main RAG pipeline function.
# It detects intent, retrieves relevant documents,
# sends context to Gemini, and returns the final answer.
# =====================================================

def generate_answer(query):

    # ─── STEP 1: INTENT DETECTION ─────────────────────────────
    # User ki query me keywords check karke category (intent) detect karta hai
    # Example: "admission process" → intent = "admission"
    intent = detect_intent(query)

    # ─── STEP 2: DOCUMENT RETRIEVAL ───────────────────────────
    # Intent ke basis pe ChromaDB se relevant document chunks retrieve karta hai
    # Pehle intent-specific collection me search hota hai,
    # agar 3 se kam results mile toh saari collections me fallback search hota hai
    try:
        docs = retrieve_docs(query, intent)
        context = "\n".join(docs)
    except Exception:
        context = ""

    # ─── LAYER 3: NO DATA FOUND ───────────────────────────────
    # Agar koi bhi relevant document nahi mila ChromaDB se,
    # toh directly "no relevant info" return kardo
    if context == "":
        return {
            "intent": intent,
            "answer": "Sorry, no relevant information found.",
            "source": "no_data"
        }

    # ─── LAYER 1: GEMINI LLM ANSWER (Best Case) ──────────────
    # Gemini available hai + context mila hai
    # → Retrieved chunks + user question ko prompt me daalke
    #   Gemini 2.0 Flash se intelligent answer generate karwata hai
    if model:

        prompt = f"""Answer using the following information only:{context}Question: {query}
Answer:
"""

        try:
            response = model.generate_content(prompt)

            return {
                "intent": intent,
                "answer": response.text,
                "source": "gemini"
            }

        except Exception:
            # Gemini API error aaya toh Layer 2 pe fall karega
            pass

    # ─── LAYER 2: RAW DATABASE CONTEXT (Fallback) ─────────────
    # Gemini available nahi hai ya API error aaya
    # → Retrieved chunks ko directly raw text me return kardo
    return {
        "intent": intent,
        "answer": context,
        "source": "database"
    }