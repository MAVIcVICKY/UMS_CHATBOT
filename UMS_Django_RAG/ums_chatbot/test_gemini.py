import os, sys
from dotenv import load_dotenv

# Explicitly load .env from chatbot/ directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot", ".env")
print(f"Loading .env from: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")
load_dotenv(env_path)

import google.generativeai as genai

key = os.getenv('GEMINI_API_KEY')
print(f'Key found: {bool(key)}')
print(f'Key preview: {key[:15]}...' if key else 'No key')

try:
    genai.configure(api_key=key)
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    r = model.generate_content('Say hello in one word')
    print(f'SUCCESS: {r.text}')
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
