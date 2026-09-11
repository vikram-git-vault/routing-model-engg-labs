import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.1")
)