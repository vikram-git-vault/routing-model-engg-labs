import os
from dotenv import load_dotenv

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5"
)

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.1")
)

# Tiered model selection.
#
# The point of the tiered router is that tiers resolve to DIFFERENT models.
# If both resolve to the same name the routing is a no-op.
#
# Gemini free-tier quota is counted per model, so spreading tasks across
# two models also spreads the daily request budget.
GEMINI_MODEL_SMALL = os.getenv(
    "GENAI_MODEL_SMALL",
    "gemini-3.5-flash-lite"
)

GEMINI_MODEL_LARGE = os.getenv(
    "GENAI_MODEL_LARGE",
    "gemini-3.6-flash"
)
