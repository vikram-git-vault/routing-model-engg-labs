import os
from pathlib import Path

from dotenv import load_dotenv


# ==================================================
# Paths
# ==================================================
# Every path is built from the folder this file lives in, so scripts work
# regardless of the directory they are launched from. Bare relative paths
# like Path("prompts") only resolve when the cwd happens to be the project
# root, which is a silent trap.

PROJECT_ROOT = Path(__file__).resolve().parent

PROMPT_DIR = PROJECT_ROOT / "prompts"
DATA_DIR = PROJECT_ROOT
EVAL_DIR = PROJECT_ROOT / "evaluation"

SOURCE_TEXT_FILE = DATA_DIR / "long_text.txt"
TEST_DATA_FILE = EVAL_DIR / "rewrite_test_cases.csv"
REPORT_FILE = EVAL_DIR / "prompt_evaluation_report.csv"


# ==================================================
# Environment
# ==================================================

load_dotenv(PROJECT_ROOT / ".env")


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


# Native google-genai client HTTP timeout.
#
# NOTE the unit difference: google.genai HttpOptions takes MILLISECONDS,
# while the LangChain chat models take SECONDS (see
# services/provider_model_factory.REQUEST_TIMEOUT_SECONDS). Mixing them up
# silently produces a deadline that never fires.
GENAI_TIMEOUT_MS = int(
    os.getenv("GENAI_TIMEOUT_MS", "30000")
)


# Per-attempt deadline for the LangChain chat models, in SECONDS.
#
# The Gemini API rejects any deadline below 10s outright:
#   400 INVALID_ARGUMENT "Manually set deadline 1s is too short.
#                         Minimum allowed deadline is 10s."
# so lower values are clamped rather than sent and refused.
#
# This covers ONE attempt. It does not cover backoff between retries, so
# total wall time can still exceed it - see services/tiered_task_executor.
MIN_TIMEOUT_SECONDS = 10

LLM_TIMEOUT_SECONDS = max(
    MIN_TIMEOUT_SECONDS,
    float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
)


def require_gemini_key() -> str:
    """Return the Gemini key, or raise with a message that says what to do.

    Raises rather than calling sys.exit so that importing a module never
    kills the process. Entry points can catch this and exit cleanly.
    """

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Copy .env.example to .env and set a real key."
        )

    return GEMINI_API_KEY


def require_anthropic_key() -> str:
    """Return the Anthropic key, or raise with a message that says what to do."""

    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not configured. Set a valid key or "
            "route the task back to Gemini."
        )

    return ANTHROPIC_API_KEY
