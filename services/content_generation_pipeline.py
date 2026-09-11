from functools import lru_cache

from google import genai

from config import GEMINI_MODEL, LLM_TEMPERATURE, require_gemini_key
from services.prompt_loader import load_prompt
from services.token_metrics import extract_token_metrics
from services.warning_policy import suppress_langchain_google_warnings


suppress_langchain_google_warnings()


MAX_OUTPUT_TOKENS = 400


# ==================================================
# Native Gemini Client
# ==================================================

@lru_cache(maxsize=1)
def get_client():
    """Build the native google-genai client on first use, then reuse it.

    This module previously raised ValueError at import time when the key
    was missing. api/content_pipeline_ui.py imports it at module level, so
    a missing key took down the whole UI with a bare traceback instead of
    failing at the point the pipeline was actually called.
    """

    return genai.Client(
        api_key=require_gemini_key()
    )


# ==================================================
# Native Gemini Chat Helper
# ==================================================

def generate_text(prompt: str):
    chat = get_client().chats.create(
        model=GEMINI_MODEL,
        config=genai.types.GenerateContentConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )

    response = chat.send_message(prompt)
    text = (response.text or "").strip()
    metrics = extract_token_metrics(response)
    return {
        "text": text,
        "metrics": metrics,
    }


# ==================================================
# Content Pipeline
# ==================================================

def run_content_pipeline(text):

    # ----------------------------------------------
    # Step 1: summary
    # ----------------------------------------------

    summary_prompt = load_prompt("summarize_v3.txt").format(text=text)
    summary_result = generate_text(summary_prompt)
    summary = summary_result["text"]
    summary_metrics = summary_result["metrics"]


    # ----------------------------------------------
    # Step 2: keywords
    # ----------------------------------------------

    keyword_prompt = load_prompt("extract_keywords_v2.txt").format(summary=summary)
    keyword_result = generate_text(keyword_prompt)
    keywords = keyword_result["text"]
    keyword_metrics = keyword_result["metrics"]


    # ----------------------------------------------
    # Step 3: headline
    # ----------------------------------------------

    headline_prompt = load_prompt("headline_v2.txt").format(
        summary=summary,
        keywords=keywords
    )
    headline_result = generate_text(headline_prompt)
    headline = headline_result["text"]
    headline_metrics = headline_result["metrics"]


    # ----------------------------------------------
    # Return pipeline result
    # ----------------------------------------------

    total_input = (
        summary_metrics["input_tokens"]
        + keyword_metrics["input_tokens"]
        + headline_metrics["input_tokens"]
    )
    total_output = (
        summary_metrics["output_tokens"]
        + keyword_metrics["output_tokens"]
        + headline_metrics["output_tokens"]
    )
    total_tokens = (
        summary_metrics["total_tokens"]
        + keyword_metrics["total_tokens"]
        + headline_metrics["total_tokens"]
    )

    return {
        "summary": summary,
        "keywords": keywords,
        "headline": headline,
        "token_metrics": {
            "summary": summary_metrics,
            "keywords": keyword_metrics,
            "headline": headline_metrics,
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_tokens,
            },
        },
    }