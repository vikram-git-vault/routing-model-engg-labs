import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from services.token_metrics import extract_token_metrics


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

if not api_key:
    raise ValueError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=api_key)


# ==================================================
# Prompt Loader
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_prompt(prompt_name):

    prompt_file = PROJECT_ROOT / "prompts" / prompt_name

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_file}"
        )

    return prompt_file.read_text(
        encoding="utf-8"
    )


# ==================================================
# Token Metrics
# ==================================================

# ==================================================
# Native Gemini Chat Helper
# ==================================================

def generate_text(prompt: str):
    chat = client.chats.create(
        model=MODEL,
        config=genai.types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=400,
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