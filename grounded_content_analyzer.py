import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from services.token_metrics import extract_token_metrics, format_token_metrics

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# --------------------------------------------------
# 2. Initialize Gemini client
# --------------------------------------------------

client = genai.Client(api_key=api_key)

MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")

# --------------------------------------------------
# 3. File locations
# --------------------------------------------------

PROMPT_DIR = Path("prompts")
TEXT_FILE = Path("long_text.txt")


# --------------------------------------------------
# 4. Load input text
# --------------------------------------------------


def load_text(file_path):
    """Read text from a file."""

    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    return file_path.read_text(encoding="utf-8").strip()


# --------------------------------------------------
# 5. Load prompt template
# --------------------------------------------------


def load_prompt(prompt_name, **variables):
    """
    Load a prompt template from the prompts/ directory
    and replace template variables.
    """

    prompt_file = PROMPT_DIR / prompt_name

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    template = prompt_file.read_text(encoding="utf-8")

    return template.format(**variables)

# --------------------------------------------------
# 6. Call LLM
# --------------------------------------------------


def call_llm(
    prompt,
    system_instruction="You are a helpful assistant.",
    response_mime_type=None,
):
    try:
        chat = client.chats.create(
            model=MODEL,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5,
                max_output_tokens=200,
                response_mime_type=response_mime_type,
            ),
        )

        response = chat.send_message(prompt)
        metrics = extract_token_metrics(response)

        print("\n--- TOKEN USAGE ---")
        print(format_token_metrics(metrics))

        return response

    except errors.ClientError as e:
        if e.code in (401, 403):
            print("Authentication Error: Please check your GEMINI_API_KEY.")
        else:
            print("API Error:", str(e))
        sys.exit(1)

    except errors.APIError as e:
        print("API Error:", str(e))
        sys.exit(1)

    except Exception as e:
        print("An error occurred:", str(e))
        sys.exit(1)


def check_hallucination(source_text, generated_text, output_type):
    """Check whether generated content is supported by the source text."""

    evaluation_prompt = f"""
You are a strict factual-grounding evaluator.

Compare the generated {output_type} with the source text. Identify claims in
the generated content that are not supported by, or contradict, the source.
Do not penalize paraphrasing, concise wording, or reasonable keywords.

Return only valid JSON in this exact shape:
{{
  "grounded": true,
  "unsupported_claims": [],
  "reason": "brief explanation"
}}

Source text:
{source_text}

Generated {output_type}:
{generated_text}
"""

    response = call_llm(
        evaluation_prompt,
        system_instruction="Return only the requested JSON object.",
        response_mime_type="application/json",
    )

    response_text = (response.text or "").strip()
    fenced_json = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        response_text,
        re.DOTALL | re.IGNORECASE
    )

    if fenced_json:
        response_text = fenced_json.group(1)

    try:
        result = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        object_start = response_text.find("{")
        object_end = response_text.rfind("}")

        if object_start == -1 or object_end <= object_start:
            print("Hallucination Check: unable to parse evaluator response.")
            return

        try:
            result = json.loads(
                response_text[object_start : object_end + 1]
            )
        except (json.JSONDecodeError, TypeError):
            print("Hallucination Check: unable to parse evaluator response.")
            return

    if not isinstance(result, dict):
        print("Hallucination Check: invalid evaluator response shape.")
        return

    grounded = result.get("grounded")
    unsupported_claims = result.get("unsupported_claims", [])
    reason = result.get("reason", "No explanation provided.")

    if not isinstance(grounded, bool) or not isinstance(unsupported_claims, list):
        print("Hallucination Check: invalid evaluator response fields.")
        return

    if grounded is True and not unsupported_claims:
        print("Hallucination Check: PASS")
    else:
        print("Hallucination Check: REVIEW")
        print("Unsupported claims:", unsupported_claims)

    print("Reason:", reason)


# --------------------------------------------------
# 7. Main application
# --------------------------------------------------


try:
    # Load input text from external file
    text = load_text(TEXT_FILE)

    print(f"Loaded text from: {TEXT_FILE}")
    print(f"Characters: {len(text)}")

    # --------------------------------------------------
    # Summarization
    # --------------------------------------------------

    summarize_prompt = load_prompt(
        "summarize_v1.txt",
        text=text
    )

    print("\n--- SUMMARIZATION PROMPT ---")
    print(summarize_prompt)

    response = call_llm(summarize_prompt)

    print("\n--- SUMMARY ---")
    print(response.text)
    check_hallucination(text, response.text, "summary")

    # --------------------------------------------------
    # Keyword extraction
    # --------------------------------------------------

    keyword_prompt = load_prompt(
        "extract_keywords.txt",
        text=text
    )

    print("\n--- KEYWORD PROMPT ---")
    print(keyword_prompt)

    response = call_llm(keyword_prompt)

    print("\n--- KEYWORDS ---")
    print(response.text)
    check_hallucination(text, response.text, "keyword list")

    # --------------------------------------------------
    # Headline generation
    # --------------------------------------------------

    headline_prompt = load_prompt(
        "headline.txt",
        text=text
    )

    print("\n--- HEADLINE PROMPT ---")
    print(headline_prompt)

    response = call_llm(headline_prompt)

    print("\n--- HEADLINE ---")
    print(response.text)
    check_hallucination(text, response.text, "headline")

except FileNotFoundError as e:
    print(f"File Error: {e}")
    sys.exit(1)

