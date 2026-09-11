import os
import csv
import random
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from services.token_metrics import extract_token_metrics, format_token_metrics


# ============================================================
# 1. Configuration
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)


MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")
REQUEST_TIMEOUT_MS = int(os.getenv("GENAI_TIMEOUT_MS", "30000"))

client = genai.Client(
    api_key=api_key,
    http_options=genai.types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=genai.types.HttpRetryOptions(attempts=1),
    ),
)

from config import PROMPT_DIR, TEST_DATA_FILE, REPORT_FILE  # noqa: E402
SAMPLE_SIZE = 5


# ============================================================
# 2. File utilities
# ============================================================

def load_prompt(prompt_name, **variables):
    """
    Load a prompt template and replace variables.
    """

    prompt_file = PROMPT_DIR / prompt_name

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}"
        )

    template = prompt_file.read_text(encoding="utf-8")

    return template.format(**variables)


def load_test_cases():
    """
    Load rewrite test cases from CSV.
    """

    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Test data file not found: {TEST_DATA_FILE}"
        )

    with open(
        TEST_DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return list(csv.DictReader(file))


# ============================================================
# 3. LLM call
# ============================================================

def call_llm(prompt):

    try:
        request_started = time.perf_counter()
        print(
            "  Waiting for Gemini response...",
            flush=True,
        )

        chat = client.chats.create(
            model=MODEL,
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a professional business writing assistant.",
                temperature=0.5,
                max_output_tokens=200,
            ),
        )

        response = chat.send_message(prompt)
        elapsed = time.perf_counter() - request_started
        print(
            f"  Gemini response received in {elapsed:.2f} seconds.",
            flush=True,
        )

        metrics = extract_token_metrics(response)
        print(
            "  Token usage: "
            f"{format_token_metrics(metrics)}",
            flush=True,
        )

        return response

    except errors.ClientError as e:
        if e.code in (401, 403):
            print("Authentication Error: Please check your GEMINI_API_KEY.")
        elif e.code == 429:
            print(
                "Quota exceeded (429). Wait for the retry period or check "
                "your Gemini plan and billing details."
            )
        else:
            print("API Error:", str(e))
        sys.exit(1)

    except errors.APIError as e:
        print("API Error:", str(e))
        sys.exit(1)

    except Exception as e:
        print("LLM Error:", str(e))
        sys.exit(1)


# ============================================================
# 4. Run experiment
# ============================================================

def run_experiment(prompt_file, test_cases):

    results = []
    selected_test_cases = random.sample(
        test_cases,
        min(SAMPLE_SIZE, len(test_cases))
    )

    print("\n")
    print("=" * 70)
    print(f"Running experiment: {prompt_file}")
    print("=" * 70)
    print(
        "Selected test cases:",
        ", ".join(test_case["id"] for test_case in selected_test_cases)
    )

    for test_case in selected_test_cases:

        test_id = test_case["id"]
        input_text = test_case["input_text"]

        print(
            f"\nStarting test case {test_id} "
            f"({len(results) + 1}/{len(selected_test_cases)})...",
            flush=True,
        )

        prompt_started = time.perf_counter()
        print("  Preparing prompt...", flush=True)
        prompt = load_prompt(
            prompt_file,
            text=input_text
        )
        prompt_elapsed = time.perf_counter() - prompt_started
        print(
            f"  Prompt ready in {prompt_elapsed:.2f} seconds.",
            flush=True,
        )

        response = call_llm(prompt)

        output = response.text.strip()

        results.append({
            "id": test_id,
            "input": input_text,
            "output": output,
            "prompt": prompt_file
        })

        print(f"\nTest Case {test_id}")
        print("Input :", input_text)
        print("Output:", output)

    return results


# ============================================================
# 5. Simple formatting evaluation
# ============================================================

def evaluate_formatting(output):
    """
    Basic automated formatting checks.

    Returns 1 if the output passes all checks,
    otherwise 0.
    """

    score = 1

    # Output should not be empty
    if not output.strip():
        score = 0

    # Should be reasonably short
    if len(output.split()) > 50:
        score = 0

    # Should not contain obvious meta text
    unwanted_phrases = [
        "here is the rewritten sentence",
        "sure",
        "certainly",
        "the rewritten sentence is"
    ]

    output_lower = output.lower()

    for phrase in unwanted_phrases:
        if phrase in output_lower:
            score = 0

    return score


# ============================================================
# 6. Save evaluation report
# ============================================================

def save_report(results):

    report_started = time.perf_counter()
    print("Saving evaluation report...", flush=True)

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "id",
            "prompt",
            "input",
            "output",
            "formatting_score"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            result["formatting_score"] = evaluate_formatting(
                result["output"]
            )

            writer.writerow(result)

    report_elapsed = time.perf_counter() - report_started
    print(
        f"\nEvaluation report saved to: {REPORT_FILE} "
        f"in {report_elapsed:.2f} seconds.",
        flush=True,
    )


# ============================================================
# 7. Main
# ============================================================

def main():

    load_started = time.perf_counter()
    print("Loading test cases...", flush=True)
    test_cases = load_test_cases()
    load_elapsed = time.perf_counter() - load_started

    print(
        f"Loaded {len(test_cases)} test cases "
        f"in {load_elapsed:.2f} seconds.",
        flush=True,
    )

    # --------------------------------------------------------
    # Zero-shot experiment
    # --------------------------------------------------------

    zero_shot_results = run_experiment(
        "rewrite_zero_shot.txt",
        test_cases
    )

    # --------------------------------------------------------
    # Few-shot experiment
    # --------------------------------------------------------

    few_shot_results = run_experiment(
        "rewrite_few_shot.txt",
        test_cases
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    all_results = (
        zero_shot_results +
        few_shot_results
    )

    save_report(all_results)


if __name__ == "__main__":
    main()