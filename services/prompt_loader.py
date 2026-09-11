from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = PROJECT_ROOT / "prompts"


def load_prompt(prompt_name: str) -> str:

    prompt_file = PROMPT_DIR / prompt_name

    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_file}"
        )

    return prompt_file.read_text(
        encoding="utf-8"
    )