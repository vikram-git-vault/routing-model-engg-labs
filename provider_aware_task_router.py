from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
)
from services.prompt_loader import load_prompt


# Model to use when a task is redirected to the other provider. The
# configured model name belongs to the original provider, so it cannot
# travel with the fallback.
FALLBACK_MODELS = {
    "gemini": GEMINI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
}


def _resolve_provider(provider: str, model: str):
    """Fall back to the other provider when the configured one has no key.

    Returns (provider, model). The model changes with the provider - routing
    a Gemini model name to Anthropic would fail at the API.
    """

    if not GEMINI_API_KEY and not ANTHROPIC_API_KEY:
        raise ValueError(
            "No provider key is configured. Set GEMINI_API_KEY or "
            "ANTHROPIC_API_KEY in .env before routing a task."
        )

    if provider == "anthropic" and not ANTHROPIC_API_KEY:
        return "gemini", FALLBACK_MODELS["gemini"]

    if provider == "gemini" and not GEMINI_API_KEY:
        return "anthropic", FALLBACK_MODELS["anthropic"]

    return provider, model


TASK_CONFIG = {

    "summarize": {
        "prompt": "summarize_v4.txt",
        "provider": "gemini",
        "model": "gemini-3.6-flash"
    },

    "rewrite": {
        "prompt": "rewrite_v3.txt",
        "provider": "gemini",
        "model": "gemini-3.6-flash"
    },

    "headline": {
        "prompt": "headline_v3.txt",
        "provider": "gemini",
        "model": "gemini-3.6-flash"
    },

    "keypoints": {
        "prompt": "keypoints.txt",
        "provider": "gemini",
        "model": "gemini-3.6-flash"
    }
}


def route_task(task: str):

    task = task.lower().strip()

    if task not in TASK_CONFIG:
        raise ValueError(
            f"Unsupported task: {task}. "
            f"Supported tasks: {', '.join(TASK_CONFIG.keys())}"
        )

    config = TASK_CONFIG[task]

    prompt = load_prompt(
        config["prompt"]
    )

    provider, model = _resolve_provider(
        config["provider"],
        config["model"]
    )

    return {
        "task": task,
        "prompt_name": config["prompt"],
        "prompt": prompt,
        "provider": provider,
        "model": model
    }