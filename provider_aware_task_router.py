from config import GEMINI_API_KEY, ANTHROPIC_API_KEY
from services.prompt_loader import load_prompt


def _resolve_provider(provider: str) -> str:
    if provider == "anthropic" and not ANTHROPIC_API_KEY:
        return "gemini"
    if provider == "gemini" and not GEMINI_API_KEY:
        return "anthropic"
    return provider


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

    return {
        "task": task,
        "prompt_name": config["prompt"],
        "prompt": prompt,
        "provider": config["provider"],
        "model": config["model"]
    }