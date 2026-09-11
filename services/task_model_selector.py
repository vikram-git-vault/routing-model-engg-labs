from config import GEMINI_MODEL_SMALL, GEMINI_MODEL_LARGE


MODEL_CONFIG = {

    "small": {
        "provider": "gemini",
        "model": GEMINI_MODEL_SMALL
    },

    "large": {
        "provider": "gemini",
        "model": GEMINI_MODEL_LARGE
    }
}


TASK_MODEL_RULES = {

    "summarize": "small",

    "headline": "small",

    "keypoints": "small",

    "rewrite": "large"
}


def select_model(task: str):

    task = task.lower().strip()

    if task not in TASK_MODEL_RULES:

        raise ValueError(
            f"No model selection rule "
            f"defined for task: {task}"
        )

    tier = TASK_MODEL_RULES[task]

    config = MODEL_CONFIG[tier]

    return {
        "tier": tier,
        "provider": config["provider"],
        "model": config["model"]
    }