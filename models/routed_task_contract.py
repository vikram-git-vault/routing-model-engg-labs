from typing import Literal

from pydantic import BaseModel


class TokenMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class TaskRequest(BaseModel):

    task: Literal[
        "summarize",
        "rewrite",
        "headline",
        "keypoints"
    ]

    text: str


class TaskResponse(BaseModel):
    """Response contract shared by every task API.

    The routing fields are optional because the three executors return
    different amounts of detail:

      basic_task_executor   -> task, result, token_metrics
      llm_task_executor     -> + provider, model
      tiered_task_executor  -> + model_tier, timed_out

    They must be declared here even so: FastAPI filters the returned dict
    against this model, so any field missing from the schema is silently
    dropped before it reaches the caller.
    """

    task: str
    result: str
    token_metrics: TokenMetrics = TokenMetrics()

    provider: str | None = None
    model: str | None = None
    model_tier: str | None = None
    timed_out: bool | None = None