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

    task: str
    result: str
    token_metrics: TokenMetrics = TokenMetrics()