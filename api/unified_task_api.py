from fastapi import FastAPI, HTTPException

from models.routed_task_contract import (TaskRequest, TaskResponse,)
from services.tiered_task_executor import execute_task


app = FastAPI(
    title="Tiered Task API",
    version="1.0.0",
    description=(
        "Prompt routing plus tier-based model selection, via "
        "tiered_model_router. Small tasks take the cheap model, rewrite "
        "takes the larger one. Adds a wall-clock timeout; returns "
        "model_tier and timed_out alongside provider and model."
    ),
)


@app.post("/task", response_model=TaskResponse)
def task_endpoint(
    request: TaskRequest
):

    try:

        result = execute_task(
            task=request.task,
            text=request.text
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )