from fastapi import FastAPI, HTTPException

from models.routed_task_contract import (TaskRequest, TaskResponse,)
from services.llm_task_executor import execute_task


app = FastAPI(
    title="Provider-Aware Task API",
    version="1.0.0",
    description=(
        "Prompt routing plus provider and model selection per task, via "
        "provider_aware_task_router. Falls back to the other provider when "
        "a key is missing. Also returns provider and model."
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