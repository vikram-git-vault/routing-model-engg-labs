from fastapi import FastAPI, HTTPException

from models.routed_task_contract import (TaskRequest, TaskResponse,)
from services.basic_task_executor import execute_task


app = FastAPI(
    title="Basic Task API",
    version="1.0.0",
    description=(
        "Prompt routing only. prompt_based_router picks a template per "
        "task; every task runs on the same Gemini model. Returns task, "
        "result and token metrics."
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