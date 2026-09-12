from fastapi import FastAPI, HTTPException

from models.task_request import TaskRequest
from basic_task_router import route_task


app = FastAPI(
    title="Legacy Task Router",
    version="1.0.0",
    description=(
        "The earliest version: returns the routing decision without "
        "calling a model. Useful for inspecting a router in isolation."
    ),
)


@app.post("/task")
def task_router(
    request: TaskRequest
):

    try:

        result = route_task(
            request.task
        )

        return result

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )