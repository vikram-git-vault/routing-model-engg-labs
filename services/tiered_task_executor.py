import multiprocessing as mp

from langchain_core.prompts import PromptTemplate

from tiered_model_router import route_task
from services.provider_model_factory import get_llm
from services.token_metrics import extract_token_metrics, normalize_text_response


mp_context = mp.get_context("fork")


def _invoke_chain(queue, route, text):
    try:
        prompt = PromptTemplate(
            template=route["prompt"],
            input_variables=["text"]
        )

        llm = get_llm(
            provider=route["provider"],
            model=route["model"]
        )

        chain = prompt | llm
        response = chain.invoke({"text": text})
        queue.put({
            "ok": True,
            "payload": {
                "task": route["task"],
                "model_tier": route["tier"],
                "provider": route["provider"],
                "model": route["model"],
                "result": normalize_text_response(response.content),
                "token_metrics": extract_token_metrics(response),
            }
        })
    except Exception as exc:  # pragma: no cover - defensive path
        queue.put({
            "ok": False,
            "error": str(exc),
        })


def execute_task(task: str, text: str, timeout_seconds: int = 20):
    route = route_task(task)
    queue = mp_context.Queue()
    process = mp_context.Process(
        target=_invoke_chain,
        args=(queue, route, text),
        daemon=True,
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "task": route["task"],
            "model_tier": route["tier"],
            "provider": route["provider"],
            "model": route["model"],
            "result": "The request timed out while waiting for the Gemini model response.",
            "token_metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "timed_out": True,
        }

    if queue.empty():
        return {
            "task": route["task"],
            "model_tier": route["tier"],
            "provider": route["provider"],
            "model": route["model"],
            "result": "The Gemini request did not return a result.",
            "token_metrics": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            "timed_out": False,
        }

    result = queue.get()
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Gemini request failed"))

    payload = result["payload"]
    payload["timed_out"] = False
    return payload