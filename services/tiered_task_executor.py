from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from langchain_core.prompts import PromptTemplate

from tiered_model_router import route_task
from services.provider_model_factory import get_llm
from services.token_metrics import extract_token_metrics, normalize_text_response


def _invoke_chain(route, text):
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
        return {
            "ok": True,
            "payload": {
                "task": route["task"],
                "model_tier": route["tier"],
                "provider": route["provider"],
                "model": route["model"],
                "result": normalize_text_response(response.content),
                "token_metrics": extract_token_metrics(response),
            }
        }
    except Exception as exc:  # pragma: no cover - defensive path
        return {
            "ok": False,
            "error": str(exc),
        }


def execute_task(task: str, text: str, timeout_seconds: int = 20):
    route = route_task(task)

    # Deliberately NOT a `with` block. Executor.__exit__ calls
    # shutdown(wait=True), which blocks until the worker finishes - so
    # returning on timeout from inside `with` still waits for the full
    # call and the timeout does nothing.
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_invoke_chain, route, text)

    try:
        result = future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        # A thread cannot be killed. shutdown(wait=False) stops US waiting;
        # the HTTP request carries on in the background until it finishes
        # on its own. That is the trade for dropping fork, which could
        # terminate the work but crashed the UI on macOS.
        executor.shutdown(wait=False)
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

    executor.shutdown(wait=False)

    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Gemini request failed"))

    payload = result["payload"]
    payload["timed_out"] = False
    return payload