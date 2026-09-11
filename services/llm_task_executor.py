from langchain_core.prompts import PromptTemplate

from provider_aware_task_router import route_task
from services.provider_model_factory import get_llm
from services.token_metrics import extract_token_metrics, normalize_text_response


def execute_task(task: str, text: str):

    # -----------------------------------------
    # 1. Router selects task configuration
    # -----------------------------------------

    route = route_task(task)


    # -----------------------------------------
    # 2. Create prompt from selected template
    # -----------------------------------------

    prompt = PromptTemplate(
        template=route["prompt"],
        input_variables=["text"]
    )


    # -----------------------------------------
    # 3. Create LLM using provider abstraction
    # -----------------------------------------

    llm = get_llm(
        provider=route["provider"],
        model=route["model"]
    )


    # -----------------------------------------
    # 4. Build chain
    # -----------------------------------------

    chain = prompt | llm


    # -----------------------------------------
    # 5. Execute
    # -----------------------------------------

    try:
        response = chain.invoke({
            "text": text
        })
    except Exception as exc:
        provider = route.get("provider", "unknown")
        model = route.get("model", "unknown")
        message = str(exc)
        if "401" in message or "authentication" in message.lower() or "api key" in message.lower():
            raise RuntimeError(
                f"Authentication failed for provider '{provider}' model '{model}'. Verify the API key is valid."
            ) from exc
        raise RuntimeError(
            f"Provider-aware executor failed for '{provider}' model '{model}': {message}"
        ) from exc


    # -----------------------------------------
    # 6. Return result
    # -----------------------------------------

    return {
        "task": route["task"],
        "provider": route["provider"],
        "model": route["model"],
        "result": normalize_text_response(response.content),
        "token_metrics": extract_token_metrics(response),
    }