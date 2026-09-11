from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import GEMINI_MODEL, LLM_TEMPERATURE, require_gemini_key
from services.token_metrics import extract_token_metrics, normalize_text_response
from services.warning_policy import suppress_langchain_google_warnings
from prompt_based_router import route_task


suppress_langchain_google_warnings()


@lru_cache(maxsize=1)
def get_llm():
    """Build the chat model on first use, then reuse it.

    Deliberately not built at import time. A module-level client means
    importing this file requires a valid API key and opens a connection,
    which makes the module impossible to import for inspection, docs or
    tests without credentials.
    """

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=require_gemini_key(),
        temperature=LLM_TEMPERATURE,
    )


def execute_task(task: str, text: str):

    # ------------------------------------------
    # 1. Ask router to select the task
    # ------------------------------------------

    route = route_task(task)


    # ------------------------------------------
    # 2. Create prompt from selected template
    # ------------------------------------------

    prompt = PromptTemplate(
        template=route["prompt"],
        input_variables=["text"]
    )


    # ------------------------------------------
    # 3. Create task-specific chain
    # ------------------------------------------

    chain = prompt | get_llm()


    # ------------------------------------------
    # 4. Execute
    # ------------------------------------------

    response = chain.invoke({
        "text": text
    })


    # ------------------------------------------
    # 5. Return result
    # ------------------------------------------

    return {
        "task": route["task"],
        "result": normalize_text_response(response.content),
        "token_metrics": extract_token_metrics(response),
    }
