from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    LLM_TEMPERATURE
)
from services.warning_policy import suppress_langchain_google_warnings


suppress_langchain_google_warnings()


def get_llm(
    provider: str,
    model: str
):

    provider = provider.lower().strip()

    if provider == "gemini":

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured"
            )

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=120,
            timeout=20000,
            max_retries=1,
        )


    if provider == "anthropic":

        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not configured. Set a valid key or switch the task back to Gemini."
            )

        try:
            return ChatAnthropic(
                model=model,
                api_key=ANTHROPIC_API_KEY,
                temperature=LLM_TEMPERATURE,
                timeout=20000,
                max_retries=1,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(
                "Anthropic client initialization failed. Check the ANTHROPIC_API_KEY value and model name."
            ) from exc


    raise ValueError(
        f"Unsupported provider: {provider}"
    )