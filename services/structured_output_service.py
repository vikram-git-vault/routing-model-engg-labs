from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from config import GEMINI_MODEL, LLM_TEMPERATURE, require_gemini_key
from models.content_summary_response import SummaryResponse
from services.prompt_loader import load_prompt
from services.token_metrics import extract_token_metrics, normalize_text_response
from services.warning_policy import suppress_langchain_google_warnings


suppress_langchain_google_warnings()


# The parser generates its own format instructions, which get appended to
# the prompt so the model is told the exact JSON shape to produce.
FORMAT_INSTRUCTIONS_SUFFIX = "\n\n{format_instructions}\n"


@lru_cache(maxsize=1)
def get_parser():
    """Parser for SummaryResponse. Pure - no key, no I/O."""

    return PydanticOutputParser(
        pydantic_object=SummaryResponse
    )


@lru_cache(maxsize=1)
def get_chain():
    """Build prompt | llm on first use, then reuse.

    Nothing here runs at import time. Previously this module read a prompt
    file and constructed a chat model when imported, so importing it
    required both a valid API key and the prompt file to exist - even for
    inspection or tests that never make a call.
    """

    # Check credentials first. Everything below reads files and builds
    # objects; there is no point doing that work only to fail on the key.
    api_key = require_gemini_key()

    parser = get_parser()

    prompt = PromptTemplate(
        template=load_prompt("structured_summary.txt") + FORMAT_INSTRUCTIONS_SUFFIX,
        input_variables=["text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
    )

    return prompt | llm


def generate_structured_response(text):

    llm_response = get_chain().invoke({
        "text": text
    })

    parsed_content = normalize_text_response(llm_response.content)
    parsed = get_parser().parse(parsed_content)
    parsed.token_metrics = extract_token_metrics(llm_response)
    return parsed
