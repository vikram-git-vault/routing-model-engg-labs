import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from models.content_summary_response import SummaryResponse
from services.token_metrics import extract_token_metrics, normalize_text_response
from services.warning_policy import suppress_langchain_google_warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


suppress_langchain_google_warnings()


load_dotenv()


# --------------------------------------------------
# 1. Create LLM
# --------------------------------------------------

MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
)


# --------------------------------------------------
# 2. Create Pydantic Output Parser
# --------------------------------------------------

parser = PydanticOutputParser(
    pydantic_object=SummaryResponse
)


# --------------------------------------------------
# 3. Load prompt
# --------------------------------------------------

prompt_template = (PROJECT_ROOT / "prompts" / "structured_summary.txt").read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# 4. Create LangChain PromptTemplate
# --------------------------------------------------

prompt = PromptTemplate(
    template=prompt_template + """

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions":
            parser.get_format_instructions()
    }
)


# --------------------------------------------------
# 5. Create chain
# --------------------------------------------------

chain = prompt | llm


# --------------------------------------------------
# 6. Generate structured response
# --------------------------------------------------

def generate_structured_response(text):

    llm_response = chain.invoke({
        "text": text
    })

    parsed_content = normalize_text_response(llm_response.content)
    parsed = parser.parse(parsed_content)
    parsed.token_metrics = extract_token_metrics(llm_response)
    return parsed