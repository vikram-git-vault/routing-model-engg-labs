import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from services.token_metrics import extract_token_metrics, normalize_text_response
from services.warning_policy import suppress_langchain_google_warnings
from prompt_based_router import route_task


suppress_langchain_google_warnings()


load_dotenv()


MODEL = os.getenv("GENAI_MODEL", "gemini-3.6-flash")


llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1
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

    chain = prompt | llm


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