from pathlib import Path

from services.content_generation_pipeline import (
    run_content_pipeline,
)
from services.token_metrics import format_token_metrics


def main():

    text = Path(
        "long_text.txt"
    ).read_text(
        encoding="utf-8"
    )


    result = run_content_pipeline(
        text
    )


    print("\n==============================")
    print("CONTENT PIPELINE RESULT")
    print("==============================")


    print("\nSUMMARY:")
    print(result["summary"])


    print("\nKEYWORDS:")
    print(result["keywords"])


    print("\nHEADLINE:")
    print(result["headline"])

    print("\nTOKEN METRICS:")
    for step_name, metrics in result["token_metrics"].items():
        if step_name == "total":
            continue
        print(f"- {step_name}: {format_token_metrics(metrics)}")
    print(f"- total: {format_token_metrics(result['token_metrics']['total'])}")


if __name__ == "__main__":
    main()