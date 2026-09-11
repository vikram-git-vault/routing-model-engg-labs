from config import SOURCE_TEXT_FILE
from services.structured_output_service import (
    generate_structured_response
)


def main():

    # ----------------------------------------------
    # Load input text
    # ----------------------------------------------

    text_file = SOURCE_TEXT_FILE

    text = text_file.read_text(
        encoding="utf-8"
    )


    # ----------------------------------------------
    # Generate structured response
    # ----------------------------------------------

    try:

        result = generate_structured_response(
            text
        )


        # ------------------------------------------
        # Display Pydantic object
        # ------------------------------------------

        print("\n--- PYDANTIC OBJECT ---")

        print(result)


        # ------------------------------------------
        # Access fields
        # ------------------------------------------

        print("\n--- SUMMARY ---")

        print(result.summary)


        print("\n--- KEYWORDS ---")

        for keyword in result.keywords:

            print("-", keyword)


        # ------------------------------------------
        # Convert to JSON
        # ------------------------------------------

        print("\n--- JSON OUTPUT ---")

        print(
            result.model_dump_json(
                indent=2
            )
        )


    except Exception as e:

        print("\nError:")
        print(e)


if __name__ == "__main__":
    main()