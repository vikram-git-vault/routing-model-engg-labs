import sys

from google import genai
from google.genai import errors

from config import GEMINI_MODEL, require_gemini_key
from services.token_metrics import extract_token_metrics, format_token_metrics


# require_gemini_key raises so that library callers are not killed by a
# sys.exit. This is a script, so catch it and exit cleanly.
try:
    api_key = require_gemini_key()
except ValueError as exc:
    print(f"Error: {exc}")
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL = GEMINI_MODEL

print("Sending prompt: ")

try:
    chat = client.chats.create(
        model=MODEL,
        config=genai.types.GenerateContentConfig(
            system_instruction="You are a helpful assistant.",
            temperature=0.5,
            max_output_tokens=100,
        ),
    )
    response = chat.send_message("Can you summarise, extract keywords and give a headline to following text: Generative AI is a branch of artificial intelligence that can create new content such as text, images, audio, video, and computer code. It uses large machine learning models trained on vast amounts of data to understand patterns and generate human-like outputs. Tools such as ChatGPT, Gemini, and Claude demonstrate how generative AI can support writing, research, programming, education, and creative work. However, generated content may contain errors, bias, or unsupported claims, so human review remains important. Organizations are increasingly using generative AI to automate tasks, improve productivity, personalize experiences, and build intelligent applications. Responsible use requires attention to privacy, security, accuracy, and ethics.")
    metrics = extract_token_metrics(response)
    print("Response:", response.text)
    print("Token usage:", format_token_metrics(metrics))

except errors.ClientError as e:
    if e.code in (401, 403):
        print("Authentication Error: Please check your GEMINI_API_KEY.")
    else:
        print("API Error:", str(e))

except errors.APIError as e:
    print("API Error:", str(e))

except Exception as e:
    print("An error occurred:", str(e))
