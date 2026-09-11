def extract_token_metrics(response):
    """Normalize Gemini/LangChain usage metadata to a consistent token dictionary."""

    def as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def find_usage_dict(data):
        if not isinstance(data, dict):
            return {}

        for key in (
            "usage_metadata",
            "token_usage",
            "usage",
            "metadata",
        ):
            value = data.get(key)
            if isinstance(value, dict):
                return value

        for nested_key in ("response_metadata", "metadata"):
            nested = data.get(nested_key)
            if isinstance(nested, dict):
                usage = find_usage_dict(nested)
                if usage:
                    return usage

        return {}

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        raw_response_metadata = getattr(response, "response_metadata", None)
        if isinstance(raw_response_metadata, dict):
            usage = find_usage_dict(raw_response_metadata)

    if usage is None and isinstance(response, dict):
        usage = find_usage_dict(response)

    if usage is None:
        usage = {}

    if not isinstance(usage, dict):
        usage = dict(usage)

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("prompt_token_count")
        or usage.get("input_token_count")
        or 0
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("completion_token_count")
        or usage.get("output_token_count")
        or usage.get("candidates_token_count")
        or 0
    )
    total_tokens = (
        usage.get("total_tokens")
        or usage.get("total_token_count")
        or 0
    )

    if total_tokens == 0:
        total_tokens = as_int(input_tokens) + as_int(output_tokens)

    return {
        "input_tokens": as_int(input_tokens),
        "output_tokens": as_int(output_tokens),
        "total_tokens": as_int(total_tokens),
    }


def normalize_text_response(response):
    """Convert Gemini/LangChain response payloads into plain string text."""

    if response is None:
        return ""

    if isinstance(response, str):
        return response.strip()

    if isinstance(response, list):
        parts = []
        for item in response:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
                else:
                    parts.append(str(item))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(response, dict):
        for key in ("text", "content", "summary", "output"):
            if key in response and response[key] is not None:
                return normalize_text_response(response[key])
        return str(response).strip()

    if hasattr(response, "content"):
        return normalize_text_response(response.content)

    if hasattr(response, "text"):
        return normalize_text_response(response.text)

    return str(response).strip()


def format_token_metrics(metrics):
    """Return a compact, readable token summary for terminal output."""

    if not isinstance(metrics, dict):
        return "input=0 | output=0 | total=0"

    return (
        f"input={metrics.get('input_tokens', 0)} | "
        f"output={metrics.get('output_tokens', 0)} | "
        f"total={metrics.get('total_tokens', 0)}"
    )
