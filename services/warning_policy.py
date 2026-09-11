import warnings


def suppress_langchain_google_warnings():
    """Suppress the noisy Google/LangChain SDK warnings that are not actionable in lab usage.

    This is intentionally narrow: it ignores only the known warning emissions from
    Google and LangChain libraries, while leaving other warnings visible.
    """

    warnings.filterwarnings(
        "ignore",
        category=Warning,
        module=r"langchain_google_genai.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        module=r"langchain_anthropic.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        module=r"google.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        message=r".*Direct use of automatic function calling.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        message=r".*uses fixed sampling defaults.*",
    )
