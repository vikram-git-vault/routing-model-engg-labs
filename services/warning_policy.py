import logging
import warnings


# Loggers that emit unactionable noise during lab runs.
#
# The AFC notice ("Direct use of automatic function calling ... is not
# recommended") is emitted through `logging`, NOT `warnings`, so
# warnings.filterwarnings cannot suppress it no matter how it is matched.
# It has to be silenced at the logger.
NOISY_LOGGERS = (
    "google_genai.models",
    "google_genai.types",
)


def suppress_langchain_google_warnings():
    """Quieten known-noisy Google/LangChain SDK output.

    Deliberately narrow on two axes:

    - Only UserWarning is filtered, not the Warning base class. Filtering
      Warning would also hide DeprecationWarning, which matters here: the
      fork-based timeout in tiered_task_executor relies on behaviour Python
      3.14 is actively deprecating, and that notice should stay visible.
    - Only the loggers listed above are raised to ERROR, so genuine errors
      from those modules still surface.
    """

    for module in (
        r"langchain_google_genai.*",
        r"langchain_anthropic.*",
        r"google.*",
    ):
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=module,
        )

    for message in (
        r".*Direct use of automatic function calling.*",
        r".*uses fixed sampling defaults.*",
    ):
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=message,
        )

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
