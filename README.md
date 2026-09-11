# routing-model-engg-labs

Model routing labs in Python. Four progressively more capable routers that
decide **which prompt and which model** should handle a given task, each
wired to a LangChain executor and a FastAPI endpoint.

The question the whole repo circles is: when a request arrives, what decides
how it gets answered? Start with a lookup table, end with tier-based model
selection behind a typed HTTP contract.

Alongside the routers are a three-stage content pipeline, a Pydantic
structured-output service, an LLM-based grounding checker, and a prompt
evaluation harness.

---

## Setup

```bash
git clone https://github.com/vikram-git-vault/routing-model-engg-labs.git
cd routing-model-engg-labs

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# open .env and paste in your real keys
```

### Watch your quota before you run anything

The Gemini **free tier allows 20 requests per day, per model**. That is the
single most limiting constraint on this repo, and it is easy to exhaust by
accident:

| What you run                   | Requests used                          |
| ------------------------------ | -------------------------------------- |
| `model_routing_dashboard.py`   | 10 (5 samples x 2 prompts)             |
| `content_pipeline_demo.py`     | 3 (one per pipeline stage)             |
| `grounded_content_analyzer.py` | 6 (3 generations + 3 grounding checks) |
| One API call via the UI        | 1                                      |

Three dashboard runs and you are done for the day. When quota runs out you
get a `429 RESOURCE_EXHAUSTED` that the SDK retries with backoff, so the
symptom is a request hanging for two or three minutes rather than a clear
error. If a call takes far longer than it should, suspect quota first.

The small and large tiers use different models, and quota is counted per
model — so tier routing spreads the budget as a side effect.

---

## Configuration

Everything lives in `.env`, which is gitignored. **`config.py` is the single
source** — no other module reads the environment.

| Variable              | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| `GEMINI_API_KEY`      | Required.                                           |
| `GENAI_MODEL`         | Default model for the non-tiered paths.             |
| `GENAI_MODEL_SMALL`   | Model for the `small` tier. Must differ from large. |
| `GENAI_MODEL_LARGE`   | Model for the `large` tier.                         |
| `ANTHROPIC_API_KEY`   | Only needed if a task is routed to Anthropic.       |
| `ANTHROPIC_MODEL`     | Model used when a task falls back to Anthropic.     |
| `LLM_TEMPERATURE`     | Sampling temperature. Default `0.1`.                |
| `LLM_TIMEOUT_SECONDS` | LangChain deadline in **seconds**. Floor of 10.     |
| `GENAI_TIMEOUT_MS`    | Native client deadline in **milliseconds**.         |

`config.py` also owns every path (`PROJECT_ROOT`, `PROMPT_DIR`,
`SOURCE_TEXT_FILE` and so on), derived from its own location, so scripts run
correctly from any working directory.

It exposes `require_gemini_key()` and `require_anthropic_key()`, which
**raise** rather than calling `sys.exit`. Importing a module must never be
able to terminate the process; entry points catch and exit on their own
terms.

---

## The routing progression

Four routers, each adding exactly one decision to the previous one.

| #   | Router                          | Decides                                       |
| --- | ------------------------------- | --------------------------------------------- |
| 1   | `basic_task_router.py`          | which prompt file, from a flat dict           |
| 2   | `prompt_based_router.py`        | same, via a config dict that can grow         |
| 3   | `provider_aware_task_router.py` | prompt + provider + model name per task       |
| 4   | `tiered_model_router.py`        | prompt + delegates model choice to a selector |

Every router returns a plain dict. Executors consume it. Nothing about a
router knows how to call an LLM, which is the point — routing is a decision,
execution is a separate concern.

### Tier selection

`services/task_model_selector.py` maps tasks to tiers, and tiers to models:

```
summarize  -> small  -> GENAI_MODEL_SMALL
headline   -> small  -> GENAI_MODEL_SMALL
keypoints  -> small  -> GENAI_MODEL_SMALL
rewrite    -> large  -> GENAI_MODEL_LARGE
```

Rewriting a customer message needs more capability than summarising, so it
gets the larger model. Everything else takes the cheap one.

---

## Running things

### The demo UI — start here

```bash
python api/content_pipeline_ui.py
# then open http://localhost:8000
```

A browser interface that fronts every service in the repo: the basic
executor, the provider-aware executor, the tiered executor, the content
pipeline, and the structured-output service. One request per click.

### The task APIs

Three FastAPI apps, all serving `POST /task`, each wired to a different
executor:

```bash
uvicorn api.basic_task_api:app --reload      # prompt routing only
uvicorn api.routed_task_api:app --reload     # + provider and model
uvicorn api.unified_task_api:app --reload    # + tier and timeout
```

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "summarize", "text": "..."}'
```

`api/legacy_task_api.py` is the earliest version — it returns the routing
decision without executing anything. Useful for seeing a router in isolation.

### Standalone scripts

```bash
python content_pipeline_demo.py        # 3-stage chain, per-stage token counts
python structured_summary_demo.py      # Pydantic-validated JSON output
python grounded_content_analyzer.py    # generation + grounding check
python model_routing_dashboard.py      # zero-shot vs few-shot evaluation
python single_prompt.py                # one raw API call
```

Run from the project root — several still use relative paths.

---

## What every file does

### Routers

| File                            | Notes                                                   |
| ------------------------------- | ------------------------------------------------------- |
| `basic_task_router.py`          | Task name to prompt file. Three tasks. No model logic.  |
| `prompt_based_router.py`        | Same idea, config dict per task so fields can be added. |
| `provider_aware_task_router.py` | Adds provider and model. Has unused fallback logic.     |
| `tiered_model_router.py`        | Delegates model choice to `task_model_selector`.        |

### Executors

| File                               | Notes                                                         |
| ---------------------------------- | ------------------------------------------------------------- |
| `services/basic_task_executor.py`  | Builds a LangChain chain from the routed prompt. Gemini only. |
| `services/llm_task_executor.py`    | Uses the provider factory. Classifies auth failures.          |
| `services/tiered_task_executor.py` | Tiered routing plus a hard wall-clock timeout. See below.     |

### Supporting services

**`services/provider_model_factory.py`**
Returns a configured chat model for a provider and model name. The single
place that knows about `ChatGoogleGenerativeAI` and `ChatAnthropic`, so
adding a provider touches one file.

**`services/task_model_selector.py`**
Maps task to tier, tier to model. Reads model names from `config.py`.

**`services/token_metrics.py`**
Normalises token counts across every naming scheme the SDKs use —
`input_tokens` / `prompt_tokens` / `prompt_token_count` and so on — into one
consistent dict. Also flattens response content into plain text, whether it
arrives as a string, a list of block dicts, or an object.

**`services/prompt_loader.py`**
Loads prompt templates from `prompts/`, resolved from the project root so it
works regardless of working directory.

**`services/content_generation_pipeline.py`**
Three stages, each feeding the next: summary, then keywords from that
summary, then a headline from both. Returns per-stage and total token counts.

**`services/structured_output_service.py`**
Prompt to model to `PydanticOutputParser`, yielding a validated
`SummaryResponse`.

**`services/warning_policy.py`**
Suppresses known-noisy SDK warnings. Currently incomplete — see Known issues.

### Standalone tools

**`grounded_content_analyzer.py`** — the most interesting thing here.
Generates a summary, keywords and a headline, then makes a *second* LLM call
that compares each output against the source and returns JSON saying whether
the claims are supported. Uses `response_mime_type="application/json"`, and
falls back to fence-stripping and brace-matching when the model wraps its
JSON anyway. That is a real evaluation technique, not a toy.

**`model_routing_dashboard.py`** — prompt evaluation harness.
Samples 5 of the 10 test cases, runs each through `rewrite_zero_shot.txt` and
`rewrite_few_shot.txt`, scores outputs on emptiness, length and meta-text
leakage, and writes `evaluation/prompt_evaluation_report.csv`. Times every
call and prints token usage.

### Models

| File                                 | Notes                                                |
| ------------------------------------ | ---------------------------------------------------- |
| `models/routed_task_contract.py`     | `TaskRequest`, `TaskResponse`, `TokenMetrics`.       |
| `models/content_summary_response.py` | Summary plus exactly 5 keywords, plus token metrics. |
| `models/task_request.py`             | Minimal request used only by the legacy API.         |

---

## Two things worth knowing about this code

**Timeout is in seconds here, milliseconds elsewhere.**
`ChatGoogleGenerativeAI` and `ChatAnthropic` take `timeout` in **seconds** and
convert internally. `google.genai.types.HttpOptions` takes **milliseconds**.
Both appear in this repo — the factory uses the first, the dashboard the
second. Passing a millisecond figure to the LangChain models requests a
multi-hour deadline that silently never fires. The API also rejects any
deadline below 10 seconds outright, so `REQUEST_TIMEOUT_SECONDS` is clamped.

That deadline covers **one attempt**. It does not cover the backoff between
retries, which is why a 20-second deadline can still produce a 160-second
call when the SDK is retrying a 429. `tiered_task_executor` exists to cap
total wall time, which the per-request deadline structurally cannot.

**FastAPI silently drops fields missing from `response_model`.**
The executors return `provider`, `model`, `model_tier` and `timed_out`. If
those are not declared on `TaskResponse`, FastAPI filters them out before the
caller ever sees them — no warning, no error. Declare every field an executor
might return, optional where not all of them do.

---

## Known issues

Honest list of what is still wrong.

- **`tiered_task_executor` uses `fork` multiprocessing.** It solves a real
  problem — capping total wall time including retry backoff, which a
  per-request deadline cannot — but forking a process holding gRPC threads
  risks deadlock on macOS, and `fork` is being deprecated in Python 3.14. A
  thread with a timeout would do the same job. Replacing it needs live calls
  to verify, so it is waiting on quota.
- **429 handling is silent.** A quota error retries with backoff for
  minutes. The API returns `retryDelay` in the response; the code should
  fail fast and report it rather than appearing to hang.
- **`max_output_tokens` varies** across files — 100, 120, 200, 400 — all
  hardcoded, none explained. 120 may truncate a rewrite.
- **Three FastAPI apps share a title.** `basic_task_api`, `routed_task_api`
  and `unified_task_api` all announce themselves as "Unified LLM Task API"
  v1.0.0 on the same `POST /task` path.
- **`models/task_request.py` duplicates `TaskRequest`** with a weaker
  definition. Only the legacy API uses it.
- **`prompts/summarize_v2.txt` is unused.**

### Fixed since the initial commit

Kept here because the reasoning is more useful than the fix.

- **Timeout was in the wrong unit.** `20000` was read as 20,000 *seconds*,
  so no deadline ever fired. Traced by the API's own error message naming
  the deadline it received.
- **Both model tiers pointed at the same model**, making the tiered router a
  no-op. The selector also hardcoded its model instead of reading config, so
  the tiered path silently used a different model from every other path.
- **`response_model` silently dropped routing fields.** FastAPI filters
  returned dicts against the schema, so `provider`, `model`, `model_tier`
  and `timed_out` never reached the caller.
- **`_resolve_provider()` was never called**, so the provider fallback the
  router is named for did not run. Wiring it up surfaced two more problems:
  the model name has to change with the provider, and falling back with no
  keys at all had to become an error rather than a second dead end.
- **The AFC notice could not be suppressed** by `warnings.filterwarnings` —
  it is emitted through `logging`. The filters were also catching
  `DeprecationWarning`, hiding the `fork` notice that applies to this repo.
- **Relative paths** meant four entry points only ran from the project root.
- **Import-time side effects.** Three services built LLM clients on import;
  `content_generation_pipeline` raised on a missing key, so importing it
  took down the UI at startup. All now build lazily behind `lru_cache`.
