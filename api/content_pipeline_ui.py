import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.basic_task_executor import execute_task as execute_basic_task
from services.content_generation_pipeline import run_content_pipeline
from services.llm_task_executor import execute_task as execute_provider_task
from services.structured_output_service import generate_structured_response
from services.tiered_task_executor import execute_task as execute_tiered_task


app = FastAPI(
    title="LLM Service Demo Dashboard",
    version="1.0.0",
    description="Generic interface for demoing the repository's content, routing, and task-execution services.",
)


class DemoRequest(BaseModel):
    module: str
    task: str = "summarize"
    text: str


SAMPLE_TEXT = """Generative AI is transforming how teams summarize large documents, extract signal from content, and create concise headlines. It can improve productivity and accelerate decisions, but outputs still need human oversight because errors and bias can appear. Responsible deployment requires attention to privacy, security, and ethics."""


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LLM Service Demo Dashboard</title>
    <style>
      :root {
        --bg: #f3f6ff;
        --panel: #ffffff;
        --panel-alt: #f8faff;
        --border: #dfe7ff;
        --primary: #2f5bff;
        --primary-dark: #2348d9;
        --text: #132238;
        --muted: #5d6d86;
        --success: #0f8a5f;
        --warning: #a35d00;
        --shadow: 0 14px 28px rgba(25, 42, 87, 0.08);
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        font-family: Arial, Helvetica, sans-serif;
        background: linear-gradient(180deg, #eef4ff, #f5f7fb);
        color: var(--text);
      }

      .container {
        max-width: 1100px;
        margin: 40px auto;
        padding: 20px;
      }

      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 28px;
      }

      h1 {
        margin: 0 0 12px;
        font-size: 2.1rem;
      }

      .subtitle {
        color: var(--muted);
        margin: 0 0 24px;
      }

      .controls {
        display: grid;
        grid-template-columns: 1.2fr 0.9fr 180px;
        gap: 12px;
        margin-bottom: 16px;
        align-items: end;
      }

      select, textarea, button {
        font: inherit;
      }

      select, textarea {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        background: #fff;
        color: var(--text);
      }

      textarea {
        min-height: 220px;
        resize: vertical;
      }

      button {
        border: none;
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s ease, opacity 0.15s ease;
        width: 180px;
        min-width: 180px;
        height: 48px;
        align-self: end;
      }

      button:hover { transform: translateY(-1px); }
      button:disabled { opacity: 0.6; cursor: wait; }

      .primary {
        background: var(--primary);
        color: white;
      }

      .secondary {
        background: #edf3ff;
        color: var(--primary-dark);
      }

      .status {
        min-height: 26px;
        color: var(--muted);
        margin: 10px 0 18px;
      }

      .results {
        margin-top: 24px;
        border-top: 1px solid var(--border);
        padding-top: 24px;
        display: none;
      }

      .result-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 18px;
      }

      .card {
        background: var(--panel-alt);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
      }

      .card h3 {
        margin: 0 0 10px;
        font-size: 1rem;
        color: var(--primary-dark);
      }

      .value-box {
        white-space: pre-wrap;
        line-height: 1.6;
        color: var(--text);
      }

      .output-box {
        min-height: 110px;
        overflow-y: auto;
        padding: 10px 12px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #fff;
      }

      .metrics-box {
        margin-top: 12px;
        padding: 6px 10px 4px;
        border: 1px solid #dfe7ff;
        border-radius: 10px;
        background: #f7f9ff;
        align-self: start;
      }

      .metrics-title {
        display: block;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: lowercase;
        color: var(--primary-dark);
        margin-bottom: 2px;
      }

      .token-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        padding: 0;
        margin: 0;
        border-bottom: 1px solid rgba(123, 136, 170, 0.22);
        font-size: 0.88rem;
        line-height: 1.05;
      }

      .token-row:last-child {
        border-bottom: none;
      }

      .token-row:last-child { border-bottom: none; }

      .token-row span { color: var(--muted); }
      .token-row strong { font-weight: 700; }

      .helper-box {
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 10px;
        background: #edf3ff;
        border: 1px solid #d7e4ff;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.5;
      }

      @media (max-width: 760px) {
        .controls, .result-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="panel">
        <h1>LLM Service Demo Dashboard</h1>
        <p class="subtitle">Run the repository's service modules individually and compare output plus usage.</p>

        <div class="controls">
          <div>
            <label for="moduleSelect">Module</label>
            <select id="moduleSelect">
              <option value="content_pipeline">Content pipeline</option>
              <option value="structured_summary">Structured summary</option>
              <option value="basic_task">Basic task executor</option>
              <option value="provider_task">Provider-aware task executor</option>
              <option value="tiered_task">Tiered task executor</option>
            </select>
            <div id="moduleDescription" class="helper-box"></div>
          </div>

          <div>
            <label for="taskSelect">Task</label>
            <select id="taskSelect">
              <option value="summarize">summarize</option>
              <option value="rewrite">rewrite</option>
              <option value="headline">headline</option>
              <option value="keypoints">keypoints</option>
            </select>
          </div>

          <button class="primary" id="runButton">Run demo</button>
        </div>

        <div style="margin: 14px 0 8px;">
          <label for="inputText" style="display: block; margin-bottom: 8px; font-weight: 600; color: var(--text);">Input text for the selected module</label>
        </div>
        <textarea id="inputText" placeholder="Paste a paragraph, article excerpt, or prompt here...">Generative AI can summarize documents, extract key ideas, and write concise headlines for busy teams. It helps speed up review workflows, but human oversight remains important to catch errors, bias, and missing context before publishing or acting on the output.</textarea>

        <div class="status" id="status"></div>

        <div class="results" id="resultsPanel">
          <div class="result-grid">
            <div class="card">
              <h3>Summary</h3>
              <div id="summaryValue" class="value-box output-box"></div>
              <div id="summaryTokens" class="value-box metrics-box"><span class="metrics-title">token usage</span></div>
            </div>

            <div class="card">
              <h3>Keywords</h3>
              <div id="keywordsValue" class="value-box output-box"></div>
              <div id="keywordsTokens" class="value-box metrics-box"><span class="metrics-title">token usage</span></div>
            </div>

            <div class="card">
              <h3>Headline</h3>
              <div id="headlineValue" class="value-box output-box"></div>
              <div id="headlineTokens" class="value-box metrics-box"><span class="metrics-title">token usage</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const moduleSelect = document.getElementById('moduleSelect');
      const taskSelect = document.getElementById('taskSelect');
      const inputText = document.getElementById('inputText');
      const status = document.getElementById('status');
      const resultsPanel = document.getElementById('resultsPanel');
      const summaryValue = document.getElementById('summaryValue');
      const keywordsValue = document.getElementById('keywordsValue');
      const headlineValue = document.getElementById('headlineValue');
      const summaryTokens = document.getElementById('summaryTokens');
      const keywordsTokens = document.getElementById('keywordsTokens');
      const headlineTokens = document.getElementById('headlineTokens');
      const runButton = document.getElementById('runButton');
      const moduleDescription = document.getElementById('moduleDescription');

      const moduleDescriptions = {
        content_pipeline: 'Runs the full pipeline: summarize the input, extract keywords, and generate a headline from the same content.',
        structured_summary: 'Returns a structured summary object with a summary, keywords, and usage metadata in a typed response format.',
        basic_task: 'Routes the request through the simple task executor to perform a basic prompt-based operation like summarize or rewrite.',
        provider_task: 'Chooses a provider and model dynamically before executing the selected task using the provider-aware router.',
        tiered_task: 'Uses tiered routing to pick the best model tier and provider for the task before generating the response.'
      };

      function updateTaskVisibility() {
        const selectedModule = moduleSelect.value;
        const requiresTask = ['basic_task', 'provider_task', 'tiered_task'].includes(selectedModule);
        taskSelect.disabled = !requiresTask;
        moduleDescription.textContent = moduleDescriptions[selectedModule] || 'Select a module to see a brief description.';
      }

      function renderTokenMetrics(metrics) {
        const safeMetrics = metrics && (
          metrics.input_tokens !== undefined ||
          metrics.output_tokens !== undefined ||
          metrics.total_tokens !== undefined
        )
          ? metrics
          : { input_tokens: 0, output_tokens: 0, total_tokens: 0 };

        return `
          <span class="metrics-title">token usage</span>
          <div class="token-row"><span>input:</span><strong>${safeMetrics.input_tokens ?? 0}</strong></div>
          <div class="token-row"><span>output:</span><strong>${safeMetrics.output_tokens ?? 0}</strong></div>
          <div class="token-row"><span>total:</span><strong>${safeMetrics.total_tokens ?? 0}</strong></div>
        `;
      }

      function renderResult(payload) {
        const metrics = payload?.token_metrics || {};

        if (payload && payload.summary) {
          const keywordText = Array.isArray(payload.keywords)
            ? payload.keywords.join(', ')
            : (payload.keywords || '');

          summaryValue.textContent = payload.summary || '';
          keywordsValue.textContent = keywordText;
          headlineValue.textContent = payload.headline || '';

          summaryTokens.innerHTML = renderTokenMetrics(metrics.summary || {});
          keywordsTokens.innerHTML = renderTokenMetrics(metrics.keywords || {});
          headlineTokens.innerHTML = renderTokenMetrics(metrics.headline || {});
        } else if (payload && payload.result) {
          summaryValue.textContent = payload.result;
          keywordsValue.textContent = '';
          headlineValue.textContent = '';
          summaryTokens.innerHTML = renderTokenMetrics(metrics || {});
          keywordsTokens.innerHTML = '';
          headlineTokens.innerHTML = '';
        } else {
          summaryValue.textContent = JSON.stringify(payload, null, 2);
          keywordsValue.textContent = '';
          headlineValue.textContent = '';
          summaryTokens.innerHTML = '';
          keywordsTokens.innerHTML = '';
          headlineTokens.innerHTML = '';
        }

        resultsPanel.style.display = 'block';
      }

      moduleSelect.addEventListener('change', updateTaskVisibility);
      updateTaskVisibility();

      runButton.addEventListener('click', async () => {
        const text = inputText.value.trim();
        const module = moduleSelect.value;
        const task = taskSelect.value;

        if (!text) {
          status.textContent = 'Please enter some text before running the demo.';
          return;
        }

        status.textContent = 'Running selected demo...';
        runButton.disabled = true;

        try {
          const response = await fetch('/api/demo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ module, task, text })
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Demo request failed.');
          }

          const data = await response.json();
          renderResult(data);
          status.textContent = `Completed: ${module}`;
        } catch (error) {
          status.textContent = `Error: ${error.message}`;
          resultsPanel.style.display = 'none';
        } finally {
          runButton.disabled = false;
        }
      });
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_PAGE)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/demo")
async def run_module_demo(payload: DemoRequest):
    text = (payload.text or "").strip()
    module_name = (payload.module or "").strip().lower()
    task = (payload.task or "summarize").strip().lower()

    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        if module_name == "content_pipeline":
            result = run_content_pipeline(text)
            return result

        if module_name == "structured_summary":
            result = generate_structured_response(text)
            return {
                "summary": result.summary,
                "keywords": result.keywords,
                "token_metrics": getattr(result, "token_metrics", {}),
            }

        if module_name == "basic_task":
            result = execute_basic_task(task=task, text=text)
            return result

        if module_name == "provider_task":
            result = execute_provider_task(task=task, text=text)
            return result

        if module_name == "tiered_task":
            result = execute_tiered_task(task=task, text=text)
            return result

        raise HTTPException(status_code=400, detail=f"Unsupported module: {module_name}")

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
