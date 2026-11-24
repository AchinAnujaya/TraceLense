---
title: TraceLens Data Analyst Agent
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
---

# TraceLens — Grounded ReAct Data Analyst Agent

Ask plain-English questions about any CSV dataset and get verified answers, publication-ready charts, and a collapsible step-by-step reasoning trace—with **zero LLM mental arithmetic** and **zero arbitrary code execution**.

**Live App:** [https://tracelense.onrender.com/](https://tracelense.onrender.com/) _(Hosted on Render free tier — initial load may take ~45s if waking from sleep)_

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          Gradio Web Interface                           │
│  [CSV Upload / Demo]  [Data Preview]  [Chat Box]  [Chart + Trace View]  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ question + per-session DataFrame
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Bounded ReAct Agent Loop + Guardrails (max 7 steps)       │
│                                                                         │
│   ┌──────────────┐   tool_call (JSON)   ┌───────────────────────────┐   │
│   │  OpenAI LLM  │ ───────────────────► │  Deterministic Tools      │   │
│   │  (Reasons)   │ ◄─────────────────── │  (Compute with pandas)    │   │
│   └──────────────┘   observation (JSON) │  1. inspect_schema()      │   │
│                                         │  2. query_data()          │   │
│                                         │  3. make_chart()          │   │
│                                         └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

- **Separation of Reasoning & Computation:** The LLM (`gpt-4o-mini`, `temperature=0`) plans queries and interprets results; deterministic `pandas` functions compute every sum, average, count, and filter.
- **Safe Structured Grammar (No `eval`):** Instead of executing model-generated Python code, `query_data` exposes a validated allowlist of filter operators (`==`, `!=`, `>`, `>=`, `<`, `<=`, `contains`, `isnull`, `notnull`) and aggregations (`sum`, `mean`, `count`, `min`, `max`, `median`, `nunique`).
- **Transparent Execution Trace:** Every `Thought → Action → Observation` cycle is logged and rendered inside a collapsible **"How I got this"** panel.
- **Production Guardrails:** Enforces a **10 MB** file size limit, **50,000-row** cap, **10-second** per-tool timeout, **7-step** loop cap, and **15-question** per-session quota via `gr.State`.

---

## Evaluation Benchmark

Tested across **15 stratified queries** (Simple aggregations, Multi-Step dirty-data filtering with trailing whitespace & nulls, Unanswerable abstention tests, and Chart generation):

- **Automated Pass Rate:** **14 / 15 (93.3%)** _(15/15 functionally accurate; Q08 returned `0.1` vs `0.10` due to 2-decimal float rounding)_
- **Abstention Accuracy (Tier 3 missing-column queries):** **3 / 3 (100%)** — zero hallucinated metrics.

| Tier                                | Focus                                                                     | Result                        |
| :---------------------------------- | :------------------------------------------------------------------------ | :---------------------------- |
| **Tier 1: Simple**                  | Row counts, totals, averages, group frequencies                           | 5 / 5 Passed                  |
| **Tier 2: Multi-Step & Messy Data** | Whitespace matching (`" North "`), null counts, multi-filter aggregations | 4 / 5 Passed _(5/5 semantic)_ |
| **Tier 3: Tricky & Visuals**        | Missing-column refusals, empty filters, Matplotlib/Seaborn charts         | 5 / 5 Passed                  |

_(Full per-question latency and step breakdown available in `outputs/eval_results.md`.)_

---

## Project Structure

```text
TraceLense/
├── app.py                  # Thin Gradio web interface & per-session state
├── requirements.txt        # Cloud deployment dependencies
├── data/
│   └── demo_sales.csv      # 2,500-row realistic sales dataset with missing values & whitespace quirks
├── scripts/
│   ├── make_demo_data.py   # Reproducible messy dataset generator
│   └── evaluate.py         # 15-question stratified benchmark runner
└── src/
    ├── tools.py            # Deterministic pandas & Matplotlib/Seaborn tools
    ├── guardrails.py       # File size, row count, timeout, and session rate limits
    └── agent.py            # Bounded OpenAI ReAct loop & JSON tool schemas
```

---

## Run Locally with `uv`

```bash
git clone [https://github.com/YOUR_USERNAME/TraceLense.git](https://github.com/YOUR_USERNAME/TraceLense.git)
cd TraceLense
uv sync
echo 'OPENAI_API_KEY=your-openai-api-key' > .env
uv run python app.py
```

---

## Limitations & Future Work

- **Multi-table joins:** Currently analyzes one CSV per session; extending `query_data` to support DuckDB SQL over multiple uploaded files would enable relational analysis.
- **Streaming trace updates:** Yielding intermediate `Thought` and `Action` steps progressively into the Gradio accordion as each tool finishes.
- **Semantic numeric eval assertions:** Replacing string-based float checks in `scripts/evaluate.py` with tolerance-based numeric parsing (`abs(pred - truth) < 1e-2`).
