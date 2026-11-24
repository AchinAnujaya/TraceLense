import json
import os
from typing import Any

import gradio as gr
import pandas as pd

from src.agent import run_agent
from src.guardrails import SessionGuard, validate_and_load_csv

DEMO_CSV_PATH = "data/demo_sales.csv"

SAMPLE_QUESTIONS = [
    "Which product category generated the highest Closed Won revenue, and by how much over the second highest?",
    "How much Closed Won revenue came from the North region? Plot a bar chart of Closed Won revenue by region.",
    "Compare average units_sold across sales channels for Closed Won deals and create a bar chart.",
    "What is the average customer satisfaction score?",
]


def _load_default_df() -> pd.DataFrame:
    return validate_and_load_csv(DEMO_CSV_PATH)


def _format_trace_md(trace: list[dict[str, Any]], steps_used: int) -> str:
    """Format the agent's ReAct trace into readable Markdown for the collapsible panel."""
    if not trace:
        return "_No reasoning steps recorded._"

    lines = [f"**Total reasoning steps:** {steps_used}\n"]
    for item in trace:
        step = item.get("step", "?")
        if item.get("type") == "final_answer":
            lines.append(f"### Step {step} — Final Answer\nSynthesized final response from tool observations.")
            continue

        thought = item.get("thought", "").strip()
        action = item.get("action", "unknown")
        action_input = json.dumps(item.get("action_input", {}), indent=2)
        obs = str(item.get("observation", ""))
        if len(obs) > 900:
            obs = obs[:900] + "\n... [truncated for display]"

        lines.append(f"### Step {step} — Action: `{action}`")
        if thought:
            lines.append(f"**Thought:** {thought}\n")
        lines.append(f"**Tool Input:**\n```json\n{action_input}\n```")
        lines.append(f"**Observation:**\n```json\n{obs}\n```\n---")

    return "\n".join(lines)


def handle_file_upload(file_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame, str, list, None, str]:
    """Validate and load an uploaded CSV, resetting chat history for the new dataset."""
    if not file_path:
        df = _load_default_df()
        return df, df.head(5), "Loaded built-in demo dataset (`demo_sales.csv`).", [], None, "_Ask a question to inspect the reasoning trace._"

    try:
        df = validate_and_load_csv(file_path)
        status = f"Loaded uploaded CSV: **{len(df):,} rows × {len(df.columns)} columns**."
        return df, df.head(5), status, [], None, "_Ask a question to inspect the reasoning trace._"
    except Exception as exc:
        fallback_df = _load_default_df()
        status = f"Upload rejected ({exc}). Reverted to built-in demo dataset."
        return fallback_df, fallback_df.head(5), status, [], None, "_Ask a question to inspect the reasoning trace._"


def use_demo_dataset() -> tuple[None, pd.DataFrame, pd.DataFrame, str, list, None, str]:
    """Reset back to the built-in demo dataset."""
    df = _load_default_df()
    status = f"Using built-in demo dataset (`demo_sales.csv`): **{len(df):,} rows × {len(df.columns)} columns**."
    return None, df, df.head(5), status, [], None, "_Ask a question to inspect the reasoning trace._"


def ask_tracelens(
    question: str,
    history: list[dict[str, str]],
    df: pd.DataFrame,
    guard: SessionGuard,
) -> tuple[str, list[dict[str, str]], str | None, str, str]:
    """Run TraceLens on the user's question and update chat, chart, trace, and session status."""
    question = (question or "").strip()
    if not question:
        return "", history, None, "_Enter a question first._", f"Questions used: {guard.count}/{guard.max_questions}"

    allowed, quota_msg = guard.check_and_increment()
    if not allowed:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"**Session Limit Reached:** {quota_msg}"},
        ]
        return "", history, None, "_Session limit reached._", quota_msg

    try:
        result = run_agent(question, df)
        answer = result["answer"]
        chart_path = result.get("chart_path")
        trace_md = _format_trace_md(result.get("trace", []), result.get("steps_used", 0))
    except Exception as exc:
        answer = f"An unexpected error occurred while running the agent: {exc}"
        chart_path = None
        trace_md = f"**Error:** {exc}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return "", history, chart_path, trace_md, quota_msg


def build_ui() -> gr.Blocks:
    initial_df = _load_default_df()

    with gr.Blocks(title="TraceLens — Data Analyst Agent") as demo:
        df_state = gr.State(initial_df)
        guard_state = gr.State(SessionGuard())

        gr.Markdown(
            "# TraceLens — Data Analyst Agent\n"
            "Ask plain-English questions about your CSV data. Every number is computed by deterministic `pandas` tools—never guessed by the LLM."
        )

        with gr.Row():
            # Left Column: Dataset Controls & Preview
            with gr.Column(scale=1):
                gr.Markdown("### 1. Dataset")
                file_input = gr.File(label="Upload CSV (max 10 MB, 50k rows)", file_types=[".csv"], type="filepath")
                demo_btn = gr.Button("Reset to Built-In Demo Dataset", variant="secondary")
                dataset_status = gr.Markdown(
                    f"Using built-in demo dataset (`demo_sales.csv`): **{len(initial_df):,} rows × {len(initial_df.columns)} columns**."
                )
                session_status = gr.Markdown("Question 0/15 (15 remaining in session).")

                gr.Markdown("### 2. Data Preview (First 5 Rows)")
                preview_table = gr.Dataframe(value=initial_df.head(5), interactive=False, wrap=True)

            # Right Column: Chat, Chart Output & Collapsible Reasoning Trace
            with gr.Column(scale=2):
                gr.Markdown("### 3. Ask a Question")
                chatbot = gr.Chatbot(label="TraceLens Analyst", height=360)

                with gr.Row():
                    question_box = gr.Textbox(
                        placeholder="e.g., Which product category generated the highest Closed Won revenue?",
                        show_label=False,
                        scale=4,
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)

                gr.Markdown("**Try a sample question:**")
                with gr.Row():
                    sample_btns = [gr.Button(q, size="sm") for q in SAMPLE_QUESTIONS[:2]]
                with gr.Row():
                    sample_btns += [gr.Button(q, size="sm") for q in SAMPLE_QUESTIONS[2:]]

                chart_output = gr.Image(label="Generated Visualization", type="filepath")

                with gr.Accordion("How I got this (Reasoning Trace)", open=False):
                    trace_output = gr.Markdown("_Ask a question to inspect the reasoning trace._")

        # Event Wiring
        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=[df_state, preview_table, dataset_status, chatbot, chart_output, trace_output],
        )

        demo_btn.click(
            fn=use_demo_dataset,
            inputs=[],
            outputs=[file_input, df_state, preview_table, dataset_status, chatbot, chart_output, trace_output],
        )

        ask_btn.click(
            fn=ask_tracelens,
            inputs=[question_box, chatbot, df_state, guard_state],
            outputs=[question_box, chatbot, chart_output, trace_output, session_status],
        )

        question_box.submit(
            fn=ask_tracelens,
            inputs=[question_box, chatbot, df_state, guard_state],
            outputs=[question_box, chatbot, chart_output, trace_output, session_status],
        )

        for btn in sample_btns:
            btn.click(fn=lambda text=btn.value: text, inputs=[], outputs=[question_box])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
    )
