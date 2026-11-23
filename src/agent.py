import json
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.guardrails import run_with_timeout
from src.tools import inspect_schema, make_chart, query_data

load_dotenv()

SYSTEM_PROMPT = """You are TraceLens, a precise, grounded Data Analyst Agent.

Follow these strict rules on every user question:
1. ALWAYS call `inspect_schema` first on a new question so you know exact column names, data types, missing value counts, and sample values.
2. NEVER calculate, estimate, or invent numbers in your head. Every single number in your final answer must come directly from `query_data` output.
3. For comparative questions (such as "which is highest and by how much?" or "compare top two"), retrieve at least the top 2+ rows (do NOT set `limit=1`) so both values come directly from `query_data`.
4. Watch for messy real-world data seen in the schema (for example, trailing spaces like " North " or missing/NaN values). Prefer the `contains` operator when filtering text columns that may have extra spaces, and mention notable missing values if relevant.
5. Call `make_chart` ONLY when the user asks for a chart/plot/graph/visual OR when a visual genuinely clarifies a comparison or trend.
6. If a requested column or metric does not exist in the dataset, explain clearly what is missing based on the schema instead of guessing.
7. Keep your final answer concise, clear, and directly grounded in the tool observations."""

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": (
                "Inspect the loaded CSV dataset's row count, column names, data types, "
                "missing value counts, numeric min/max ranges, sample categorical values, and first 3 rows. "
                "Always call this first before querying or charting."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Run a read-only filter, group-by, aggregation, sort, and limit query on the dataset. "
                "Use this for all counts, sums, averages, comparisons, and row lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "array",
                        "description": "Optional list of AND filter conditions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string", "description": "Exact column name from inspect_schema."},
                                "op": {
                                    "type": "string",
                                    "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "isnull", "notnull"],
                                    "description": "Comparison operator. Use 'contains' for case-insensitive partial text match.",
                                },
                                "value": {
                                    "type": ["string", "number", "boolean", "null"],
                                    "description": "Value to compare against (omit or null for isnull/notnull).",
                                },
                            },
                            "required": ["column", "op"],
                        },
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names to group by.",
                    },
                    "metric_col": {
                        "type": "string",
                        "description": "Numeric or target column to aggregate (e.g., 'revenue', 'units_sold').",
                    },
                    "agg": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "min", "max", "median", "nunique"],
                        "description": "Aggregation function to apply. Defaults to 'sum'.",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Column name to sort results by.",
                    },
                    "ascending": {
                        "type": "boolean",
                        "description": "Sort ascending (true) or descending (false, default).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default 15).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Generate and save a Matplotlib/Seaborn chart image from the dataset. "
                "For 'bar' and 'line' charts, data is automatically grouped by x_col (and optional hue_col) using agg."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "hist", "scatter", "box"],
                        "description": "Type of visualization to generate.",
                    },
                    "x_col": {"type": "string", "description": "Column name for the x-axis."},
                    "y_col": {"type": "string", "description": "Column name for the y-axis (optional for bar/line counts and hist)."},
                    "hue_col": {"type": "string", "description": "Optional categorical column for color grouping."},
                    "agg": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "min", "max", "median"],
                        "description": "Aggregation for bar/line charts (default 'sum').",
                    },
                    "title": {"type": "string", "description": "Clear, descriptive chart title."},
                    "filters": {
                        "type": "array",
                        "description": "Optional filter conditions applied before plotting.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "op": {
                                    "type": "string",
                                    "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "isnull", "notnull"],
                                },
                                "value": {"type": ["string", "number", "boolean", "null"]},
                            },
                            "required": ["column", "op"],
                        },
                    },
                },
                "required": ["chart_type", "x_col"],
            },
        },
    },
]


def _execute_tool(df: pd.DataFrame, tool_name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call with timeout protection and crash-to-JSON error recovery."""
    try:
        if tool_name == "inspect_schema":
            return run_with_timeout(inspect_schema, df)
        if tool_name == "query_data":
            return run_with_timeout(query_data, df, **args)
        if tool_name == "make_chart":
            return run_with_timeout(make_chart, df, **args)
        return json.dumps({"status": "error", "message": f"Unknown tool '{tool_name}'."})
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Tool execution error: {exc}"})


def run_agent(
    question: str,
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    max_steps: int = 7,
) -> dict[str, Any]:
    """Run a bounded ReAct loop to answer a plain-English question about `df`."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    trace: list[dict[str, Any]] = []
    chart_path: str | None = None

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            final_answer = msg.content or "No answer generated."
            trace.append({"step": step, "type": "final_answer", "content": final_answer})
            return {
                "answer": final_answer,
                "chart_path": chart_path,
                "trace": trace,
                "steps_used": step,
            }

        assistant_record: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        }
        messages.append(assistant_record)

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                observation = json.dumps({"status": "error", "message": f"Invalid JSON arguments: {exc}"})
                args = {}
            else:
                observation = _execute_tool(df, tool_name, args)

            if tool_name == "make_chart":
                try:
                    obs_data = json.loads(observation)
                    if obs_data.get("status") == "ok" and obs_data.get("chart_path"):
                        chart_path = obs_data["chart_path"]
                except json.JSONDecodeError:
                    pass

            trace.append(
                {
                    "step": step,
                    "thought": msg.content or "",
                    "action": tool_name,
                    "action_input": args,
                    "observation": observation,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                }
            )

    fallback = f"Stopped after reaching the maximum limit of {max_steps} reasoning steps without a final answer."
    return {
        "answer": fallback,
        "chart_path": chart_path,
        "trace": trace,
        "steps_used": max_steps,
    }


if __name__ == "__main__":
    demo_df = pd.read_csv("data/demo_sales.csv")
    q1 = "Which product category generated the highest Closed Won revenue, and by how much over the second highest?"
    print(f"QUESTION: {q1}")
    res = run_agent(q1, demo_df)
    print(f"\nANSWER:\n{res['answer']}")
