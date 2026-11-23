import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.agent import run_agent
from src.guardrails import validate_and_load_csv


def _num_in_text(text: str, value: float, decimals: int = 2) -> bool:
    """Check if a number appears in text either with commas (1,234.56) or raw (1234.56)."""
    clean = text.replace(",", "")
    formatted_raw = f"{value:.{decimals}f}"
    formatted_int = f"{int(round(value))}"
    return (formatted_raw in clean) or (decimals == 0 and formatted_int in clean)


def build_eval_suite(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Define 15 evaluation cases across 3 difficulty tiers with deterministic ground truth."""
    total_rows = len(df)
    total_rev = float(df["revenue"].sum())
    refunded_count = int((df["deal_status"] == "Refunded").sum())
    avg_unit_price = float(df["unit_price"].mean())
    top_channel = str(df["sales_channel"].value_counts().idxmax())

    north_cw_rev = float(
        df[df["region"].astype(str).str.contains("North", case=False, na=False) & (df["deal_status"] == "Closed Won")][
            "revenue"
        ].sum()
    )
    top_cw_cat = str(
        df[df["deal_status"] == "Closed Won"].groupby("product_category")["revenue"].sum().idxmax()
    )
    ai_self_disc = float(
        df[(df["product_category"] == "AI Copilot") & (df["sales_channel"] == "Self-Serve")]["discount_rate"].mean()
    )
    missing_rev_count = int(df["revenue"].isna().sum())
    max_direct_rev = float(df[df["sales_channel"] == "Direct"]["revenue"].max())

    return [
        # --- TIER 1: SIMPLE ---
        {
            "id": 1,
            "tier": "Simple",
            "question": "How many total rows are in the dataset?",
            "expected": f"{total_rows:,} rows",
            "check": lambda res: _num_in_text(res["answer"], total_rows, decimals=0),
        },
        {
            "id": 2,
            "tier": "Simple",
            "question": "What is the total revenue across all orders?",
            "expected": f"${total_rev:,.2f}",
            "check": lambda res: _num_in_text(res["answer"], total_rev, decimals=2),
        },
        {
            "id": 3,
            "tier": "Simple",
            "question": "How many orders have a deal_status of Refunded?",
            "expected": f"{refunded_count} orders",
            "check": lambda res: _num_in_text(res["answer"], refunded_count, decimals=0),
        },
        {
            "id": 4,
            "tier": "Simple",
            "question": "What is the average unit_price across all rows?",
            "expected": f"${avg_unit_price:,.2f}",
            "check": lambda res: _num_in_text(res["answer"], avg_unit_price, decimals=2),
        },
        {
            "id": 5,
            "tier": "Simple",
            "question": "Which sales_channel has the highest number of orders?",
            "expected": top_channel,
            "check": lambda res: top_channel.lower() in res["answer"].lower(),
        },
        # --- TIER 2: MULTI-STEP & DIRTY DATA ---
        {
            "id": 6,
            "tier": "Multi-Step",
            "question": "What is the total Closed Won revenue from the North region?",
            "expected": f"${north_cw_rev:,.2f} (including whitespace variants)",
            "check": lambda res: _num_in_text(res["answer"], north_cw_rev, decimals=2),
        },
        {
            "id": 7,
            "tier": "Multi-Step",
            "question": "Which product_category generated the highest Closed Won revenue?",
            "expected": top_cw_cat,
            "check": lambda res: top_cw_cat.lower() in res["answer"].lower(),
        },
        {
            "id": 8,
            "tier": "Multi-Step",
            "question": "What is the average discount_rate for AI Copilot orders in the Self-Serve channel?",
            "expected": f"{ai_self_disc:.2f} (or {ai_self_disc*100:.1f}%)",
            "check": lambda res: _num_in_text(res["answer"], ai_self_disc, decimals=2)
            or _num_in_text(res["answer"], ai_self_disc * 100, decimals=1),
        },
        {
            "id": 9,
            "tier": "Multi-Step",
            "question": "How many rows have missing (null) values in the revenue column?",
            "expected": f"{missing_rev_count} rows",
            "check": lambda res: _num_in_text(res["answer"], missing_rev_count, decimals=0),
        },
        {
            "id": 10,
            "tier": "Multi-Step",
            "question": "What is the highest single-order revenue in the Direct sales channel?",
            "expected": f"${max_direct_rev:,.2f}",
            "check": lambda res: _num_in_text(res["answer"], max_direct_rev, decimals=2),
        },
        # --- TIER 3: TRICKY, REFUSALS & CHARTS ---
        {
            "id": 11,
            "tier": "Tricky",
            "question": "What is the average customer satisfaction score?",
            "expected": "Graceful refusal (column missing)",
            "check": lambda res: any(
                k in res["answer"].lower() for k in ["not exist", "not present", "no column", "does not contain", "missing"]
            ),
        },
        {
            "id": 12,
            "tier": "Tricky",
            "question": "What is the total revenue from the Antarctica region?",
            "expected": "Graceful handling (no matching rows / 0)",
            "check": lambda res: any(
                k in res["answer"].lower() for k in ["no ", "not ", "0", "zero", "none", "antarctica"]
            ),
        },
        {
            "id": 13,
            "tier": "Tricky",
            "question": "What is the net profit margin after employee salaries?",
            "expected": "Graceful refusal (cost/salary data missing)",
            "check": lambda res: any(
                k in res["answer"].lower() for k in ["cannot", "can't", "does not contain", "no ", "not available", "missing"]
            ),
        },
        {
            "id": 14,
            "tier": "Tricky",
            "question": "Plot a bar chart of total revenue by product_category.",
            "expected": "Valid chart PNG generated",
            "check": lambda res: bool(res.get("chart_path")) and Path(res["chart_path"]).exists(),
        },
        {
            "id": 15,
            "tier": "Tricky",
            "question": "Compare average units_sold across sales channels for Closed Won deals and create a bar chart.",
            "expected": "Chart generated + channel names cited",
            "check": lambda res: bool(res.get("chart_path"))
            and Path(res["chart_path"]).exists()
            and ("direct" in res["answer"].lower()),
        },
    ]


def run_evaluation() -> None:
    df = validate_and_load_csv("data/demo_sales.csv")
    suite = build_eval_suite(df)

    results_rows: list[dict[str, Any]] = []
    passed_count = 0

    print(f"Running {len(suite)}-question TraceLens evaluation suite...\n")
    for item in suite:
        start = time.perf_counter()
        res = run_agent(item["question"], df)
        elapsed = round(time.perf_counter() - start, 2)

        checker: Callable[[dict[str, Any]], bool] = item["check"]
        passed = bool(checker(res))
        if passed:
            passed_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Q{item['id']:02d} ({item['tier']}) | Steps: {res['steps_used']} | {elapsed}s — {item['question']}")
        if not passed:
            print(f"       Expected: {item['expected']}")
            print(f"       Got:      {res['answer'][:180]}...")

        results_rows.append(
            {
                "ID": f"Q{item['id']:02d}",
                "Tier": item["tier"],
                "Question": item["question"],
                "Expected": item["expected"],
                "Steps": res["steps_used"],
                "Latency (s)": elapsed,
                "Status": status,
            }
        )

    pass_rate = (passed_count / len(suite)) * 100
    print("\n" + "=" * 70)
    print(f"FINAL SCORE: {passed_count}/{len(suite)} passed ({pass_rate:.1f}%)")
    print("=" * 70)

    # Export Markdown table for README
    Path("outputs").mkdir(exist_ok=True)
    eval_df = pd.DataFrame(results_rows)
    md_report = (
        f"# TraceLens Evaluation Report\n\n"
        f"**Overall Pass Rate:** {passed_count}/{len(suite)} ({pass_rate:.1f}%)\n\n"
        + eval_df.to_markdown(index=False)
        + "\n"
    )
    Path("outputs/eval_results.md").write_text(md_report, encoding="utf-8")
    print("Saved Markdown report to outputs/eval_results.md")


if __name__ == "__main__":
    run_evaluation()
