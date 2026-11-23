import json
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (prevents GUI thread crashes on macOS/servers)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

VALID_AGGS = {"sum", "mean", "count", "min", "max", "median", "nunique"}
VALID_OPS = {"==", "!=", ">", ">=", "<", "<=", "contains", "isnull", "notnull"}
VALID_CHARTS = {"bar", "line", "hist", "scatter", "box"}


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Safely filter a DataFrame using structured filter rules."""
    if not filters:
        return df

    filtered = df.copy()
    for f in filters:
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")

        if col not in filtered.columns:
            raise ValueError(f"Column '{col}' does not exist. Available columns: {list(filtered.columns)}")
        if op not in VALID_OPS:
            raise ValueError(f"Unsupported operator '{op}'. Use one of: {sorted(VALID_OPS)}")

        series = filtered[col]
        if op == "isnull":
            filtered = filtered[series.isna()]
        elif op == "notnull":
            filtered = filtered[series.notna()]
        elif op == "==":
            filtered = filtered[series == val]
        elif op == "!=":
            filtered = filtered[series != val]
        elif op == ">":
            filtered = filtered[series > val]
        elif op == ">=":
            filtered = filtered[series >= val]
        elif op == "<":
            filtered = filtered[series < val]
        elif op == "<=":
            filtered = filtered[series <= val]
        elif op == "contains":
            filtered = filtered[series.astype(str).str.contains(str(val), case=False, na=False)]

    return filtered


def inspect_schema(df: pd.DataFrame) -> str:
    """Return dataset shape, column types, missing counts, sample values, and preview rows."""
    columns_info = []
    for col in df.columns:
        s = df[col]
        info: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "unique_count": int(s.nunique(dropna=True)),
        }
        non_null = s.dropna()
        if pd.api.types.is_numeric_dtype(s) and not non_null.empty:
            info["min"] = round(float(non_null.min()), 2)
            info["max"] = round(float(non_null.max()), 2)
        else:
            # Show up to 6 distinct values so the LLM sees exact spelling/whitespace
            info["sample_values"] = [str(v) for v in non_null.unique()[:6]]
        columns_info.append(info)

    result = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
        "sample_rows": df.head(3).fillna("NaN").to_dict(orient="records"),
    }
    return json.dumps(result, indent=2, default=str)


def query_data(
    df: pd.DataFrame,
    filters: list[dict[str, Any]] | None = None,
    group_by: list[str] | None = None,
    metric_col: str | None = None,
    agg: str = "sum",
    sort_by: str | None = None,
    ascending: bool = False,
    limit: int = 15,
) -> str:
    """Run a safe filter, group-by, aggregation, sort, and limit query on the DataFrame."""
    try:
        working_df = _apply_filters(df, filters)
        if working_df.empty:
            return json.dumps({"status": "empty", "message": "No rows matched the given filters.", "rows": []})

        if agg not in VALID_AGGS:
            return json.dumps({"status": "error", "message": f"Invalid agg '{agg}'. Valid: {sorted(VALID_AGGS)}"})

        # Case 1: Grouped aggregation
        if group_by:
            missing = [c for c in group_by if c not in working_df.columns]
            if missing:
                return json.dumps({"status": "error", "message": f"Unknown group_by columns: {missing}"})

            if agg == "count" and not metric_col:
                grouped = working_df.groupby(group_by, dropna=False).size().reset_index(name="row_count")
                default_sort = "row_count"
            else:
                if not metric_col or metric_col not in working_df.columns:
                    return json.dumps({"status": "error", "message": f"Valid metric_col required for agg='{agg}'."})
                out_col = f"{agg}_{metric_col}"
                grouped = (
                    working_df.groupby(group_by, dropna=False)[metric_col]
                    .agg(agg)
                    .reset_index(name=out_col)
                )
                default_sort = out_col

            target_sort = sort_by if (sort_by and sort_by in grouped.columns) else default_sort
            grouped = grouped.sort_values(by=target_sort, ascending=ascending).head(limit)

            # Round floats for clean LLM reading
            for c in grouped.select_dtypes(include="float").columns:
                grouped[c] = grouped[c].round(2)

            return json.dumps(
                {
                    "status": "ok",
                    "matched_rows": len(working_df),
                    "returned_rows": len(grouped),
                    "results": grouped.fillna("NaN").to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )

        # Case 2: Overall scalar aggregation (no group_by)
        if metric_col:
            if metric_col not in working_df.columns:
                return json.dumps({"status": "error", "message": f"Unknown metric_col '{metric_col}'."})
            val = getattr(working_df[metric_col], agg)()
            if isinstance(val, float):
                val = round(val, 2)
            return json.dumps(
                {
                    "status": "ok",
                    "matched_rows": len(working_df),
                    "metric": f"{agg}({metric_col})",
                    "value": val,
                },
                indent=2,
                default=str,
            )

        # Case 3: Filtered row preview (no group_by, no metric_col)
        if sort_by and sort_by in working_df.columns:
            working_df = working_df.sort_values(by=sort_by, ascending=ascending)
        preview = working_df.head(limit).fillna("NaN")
        return json.dumps(
            {
                "status": "ok",
                "matched_rows": len(working_df),
                "returned_rows": len(preview),
                "results": preview.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )

    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


def make_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str | None = None,
    hue_col: str | None = None,
    agg: str = "sum",
    title: str | None = None,
    filters: list[dict[str, Any]] | None = None,
    output_dir: str = "outputs",
) -> str:
    """Create a Matplotlib/Seaborn chart, save it to disk, and return the image path."""
    try:
        if chart_type not in VALID_CHARTS:
            return json.dumps({"status": "error", "message": f"Invalid chart_type '{chart_type}'. Use {sorted(VALID_CHARTS)}"})

        working_df = _apply_filters(df, filters)
        if working_df.empty:
            return json.dumps({"status": "error", "message": "No data left after applying filters."})

        for col in [c for c in [x_col, y_col, hue_col] if c]:
            if col not in working_df.columns:
                return json.dumps({"status": "error", "message": f"Column '{col}' not found."})

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        chart_path = str(Path(output_dir) / f"chart_{uuid.uuid4().hex[:8]}.png")

        fig, ax = plt.subplots(figsize=(9, 5))

        if chart_type == "hist":
            sns.histplot(data=working_df, x=x_col, hue=hue_col, bins=25, kde=True, ax=ax)

        elif chart_type == "scatter":
            if not y_col:
                plt.close(fig)
                return json.dumps({"status": "error", "message": "scatter chart requires y_col."})
            sns.scatterplot(data=working_df, x=x_col, y=y_col, hue=hue_col, alpha=0.65, ax=ax)

        elif chart_type == "box":
            if not y_col:
                plt.close(fig)
                return json.dumps({"status": "error", "message": "box chart requires y_col."})
            sns.boxplot(data=working_df, x=x_col, y=y_col, hue=hue_col, ax=ax)

        elif chart_type in {"bar", "line"}:
            group_cols = [x_col] + ([hue_col] if hue_col else [])
            if y_col:
                plot_df = working_df.groupby(group_cols, dropna=False)[y_col].agg(agg).reset_index()
                metric_name = y_col
            else:
                plot_df = working_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
                metric_name = "count"

            # Limit top 15 categories on bar charts so labels stay readable
            if chart_type == "bar" and plot_df[x_col].nunique() > 15:
                top_x = plot_df.groupby(x_col)[metric_name].sum().nlargest(15).index
                plot_df = plot_df[plot_df[x_col].isin(top_x)]

            if chart_type == "bar":
                sns.barplot(data=plot_df, x=x_col, y=metric_name, hue=hue_col, ax=ax)
            else:
                plot_df = plot_df.sort_values(by=x_col)
                sns.lineplot(data=plot_df, x=x_col, y=metric_name, hue=hue_col, marker="o", ax=ax)

        ax.set_title(title or f"{chart_type.title()} Chart: {y_col or 'Count'} by {x_col}", fontsize=13, pad=12)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        return json.dumps({"status": "ok", "chart_path": chart_path, "summary": f"Saved {chart_type} chart to {chart_path}"})

    except Exception as exc:
        plt.close("all")
        return json.dumps({"status": "error", "message": str(exc)})


if __name__ == "__main__":
    sample_df = pd.read_csv("data/demo_sales.csv")

    print("=== 1. TEST inspect_schema ===")
    print(inspect_schema(sample_df))

    print("\n=== 2. TEST query_data (Closed Won revenue by product_category) ===")
    print(
        query_data(
            sample_df,
            filters=[{"column": "deal_status", "op": "==", "value": "Closed Won"}],
            group_by=["product_category"],
            metric_col="revenue",
            agg="sum",
        )
    )

    print("\n=== 3. TEST make_chart ===")
    print(
        make_chart(
            sample_df,
            chart_type="bar",
            x_col="product_category",
            y_col="revenue",
            hue_col="deal_status",
            agg="sum",
            title="Total Revenue by Category & Deal Status",
        )
    )
