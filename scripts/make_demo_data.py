import numpy as np
import pandas as pd


def generate_demo_sales(path: str = "data/demo_sales.csv", n_rows: int = 2500) -> None:
    rng = np.random.default_rng(42)

    dates = pd.date_range("2025-01-01", "2026-10-31", freq="D")
    regions = ["North", "South", "East", "West", " North "]  # intentional whitespace quirk
    categories = ["Cloud Analytics", "Security Suite", "Data Pipeline", "AI Copilot"]
    channels = ["Direct", "Partner", "Self-Serve"]
    statuses = ["Closed Won", "Closed Won", "Closed Won", "Refunded", "Pending"]

    units = rng.integers(1, 25, size=n_rows)
    unit_price = rng.choice([120.0, 250.0, 499.0, 999.0, 1499.0], size=n_rows)
    discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.25], size=n_rows)
    revenue = np.round(units * unit_price * (1 - discount), 2)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{10000 + i}" for i in range(n_rows)],
            "order_date": rng.choice(dates, size=n_rows),
            "region": rng.choice(regions, size=n_rows),
            "product_category": rng.choice(categories, size=n_rows),
            "sales_channel": rng.choice(channels, size=n_rows),
            "units_sold": units.astype(float),
            "unit_price": unit_price,
            "discount_rate": discount,
            "revenue": revenue,
            "deal_status": rng.choice(statuses, size=n_rows),
        }
    )

    # Inject realistic missing values (~4% across key columns)
    for col in ["region", "units_sold", "revenue", "sales_channel"]:
        mask = rng.random(n_rows) < 0.04
        df.loc[mask, col] = np.nan

    df = df.sort_values("order_date").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows and {len(df.columns)} columns to {path}")
    print("\nMissing values per column:\n", df.isna().sum())


if __name__ == "__main__":
    generate_demo_sales()
