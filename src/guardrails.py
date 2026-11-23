import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Callable

import pandas as pd

MAX_FILE_SIZE_MB = 10
MAX_ROWS = 50_000
MAX_QUESTIONS_PER_SESSION = 15
TOOL_TIMEOUT_SECONDS = 10


def validate_and_load_csv(file_path: str | Path) -> pd.DataFrame:
    """Safely validate file size and row limits before loading a CSV into memory."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"CSV is {size_mb:.1f} MB, exceeding the {MAX_FILE_SIZE_MB} MB limit.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The uploaded CSV is empty (0 rows).")
    if len(df) > MAX_ROWS:
        raise ValueError(f"CSV has {len(df):,} rows, exceeding the {MAX_ROWS:,}-row limit.")

    return df


def run_with_timeout(func: Callable[..., str], *args: Any, timeout: int = TOOL_TIMEOUT_SECONDS, **kwargs: Any) -> str:
    """Run a tool function with a hard timeout, returning a JSON error observation on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool timed out after {timeout} seconds. Try a simpler filter or aggregation.",
                }
            )


class SessionGuard:
    """Track and enforce per-session question limits to protect API credits."""

    def __init__(self, max_questions: int = MAX_QUESTIONS_PER_SESSION) -> None:
        self.max_questions = max_questions
        self.count = 0

    def check_and_increment(self) -> tuple[bool, str]:
        if self.count >= self.max_questions:
            return False, f"Session limit reached ({self.max_questions} questions max). Refresh to start a new session."
        self.count += 1
        remaining = self.max_questions - self.count
        return True, f"Question {self.count}/{self.max_questions} ({remaining} remaining in session)."


if __name__ == "__main__":
    # Quick sanity check on our guardrails
    df = validate_and_load_csv("data/demo_sales.csv")
    print(f"1. CSV validation passed: {len(df)} rows loaded.")

    guard = SessionGuard(max_questions=2)
    print("2. Session check 1:", guard.check_and_increment())
    print("   Session check 2:", guard.check_and_increment())
    print("   Session check 3 (blocked):", guard.check_and_increment())
