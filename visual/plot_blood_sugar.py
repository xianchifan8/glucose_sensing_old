#!/usr/bin/env python3
"""Plot blood sugar from a SQLite DB table."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_blood_sugar(db_path: Path, table: str = "blood_sugar") -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
    if "blood_sugar" not in df.columns:
        raise ValueError("blood_sugar column not found in table")
    if "create_time" not in df.columns:
        raise ValueError("create_time column not found in table")
    return df


def _infer_epoch_unit(series: pd.Series) -> str:
    max_val = series.dropna().max()
    if max_val > 1e12:
        return "ms"
    if max_val > 1e10:
        return "s"
    return "s"


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unit = _infer_epoch_unit(df["create_time"])
    df["dt"] = pd.to_datetime(df["create_time"], unit=unit, utc=True).dt.tz_convert(None)
    df["date"] = df["dt"].dt.date
    return df


def _print_date_options(dates: list[str]) -> None:
    if not dates:
        print("No dates available in data.")
        return
    print(f"Available date range: {dates[0]} to {dates[-1]}")
    if len(dates) <= 20:
        print("Available dates:", ", ".join(dates))
    else:
        print("Available dates (first 20):", ", ".join(dates[:20]), "...")


def plot_blood_sugar(df: pd.DataFrame, selected_date: str, out_path: Path | None = None) -> None:
    selected_dt = pd.to_datetime(selected_date, format="%Y-%m-%d").date()
    day_df = df[df["date"] == selected_dt]
    if day_df.empty:
        raise ValueError(f"No data found for date: {selected_date}")

    x = day_df["dt"]

    plt.figure(figsize=(10, 4))
    plt.plot(x, day_df["blood_sugar"], label="Blood Sugar")
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Blood Sugar (mmol/L)")
    plt.title(f"{selected_date} Blood Sugar")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=150)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot blood sugar from a SQLite database.")
    parser.add_argument("--db", required=True, help="Path to .db file")
    parser.add_argument("--table", default="blood_sugar", help="Table name (default: blood_sugar)")
    parser.add_argument("--out", default=None, help="Optional output image path")
    args = parser.parse_args()

    db_path = Path(args.db)
    df = load_blood_sugar(db_path, table=args.table)
    df = _ensure_datetime(df)

    dates = sorted({d.isoformat() for d in df["date"].dropna().unique()})
    _print_date_options(dates)
    if not dates:
        return

    selected_date = input("Enter date to plot (YYYY-MM-DD): ").strip()
    if selected_date not in dates:
        raise ValueError(f"Invalid date selection: {selected_date}")

    out_path = Path(args.out) if args.out else None
    plot_blood_sugar(df, selected_date=selected_date, out_path=out_path)


if __name__ == "__main__":
    main()
