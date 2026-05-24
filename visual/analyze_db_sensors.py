#!/usr/bin/env python3
"""Analyze SQLite .db files for sensor data and spectrum payloads.

This script introspects tables, reports schema, samples rows, and attempts
heuristic decoding of spectrum data blobs for later visualization.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sa


def _connect(db_path: Path) -> sa.Engine:
    return sa.create_engine(f"sqlite:///{db_path}")


def _list_tables(engine: sa.Engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"))
        return [r[0] for r in rows]


def _table_info(engine: sa.Engine, table: str) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(sa.text(f"PRAGMA table_info('{table}')"))
        return [
            {
                "cid": r[0],
                "name": r[1],
                "type": r[2],
                "notnull": r[3],
                "dflt_value": r[4],
                "pk": r[5],
            }
            for r in rows
        ]


def _row_count(engine: sa.Engine, table: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(sa.text(f"SELECT COUNT(*) FROM '{table}'")).scalar() or 0)


def _sample_table(engine: sa.Engine, table: str, limit: int) -> pd.DataFrame:
    query = f"SELECT * FROM '{table}' LIMIT {int(limit)}"
    return pd.read_sql_query(query, engine)


def _looks_hex_string(value: str) -> bool:
    s = value.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    s = s.replace(" ", "")
    if len(s) < 2 or len(s) % 2 != 0:
        return False
    return all(c in "0123456789abcdef" for c in s)


def _decode_to_bytes(value: Any) -> Optional[bytes]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        s = value.strip()
        if _looks_hex_string(s):
            s = s[2:] if s.lower().startswith("0x") else s
            s = s.replace(" ", "")
            try:
                return bytes.fromhex(s)
            except ValueError:
                return None
        # Try base64
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            return None
    return None


def _guess_blob_dtype(blob: bytes, count: Optional[int]) -> Optional[np.dtype]:
    if blob is None:
        return None
    byte_len = len(blob)
    candidates = [np.dtype("<i2"), np.dtype("<i4"), np.dtype("<f4"), np.dtype("<f8")]
    if count is not None and count > 0:
        for dt in candidates:
            if byte_len == count * dt.itemsize:
                return dt
        return None
    # No count column or invalid; try to find a reasonable length
    for dt in candidates:
        if byte_len % dt.itemsize == 0:
            length = byte_len // dt.itemsize
            if 8 <= length <= 4096:
                return dt
    return None


def _decode_blob(blob: bytes, count: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    dt = _guess_blob_dtype(blob, count)
    if dt is None:
        return None, None
    arr = np.frombuffer(blob, dtype=dt)
    if count is not None and count > 0 and len(arr) != count:
        return None, None
    return arr, str(dt)


def _detect_spectrum_table(table_infos: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    target_cols = {"ts", "dev_ms", "count", "data"}
    for table, info in table_infos.items():
        cols = {c["name"].lower() for c in info}
        if target_cols.issubset(cols):
            return table
    # Fallback: table with a "data" column and timestamp-like column
    for table, info in table_infos.items():
        cols = {c["name"].lower() for c in info}
        if "data" in cols and ("ts" in cols or "timestamp" in cols or "time" in cols):
            return table
    return None


def summarize_database(db_path: Path, out_dir: Path, sample_limit: int, export_csv: bool) -> Dict[str, Any]:
    engine = _connect(db_path)
    tables = _list_tables(engine)
    table_infos = {t: _table_info(engine, t) for t in tables}

    summary: Dict[str, Any] = {
        "db_path": str(db_path),
        "tables": {},
    }

    for table in tables:
        info = table_infos[table]
        row_count = _row_count(engine, table)
        summary["tables"][table] = {
            "row_count": row_count,
            "columns": info,
        }

        if sample_limit > 0:
            df_sample = _sample_table(engine, table, sample_limit)
            summary["tables"][table]["sample_head"] = df_sample.head(5).to_dict(orient="records")
            if export_csv:
                sample_path = out_dir / f"{db_path.stem}__{table}__sample.csv"
                df_sample.to_csv(sample_path, index=False)

    # Spectrum heuristic decoding
    spectrum_table = _detect_spectrum_table(table_infos)
    if spectrum_table:
        df_spec = _sample_table(engine, spectrum_table, sample_limit)
        decoded_rows = []
        for _, row in df_spec.iterrows():
            blob = _decode_to_bytes(row.get("data"))
            count = row.get("count") if "count" in row else None
            count = int(count) if count is not None and pd.notna(count) else None
            arr, dtype_str = _decode_blob(blob, count)
            preview = arr[:10].tolist() if arr is not None else None
            decoded_rows.append(
                {
                    "ts": row.get("ts"),
                    "dev_ms": row.get("dev_ms"),
                    "count": count,
                    "data_len": len(blob) if blob is not None else None,
                    "dtype": dtype_str,
                    "preview": preview,
                }
            )
        summary["spectrum_guess"] = {
            "table": spectrum_table,
            "decoded_preview": decoded_rows,
        }

    # Write summary report
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{db_path.stem}__schema_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SQLite sensor DB schema and spectrum blobs.")
    parser.add_argument("--db", required=True, help="Path to .db file")
    parser.add_argument("--out-dir", default="db_reports", help="Output directory for reports")
    parser.add_argument("--sample-limit", type=int, default=200, help="Rows to sample per table")
    parser.add_argument("--export-csv", action="store_true", help="Export per-table sample CSVs")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not db_path.exists():
        raise FileNotFoundError(db_path)

    print(f"Analyzing DB: {db_path}")
    summary = summarize_database(db_path, out_dir, args.sample_limit, args.export_csv)

    print("\nTables:")
    for name, meta in summary["tables"].items():
        print(f"- {name}: {meta['row_count']} rows, {len(meta['columns'])} columns")

    if "spectrum_guess" in summary:
        guess = summary["spectrum_guess"]
        print("\nSpectrum guess:")
        print(f"- table: {guess['table']}")
        if guess["decoded_preview"]:
            first = guess["decoded_preview"][0]
            print(f"- first row dtype: {first.get('dtype')}, data_len: {first.get('data_len')}")

    print(f"\nReport written to: {out_dir}")


if __name__ == "__main__":
    main()
