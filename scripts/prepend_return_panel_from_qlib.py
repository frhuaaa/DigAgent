from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qlib_code(panel_code: str) -> str:
    code, suffix = panel_code.split("_", 1)
    if suffix == "XSHE":
        return f"SZ{code}"
    if suffix == "XSHG":
        return f"SH{code}"
    raise ValueError(f"Unsupported panel instrument: {panel_code}")


def first_column(path: Path) -> pd.Index:
    values = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str)
    return pd.Index(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepend missing close-to-close return dates from Qlib without rewriting existing rows."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    repo = args.repo.resolve()
    provider = repo / "data" / "cn_data"
    panel = repo / "data" / "portfolio" / "c_2_c_1D.csv"
    masks = [
        repo / "data" / "portfolio" / "mask_limit_up_1D.csv",
        repo / "data" / "portfolio" / "mask_limit_down_1D.csv",
    ]

    with panel.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
    panel_columns = header[1:]
    if not panel_columns or len(panel_columns) != len(set(panel_columns)):
        raise RuntimeError("Return panel header is empty or contains duplicate instruments")

    existing_dates = first_column(panel)
    first_existing = pd.to_datetime(existing_dates[0])
    requested_start = pd.Timestamp(args.start)
    if requested_start >= first_existing:
        print("Return panel already starts on or before the requested period.")
        return 0

    qlib.init(provider_uri=str(provider), region="cn", kernels=8)
    calendar = pd.DatetimeIndex(
        D.calendar(
            start_time=requested_start - timedelta(days=40),
            end_time=first_existing,
            freq="day",
        )
    )
    target_dates = calendar[(calendar >= requested_start) & (calendar < first_existing)]
    if target_dates.empty:
        raise RuntimeError("No Qlib trading dates found to prepend")
    prior_dates = calendar[calendar < target_dates[0]]
    if prior_dates.empty:
        raise RuntimeError("No prior Qlib trading day is available for the first return")
    fetch_start = prior_dates[-1]

    target_keys = pd.Index(target_dates.strftime("%Y%m%d"))
    for mask in masks:
        missing_dates = target_keys.difference(first_column(mask))
        if not missing_dates.empty:
            raise RuntimeError(f"{mask.name} lacks {len(missing_dates)} target dates; first={missing_dates[0]}")

    qlib_codes = [qlib_code(column) for column in panel_columns]
    reverse = dict(zip(qlib_codes, panel_columns))
    matrix = pd.DataFrame(index=target_dates, columns=panel_columns, dtype="float64")
    for offset in range(0, len(qlib_codes), args.chunk_size):
        chunk = qlib_codes[offset : offset + args.chunk_size]
        close = D.features(
            chunk,
            ["$close"],
            start_time=fetch_start,
            end_time=target_dates[-1],
            freq="day",
        )
        if close.empty:
            continue
        for instrument, values in close["$close"].groupby(level="instrument", sort=False):
            column = reverse.get(str(instrument))
            if column is None:
                continue
            series = values.droplevel("instrument").sort_index()
            returns = series.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA)
            matrix.loc[:, column] = returns.reindex(target_dates).astype("float64")

    finite_count = int(matrix.notna().sum().sum())
    dates_with_values = int(matrix.notna().any(axis=1).sum())
    if dates_with_values != len(target_dates):
        missing = matrix.index[~matrix.notna().any(axis=1)]
        raise RuntimeError(f"Generated return panel has empty dates: {missing.strftime('%Y%m%d').tolist()[:5]}")

    created = datetime.now(timezone.utc)
    repair_dir = repo / "data_repairs" / f"c2c_prepend_2021_{created.strftime('%Y%m%dT%H%M%SZ')}"
    repair_dir.mkdir(parents=True, exist_ok=False)
    backup = repair_dir / "c_2_c_1D.csv.before"
    shutil.copy2(panel, backup)
    before_hash = sha256_file(panel)

    temp = panel.with_suffix(".csv.prepending")
    with panel.open("rb") as source, temp.open("wb") as target:
        header_line = source.readline()
        target.write(header_line)
        for date, row in matrix.iterrows():
            fields = []
            for value in row.to_numpy():
                fields.append(repr(float(value)) if pd.notna(value) and math.isfinite(float(value)) else "")
            target.write((date.strftime("%Y%m%d") + "," + ",".join(fields) + "\n").encode("utf-8"))
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temp, panel)

    repaired_dates = first_column(panel)
    expected_prefix = target_keys.append(pd.Index([existing_dates[0]]))
    if not repaired_dates[: len(expected_prefix)].equals(expected_prefix):
        raise RuntimeError("Repaired return-panel date prefix failed validation")

    manifest = {
        "created_at_utc": created.isoformat(),
        "method": "Qlib adjusted $close pct_change(fill_method=None), stored on realized-return date",
        "source_provider": provider.as_posix(),
        "target": panel.as_posix(),
        "backup": backup.as_posix(),
        "requested_start": requested_start.strftime("%Y-%m-%d"),
        "first_added_date": target_dates[0].strftime("%Y-%m-%d"),
        "last_added_date": target_dates[-1].strftime("%Y-%m-%d"),
        "rows_added": len(target_dates),
        "columns": len(panel_columns),
        "finite_values_added": finite_count,
        "sha256_before": before_hash,
        "sha256_after": sha256_file(panel),
    }
    (repair_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
