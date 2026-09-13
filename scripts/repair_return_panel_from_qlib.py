from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
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


def panel_code(instrument: str) -> str:
    if instrument.startswith("SZ"):
        return f"{instrument[2:]}_XSHE"
    if instrument.startswith("SH"):
        return f"{instrument[2:]}_XSHG"
    raise ValueError(f"Unsupported Qlib instrument: {instrument}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append missing Qlib close-to-close returns to the frozen portfolio panel.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--market", default="csi1000")
    args = parser.parse_args()

    repo = args.repo.resolve()
    provider = repo / "data" / "cn_data"
    panel = repo / "data" / "portfolio" / "c_2_c_1D.csv"
    masks = [
        repo / "data" / "portfolio" / "mask_limit_up_1D.csv",
        repo / "data" / "portfolio" / "mask_limit_down_1D.csv",
    ]

    with panel.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    dates = pd.read_csv(panel, usecols=[0]).iloc[:, 0].astype(str)
    start, end = pd.to_datetime(dates.iloc[0]), pd.to_datetime(dates.iloc[-1])
    existing = set(header[1:])

    qlib.init(provider_uri=str(provider), region="cn", kernels=1)
    market = D.instruments(market=args.market)
    spans = D.list_instruments(market, start_time=start, end_time=end, as_list=False)
    missing_instruments = sorted(instrument for instrument in spans if panel_code(instrument) not in existing)
    missing_columns = [panel_code(instrument) for instrument in missing_instruments]
    if not missing_instruments:
        print("Return panel already covers the selected market.")
        return 0

    for mask in masks:
        with mask.open("r", encoding="utf-8-sig", newline="") as handle:
            mask_columns = set(next(csv.reader(handle))[1:])
        absent = sorted(set(missing_columns) - mask_columns)
        if absent:
            raise RuntimeError(f"Mask {mask.name} lacks required columns: {absent}")

    close = D.features(missing_instruments, ["$close"], start_time=start, end_time=end, freq="day")["$close"]
    returns_by_date: dict[str, dict[str, float]] = {}
    non_null_counts: dict[str, int] = {}
    for instrument, column in zip(missing_instruments, missing_columns):
        series = close.xs(instrument, level="instrument").sort_index()
        values = series.pct_change(fill_method=None).replace([float("inf"), float("-inf")], pd.NA)
        mapped = {pd.Timestamp(date).strftime("%Y%m%d"): float(value) for date, value in values.dropna().items()}
        if not mapped:
            raise RuntimeError(f"No finite close-to-close returns for {instrument}")
        returns_by_date[column] = mapped
        non_null_counts[column] = len(mapped)

    repair_dir = repo / "data_repairs" / "20260909_csi1000_return_panel"
    repair_dir.mkdir(parents=True, exist_ok=True)
    backup = repair_dir / "c_2_c_1D.csv.before"
    if not backup.exists():
        shutil.copy2(panel, backup)
    before_hash = sha256_file(panel)

    temp = panel.with_suffix(".csv.repairing")
    with panel.open("r", encoding="utf-8-sig", newline="") as source, temp.open("w", encoding="utf-8", newline="") as target:
        first = source.readline().rstrip("\r\n")
        target.write(first + "," + ",".join(missing_columns) + "\n")
        for line in source:
            stripped = line.rstrip("\r\n")
            date = stripped.split(",", 1)[0]
            appended = []
            for column in missing_columns:
                value = returns_by_date[column].get(date)
                appended.append("" if value is None or not math.isfinite(value) else repr(value))
            target.write(stripped + "," + ",".join(appended) + "\n")
    os.replace(temp, panel)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "market": args.market,
        "method": "Qlib adjusted $close pct_change(fill_method=None), stored on the realized-return date",
        "source_provider": provider.as_posix(),
        "target": panel.as_posix(),
        "backup": backup.as_posix(),
        "columns_added": missing_columns,
        "finite_values_added": non_null_counts,
        "sha256_before": before_hash,
        "sha256_after": sha256_file(panel),
    }
    (repair_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
