from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path


TARGET_DATE = "2025-04-16"
RAW_CLOSE = 5835.35
SOURCE_URLS = [
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000852,day,2025-04-10,2025-04-18,20,qfq",
    "https://file.iyanbao.com/pdf/ea241-0ad8d34b-5a17-42e2-aacc-3142f49cb32d.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_float32(path: Path, position: int) -> float:
    with path.open("rb") as stream:
        stream.seek(position * 4)
        return struct.unpack("<f", stream.read(4))[0]


def write_float32(path: Path, position: int, value: float) -> None:
    with path.open("r+b") as stream:
        stream.seek(position * 4)
        stream.write(struct.pack("<f", value))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    data_root = (repo / "data" / "cn_data").resolve()
    feature_dir = (data_root / "features" / "sh000852").resolve()
    if not feature_dir.is_relative_to(data_root):
        raise RuntimeError("Resolved feature directory escaped data root")

    calendar_path = data_root / "calendars" / "day.txt"
    calendar = [line.strip() for line in calendar_path.read_text().splitlines() if line.strip()]
    calendar_index = calendar.index(TARGET_DATE)

    paths = {
        "close": feature_dir / "close.day.bin",
        "adjclose": feature_dir / "adjclose.day.bin",
        "factor": feature_dir / "factor.day.bin",
    }
    starts = {name: int(read_float32(path, 0)) for name, path in paths.items()}
    if len(set(starts.values())) != 1:
        raise RuntimeError(f"Feature files have inconsistent start indices: {starts}")
    position = 1 + calendar_index - starts["close"]

    old_close = read_float32(paths["close"], position)
    old_adjclose = read_float32(paths["adjclose"], position)
    factor = read_float32(paths["factor"], position)
    scaled_close = RAW_CLOSE * factor

    expected_scaled = struct.unpack("<f", struct.pack("<f", scaled_close))[0]
    expected_adjclose = struct.unpack("<f", struct.pack("<f", RAW_CLOSE))[0]
    already_repaired = (
        math.isclose(old_close, expected_scaled, rel_tol=0.0, abs_tol=1e-7)
        and math.isclose(old_adjclose, expected_adjclose, rel_tol=0.0, abs_tol=1e-3)
    )
    if not already_repaired and (not math.isnan(old_close) or not math.isnan(old_adjclose)):
        raise RuntimeError("Target cells are not both missing; refusing to overwrite existing data")

    repair_dir = repo / "data_repairs" / "sh000852_20250416"
    backup_dir = repair_dir / "before"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for field in ("close", "adjclose"):
        backup = backup_dir / paths[field].name
        if not backup.exists():
            shutil.copy2(paths[field], backup)

    before_hashes = {field: sha256(paths[field]) for field in ("close", "adjclose")}
    if not already_repaired:
        write_float32(paths["close"], position, scaled_close)
        write_float32(paths["adjclose"], position, RAW_CLOSE)

    observed_close = read_float32(paths["close"], position)
    observed_adjclose = read_float32(paths["adjclose"], position)
    if not math.isclose(observed_close, expected_scaled, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError("Stored normalized close failed verification")
    if not math.isclose(observed_adjclose, expected_adjclose, rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError("Stored raw close failed verification")

    manifest = {
        "instrument": "SH000852",
        "date": TARGET_DATE,
        "source_close": RAW_CLOSE,
        "factor": factor,
        "stored_close": observed_close,
        "stored_adjclose": observed_adjclose,
        "calendar_index": calendar_index,
        "binary_position": position,
        "source_urls": SOURCE_URLS,
        "before_sha256": before_hashes,
        "after_sha256": {field: sha256(paths[field]) for field in ("close", "adjclose")},
        "backup_directory": str(backup_dir.relative_to(repo)).replace("\\", "/"),
    }
    (repair_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
