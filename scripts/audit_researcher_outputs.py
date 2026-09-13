from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.io_utils import atomic_write_json, load_json


REQUIRED_TEST_ARTIFACTS = (
    "result.json",
    "test_alpha.csv",
    "portfolio_daily_diagnostics.csv",
    "researcher_complete.json",
    "epoch_metrics.jsonl",
)


def build_researcher_outputs_audit(run_root: Path) -> dict:
    run_root = run_root.resolve()
    records = []
    experiments_root = run_root / "experiments"
    for experiment_dir in sorted(experiments_root.glob("EXP_[0-9][0-9][0-9]")):
        if not (experiment_dir / "logs" / "execution_complete.json").is_file():
            continue
        test_dir = experiment_dir / "test"
        missing = [
            name
            for name in REQUIRED_TEST_ARTIFACTS
            if not (test_dir / name).is_file() or (test_dir / name).stat().st_size == 0
        ]
        completion_valid = False
        completion_path = test_dir / "researcher_complete.json"
        if completion_path.is_file() and completion_path.stat().st_size > 0:
            try:
                completion_valid = load_json(completion_path).get("status") == "COMPLETE"
            except Exception:
                completion_valid = False
        if not completion_valid and "researcher_complete.json" not in missing:
            missing.append("researcher_complete.json:invalid_status")
        records.append({
            "experiment_id": experiment_dir.name,
            "complete": not missing,
            "missing_or_invalid": sorted(missing),
        })
    incomplete = [item["experiment_id"] for item in records if not item["complete"]]
    return {
        "researcher_only": True,
        "adaptive_state_affected": False,
        "executed_experiment_count": len(records),
        "complete_experiment_count": len(records) - len(incomplete),
        "researcher_outputs_complete": bool(records) and not incomplete,
        "incomplete_experiments": incomplete,
        "experiments": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit isolated researcher-only test outputs.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output = run_root / "test" / "researcher_outputs_audit.json"
    atomic_write_json(output, build_researcher_outputs_audit(run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
