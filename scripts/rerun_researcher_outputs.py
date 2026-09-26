from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.io_utils import load_json


def resolve_training_source(run_root: Path, experiment_dir: Path) -> Path | None:
    """Follow RAPA reuse links to the experiment that owns the trained checkpoint."""
    current = experiment_dir
    visited: set[str] = set()
    while True:
        if current.name in visited:
            raise RuntimeError(f"checkpoint reuse cycle detected at {current.name}")
        visited.add(current.name)
        selected_path = current / "artifacts" / "selected_checkpoint.json"
        if not selected_path.is_file():
            return None if current == experiment_dir else current
        reused_from = load_json(selected_path).get("reused_from")
        if not reused_from:
            return None if current == experiment_dir else current
        current = run_root / "experiments" / reused_from
        if not current.is_dir():
            raise RuntimeError(f"checkpoint reuse source does not exist: {current}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild isolated researcher outputs without adaptive side effects.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--experiment",
        help="Optionally rebuild one EXP_NNN directory instead of every executed experiment.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the run-level audit (useful for parallel per-experiment jobs).",
    )
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    failures: list[str] = []
    experiment_dirs = sorted((run_root / "experiments").glob("EXP_[0-9][0-9][0-9]"))
    if args.experiment:
        experiment_dirs = [path for path in experiment_dirs if path.name == args.experiment]
        if not experiment_dirs:
            parser.error(f"experiment not found under run root: {args.experiment}")
    for experiment_dir in experiment_dirs:
        marker_path = experiment_dir / "logs" / "execution_complete.json"
        if not marker_path.is_file():
            continue
        marker = load_json(marker_path)
        records = marker.get("selected_checkpoints")
        checkpoints = (
            [(REPO_ROOT / item["repository_path"]).resolve() for item in records]
            if records is not None
            else [(REPO_ROOT / marker["selected_checkpoint"]).resolve()]
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_researcher_test.py"),
            "--mode", "full",
            "--config", str(experiment_dir / "config.json"),
            "--checkpoints-json", json.dumps([str(checkpoint) for checkpoint in checkpoints]),
        ]
        training_source = resolve_training_source(run_root, experiment_dir)
        if training_source is not None:
            command.extend(["--source-experiment-dir", str(training_source)])
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        status = "OK" if completed.returncode == 0 else f"FAILED({completed.returncode})"
        print(f"{experiment_dir.name}: {status}", flush=True)
        if completed.returncode != 0:
            failures.append(experiment_dir.name)
    if not args.skip_audit:
        subprocess.run([
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_researcher_outputs.py"),
            "--run-root", str(run_root),
        ], cwd=REPO_ROOT, check=False)
    if failures:
        print("researcher failures: " + ",".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
