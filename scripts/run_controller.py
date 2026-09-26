from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.slurm_controller import ControllerOptions, SlurmTrajectoryController
from core.configuration import resolve_run_layout


def materialize_on_worker(submit_path: Path, slurm_script: Path) -> None:
    """Keep Qlib/Alpha158 materialization off the login node."""
    resolved_submit = submit_path.resolve()
    with resolved_submit.open("r", encoding="utf-8") as handle:
        seed = json.load(handle)
    task_name = seed["task"]["name"]
    _, run_root, _ = resolve_run_layout(REPO_ROOT, seed, resolved_submit)
    initial_path = run_root / "configs" / "initial.json"
    if initial_path.is_file():
        return
    command = [
        "sbatch", "--wait", "--parsable",
        "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G",
        "--job-name", f"diagagent-{task_name}-materialize",
        str(slurm_script.resolve()), str(resolved_submit), "EXP_000", "materialize", "support",
    ]
    print("Submitting initial Qlib/RASS materialization to Slurm", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"INITIAL_MATERIALIZATION_FAILED: {detail}")
    print(f"materialization_job={completed.stdout.strip()}", flush=True)
    if not initial_path.is_file():
        raise RuntimeError("INITIAL_MATERIALIZATION_COMPLETED_WITHOUT_INITIAL_CONFIG")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the login-node Agent controller and submit offline Slurm workers one round at a time."
    )
    parser.add_argument("--submit", required=True)
    parser.add_argument("--slurm-script", default="slurm/diagagent_worker.slurm")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--agent-retry-seconds", type=float, default=30.0)
    parser.add_argument("--max-worker-attempts", type=int, default=2)
    parser.add_argument(
        "--run-timestamp",
        help="MMDDHHMM run directory; omit for the current local launch minute, or pass the original value to resume",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.agent_retry_seconds <= 0:
        parser.error("poll intervals must be positive")
    if args.max_worker_attempts < 1:
        parser.error("--max-worker-attempts must be at least 1")
    run_timestamp = args.run_timestamp or datetime.now().astimezone().strftime("%m%d%H%M")
    if not re.fullmatch(r"\d{8}", run_timestamp):
        parser.error("--run-timestamp must be MMDDHHMM")
    os.environ["DIAGAGENT_RUN_TIMESTAMP"] = run_timestamp
    print(f"run_timestamp={run_timestamp}", flush=True)
    submit_path = Path(args.submit)
    slurm_script = Path(args.slurm_script)
    materialize_on_worker(submit_path, slurm_script)
    controller = SlurmTrajectoryController(
        REPO_ROOT,
        submit_path,
        slurm_script,
        ControllerOptions(
            poll_seconds=args.poll_seconds,
            agent_retry_seconds=args.agent_retry_seconds,
            max_worker_attempts=args.max_worker_attempts,
        ),
    )
    final_path = controller.run()
    print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
