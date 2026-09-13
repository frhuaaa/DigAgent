from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


class ResearcherBoundary:
    """Write-only dispatch boundary: no test metric or test path is returned."""

    def __init__(self, repo_root: Path, experiment_config: Path, status_log: Path):
        self.repo_root = repo_root.resolve()
        self.experiment_config = experiment_config.resolve()
        self.status_log = status_log

    def _run(self, arguments: list[str], event: dict) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.repo_root) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, str(self.repo_root / "scripts" / "run_researcher_test.py"), *arguments]
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.status_log.parent.mkdir(parents=True, exist_ok=True)
        with self.status_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({**event, "exit_code": completed.returncode}, sort_keys=True, separators=(",", ":")) + "\n")

    def epoch(self, epoch: int, checkpoint: Path, train_metrics: dict, valid_metrics: dict) -> None:
        self._run(
            [
                "--mode", "epoch",
                "--config", str(self.experiment_config),
                "--checkpoint", str(checkpoint.resolve()),
                "--epoch", str(epoch),
                "--train-metrics-json", json.dumps(train_metrics, separators=(",", ":")),
                "--valid-metrics-json", json.dumps(valid_metrics, separators=(",", ":")),
            ],
            {"event": "researcher_epoch_dispatched", "epoch": epoch},
        )

    def full(self, checkpoint: Path, source_experiment_dir: Path | None = None) -> None:
        arguments = [
            "--mode", "full",
            "--config", str(self.experiment_config),
            "--checkpoint", str(checkpoint.resolve()),
        ]
        if source_experiment_dir is not None:
            arguments.extend(["--source-experiment-dir", str(source_experiment_dir.resolve())])
        self._run(arguments, {"event": "researcher_full_dispatched"})


def dispatch_researcher_outputs_audit(repo_root: Path, run_root: Path) -> None:
    """Run a post-trajectory audit without returning test-derived state."""
    command = [
        sys.executable,
        str(repo_root.resolve() / "scripts" / "audit_researcher_outputs.py"),
        "--run-root",
        str(run_root.resolve()),
    ]
    subprocess.run(
        command,
        cwd=repo_root.resolve(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
