from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


RESEARCHER_STARTUP_TIMEOUT_SECONDS = 600
RESEARCHER_REQUEST_TIMEOUT_SECONDS = 900
RESEARCHER_FULL_TIMEOUT_SECONDS = 1800


class ResearcherBoundary:
    """Write-only dispatch boundary: no test metric or test path is returned."""

    def __init__(
        self,
        repo_root: Path,
        experiment_config: Path,
        status_log: Path,
        *,
        seed: int | None = None,
        startup_timeout: float = RESEARCHER_STARTUP_TIMEOUT_SECONDS,
        request_timeout: float = RESEARCHER_REQUEST_TIMEOUT_SECONDS,
        full_timeout: float = RESEARCHER_FULL_TIMEOUT_SECONDS,
    ):
        self.repo_root = repo_root.resolve()
        self.experiment_config = experiment_config.resolve()
        self.status_log = status_log
        self.seed = seed
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.full_timeout = full_timeout
        self._epoch_worker: subprocess.Popen[str] | None = None
        self._epoch_worker_disabled = False

    def _record(self, event: dict) -> None:
        self.status_log.parent.mkdir(parents=True, exist_ok=True)
        with self.status_log.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    def _worker_command(self) -> list[str]:
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "run_researcher_test.py"),
            "--mode", "epoch-stream",
            "--config", str(self.experiment_config),
        ]
        if self.seed is not None:
            command.extend(["--seed", str(self.seed)])
        return command

    def _start_epoch_worker(self) -> subprocess.Popen[str]:
        worker = self._epoch_worker
        if worker is not None and worker.poll() is None:
            return worker
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.repo_root) + os.pathsep + environment.get("PYTHONPATH", "")
        worker = subprocess.Popen(
            self._worker_command(),
            cwd=self.repo_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._epoch_worker = worker
        try:
            self._wait_for_ack(worker, "worker_ready", self.startup_timeout)
        except Exception as error:
            self._record({
                "event": "researcher_epoch_worker_start_failed",
                "pid": worker.pid,
                "error": type(error).__name__,
            })
            self.close_epoch_worker(force=True)
            raise
        self._record({"event": "researcher_epoch_worker_started", "pid": worker.pid, "ready": True, "seed": self.seed})
        return worker

    @staticmethod
    def _wait_for_ack(worker: subprocess.Popen[str], expected_event: str, timeout: float) -> dict:
        """Read a generic ACK without allowing a broken researcher to block training."""
        if worker.stdout is None:
            raise RuntimeError("RESEARCHER_STREAM_PIPE_UNAVAILABLE")
        result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def read_ack() -> None:
            try:
                while True:
                    line = worker.stdout.readline()
                    if not line:
                        raise RuntimeError("RESEARCHER_STREAM_ENDED_WITHOUT_ACK")
                    if not line.startswith("DIAGAGENT_ACK "):
                        continue
                    acknowledgement = json.loads(line[len("DIAGAGENT_ACK ") :])
                    allowed_keys = {"event", "ok", "epoch"}
                    if set(acknowledgement) - allowed_keys:
                        raise RuntimeError("RESEARCHER_ACK_CONTAINS_UNEXPECTED_FIELDS")
                    if acknowledgement.get("event") != expected_event:
                        raise RuntimeError("RESEARCHER_ACK_EVENT_MISMATCH")
                    if acknowledgement.get("ok") is not True:
                        raise RuntimeError("RESEARCHER_ACK_REPORTED_FAILURE")
                    result.put((True, acknowledgement))
                    return
            except BaseException as error:
                result.put((False, error))

        reader = threading.Thread(target=read_ack, name="researcher-ack-reader", daemon=True)
        reader.start()
        reader.join(timeout=max(0.0, float(timeout)))
        if reader.is_alive():
            raise TimeoutError(f"RESEARCHER_ACK_TIMEOUT: {expected_event}")
        succeeded, value = result.get_nowait()
        if not succeeded:
            raise value  # type: ignore[misc]
        return value  # type: ignore[return-value]

    def _stream_request(self, payload: dict, expected_event: str) -> bool:
        try:
            worker = self._start_epoch_worker()
            if worker.stdin is None or worker.stdout is None:
                raise RuntimeError("RESEARCHER_STREAM_PIPE_UNAVAILABLE")
            worker.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            worker.stdin.flush()
            self._wait_for_ack(worker, expected_event, self.request_timeout)
            return True
        except Exception as error:
            self._record({"event": "researcher_epoch_worker_failed", "error": type(error).__name__})
            self._epoch_worker_disabled = True
            self.close_epoch_worker(force=True)
            return False

    def _run(self, arguments: list[str], event: dict) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.repo_root) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, str(self.repo_root / "scripts" / "run_researcher_test.py"), *arguments]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=self.full_timeout,
            )
            self._record({**event, "exit_code": completed.returncode})
        except subprocess.TimeoutExpired:
            self._record({**event, "exit_code": 124, "error": "TimeoutExpired"})

    def epoch(self, epoch: int, checkpoint: Path, train_metrics: dict, train_valid_metrics: dict) -> None:
        if self._epoch_worker_disabled:
            self._record({
                "event": "researcher_epoch_skipped",
                "epoch": epoch,
                "reason": "worker_disabled_after_failure",
            })
            return
        success = self._stream_request(
            {
                "command": "epoch",
                "checkpoint": str(checkpoint.resolve()),
                "epoch": epoch,
                "train_metrics": train_metrics,
                "train_valid_metrics": train_valid_metrics,
            },
            "epoch_complete",
        )
        self._record({
            "event": "researcher_epoch_dispatched",
            "epoch": epoch,
            "exit_code": 0 if success else 1,
            "persistent_worker": True,
        })

    def full(self, checkpoint: Path | list[Path], source_experiment_dir: Path | None = None) -> None:
        checkpoints = [checkpoint] if isinstance(checkpoint, Path) else list(checkpoint)
        if not checkpoints:
            raise RuntimeError("RESEARCHER_CHECKPOINTS_EMPTY")
        worker = self._epoch_worker
        if not self._epoch_worker_disabled and worker is not None and worker.poll() is None:
            success = self._stream_request(
                {
                    "command": "full",
                    "checkpoints": [str(item.resolve()) for item in checkpoints],
                    "source_experiment_dir": None if source_experiment_dir is None else str(source_experiment_dir.resolve()),
                },
                "full_complete",
            )
            self._record({
                "event": "researcher_full_dispatched",
                "exit_code": 0 if success else 1,
                "persistent_worker": True,
            })
            if success:
                return
        arguments = [
            "--mode", "full",
            "--config", str(self.experiment_config),
            "--checkpoints-json", json.dumps([str(item.resolve()) for item in checkpoints]),
        ]
        if source_experiment_dir is not None:
            arguments.extend(["--source-experiment-dir", str(source_experiment_dir.resolve())])
        self._run(arguments, {"event": "researcher_full_dispatched"})

    def close_epoch_worker(self, *, force: bool = False) -> None:
        worker = self._epoch_worker
        self._epoch_worker = None
        if worker is None:
            return
        try:
            if worker.poll() is None and not force and worker.stdin is not None:
                worker.stdin.write('{"command":"close"}\n')
                worker.stdin.flush()
                worker.wait(timeout=30)
            elif worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
        except Exception:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)
        finally:
            for stream in (worker.stdin, worker.stdout):
                if stream is not None:
                    stream.close()

    def __del__(self) -> None:
        try:
            self.close_epoch_worker(force=True)
        except Exception:
            pass


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
