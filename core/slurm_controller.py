from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core.errors import ContractError, ExternalServiceError, OfflineEvidenceRequired, ResumeError, WorkerError
from core.io_utils import atomic_write_json, load_json
from core.orchestrator import Orchestrator
from core.process_lock import advisory_lock
from core.researcher_boundary import dispatch_researcher_outputs_audit


ACTIVE_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING",
    "REQUEUED", "REQUEUE_FED", "SIGNALING", "STAGE_OUT",
}
SUCCESS_STATES = {"COMPLETED"}
WORKER_PROFILES = {"training", "portfolio", "support"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_state(value: str) -> str:
    return value.strip().upper().split()[0].rstrip("+") if value.strip() else "UNKNOWN"


class SlurmClient:
    def __init__(self, repo_root: Path, script: Path, runner: Callable = subprocess.run):
        self.repo_root = repo_root.resolve()
        self.script = script.resolve()
        self.runner = runner
        (self.repo_root / "slurm_logs").mkdir(parents=True, exist_ok=True)
        if not self.script.is_file():
            raise ResumeError(f"SLURM_SCRIPT_MISSING: {self.script}")

    def _run(self, command: list[str]) -> subprocess.CompletedProcess:
        return self.runner(
            command,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

    def submit(
        self,
        submit_path: Path,
        experiment_id: str,
        job_name: str,
        stage: str,
        worker_profile: str = "support",
    ) -> str:
        if worker_profile not in WORKER_PROFILES:
            raise WorkerError(f"WORKER_PROFILE_INVALID: {worker_profile}")
        if worker_profile == "portfolio":
            # This cluster permits the account only on the gpu partition.  An empty
            # command-line GRES value overrides the script's default gpu:3 request.
            resource_override = ["--gres=", "--cpus-per-task=8", "--mem=64G"]
        elif worker_profile == "training":
            resource_override = []
        else:
            resource_override = ["--gres=gpu:1", "--cpus-per-task=8", "--mem=64G"]
        completed = self._run([
            "sbatch", "--parsable", *resource_override, "--job-name", job_name,
            str(self.script), str(submit_path.resolve()), experiment_id, stage, worker_profile,
        ])
        if completed.returncode != 0:
            raise WorkerError(f"SBATCH_FAILED: {completed.stderr.strip() or completed.stdout.strip()}")
        job_id = completed.stdout.strip().splitlines()[-1].split(";", 1)[0].strip()
        if not job_id.isdigit():
            raise WorkerError(f"SBATCH_JOB_ID_INVALID: {completed.stdout.strip()}")
        return job_id

    def find_active_job(self, job_name: str) -> str | None:
        completed = self._run(["squeue", "--noheader", "--name", job_name, "--format", "%A|%T"])
        if completed.returncode != 0:
            return None
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split("|", 1)]
            if len(fields) == 2 and fields[0].isdigit() and _base_state(fields[1]) in ACTIVE_STATES:
                return fields[0]
        return None

    def status(self, job_id: str) -> tuple[str, str | None]:
        completed = self._run([
            "sacct", "-X", "--noheader", "--parsable2", "--jobs", job_id,
            "--format", "JobIDRaw,State,ExitCode",
        ])
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split("|")]
                if len(fields) >= 3 and fields[0] == job_id:
                    return _base_state(fields[1]), fields[2] or None
        queued = self._run(["squeue", "--noheader", "--jobs", job_id, "--format", "%T"])
        if queued.returncode == 0 and queued.stdout.strip():
            return _base_state(queued.stdout.splitlines()[0]), None
        return "UNKNOWN", None


@dataclass
class ControllerOptions:
    poll_seconds: float = 30.0
    agent_retry_seconds: float = 30.0
    max_worker_attempts: int = 2
    max_unknown_polls: int = 20


class SlurmTrajectoryController:
    """Login-node controller. It never performs model execution itself."""

    def __init__(
        self,
        repo_root: Path,
        submit_path: Path,
        slurm_script: Path,
        options: ControllerOptions | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        slurm: SlurmClient | None = None,
    ):
        self.repo_root = repo_root.resolve()
        self.submit_path = submit_path.resolve()
        self.orchestrator = Orchestrator(self.repo_root, self.submit_path)
        self.options = options or ControllerOptions()
        self.sleeper = sleeper
        self.slurm = slurm or SlurmClient(self.repo_root, slurm_script)
        self.status_path = self.orchestrator.run_root / "logs" / "controller_status.json"

    def _status(self, phase: str, **fields: object) -> None:
        atomic_write_json(self.status_path, {
            "phase": phase,
            "updated_at": _now(),
            "contains_test_derived_data": False,
            **fields,
        })

    def _dispatch_path(self, experiment_id: str, stage: str) -> Path:
        suffix = stage.replace("-", "_")
        return self.orchestrator.experiments_dir / experiment_id / "logs" / f"dispatch_{suffix}.json"

    def _stage_complete(self, experiment_id: str, stage: str) -> bool:
        filename = "execution_complete.json" if stage == "experiment" else "rass_shortlist_evidence.json"
        return (self.orchestrator.experiments_dir / experiment_id / "logs" / filename).is_file()

    def _worker_profile(self, experiment_id: str, stage: str) -> str:
        if stage != "experiment":
            return "support"
        decision_path = self.orchestrator.experiments_dir / experiment_id / "decision.json"
        if not decision_path.is_file():
            return "training"
        decision = load_json(decision_path)
        specialist = decision.get("specialist")
        return "portfolio" if isinstance(specialist, dict) and specialist.get("agent") == "RAPA" else "training"

    def _wait_for_worker(self, experiment_id: str, stage: str = "experiment") -> None:
        if self._stage_complete(experiment_id, stage):
            return
        dispatch_path = self._dispatch_path(experiment_id, stage)
        dispatch = load_json(dispatch_path) if dispatch_path.is_file() else {}
        attempt = int(dispatch.get("attempt", 0))
        stage_tag = "exec" if stage == "experiment" else "rass"
        worker_profile = self._worker_profile(experiment_id, stage)
        job_name = f"diagagent-{experiment_id.lower()}-{stage_tag}"
        job_id = str(dispatch.get("job_id", "")) or None
        if not job_id and dispatch.get("status") == "SUBMITTING":
            job_id = self.slurm.find_active_job(job_name)

        while True:
            if self._stage_complete(experiment_id, stage):
                dispatch.update({"status": "COMPLETE", "completed_at": _now()})
                atomic_write_json(dispatch_path, dispatch)
                return
            if not job_id:
                if attempt >= self.options.max_worker_attempts:
                    raise WorkerError(f"WORKER_ATTEMPTS_EXHAUSTED: {experiment_id}")
                attempt += 1
                dispatch = {
                    "experiment_id": experiment_id,
                    "stage": stage,
                    "worker_profile": worker_profile,
                    "job_name": job_name,
                    "attempt": attempt,
                    "status": "SUBMITTING",
                    "submitted_at": _now(),
                    "contains_test_derived_data": False,
                }
                atomic_write_json(dispatch_path, dispatch)
                job_id = self.slurm.submit(
                    self.submit_path,
                    experiment_id,
                    job_name,
                    stage,
                    worker_profile,
                )
                dispatch.update({"job_id": job_id, "status": "SUBMITTED"})
                atomic_write_json(dispatch_path, dispatch)

            unknown_polls = 0
            while True:
                state, exit_code = self.slurm.status(job_id)
                dispatch.update({
                    "job_id": job_id,
                    "slurm_state": state,
                    "exit_code": exit_code,
                    "status": "RUNNING" if state in ACTIVE_STATES else state,
                    "last_polled_at": _now(),
                })
                atomic_write_json(dispatch_path, dispatch)
                self._status("WAITING_WORKER", experiment_id=experiment_id, job_id=job_id, slurm_state=state)
                if self._stage_complete(experiment_id, stage):
                    dispatch.update({"status": "COMPLETE", "completed_at": _now()})
                    atomic_write_json(dispatch_path, dispatch)
                    return
                if state in SUCCESS_STATES:
                    raise WorkerError(f"WORKER_COMPLETED_WITHOUT_MARKER: {experiment_id} job={job_id}")
                if state not in ACTIVE_STATES and state != "UNKNOWN":
                    job_id = None
                    break
                if state == "UNKNOWN":
                    unknown_polls += 1
                    if unknown_polls >= self.options.max_unknown_polls:
                        raise WorkerError(f"WORKER_STATUS_UNKNOWN: {experiment_id} job={job_id}")
                else:
                    unknown_polls = 0
                self.sleeper(self.options.poll_seconds)

    def run(self) -> Path:
        lock_path = self.orchestrator.run_root / "logs" / "controller.lock"
        with advisory_lock(lock_path, blocking=False):
            while True:
                state = self.orchestrator.store.state()
                if state["status"] == "FINISHED":
                    dispatch_researcher_outputs_audit(self.repo_root, self.orchestrator.run_root)
                    final_path = self.orchestrator.run_root / "configs" / "final_frozen.json"
                    self._status("FINISHED", final_path=str(final_path))
                    return final_path
                if state["status"] == "FAILED":
                    raise ContractError(f"TRAJECTORY_ALREADY_FAILED: {state.get('failure_reason')}")
                if state["accepted_experiment_id"] is None:
                    experiment_id = self.orchestrator.prepare_round0()
                    self._wait_for_worker(experiment_id)
                    self.orchestrator.finalize_round0()
                    continue

                round_number = int(state["next_round"])
                if round_number > int(self.orchestrator.initial_config["task"]["trials"]):
                    final_path = self.orchestrator._finish("ADAPTIVE_ROUND_BUDGET_EXHAUSTED")
                    self._status("FINISHED", final_path=str(final_path))
                    return final_path
                try:
                    self._status("PREPARING", round=round_number)
                    prepared = self.orchestrator.prepare_adaptive_round(round_number, defer_rass_evidence=True)
                except OfflineEvidenceRequired:
                    experiment_id = f"EXP_{round_number:03d}"
                    self._status("WAITING_RASS_EVIDENCE", round=round_number, experiment_id=experiment_id)
                    self._wait_for_worker(experiment_id, "rass-evidence")
                    continue
                except ExternalServiceError as error:
                    self._status("WAITING_AGENT_API", round=round_number, error=str(error))
                    self.sleeper(self.options.agent_retry_seconds)
                    continue
                if prepared is None:
                    continue
                self._wait_for_worker(prepared)
                self.orchestrator.finalize_adaptive_round(round_number)
