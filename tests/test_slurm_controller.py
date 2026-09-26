from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.slurm_controller import SlurmClient, SlurmTrajectoryController


class FakeRunner:
    def __init__(self, responses: list[subprocess.CompletedProcess]):
        self.responses = list(responses)
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        return self.responses.pop(0)


class SlurmClientTests(unittest.TestCase):
    def _client(self, root: Path, runner: FakeRunner) -> SlurmClient:
        script = root / "worker.slurm"
        script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        return SlurmClient(root, script, runner=runner)

    def test_submit_uses_explicit_experiment_and_parses_federated_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submit = root / "submit.json"
            submit.write_text("{}", encoding="utf-8")
            runner = FakeRunner([subprocess.CompletedProcess([], 0, "3485001;cluster\n", "")])
            client = self._client(root, runner)
            job_id = client.submit(submit, "EXP_001", "diagagent-exp_001", "rass-evidence")
            self.assertEqual(job_id, "3485001")
            self.assertEqual(
                runner.commands[0][-4:],
                [str(submit.resolve()), "EXP_001", "rass-evidence", "support"],
            )
            self.assertNotIn(".env", " ".join(runner.commands[0]))

    def test_portfolio_submit_requests_cpu_only_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            submit = root / "submit.json"
            submit.write_text("{}", encoding="utf-8")
            runner = FakeRunner([subprocess.CompletedProcess([], 0, "3485002\n", "")])
            client = self._client(root, runner)
            client.submit(submit, "EXP_008", "diagagent-exp_008", "experiment", "portfolio")
            command = runner.commands[0]
            self.assertIn("--gres=", command)
            self.assertEqual(command[-1], "portfolio")

    def test_status_prefers_exact_allocation_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner([
                subprocess.CompletedProcess([], 0, "3485001|COMPLETED|0:0\n", ""),
            ])
            client = self._client(Path(tmp), runner)
            self.assertEqual(client.status("3485001"), ("COMPLETED", "0:0"))

    def test_status_falls_back_to_squeue_during_accounting_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = FakeRunner([
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 0, "RUNNING\n", ""),
            ])
            client = self._client(Path(tmp), runner)
            self.assertEqual(client.status("3485001"), ("RUNNING", None))


class WorkerProfileTests(unittest.TestCase):
    def test_rapa_decision_routes_to_cpu_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments = Path(tmp) / "experiments"
            decision = experiments / "EXP_008" / "decision.json"
            decision.parent.mkdir(parents=True)
            decision.write_text('{"specialist":{"agent":"RAPA"}}', encoding="utf-8")
            controller = object.__new__(SlurmTrajectoryController)
            controller.orchestrator = SimpleNamespace(experiments_dir=experiments)
            self.assertEqual(controller._worker_profile("EXP_008", "experiment"), "portfolio")
            self.assertEqual(controller._worker_profile("EXP_008", "rass-evidence"), "support")
            self.assertEqual(controller._worker_profile("EXP_000", "experiment"), "training")


class WorkerScriptTests(unittest.TestCase):
    def test_initial_materialization_uses_support_profile(self):
        controller_script = Path(__file__).resolve().parents[1] / "scripts" / "run_controller.py"
        text = controller_script.read_text(encoding="utf-8")
        self.assertIn('"EXP_000", "materialize", "support"', text)

    def test_generic_worker_is_offline_execution_only(self):
        script = Path(__file__).resolve().parents[1] / "slurm" / "diagagent_worker.slurm"
        text = script.read_text(encoding="utf-8")
        self.assertIn("run_execution_stage.py", text)
        self.assertNotIn("run_adaptation.py", text)
        self.assertNotIn('".env"', text)
        self.assertIn('[[ "${WORKER_PROFILE}" != "portfolio" ]]', text)


if __name__ == "__main__":
    unittest.main()
