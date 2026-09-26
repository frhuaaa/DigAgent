from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.researcher_boundary import ResearcherBoundary
from scripts.rerun_researcher_outputs import resolve_training_source
from scripts import run_researcher_test as researcher_test
from scripts.run_researcher_test import (
    configure_researcher_temp,
    ensure_recovered_epoch_record,
    researcher_runtime_config,
    zip_equal_lengths,
)
from training.model_trainer import epoch_shuffle_seed


class ResearcherRuntimeTests(unittest.TestCase):
    def test_equal_length_zip_preserves_strict_length_validation(self):
        self.assertEqual(list(zip_equal_lengths([1, 2], ["a", "b"])), [(1, "a"), (2, "b")])
        with self.assertRaisesRegex(RuntimeError, "RESEARCHER_ENSEMBLE_MEMBER_COUNT_MISMATCH"):
            list(zip_equal_lengths([1], ["a", "b"]))

    def test_shuffle_seed_is_reproducible_without_persisting_permutation(self):
        self.assertEqual([epoch_shuffle_seed(7, epoch) for epoch in range(3)], [7, 7, 7])

    def test_persistent_worker_loads_test_context_once_for_epochs_and_full(self):
        messages = [
            {"command": "epoch", "checkpoint": "a.pt", "epoch": 0, "train_metrics": {}, "train_valid_metrics": {}},
            {"command": "epoch", "checkpoint": "b.pt", "epoch": 1, "train_metrics": {}, "train_valid_metrics": {}},
            {"command": "full", "checkpoint": "b.pt", "source_experiment_dir": None},
            {"command": "close"},
        ]
        input_stream = StringIO("".join(json.dumps(message) + "\n" for message in messages))
        output_stream = StringIO()
        context = (object(), object(), object())
        with patch.object(researcher_test, "prepare_test_context", return_value=context) as prepare, \
             patch.object(researcher_test, "run_epoch_prepared") as epoch_run, \
             patch.object(researcher_test, "run_full") as full_run, \
             patch.object(researcher_test.sys, "stdin", input_stream), \
             patch.object(researcher_test.sys, "stdout", output_stream):
            researcher_test.run_epoch_stream({}, Path("EXP_000"))
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(epoch_run.call_count, 2)
        self.assertEqual(full_run.call_count, 1)
        self.assertIs(full_run.call_args.kwargs["prepared_test_context"], context)
        acknowledgements = output_stream.getvalue()
        self.assertEqual(acknowledgements.count("DIAGAGENT_ACK "), 5)
        self.assertIn('"event":"worker_ready"', acknowledgements)
        self.assertNotIn("test_ic", acknowledgements)

    def test_boundary_reuses_one_subprocess_and_receives_only_acknowledgements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            process = MagicMock()
            process.pid = 123
            process.poll.return_value = None
            process.stdin = StringIO()
            process.stdout = StringIO(
                'DIAGAGENT_ACK {"event":"worker_ready","ok":true}\n'
                'DIAGAGENT_ACK {"event":"epoch_complete","ok":true,"epoch":0}\n'
                'DIAGAGENT_ACK {"event":"epoch_complete","ok":true,"epoch":1}\n'
            )
            process.wait.return_value = 0
            with patch("core.researcher_boundary.subprocess.Popen", return_value=process) as popen:
                boundary = ResearcherBoundary(root, config, root / "dispatch.jsonl")
                boundary.epoch(0, root / "a.pt", {}, {})
                boundary.epoch(1, root / "b.pt", {}, {})
                self.assertEqual(popen.call_count, 1)
                boundary.close_epoch_worker()

    def test_researcher_runtime_forces_single_process_without_mutating_config(self):
        config = {"z_model": {"base_params": {"n_jobs": 8}}}
        runtime = researcher_runtime_config(config)
        self.assertEqual(runtime["z_model"]["base_params"]["n_jobs"], 0)
        self.assertEqual(config["z_model"]["base_params"]["n_jobs"], 8)

    def test_rapa_full_reuses_parent_test_alpha_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "EXP_008"
            source = Path(tmp) / "EXP_000"
            (experiment / "test").mkdir(parents=True)
            with patch.object(researcher_test, "reuse_parent_test_alpha", return_value=True) as reuse, \
                 patch.object(researcher_test, "prepare_test_context") as prepare, \
                 patch.object(researcher_test, "load_model_from_checkpoint") as load_model:
                researcher_test.run_full({}, experiment, [Path(tmp) / "unused.pt"], source)
            reuse.assert_called_once_with({}, experiment, source)
            prepare.assert_not_called()
            load_model.assert_not_called()

    def test_failed_worker_is_disabled_so_later_epochs_do_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            process = MagicMock()
            process.pid = 123
            process.poll.return_value = None
            process.stdin = StringIO()
            process.stdout = StringIO("")
            process.wait.return_value = 0
            with patch("core.researcher_boundary.subprocess.Popen", return_value=process) as popen:
                boundary = ResearcherBoundary(root, config, root / "dispatch.jsonl", startup_timeout=0.01)
                boundary.epoch(0, root / "a.pt", {}, {})
                boundary.epoch(1, root / "b.pt", {}, {})
                self.assertEqual(popen.call_count, 1)
            events = [json.loads(line) for line in (root / "dispatch.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["event"] == "researcher_epoch_worker_failed" for event in events))
            self.assertTrue(any(event["event"] == "researcher_epoch_skipped" and event["epoch"] == 1 for event in events))

    def test_joblib_temp_is_inside_isolated_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test"
            original = {key: os.environ.get(key) for key in ("TMP", "TEMP", "TMPDIR", "JOBLIB_TEMP_FOLDER")}
            original_tempdir = tempfile.tempdir
            try:
                temporary = configure_researcher_temp(output)
                self.assertEqual(Path(os.environ["JOBLIB_TEMP_FOLDER"]), temporary)
                self.assertTrue(temporary.is_relative_to(output))
            finally:
                for key, value in original.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                tempfile.tempdir = original_tempdir

    def test_historical_recovery_is_explicitly_selected_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "EXP_001"
            output = experiment / "test"
            (experiment / "logs").mkdir(parents=True)
            output.mkdir(parents=True)
            records = [
                {"epoch": 3, "train": {"ic": 0.2}, "train_valid": {"ic": 0.04}},
                {"event": "checkpoint_selected", "epoch": 3, "metric": "train_valid.ic", "score": 0.04},
            ]
            (experiment / "logs" / "training_metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            ensure_recovered_epoch_record(experiment, output, experiment / "artifacts" / "best.pt", {"ic": 0.03})
            recovered = json.loads((output / "epoch_metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(recovered["recovery_scope"], "selected_checkpoint_only")
            self.assertEqual(recovered["epoch"], 3)
            self.assertEqual(recovered["test_ic"], 0.03)

    def test_training_source_resolution_follows_reuse_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            experiments = run_root / "experiments"
            source = experiments / "EXP_001"
            middle = experiments / "EXP_007"
            candidate = experiments / "EXP_008"
            for path in (source, middle, candidate):
                (path / "artifacts").mkdir(parents=True)
            (source / "artifacts" / "selected_checkpoint.json").write_text(
                '{"repository_path":"best.pt"}', encoding="utf-8"
            )
            (middle / "artifacts" / "selected_checkpoint.json").write_text(
                '{"reused_from":"EXP_001"}', encoding="utf-8"
            )
            (candidate / "artifacts" / "selected_checkpoint.json").write_text(
                '{"reused_from":"EXP_007"}', encoding="utf-8"
            )

            self.assertEqual(resolve_training_source(run_root, candidate), source)


if __name__ == "__main__":
    unittest.main()
