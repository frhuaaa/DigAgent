from __future__ import annotations

import os
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from core.configuration import (
    ANCHOR_FEATURES,
    _set_round0_alpha_baseline,
    _validated_cached_rass_evidence,
    rass_evidence_path,
    resolve_run_layout,
)
from core.errors import ContractError
from core.io_utils import canonical_json_bytes, sha256_bytes, sha256_file


class RunLayoutTests(unittest.TestCase):
    def seed(self, *, agent_name: str | None = None, prediction_mode: str = "c2c") -> dict:
        agent = {"model": "deepseek-v4-pro"}
        if agent_name is not None:
            agent["name"] = agent_name
        return {
            "task": {"instruments": "csi1000", "prediction_mode": prediction_mode},
            "agent": agent,
            "z_model": {"model_name": "lstm"},
        }

    def test_timestamped_run_root_uses_agent_model_fallback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs" / "agent_models.json").write_text(
                json.dumps({"models": {"deepseek-v4-pro": {"provider": "deepseek"}}}),
                encoding="utf-8",
            )
            submit = root / "submit.json"
            submit.write_text("{}", encoding="utf-8")
            os.utime(submit, (1_700_000_000, 1_700_000_000))
            with patch.dict(os.environ, {"DIAGAGENT_RUN_TIMESTAMP": "09161234"}):
                namespace, run_root, layout = resolve_run_layout(root, self.seed(), submit)
            self.assertEqual(namespace, root / "runs" / "csi1000_deepseek_c2c")
            self.assertEqual(run_root.parent, namespace / "lstm")
            self.assertEqual(run_root.name, "09161234")
            self.assertEqual(layout["agent_name"], "deepseek")

    def test_explicit_agent_name_controls_shared_namespace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submit = root / "submit.json"
            submit.write_text("{}", encoding="utf-8")
            namespace, _, layout = resolve_run_layout(root, self.seed(agent_name="deepseek"), submit)
            self.assertEqual(namespace.name, "csi1000_deepseek_c2c")
            self.assertEqual(layout["agent_name"], "deepseek")

    def test_prediction_mode_separates_run_and_rass_namespaces(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submit = root / "submit.json"
            submit.write_text("{}", encoding="utf-8")
            c2c_namespace, _, _ = resolve_run_layout(
                root, self.seed(agent_name="deepseek", prediction_mode="c2c"), submit
            )
            o2o_namespace, _, layout = resolve_run_layout(
                root, self.seed(agent_name="deepseek", prediction_mode="o2o"), submit
            )
            self.assertEqual(c2c_namespace.name, "csi1000_deepseek_c2c")
            self.assertEqual(o2o_namespace.name, "csi1000_deepseek_o2o")
            self.assertEqual(layout["prediction_mode"], "o2o")

    def test_explicit_run_id_is_parent_of_submission_timestamp(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            submit = root / "submit.json"
            submit.write_text("{}", encoding="utf-8")
            seed = self.seed(agent_name="deepseek")
            seed["task"]["run_id"] = "2024"
            with patch.dict(os.environ, {"DIAGAGENT_RUN_TIMESTAMP": "09181234"}):
                namespace, run_root, layout = resolve_run_layout(root, seed, submit)
            self.assertEqual(run_root, namespace / "2024" / "lstm" / "09181234")
            self.assertEqual(layout["run_id"], "2024")
            self.assertEqual(layout["run_id_source"], "task.run_id")
            self.assertEqual(layout["study_id"], "2024")
            self.assertEqual(layout["run_timestamp"], "09181234")


class RassEvidenceCacheTests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "task": {
                "instruments": "csi1000",
                "run_id": "2024",
                "prediction_mode": "c2c",
                "train_start_time": "2018-01-01",
                "train_end_time": "2023-12-31",
            },
            "agent": {"name": "deepseek", "model": "deepseek-v4-pro"},
            "z_alpha": {"feature_pool": "Alpha158", "selected_features": list(ANCHOR_FEATURES)},
            "provenance": {"input_hashes": {"qlib_relevant_manifest": "data-hash"}},
        }

    def periods(self) -> tuple[dict, ...]:
        return tuple({
            "period_id": str(year),
            "start_time": f"{year}-01-01",
            "end_time": f"{year}-12-31",
            "signal_end_time": f"{year}-12-27",
        } for year in (2021, 2022, 2023))

    def evidence(self, root: Path, catalog_hash: str) -> dict:
        target = "Ref($close, -2) / Ref($close, -1) - 1"
        periods = list(self.periods())
        context = {
            "version": 1,
            "instruments": "csi1000",
            "run_id": "2024",
            "prediction_mode": "c2c",
            "feature_pool": "Alpha158",
            "initial_features": list(ANCHOR_FEATURES),
            "configured_training_period": ["2018-01-01", "2023-12-31"],
            "evidence_periods": periods,
            "target_label": target,
            "label_hash": sha256_bytes(target.encode("utf-8")),
            "input_data_hash": "data-hash",
            "catalog_hash": catalog_hash,
            "evidence_code_hash": sha256_file(root / "core" / "evidence_builder.py"),
        }
        payload = {
            "method": "deterministic_rass_development_factor_evidence_v5",
            "data_split": "rass_development",
            "evidence_context": context,
            "rass_evidence_context_hash": sha256_bytes(canonical_json_bytes(context)),
            "evidence_period": ["2021-01-01", "2023-12-27"],
            "configured_training_period": ["2018-01-01", "2023-12-31"],
            "prediction_mode": "c2c",
            "target_label": target,
            "label_hash": sha256_bytes(target.encode("utf-8")),
            "initial_features": list(ANCHOR_FEATURES),
            "catalog_hash": catalog_hash,
            "input_data_hash": "data-hash",
            "evidence_code_hash": sha256_file(root / "core" / "evidence_builder.py"),
            "candidate_count": 1,
            "records": [{"id": "KLEN", "eligible": True}],
            "rolling_evidence_protocol": {"periods": periods, "test_used": False},
            "contains_test_derived_data": False,
        }
        payload["evidence_hash"] = sha256_bytes(canonical_json_bytes(payload))
        return payload

    def test_round0_always_keeps_five_submitted_anchors(self):
        config = {
            "z_alpha": {"selected_features": list(ANCHOR_FEATURES), "alpha_frozen": True},
            "z_model": {"derived": {"d_feat": 8}},
        }
        _set_round0_alpha_baseline(config)
        self.assertEqual(config["z_alpha"]["selected_features"], ANCHOR_FEATURES)
        self.assertFalse(config["z_alpha"]["alpha_frozen"])
        self.assertEqual(config["z_model"]["derived"]["d_feat"], 5)

    def test_shared_path_contains_only_rass_evidence(self):
        with TemporaryDirectory() as tmp:
            path = rass_evidence_path(Path(tmp), self.config())
            self.assertEqual(path.name, "rass_train_evidence.json")
            self.assertNotIn("rass_frozen", path.as_posix())

    def test_matching_evidence_is_reusable_and_tampering_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core").mkdir()
            (root / "core" / "evidence_builder.py").write_text("# frozen evidence code\n", encoding="utf-8")
            evidence = self.evidence(root, "catalog-hash")
            with patch("core.configuration.rass_evidence_context", return_value=evidence["evidence_context"]):
                validated = _validated_cached_rass_evidence(root, self.config(), evidence, "catalog-hash")
                self.assertEqual(validated, evidence)
                evidence["records"][0]["id"] = "TAMPERED"
                with self.assertRaisesRegex(ContractError, "HASH_MISMATCH"):
                    _validated_cached_rass_evidence(root, self.config(), evidence, "catalog-hash")

    def test_cross_mode_or_test_derived_evidence_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core").mkdir()
            (root / "core" / "evidence_builder.py").write_text("# frozen evidence code\n", encoding="utf-8")
            evidence = self.evidence(root, "catalog-hash")
            evidence["prediction_mode"] = "o2o"
            unhashed = dict(evidence)
            unhashed.pop("evidence_hash")
            evidence["evidence_hash"] = sha256_bytes(canonical_json_bytes(unhashed))
            with patch("core.configuration.rass_evidence_context", return_value=evidence["evidence_context"]):
                with self.assertRaisesRegex(ContractError, "prediction_mode"):
                    _validated_cached_rass_evidence(root, self.config(), evidence, "catalog-hash")
                evidence["prediction_mode"] = "c2c"
                evidence["contains_test_derived_data"] = True
                with self.assertRaisesRegex(ContractError, "CONTRACT_INVALID"):
                    _validated_cached_rass_evidence(root, self.config(), evidence, "catalog-hash")


if __name__ == "__main__":
    unittest.main()
