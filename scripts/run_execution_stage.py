from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestrator import Orchestrator
from core.process_lock import advisory_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one prepared DiagAgent execution stage without any Agent API calls.")
    parser.add_argument("--submit", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", choices=["materialize", "experiment", "rass-evidence"], default="experiment")
    args = parser.parse_args()
    orchestrator = Orchestrator(REPO_ROOT, Path(args.submit))
    experiment_dir = orchestrator.experiments_dir / args.experiment_id
    lock_path = experiment_dir / "logs" / f"{args.stage.replace('-', '_')}.lock"
    with advisory_lock(lock_path):
        if args.stage == "materialize":
            prepared = orchestrator.prepare_round0()
            print(orchestrator.run_root / "configs" / "initial.json")
            print(orchestrator.experiments_dir / prepared / "config.json")
        elif args.stage == "rass-evidence":
            output_path = orchestrator.execute_rass_evidence_stage(args.experiment_id)
            print(output_path)
        else:
            outcome = orchestrator.execute_prepared_round(args.experiment_id)
            print(experiment_dir / "logs" / "execution_complete.json")
            print(f"validation_sharpe={outcome.result.get('sharpe_ratio')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
