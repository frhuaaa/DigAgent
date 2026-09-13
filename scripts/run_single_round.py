from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestrator import Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the next DiagAgent adaptive round.")
    parser.add_argument("--submit", required=True)
    args = parser.parse_args()
    orchestrator = Orchestrator(REPO_ROOT, Path(args.submit))
    state = orchestrator.store.state()
    if state["status"] == "FINISHED":
        print(orchestrator.run_root / "configs" / "final_frozen.json")
        return 0
    if state["accepted_experiment_id"] is None:
        raise RuntimeError("Round 0 has not completed")
    if int(state["next_round"]) > int(orchestrator.initial_config["task"]["trials"]):
        print(orchestrator.store.finalize("ADAPTIVE_ROUND_BUDGET_EXHAUSTED"))
        return 0
    orchestrator.run_adaptive_round(int(state["next_round"]))
    print(orchestrator.store.state())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
