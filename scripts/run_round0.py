from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestrator import Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume DiagAgent Round 0.")
    parser.add_argument("--submit", required=True)
    args = parser.parse_args()
    outcome = Orchestrator(REPO_ROOT, Path(args.submit)).run_round0()
    print(outcome.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

