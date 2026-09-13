from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.orchestrator import run_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or resume a formal DiagAgent trajectory.")
    parser.add_argument("--submit", required=True, help="Explicit repository submit JSON path")
    args = parser.parse_args()
    final_path = run_submission(REPO_ROOT, Path(args.submit))
    print(final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

