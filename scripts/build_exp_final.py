from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.io_utils import atomic_write_bytes, atomic_write_json, load_json, sha256_file


EXPERIMENT_ID_PATTERN = re.compile(r"EXP_[0-9]{3}")
REPORT_SECTIONS = (
    ("signal", (("IC", "ic", False), ("ICIR", "icir", False), ("Rank IC", "rank_ic", False), ("Rank ICIR", "rank_icir", False))),
    ("net_return", (("ARR", "annual_return", True), ("Vol", "annual_volatility", True), ("MDD", "max_drawdown", True), ("Sharpe", "sharpe_ratio", False), ("Calmar", "calmar_ratio", False), ("Sortino", "sortino_ratio", False), ("Win rate", "win_rate", True))),
    ("excess_return", (("ARR", "excess_annual_return", True), ("Vol", "excess_annual_volatility", True), ("MDD", "excess_max_drawdown", True), ("Sharpe (IR)", "information_ratio", False), ("Calmar", "excess_calmar_ratio", False), ("Sortino", "excess_sortino_ratio", False), ("Win rate", "excess_win_rate", True))),
    ("turnover", (("One-way turnover", "one_way_turnover_mean", True),)),
)


def _report_payload(source_experiment_id: str, result: dict) -> dict:
    required = {key for _, metrics in REPORT_SECTIONS for _, key, _ in metrics}
    missing = sorted(required - set(result))
    if missing:
        raise RuntimeError(f"EXP_FINAL_RESULT_FIELDS_MISSING: {missing}")
    return {
        "researcher_only": True,
        "adaptive_state_affected": False,
        "source_experiment_id": source_experiment_id,
        "sections": {
            section: [
                {"label": label, "key": key, "value": result[key], "display_as_percent": percent}
                for label, key, percent in metrics
            ]
            for section, metrics in REPORT_SECTIONS
        },
    }


def _csv_bytes(report: dict) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("section", "label", "metric_key", "value", "display_as_percent"))
    for section, metrics in report["sections"].items():
        for metric in metrics:
            writer.writerow((section, metric["label"], metric["key"], "" if metric["value"] is None else metric["value"], str(metric["display_as_percent"]).lower()))
    return output.getvalue().encode("utf-8-sig")


def _display(value: object, percent: bool) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number * 100.0:.3f}%" if percent else f"{number:.3f}"


def _markdown_bytes(report: dict) -> bytes:
    sections = report["sections"]
    signal = sections["signal"]
    net = sections["net_return"]
    excess = sections["excess_return"]
    turnover = sections["turnover"][0]
    lines = [
        "# EXP_FINAL isolated test report", "",
        f"Source accepted experiment: `{report['source_experiment_id']}`", "",
        "## Signal statistics", "",
        "| " + " | ".join(item["label"] for item in signal) + " |",
        "| " + " | ".join("---" for _ in signal) + " |",
        "| " + " | ".join(_display(item["value"], item["display_as_percent"]) for item in signal) + " |", "",
        "## Portfolio statistics", "",
        "| Return type | ARR | Vol | MDD | Sharpe / Sharpe (IR) | Calmar | Sortino | Win rate |",
        "| --- | " + " | ".join("---" for _ in net) + " |",
        "| Net return | " + " | ".join(_display(item["value"], item["display_as_percent"]) for item in net) + " |",
        "| Excess return | " + " | ".join(_display(item["value"], item["display_as_percent"]) for item in excess) + " |", "",
        "## Turnover", "",
        f"One-way turnover mean: **{_display(turnover['value'], turnover['display_as_percent'])}**", "",
        "> Researcher-only post-hoc output. It is never available to adaptive agents, routing, validation, memory, promotion, rollback, or stopping.", "",
    ]
    return "\n".join(lines).encode("utf-8")


def materialize_exp_final(run_root: Path) -> Path:
    """Create a researcher-only snapshot of the final accepted test result."""
    run_root = run_root.resolve()
    state = load_json(run_root / "configs" / "state.json")
    final_config_path = run_root / "configs" / "final_frozen.json"
    if state.get("status") != "FINISHED" or not final_config_path.is_file():
        raise RuntimeError("EXP_FINAL_REQUIRES_FINISHED_TRAJECTORY")
    source_experiment_id = state.get("accepted_experiment_id")
    if not isinstance(source_experiment_id, str) or EXPERIMENT_ID_PATTERN.fullmatch(source_experiment_id) is None:
        raise RuntimeError("EXP_FINAL_ACCEPTED_EXPERIMENT_INVALID")
    source_test = run_root / "experiments" / source_experiment_id / "test"
    if load_json(source_test / "researcher_complete.json").get("status") != "COMPLETE":
        raise RuntimeError("EXP_FINAL_SOURCE_TEST_INCOMPLETE")
    source_result_path = source_test / "result.json"
    report = _report_payload(source_experiment_id, load_json(source_result_path))

    final_root = run_root / "experiments" / "EXP_FINAL"
    final_test = final_root / "test"
    result_path = final_test / "result.json"
    report_path = final_test / "final_report.json"
    csv_path = final_test / "final_report.csv"
    markdown_path = final_test / "final_report.md"
    completion_path = final_test / "researcher_complete.json"
    atomic_write_bytes(result_path, source_result_path.read_bytes())
    atomic_write_json(report_path, report)
    atomic_write_bytes(csv_path, _csv_bytes(report))
    atomic_write_bytes(markdown_path, _markdown_bytes(report))
    atomic_write_json(completion_path, {"status": "COMPLETE", "researcher_only": True, "adaptive_state_affected": False, "source_experiment_id": source_experiment_id})
    atomic_write_json(final_root / "manifest.json", {
        "experiment_id": "EXP_FINAL",
        "source_accepted_experiment_id": source_experiment_id,
        "stop_reason": state.get("stop_reason"),
        "researcher_only": True,
        "adaptive_state_affected": False,
        "test_metrics_used_by_adaptation": False,
        "final_config_sha256": sha256_file(final_config_path),
        "files": {path.relative_to(final_root).as_posix(): sha256_file(path) for path in (result_path, report_path, csv_path, markdown_path, completion_path)},
    })
    return final_root


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Materialize the final accepted isolated test report.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    materialize_exp_final(Path(args.run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
