from pathlib import Path

import pytest

from agentforge.baseline import run_baseline
from agentforge.benchmarks import load_benchmark


BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "technical-support.json"


def test_technical_support_benchmark_has_realistic_coverage() -> None:
    tasks = load_benchmark(BENCHMARK)
    categories = {task.metadata["category"] for task in tasks}

    assert len(tasks) == 7
    assert {"http-auth", "sqlite", "python", "networking", "git", "configuration"} <= categories


def test_baseline_execution_is_reproducible_and_preserves_failure() -> None:
    first = run_baseline(BENCHMARK)
    second = run_baseline(BENCHMARK)

    assert first == second
    assert (first["total"], first["passed"], first["failed"]) == (7, 6, 1)
    assert first["aggregate_score"] == pytest.approx(34 / 35)
    failures = [result for result in first["results"] if not result["passed"]]
    assert failures == [
        {
            "task_id": "http-timeout-debugging",
            "score": 0.8,
            "passed": False,
            "details": "matched 4/5 required keywords; threshold=1.00; missing: firewall",
        }
    ]
