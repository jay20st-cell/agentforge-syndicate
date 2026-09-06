import json
from pathlib import Path

import pytest

from agentforge.benchmarks import load_benchmark


def test_loads_sample_benchmark() -> None:
    sample = Path(__file__).parents[1] / "benchmarks" / "sample.json"

    tasks = load_benchmark(sample)

    assert len(tasks) == 1
    assert tasks[0].id == "arithmetic-addition"
    assert tasks[0].expected_output == "4"
    assert tasks[0].metadata == {"category": "arithmetic"}


def test_rejects_missing_required_fields(tmp_path: Path) -> None:
    benchmark = tmp_path / "invalid.json"
    benchmark.write_text(json.dumps({"tasks": [{"id": "incomplete"}]}))

    with pytest.raises(ValueError, match="expected_output, prompt"):
        load_benchmark(benchmark)


def test_rejects_duplicate_task_ids(tmp_path: Path) -> None:
    benchmark = tmp_path / "duplicates.json"
    task = {"id": "same", "prompt": "Prompt", "expected_output": "Output"}
    benchmark.write_text(json.dumps({"tasks": [task, task]}))

    with pytest.raises(ValueError, match="must be unique"):
        load_benchmark(benchmark)
