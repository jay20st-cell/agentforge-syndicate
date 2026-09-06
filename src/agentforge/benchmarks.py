"""JSON benchmark loading with strict, predictable validation."""

from __future__ import annotations

import json
from pathlib import Path

from agentforge.models import BenchmarkTask


def load_benchmark(path: str | Path) -> tuple[BenchmarkTask, ...]:
    """Load tasks from a JSON object containing a ``tasks`` array."""

    benchmark_path = Path(path)
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark JSON: {exc.msg}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("benchmark must be an object containing a tasks array")

    tasks: list[BenchmarkTask] = []
    for index, item in enumerate(payload["tasks"]):
        if not isinstance(item, dict):
            raise ValueError(f"task at index {index} must be an object")
        required = {"id", "prompt", "expected_output"}
        missing = required.difference(item)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"task at index {index} is missing: {names}")
        try:
            tasks.append(
                BenchmarkTask(
                    id=item["id"],
                    prompt=item["prompt"],
                    expected_output=item["expected_output"],
                    metadata=item.get("metadata", {}),
                )
            )
        except ValueError as exc:
            raise ValueError(f"invalid task at index {index}: {exc}") from exc

    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark task ids must be unique")
    return tuple(tasks)
