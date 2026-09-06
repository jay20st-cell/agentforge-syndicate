"""Immutable value objects used by the benchmark pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """A single, self-contained benchmark case."""

    id: str
    prompt: str
    expected_output: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.id, "id")
        _require_text(self.prompt, "prompt")
        if not isinstance(self.expected_output, str):
            raise ValueError("expected_output must be a string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class AgentVersion:
    """Identity and configuration for one reproducible agent version."""

    name: str
    version: str
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.version, "version")
        if not isinstance(self.configuration, Mapping):
            raise ValueError("configuration must be an object")
        object.__setattr__(
            self, "configuration", MappingProxyType(dict(self.configuration))
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """The deterministic score assigned to one task execution."""

    task_id: str
    agent_version: str
    score: float
    passed: bool
    details: str = ""

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.agent_version, "agent_version")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a number")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a boolean")
        if not isinstance(self.details, str):
            raise ValueError("details must be a string")


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """Aggregate and per-task results from one benchmark suite run."""

    agent_version: str
    results: tuple[EvaluationResult, ...]
    total: int = field(init=False)
    passed: int = field(init=False)
    failed: int = field(init=False)
    aggregate_score: float = field(init=False)

    def __post_init__(self) -> None:
        _require_text(self.agent_version, "agent_version")
        results = tuple(self.results)
        if any(not isinstance(result, EvaluationResult) for result in results):
            raise ValueError("results must contain only EvaluationResult objects")
        if any(result.agent_version != self.agent_version for result in results):
            raise ValueError("all results must match agent_version")

        total = len(results)
        passed = sum(result.passed for result in results)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "failed", total - passed)
        object.__setattr__(
            self,
            "aggregate_score",
            sum(result.score for result in results) / total if total else 0.0,
        )


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """An explicit decision comparing a candidate with a baseline."""

    candidate_version: str
    baseline_version: str
    promoted: bool
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.candidate_version, "candidate_version")
        _require_text(self.baseline_version, "baseline_version")
        if not isinstance(self.promoted, bool):
            raise ValueError("promoted must be a boolean")
        _require_text(self.reason, "reason")
