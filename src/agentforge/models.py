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
    matched_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()

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
        matched = tuple(self.matched_requirements)
        missing = tuple(self.missing_requirements)
        if any(not isinstance(item, str) or not item.strip() for item in matched + missing):
            raise ValueError("requirements must be non-empty strings")
        object.__setattr__(self, "matched_requirements", matched)
        object.__setattr__(self, "missing_requirements", missing)


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


@dataclass(frozen=True, slots=True)
class FailureDiagnosis:
    """Structured evidence explaining one failed benchmark task."""

    task_id: str
    category: str
    baseline_score: float
    missing_requirements: tuple[str, ...]
    critical: bool
    explanation: str

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.category, "category")
        if isinstance(self.baseline_score, bool) or not isinstance(
            self.baseline_score, (int, float)
        ) or not 0.0 <= self.baseline_score <= 1.0:
            raise ValueError("baseline_score must be between 0.0 and 1.0")
        missing = tuple(self.missing_requirements)
        if any(not isinstance(item, str) or not item.strip() for item in missing):
            raise ValueError("missing_requirements must contain non-empty strings")
        if not isinstance(self.critical, bool):
            raise ValueError("critical must be a boolean")
        _require_text(self.explanation, "explanation")
        object.__setattr__(self, "missing_requirements", missing)


@dataclass(frozen=True, slots=True)
class RegressionAnalysis:
    """Per-task and aggregate comparison of a baseline and candidate."""

    improved_tasks: tuple[str, ...]
    unchanged_tasks: tuple[str, ...]
    regressed_tasks: tuple[str, ...]
    baseline_aggregate_score: float
    candidate_aggregate_score: float
    baseline_failed_count: int
    candidate_failed_count: int
    regression_count: int = field(init=False)
    score_delta: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "improved_tasks", tuple(self.improved_tasks))
        object.__setattr__(self, "unchanged_tasks", tuple(self.unchanged_tasks))
        object.__setattr__(self, "regressed_tasks", tuple(self.regressed_tasks))
        object.__setattr__(self, "regression_count", len(self.regressed_tasks))
        object.__setattr__(
            self,
            "score_delta",
            self.candidate_aggregate_score - self.baseline_aggregate_score,
        )
