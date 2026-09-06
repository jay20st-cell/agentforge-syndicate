"""Extension points for running agents and evaluating their output."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentforge.models import AgentVersion, BenchmarkTask, EvaluationResult


class AgentRunner(ABC):
    """Executes a specified agent version for one benchmark task."""

    @abstractmethod
    def run(self, task: BenchmarkTask, agent: AgentVersion) -> str:
        """Return the agent's textual output."""


class Evaluator(ABC):
    """Scores an agent output against a benchmark task."""

    @abstractmethod
    def evaluate(
        self, task: BenchmarkTask, agent: AgentVersion, output: str
    ) -> EvaluationResult:
        """Return a deterministic evaluation result."""
