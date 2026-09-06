"""Concrete deterministic execution and evaluation pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentforge.interfaces import AgentRunner, Evaluator
from agentforge.models import AgentVersion, BenchmarkTask, EvaluationResult, SuiteResult


class DeterministicAgentRunner(AgentRunner):
    """Return task responses stored in an agent version's configuration.

    ``configuration["responses"]`` maps task IDs to strings. Keeping execution
    data-only makes repeated local runs reproducible and avoids external calls.
    """

    def run(self, task: BenchmarkTask, agent: AgentVersion) -> str:
        responses = agent.configuration.get("responses")
        if not isinstance(responses, Mapping):
            raise ValueError("agent configuration must contain a responses object")
        if task.id not in responses:
            raise ValueError(f"no configured response for task {task.id!r}")
        output = responses[task.id]
        if not isinstance(output, str):
            raise ValueError(f"configured response for task {task.id!r} must be a string")
        return output


class DeterministicEvaluator(Evaluator):
    """Evaluate exact matches or required keywords with an explicit threshold."""

    def __init__(self, pass_threshold: float = 0.8) -> None:
        if isinstance(pass_threshold, bool) or not isinstance(pass_threshold, (int, float)):
            raise ValueError("pass_threshold must be a number")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        self.pass_threshold = float(pass_threshold)

    def evaluate(
        self, task: BenchmarkTask, agent: AgentVersion, output: str
    ) -> EvaluationResult:
        if not isinstance(output, str):
            raise ValueError("output must be a string")

        policy = task.metadata.get("evaluation", "exact_match")
        if policy == "exact_match":
            return self._exact_match(task, agent, output)
        if policy == "required_keywords":
            return self._required_keywords(task, agent, output)
        raise ValueError(f"unsupported evaluation policy: {policy!r}")

    def _exact_match(
        self, task: BenchmarkTask, agent: AgentVersion, output: str
    ) -> EvaluationResult:
        actual = output.strip()
        expected = task.expected_output.strip()
        passed = actual == expected
        details = "exact match" if passed else f"expected {expected!r}; received {actual!r}"
        return EvaluationResult(
            task.id,
            agent.version,
            float(passed),
            passed,
            details,
            (expected,) if passed else (),
            () if passed else (expected,),
        )

    def _required_keywords(
        self, task: BenchmarkTask, agent: AgentVersion, output: str
    ) -> EvaluationResult:
        keywords = task.metadata.get("required_keywords")
        if (
            not isinstance(keywords, Sequence)
            or isinstance(keywords, (str, bytes))
            or not keywords
            or any(not isinstance(keyword, str) or not keyword.strip() for keyword in keywords)
        ):
            raise ValueError("required_keywords policy needs a non-empty string list")

        normalized_output = output.casefold()
        normalized_keywords = [keyword.strip().casefold() for keyword in keywords]
        matched = [keyword for keyword in normalized_keywords if keyword in normalized_output]
        missing = [keyword for keyword in normalized_keywords if keyword not in normalized_output]
        score = len(matched) / len(normalized_keywords)
        passed = score >= self.pass_threshold
        details = (
            f"matched {len(matched)}/{len(normalized_keywords)} required keywords; "
            f"threshold={self.pass_threshold:.2f}"
        )
        if missing:
            details += f"; missing: {', '.join(missing)}"
        return EvaluationResult(
            task.id, agent.version, score, passed, details, tuple(matched), tuple(missing)
        )


class SuiteRunner:
    """Run one agent version over a suite in input order."""

    def __init__(self, runner: AgentRunner, evaluator: Evaluator) -> None:
        self.runner = runner
        self.evaluator = evaluator

    def run(
        self, tasks: Sequence[BenchmarkTask], agent: AgentVersion
    ) -> SuiteResult:
        results = tuple(
            self.evaluator.evaluate(task, agent, self.runner.run(task, agent))
            for task in tasks
        )
        return SuiteResult(agent.version, results)
