"""Deterministic diagnosis, improvement, comparison, and promotion components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agentforge.interfaces import CandidateImprover
from agentforge.models import (
    AgentVersion,
    BenchmarkTask,
    FailureDiagnosis,
    PromotionDecision,
    RegressionAnalysis,
    SuiteResult,
)


class FailureDiagnoser:
    """Translate evaluator evidence into structured failure diagnoses."""

    def diagnose(
        self, tasks: Sequence[BenchmarkTask], result: SuiteResult
    ) -> tuple[FailureDiagnosis, ...]:
        task_by_id = {task.id: task for task in tasks}
        diagnoses = []
        for evaluation in result.results:
            if evaluation.passed:
                continue
            if evaluation.task_id not in task_by_id:
                raise ValueError(f"result references unknown task {evaluation.task_id!r}")
            task = task_by_id[evaluation.task_id]
            category = task.metadata.get("category", "uncategorized")
            critical = task.metadata.get("critical", False)
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"task {task.id!r} category must be a non-empty string")
            if not isinstance(critical, bool):
                raise ValueError(f"task {task.id!r} critical must be a boolean")
            missing = evaluation.missing_requirements
            explanation = (
                f"Missing {len(missing)} requirement(s): {', '.join(missing)}"
                if missing
                else "Task did not meet its pass threshold."
            )
            diagnoses.append(
                FailureDiagnosis(
                    task.id, category, evaluation.score, missing, critical, explanation
                )
            )
        return tuple(diagnoses)


class DeterministicCandidateImprover(CandidateImprover):
    """Augment failed keyword responses using structured evaluator evidence."""

    def __init__(self, candidate_version: str = "2.0.0") -> None:
        self.candidate_version = candidate_version

    def improve(
        self,
        baseline: AgentVersion,
        tasks: Sequence[BenchmarkTask],
        baseline_result: SuiteResult,
        diagnoses: Sequence[FailureDiagnosis],
    ) -> AgentVersion:
        if self.candidate_version == baseline.version:
            raise ValueError("candidate version must differ from baseline version")
        responses = baseline.configuration.get("responses")
        if not isinstance(responses, Mapping):
            raise ValueError("baseline configuration must contain a responses object")
        failed_ids = {result.task_id for result in baseline_result.results if not result.passed}
        diagnosed_ids = {diagnosis.task_id for diagnosis in diagnoses}
        if failed_ids != diagnosed_ids:
            raise ValueError("diagnoses must correspond exactly to baseline failures")

        task_by_id = {task.id: task for task in tasks}
        if any(task_id not in task_by_id for task_id in diagnosed_ids):
            raise ValueError("diagnosis references an unknown benchmark task")

        candidate_responses = dict(responses)
        for diagnosis in diagnoses:
            task = task_by_id[diagnosis.task_id]
            if task.metadata.get("evaluation", "exact_match") != "required_keywords":
                # Exact matching has no safe repair without an answer key. Fail closed
                # by retaining the baseline response exactly.
                continue
            if not diagnosis.missing_requirements:
                continue
            baseline_response = candidate_responses[diagnosis.task_id]
            if not isinstance(baseline_response, str):
                raise ValueError(
                    f"baseline response for task {diagnosis.task_id!r} must be a string"
                )
            separator = " " if baseline_response and not baseline_response.endswith((" ", "\n")) else ""
            additions = ", ".join(diagnosis.missing_requirements)
            candidate_responses[diagnosis.task_id] = (
                f"{baseline_response}{separator}Additional checks: {additions}."
            )
        configuration = dict(baseline.configuration)
        configuration["responses"] = candidate_responses
        return AgentVersion(baseline.name, self.candidate_version, configuration)


def compare_suites(baseline: SuiteResult, candidate: SuiteResult) -> RegressionAnalysis:
    """Compare task scores; task sets must match exactly."""

    baseline_by_id = {result.task_id: result for result in baseline.results}
    candidate_by_id = {result.task_id: result for result in candidate.results}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate task sets must match")

    improved, unchanged, regressed = [], [], []
    for result in baseline.results:
        delta = candidate_by_id[result.task_id].score - result.score
        target = improved if delta > 0 else regressed if delta < 0 else unchanged
        target.append(result.task_id)
    return RegressionAnalysis(
        tuple(improved),
        tuple(unchanged),
        tuple(regressed),
        baseline.aggregate_score,
        candidate.aggregate_score,
        baseline.failed,
        candidate.failed,
    )


class PromotionPolicy:
    """Default deterministic gate: improve, remain regression-free, and be safe."""

    def decide(
        self,
        baseline: SuiteResult,
        candidate: SuiteResult,
        comparison: RegressionAnalysis,
        candidate_diagnoses: Sequence[FailureDiagnosis],
    ) -> PromotionDecision:
        reasons = []
        if candidate.aggregate_score <= baseline.aggregate_score:
            reasons.append("candidate aggregate score did not improve")
        if comparison.regression_count:
            reasons.append(f"candidate introduced {comparison.regression_count} regression(s)")
        critical = tuple(d.task_id for d in candidate_diagnoses if d.critical)
        if critical:
            reasons.append(f"candidate has critical failure(s): {', '.join(critical)}")
        promoted = not reasons
        reason = (
            "aggregate score improved with zero regressions and zero critical failures"
            if promoted
            else "; ".join(reasons)
        )
        return PromotionDecision(candidate.agent_version, baseline.agent_version, promoted, reason)
