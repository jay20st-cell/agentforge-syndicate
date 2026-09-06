from copy import deepcopy

import pytest

from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.improvement import (
    DeterministicCandidateImprover,
    FailureDiagnoser,
    PromotionPolicy,
    compare_suites,
)
from agentforge.improvement_demo import run_demo
from agentforge.models import AgentVersion, BenchmarkTask, EvaluationResult, SuiteResult


def test_evaluator_exposes_structured_failure_evidence() -> None:
    task = BenchmarkTask("t", "p", "e", {"evaluation": "required_keywords", "required_keywords": ["one", "two"]})
    result = DeterministicEvaluator(1.0).evaluate(task, AgentVersion("a", "1"), "ONE")

    assert result.matched_requirements == ("one",)
    assert result.missing_requirements == ("two",)


def test_diagnosis_uses_structured_evidence_and_metadata() -> None:
    task = BenchmarkTask("t", "p", "e", {"category": "safety", "critical": True})
    suite = SuiteResult("1", (EvaluationResult("t", "1", 0.0, False, "opaque", (), ("e",)),))

    diagnosis = FailureDiagnoser().diagnose((task,), suite)[0]

    assert diagnosis.task_id == "t"
    assert diagnosis.category == "safety"
    assert diagnosis.baseline_score == 0.0
    assert diagnosis.missing_requirements == ("e",)
    assert diagnosis.critical is True
    assert "e" in diagnosis.explanation


def test_demo_candidate_is_new_and_only_changes_failed_tasks() -> None:
    run = run_demo()
    baseline = run["baseline_agent"]
    candidate = run["candidate_agent"]
    baseline_result = run["baseline_result"]
    original = deepcopy(dict(baseline.configuration["responses"]))
    failed = {result.task_id for result in baseline_result.results if not result.passed}

    assert candidate is not baseline
    assert candidate.version == "2.0.0"
    assert baseline.version == "1.0.0"
    assert dict(baseline.configuration["responses"]) == original
    for task_id, response in original.items():
        if task_id not in failed:
            assert candidate.configuration["responses"][task_id] == response


def test_keyword_improvement_does_not_depend_on_expected_output() -> None:
    task = BenchmarkTask(
        "database",
        "Diagnose a database lock",
        "UNRELATED ANSWER-KEY SENTINEL",
        {
            "category": "database",
            "evaluation": "required_keywords",
            "required_keywords": ["transactions", "busy timeout", "wal"],
        },
    )
    baseline = AgentVersion(
        "support", "1.0.0", {"responses": {"database": "Keep transactions short."}}
    )
    suite_runner = SuiteRunner(
        DeterministicAgentRunner(), DeterministicEvaluator(pass_threshold=1.0)
    )
    baseline_result = suite_runner.run((task,), baseline)
    diagnoses = FailureDiagnoser().diagnose((task,), baseline_result)

    candidate = DeterministicCandidateImprover().improve(
        baseline, (task,), baseline_result, diagnoses
    )
    candidate_result = suite_runner.run((task,), candidate)

    assert candidate_result.results[0].passed is True
    assert candidate.configuration["responses"]["database"] == (
        "Keep transactions short. Additional checks: busy timeout, wal."
    )
    assert "UNRELATED ANSWER-KEY SENTINEL" not in candidate.configuration["responses"]["database"]


def test_failed_exact_match_is_left_unchanged() -> None:
    task = BenchmarkTask("exact", "p", "secret answer")
    baseline = AgentVersion("a", "1", {"responses": {"exact": "unsupported guess"}})
    suite_runner = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator())
    result = suite_runner.run((task,), baseline)
    diagnoses = FailureDiagnoser().diagnose((task,), result)

    candidate = DeterministicCandidateImprover("2").improve(
        baseline, (task,), result, diagnoses
    )

    assert candidate.configuration["responses"]["exact"] == "unsupported guess"


def test_demo_improves_failures_keeps_passing_scores_and_promotes() -> None:
    run = run_demo()
    baseline = run["baseline_result"]
    candidate = run["candidate_result"]
    comparison = run["comparison"]

    assert comparison.improved_tasks == (
        "sqlite-database-locked", "git-merge-conflict", "http-timeout-debugging"
    )
    assert comparison.regressed_tasks == ()
    assert comparison.regression_count == 0
    assert comparison.unchanged_tasks == (
        "http-bearer-auth", "python-none-attribute", "dns-resolution-failure", "missing-environment-variable"
    )
    assert (baseline.passed, candidate.passed) == (4, 7)
    assert run["decision"].promoted is True


def test_regression_detection_and_rejection() -> None:
    baseline = SuiteResult("1", (EvaluationResult("a", "1", 0.5, False), EvaluationResult("b", "1", 1.0, True)))
    candidate = SuiteResult("2", (EvaluationResult("a", "2", 1.0, True), EvaluationResult("b", "2", 0.75, False)))
    comparison = compare_suites(baseline, candidate)
    decision = PromotionPolicy().decide(baseline, candidate, comparison, ())

    assert comparison.regressed_tasks == ("b",)
    assert comparison.regression_count == 1
    assert decision.promoted is False
    assert "regression" in decision.reason


def test_rejection_without_score_improvement() -> None:
    baseline = SuiteResult("1", (EvaluationResult("a", "1", 1.0, True),))
    candidate = SuiteResult("2", (EvaluationResult("a", "2", 1.0, True),))
    decision = PromotionPolicy().decide(baseline, candidate, compare_suites(baseline, candidate), ())
    assert decision.promoted is False
    assert "did not improve" in decision.reason


def test_rejection_on_candidate_critical_failure() -> None:
    tasks = (BenchmarkTask("critical", "p", "yes", {"critical": True}), BenchmarkTask("gain", "p", "yes"))
    baseline = SuiteResult("1", (EvaluationResult("critical", "1", 0.0, False), EvaluationResult("gain", "1", 0.0, False)))
    candidate = SuiteResult("2", (EvaluationResult("critical", "2", 0.0, False, "", (), ("yes",)), EvaluationResult("gain", "2", 1.0, True)))
    diagnoses = FailureDiagnoser().diagnose(tasks, candidate)
    decision = PromotionPolicy().decide(baseline, candidate, compare_suites(baseline, candidate), diagnoses)
    assert decision.promoted is False
    assert "critical failure" in decision.reason


def test_repeated_demo_execution_is_deterministic() -> None:
    first = run_demo()
    second = run_demo()
    for key in ("baseline_result", "candidate_result", "baseline_diagnoses", "candidate_diagnoses", "comparison", "decision"):
        assert first[key] == second[key]


def test_improver_requires_a_new_version() -> None:
    with pytest.raises(ValueError, match="must differ"):
        DeterministicCandidateImprover("1").improve(
            AgentVersion("a", "1", {"responses": {}}), (), SuiteResult("1", ()), ()
        )
