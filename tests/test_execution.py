import pytest

from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.models import AgentVersion, BenchmarkTask


def make_agent(**responses: str) -> AgentVersion:
    return AgentVersion("test-agent", "1.2.3", {"responses": responses})


def test_runner_returns_configured_response_deterministically() -> None:
    task = BenchmarkTask("dns", "Diagnose", "DNS")
    runner = DeterministicAgentRunner()
    agent = make_agent(dns="DNS")

    assert runner.run(task, agent) == "DNS"
    assert runner.run(task, agent) == "DNS"


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        (AgentVersion("agent", "1", {}), "responses object"),
        (make_agent(other="answer"), "no configured response"),
        (AgentVersion("agent", "1", {"responses": {"dns": 3}}), "must be a string"),
    ],
)
def test_runner_rejects_invalid_or_missing_responses(agent, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeterministicAgentRunner().run(BenchmarkTask("dns", "p", "e"), agent)


def test_exact_match_is_whitespace_tolerant_but_case_sensitive() -> None:
    task = BenchmarkTask("exact", "p", "Expected")
    agent = make_agent(exact=" Expected\n")
    evaluator = DeterministicEvaluator()

    passed = evaluator.evaluate(task, agent, " Expected\n")
    failed = evaluator.evaluate(task, agent, "expected")

    assert (passed.score, passed.passed, passed.details) == (1.0, True, "exact match")
    assert failed.score == 0.0
    assert failed.passed is False
    assert "expected 'Expected'" in failed.details
    assert "received 'expected'" in failed.details


def test_keyword_score_is_normalized_case_insensitive_and_threshold_explicit() -> None:
    task = BenchmarkTask(
        "network",
        "p",
        "e",
        {"evaluation": "required_keywords", "required_keywords": ["DNS", "dig", "nameserver"]},
    )
    result = DeterministicEvaluator(pass_threshold=2 / 3).evaluate(
        task, make_agent(network=""), "Check dns with DIG"
    )

    assert result.score == pytest.approx(2 / 3)
    assert result.passed is True
    assert "missing: nameserver" in result.details
    assert "threshold=0.67" in result.details


def test_keyword_failure_preserves_missing_terms() -> None:
    task = BenchmarkTask(
        "sqlite",
        "p",
        "e",
        {"evaluation": "required_keywords", "required_keywords": ["WAL", "busy timeout"]},
    )
    result = DeterministicEvaluator(pass_threshold=1.0).evaluate(
        task, make_agent(sqlite="WAL"), "Enable WAL"
    )

    assert result.score == 0.5
    assert result.passed is False
    assert result.details.endswith("missing: busy timeout")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.8"])
def test_evaluator_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="pass_threshold"):
        DeterministicEvaluator(threshold)


def test_evaluator_rejects_unknown_policy_and_bad_keywords() -> None:
    evaluator = DeterministicEvaluator()
    agent = make_agent(task="output")

    with pytest.raises(ValueError, match="unsupported"):
        evaluator.evaluate(BenchmarkTask("task", "p", "e", {"evaluation": "semantic"}), agent, "x")
    with pytest.raises(ValueError, match="non-empty string list"):
        evaluator.evaluate(
            BenchmarkTask("task", "p", "e", {"evaluation": "required_keywords", "required_keywords": []}),
            agent,
            "x",
        )


def test_suite_runner_reports_counts_average_and_per_task_details() -> None:
    tasks = (
        BenchmarkTask("one", "p", "yes"),
        BenchmarkTask("two", "p", "yes"),
    )
    report = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator()).run(
        tasks, make_agent(one="yes", two="no")
    )

    assert (report.total, report.passed, report.failed) == (2, 1, 1)
    assert report.aggregate_score == 0.5
    assert [result.task_id for result in report.results] == ["one", "two"]
    assert report.results[1].details == "expected 'yes'; received 'no'"


def test_empty_suite_has_zero_aggregate() -> None:
    report = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator()).run((), make_agent())
    assert (report.total, report.passed, report.failed, report.aggregate_score) == (0, 0, 0, 0.0)
