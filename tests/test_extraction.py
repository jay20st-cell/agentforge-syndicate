import json

from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicEvaluator
from agentforge.extraction_demo import run_demo
from agentforge.improvement import FailureDiagnoser, PromotionPolicy, compare_suites
from agentforge.models import AgentVersion, BenchmarkTask, SuiteResult


class RecordingTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, model, prompt, timeout):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _task(expected='{"invoice_id":"INV-9","amount":"50.00 USD"}'):
    return BenchmarkTask(
        "invoice",
        "Invoice INV-9 has an amount of 50.00 USD.",
        expected,
        {
            "evaluation": "json_fields",
            "required_fields": ["invoice_id", "amount"],
            "category": "invoice",
        },
    )


def test_valid_json_fields_score_individually_by_name() -> None:
    result = DeterministicEvaluator(1.0).evaluate(
        _task(), AgentVersion("a", "1"), '{"invoice_id":"INV-9","amount":"wrong"}'
    )

    assert result.score == 0.5
    assert result.passed is False
    assert result.matched_requirements == ("invoice_id",)
    assert result.missing_requirements == ("amount",)
    assert "50.00 USD" not in result.details


def test_malformed_json_fails_safely_with_field_names_only() -> None:
    result = DeterministicEvaluator(0.5).evaluate(
        _task(), AgentVersion("a", "1"), "not JSON"
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.missing_requirements == ("invoice_id", "amount")
    assert "INV-9" not in result.details
    assert "50.00 USD" not in result.details


def test_expected_values_do_not_appear_in_failure_diagnosis() -> None:
    task = _task()
    evaluation = DeterministicEvaluator(1.0).evaluate(
        task, AgentVersion("a", "1"), '{"invoice_id":"wrong"}'
    )
    diagnosis = FailureDiagnoser().diagnose((task,), SuiteResult("1", (evaluation,)))[0]

    serialized = repr(diagnosis)
    assert diagnosis.missing_requirements == ("invoice_id", "amount")
    assert "INV-9" not in serialized
    assert "50.00 USD" not in serialized


def test_extraction_demo_improves_and_never_sends_expected_values() -> None:
    tasks = load_benchmark("benchmarks/structured-extraction.json")
    by_id = {task.id: json.loads(task.expected_output) for task in tasks}
    transport = RecordingTransport(
        [json.dumps(by_id["purchase-order-copperline"]), json.dumps(by_id["support-ticket-access"])]
    )
    run = run_demo(model="fake", transport=transport)

    assert run["baseline_result"].aggregate_score == 0.75
    assert run["baseline_result"].failed == 2
    assert run["comparison"].improved_tasks == (
        "purchase-order-copperline",
        "support-ticket-access",
    )
    assert run["comparison"].regressed_tasks == ()
    assert run["decision"].promoted is True
    assert run["candidate_agent"].configuration["responses"]["invoice-northwind"] == run[
        "baseline_agent"
    ].configuration["responses"]["invoice-northwind"]
    for task, prompt in zip((tasks[1], tasks[2]), transport.prompts):
        assert "expected_output" not in prompt
        for value in json.loads(task.expected_output).values():
            # Values present in the source document are necessarily recoverable from
            # task_prompt; this checks that no evaluator-only answer payload is sent.
            assert prompt.count(str(value)) <= task.prompt.count(str(value)) + json.loads(prompt.split("INPUT:\n", 1)[1])["baseline_response"].count(str(value))


def test_extraction_regression_causes_reject() -> None:
    tasks = load_benchmark("benchmarks/structured-extraction.json")
    expected = [json.loads(task.expected_output) for task in tasks]
    transport = RecordingTransport(
        [json.dumps(expected[1]), json.dumps({**expected[2], "priority": "low"})]
    )
    run = run_demo(model="fake", transport=transport)
    candidate = run["candidate_result"]
    # Compare against a deliberately stronger reference on one task to exercise
    # the unchanged promotion gate's regression rule with extraction results.
    responses = dict(run["baseline_agent"].configuration["responses"])
    responses["support-ticket-access"] = json.dumps(expected[2])
    from agentforge.execution import DeterministicAgentRunner, SuiteRunner
    stronger = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator(1.0)).run(
        tasks, AgentVersion("structured-extraction-agent", "strong", {"responses": responses})
    )
    comparison = compare_suites(stronger, candidate)
    decision = PromotionPolicy().decide(stronger, candidate, comparison, ())

    assert "support-ticket-access" in comparison.regressed_tasks
    assert decision.promoted is False
