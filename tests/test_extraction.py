import json

from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.extraction_demo import run_demo
from agentforge.improvement import FailureDiagnoser, PromotionPolicy, compare_suites
from agentforge.models import AgentVersion, BenchmarkTask, EvaluationResult, SuiteResult
from agentforge.ollama import OllamaCandidateImprover


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


def _repair(baseline_response, model_response):
    task = _task()
    runner = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator(1.0))
    baseline = AgentVersion("a", "1", {"responses": {task.id: baseline_response}})
    baseline_result = runner.run((task,), baseline)
    diagnoses = FailureDiagnoser().diagnose((task,), baseline_result)
    transport = RecordingTransport([model_response])
    improver = OllamaCandidateImprover(
        "fake", transport=transport, candidate_version="2"
    )
    candidate = improver.improve(baseline, (task,), baseline_result, diagnoses)
    return candidate.configuration["responses"][task.id], transport, improver


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
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        assert "baseline_response" not in payload
        for value in json.loads(task.expected_output).values():
            # Values present in the source document are necessarily recoverable from
            # task_prompt; no evaluator-only answer payload is sent.
            assert prompt.count(str(value)) <= task.prompt.count(str(value))


def test_json_repair_only_replaces_diagnosed_fields() -> None:
    repaired, _, _ = _repair(
        '{"invoice_id":"INV-9","amount":"wrong","note":"keep exactly"}',
        '{"invoice_id":"ATTACK","amount":"50.00 USD","note":"overwrite","extra":1}',
    )

    parsed = json.loads(repaired)
    assert parsed == {
        "invoice_id": "INV-9",
        "amount": "50.00 USD",
        "note": "keep exactly",
    }


def test_malformed_json_proposal_retains_baseline_safely() -> None:
    baseline = '{"invoice_id":"INV-9","amount":"wrong"}'
    repaired, _, improver = _repair(baseline, "not JSON")

    assert repaired == baseline
    assert "model response is not valid JSON" in improver.generation_failures[0]


def test_partial_json_proposal_preserves_unrepaired_baseline_values() -> None:
    task = BenchmarkTask(
        "invoice",
        "Invoice INV-9 is for Acme and totals 50.00 USD.",
        '{"invoice_id":"INV-9","customer":"Acme","amount":"50.00 USD"}',
        {
            "evaluation": "json_fields",
            "required_fields": ["invoice_id", "customer", "amount"],
        },
    )
    baseline_text = '{"invoice_id":"INV-9","customer":"wrong","amount":"also wrong"}'
    runner = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator(1.0))
    baseline = AgentVersion("a", "1", {"responses": {task.id: baseline_text}})
    result = runner.run((task,), baseline)
    diagnosis = FailureDiagnoser().diagnose((task,), result)
    candidate = OllamaCandidateImprover(
        "fake", transport=RecordingTransport(['{"customer":"Acme"}']), candidate_version="2"
    ).improve(baseline, (task,), result, diagnosis)

    assert json.loads(candidate.configuration["responses"][task.id]) == {
        "invoice_id": "INV-9",
        "customer": "Acme",
        "amount": "also wrong",
    }


def test_extraction_regression_causes_reject() -> None:
    baseline = SuiteResult(
        "1",
        (
            EvaluationResult("gain", "1", 0.5, False),
            EvaluationResult("loss", "1", 1.0, True),
        ),
    )
    candidate = SuiteResult(
        "2",
        (
            EvaluationResult("gain", "2", 1.0, True),
            EvaluationResult("loss", "2", 0.5, False),
        ),
    )
    comparison = compare_suites(baseline, candidate)
    decision = PromotionPolicy().decide(baseline, candidate, comparison, ())

    assert comparison.regressed_tasks == ("loss",)
    assert decision.promoted is False
