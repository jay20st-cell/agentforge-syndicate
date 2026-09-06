import json

from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.improvement import FailureDiagnoser, PromotionPolicy, compare_suites
from agentforge.models import AgentVersion, BenchmarkTask
from agentforge.ollama import OllamaCandidateImprover, UrllibOllamaTransport


class FakeTransport:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, model: str, prompt: str, timeout: float) -> str:
        self.calls.append((model, prompt, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_http_transport_sets_temperature_zero(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"response":"ok"}'

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert UrllibOllamaTransport().generate("model", "prompt", 7) == "ok"
    assert captured["body"]["options"] == {"temperature": 0}


def _run(tasks, responses, generated):
    runner = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator(1.0))
    baseline = AgentVersion("agent", "1", {"responses": responses})
    baseline_result = runner.run(tasks, baseline)
    diagnoses = FailureDiagnoser().diagnose(tasks, baseline_result)
    transport = FakeTransport(generated)
    improver = OllamaCandidateImprover("test-model", transport=transport, candidate_version="2")
    candidate = improver.improve(baseline, tasks, baseline_result, diagnoses)
    return runner, baseline, baseline_result, diagnoses, transport, improver, candidate


def test_model_boundary_excludes_expected_output_and_preserves_passing_response() -> None:
    sentinel = "SECRET_EXPECTED_OUTPUT_SENTINEL"
    tasks = (
        BenchmarkTask("pass", "Already correct", "exact bytes"),
        BenchmarkTask(
            "fail",
            "Repair this",
            sentinel,
            {
                "evaluation": "required_keywords",
                "required_keywords": ["needed"],
                "category": "support",
                "unsafe_note": sentinel,
            },
        ),
    )
    result = _run(tasks, {"pass": "exact bytes", "fail": "baseline"}, ["baseline needed"])
    _, _, _, _, transport, _, candidate = result

    assert len(transport.calls) == 1
    prompt = transport.calls[0][1]
    assert sentinel not in prompt
    assert "expected_output" not in prompt
    assert candidate.configuration["responses"]["pass"] == "exact bytes"
    assert candidate.configuration["responses"]["fail"] == "baseline needed"


def test_only_diagnosed_failures_are_proposed_and_exact_match_fails_closed() -> None:
    tasks = (
        BenchmarkTask("pass", "p", "ok"),
        BenchmarkTask("exact", "exact prompt", "TOP SECRET"),
        BenchmarkTask(
            "keywords",
            "keyword prompt",
            "answer",
            {"evaluation": "required_keywords", "required_keywords": ["missing"]},
        ),
    )
    result = _run(
        tasks,
        {"pass": "ok", "exact": "wrong", "keywords": "base"},
        ["base missing"],
    )
    _, _, _, diagnoses, transport, _, candidate = result

    assert {item.task_id for item in diagnoses} == {"exact", "keywords"}
    assert len(transport.calls) == 1
    assert "keyword prompt" in transport.calls[0][1]
    assert "TOP SECRET" not in transport.calls[0][1]
    assert candidate.configuration["responses"]["pass"] == "ok"
    assert candidate.configuration["responses"]["exact"] == "wrong"


def test_model_failure_retains_baseline_response() -> None:
    task = BenchmarkTask(
        "fail", "p", "answer", {"evaluation": "required_keywords", "required_keywords": ["needed"]}
    )
    result = _run((task,), {"fail": "original"}, [RuntimeError("offline")])
    _, baseline, _, _, _, improver, candidate = result

    assert candidate.configuration["responses"]["fail"] == "original"
    assert baseline.configuration["responses"]["fail"] == "original"
    assert improver.generation_failures == ("fail: offline",)


def test_malformed_empty_generation_retains_baseline_response() -> None:
    task = BenchmarkTask(
        "fail", "p", "answer", {"evaluation": "required_keywords", "required_keywords": ["needed"]}
    )
    result = _run((task,), {"fail": "original"}, ["   "])
    candidate = result[6]

    assert candidate.configuration["responses"]["fail"] == "original"
    assert "no non-empty response" in result[5].generation_failures[0]


def test_regression_gate_rejects_llm_candidate() -> None:
    tasks = (
        BenchmarkTask(
            "gain", "p1", "unused", {"evaluation": "required_keywords", "required_keywords": ["a", "b"]}
        ),
        BenchmarkTask(
            "regress", "p2", "unused", {"evaluation": "required_keywords", "required_keywords": ["x", "y"]}
        ),
    )
    result = _run(tasks, {"gain": "", "regress": "x"}, ["a b", "unhelpful"])
    runner, _, baseline_result, _, _, _, candidate = result
    candidate_result = runner.run(tasks, candidate)
    comparison = compare_suites(baseline_result, candidate_result)
    decision = PromotionPolicy().decide(baseline_result, candidate_result, comparison, ())

    assert comparison.improved_tasks == ("gain",)
    assert comparison.regressed_tasks == ("regress",)
    assert decision.promoted is False


def test_critical_failure_still_blocks_promotion() -> None:
    tasks = (
        BenchmarkTask(
            "gain", "p1", "unused", {"evaluation": "required_keywords", "required_keywords": ["a"]}
        ),
        BenchmarkTask(
            "critical",
            "p2",
            "unused",
            {"evaluation": "required_keywords", "required_keywords": ["safe"], "critical": True},
        ),
    )
    result = _run(tasks, {"gain": "no", "critical": "no"}, ["a", "still no"])
    runner, _, baseline_result, _, _, _, candidate = result
    candidate_result = runner.run(tasks, candidate)
    diagnoses = FailureDiagnoser().diagnose(tasks, candidate_result)
    decision = PromotionPolicy().decide(
        baseline_result, candidate_result, compare_suites(baseline_result, candidate_result), diagnoses
    )

    assert decision.promoted is False
    assert "critical failure" in decision.reason


def test_prompt_is_structured_and_requests_response_only() -> None:
    task = BenchmarkTask(
        "fail", "Do it", "unused", {"evaluation": "required_keywords", "required_keywords": ["needed"]}
    )
    result = _run((task,), {"fail": "start"}, ["start needed"])
    prompt = result[4].calls[0][1]
    payload = json.loads(prompt.split("INPUT:\n", 1)[1])

    assert prompt.startswith("Return ONLY the improved response")
    assert payload["task_prompt"] == "Do it"
    assert payload["baseline_response"] == "start"
    assert payload["missing_requirements"] == ["needed"]
