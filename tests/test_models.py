import pytest

from agentforge.models import (
    AgentVersion,
    BenchmarkTask,
    EvaluationResult,
    PromotionDecision,
)


def test_models_store_deterministic_values() -> None:
    task = BenchmarkTask("task-1", "Say yes", "yes", {"category": "exact"})
    agent = AgentVersion("baseline", "1.0.0", {"temperature": 0})
    result = EvaluationResult(task.id, agent.version, 1.0, True, "exact match")
    decision = PromotionDecision("1.1.0", agent.version, True, "score improved")

    assert task.metadata["category"] == "exact"
    assert agent.configuration["temperature"] == 0
    assert result.passed is True
    assert decision.promoted is True


def test_mapping_fields_are_defensive_and_read_only() -> None:
    metadata = {"category": "exact"}
    task = BenchmarkTask("task-1", "Say yes", "yes", metadata)
    metadata["category"] = "changed"

    assert task.metadata["category"] == "exact"
    with pytest.raises(TypeError):
        task.metadata["category"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: BenchmarkTask("", "prompt", "answer"), "id"),
        (lambda: AgentVersion("agent", "", {}), "version"),
        (lambda: EvaluationResult("task", "1", 1.1, True), "score"),
        (lambda: PromotionDecision("2", "1", True, ""), "reason"),
    ],
)
def test_models_reject_invalid_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
