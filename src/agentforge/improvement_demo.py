"""Run AgentForge Mini's deterministic V1-to-V2 improvement loop."""

from __future__ import annotations

from pathlib import Path

from agentforge.baseline import BASELINE_RESPONSES
from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.improvement import (
    DeterministicCandidateImprover,
    FailureDiagnoser,
    PromotionPolicy,
    compare_suites,
)
from agentforge.models import AgentVersion


def run_demo(path: str | Path | None = None) -> dict[str, object]:
    """Execute the complete loop and return its components for reuse and tests."""

    benchmark_path = path or Path(__file__).resolve().parents[2] / "benchmarks" / "technical-support.json"
    tasks = load_benchmark(benchmark_path)
    suite_runner = SuiteRunner(
        DeterministicAgentRunner(), DeterministicEvaluator(pass_threshold=1.0)
    )
    diagnoser = FailureDiagnoser()
    baseline = AgentVersion("support-agent", "1.0.0", {"responses": BASELINE_RESPONSES})
    baseline_result = suite_runner.run(tasks, baseline)
    baseline_diagnoses = diagnoser.diagnose(tasks, baseline_result)
    candidate = DeterministicCandidateImprover().improve(
        baseline, tasks, baseline_result, baseline_diagnoses
    )
    candidate_result = suite_runner.run(tasks, candidate)
    candidate_diagnoses = diagnoser.diagnose(tasks, candidate_result)
    comparison = compare_suites(baseline_result, candidate_result)
    decision = PromotionPolicy().decide(
        baseline_result, candidate_result, comparison, candidate_diagnoses
    )
    return {
        "baseline_agent": baseline,
        "candidate_agent": candidate,
        "baseline_result": baseline_result,
        "candidate_result": candidate_result,
        "baseline_diagnoses": baseline_diagnoses,
        "candidate_diagnoses": candidate_diagnoses,
        "comparison": comparison,
        "decision": decision,
    }


def main() -> None:
    run = run_demo()
    baseline = run["baseline_result"]
    candidate = run["candidate_result"]
    diagnoses = run["baseline_diagnoses"]
    comparison = run["comparison"]
    decision = run["decision"]

    print("BASELINE V1")
    print(f"score: {baseline.aggregate_score:.4f}")
    print(f"passed/failed: {baseline.passed}/{baseline.failed}")
    print("failure diagnoses:")
    for diagnosis in diagnoses:
        print(
            f"- {diagnosis.task_id} [{diagnosis.category}] score={diagnosis.baseline_score:.4f} "
            f"critical={str(diagnosis.critical).lower()} missing={', '.join(diagnosis.missing_requirements)}"
        )
    print("\nCANDIDATE V2")
    print(f"score: {candidate.aggregate_score:.4f}")
    print(f"passed/failed: {candidate.passed}/{candidate.failed}")
    print(f"improvements: {', '.join(comparison.improved_tasks) or 'none'}")
    print(f"regressions: {', '.join(comparison.regressed_tasks) or 'none'}")
    print("\nDECISION")
    print("PROMOTE" if decision.promoted else "REJECT")
    print(f"reason: {decision.reason}")


if __name__ == "__main__":
    main()
