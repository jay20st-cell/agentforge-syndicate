"""Run AgentForge Mini's optional local Ollama improvement loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentforge.baseline import BASELINE_RESPONSES
from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.improvement import FailureDiagnoser, PromotionPolicy, compare_suites
from agentforge.models import AgentVersion
from agentforge.ollama import OllamaCandidateImprover, OllamaTransport


def run_demo(
    path: str | Path | None = None,
    *,
    model: str | None = None,
    endpoint: str = "http://localhost:11434/api/generate",
    timeout: float = 30.0,
    transport: OllamaTransport | None = None,
) -> dict[str, object]:
    benchmark_path = path or Path(__file__).resolve().parents[2] / "benchmarks" / "technical-support.json"
    tasks = load_benchmark(benchmark_path)
    runner = SuiteRunner(DeterministicAgentRunner(), DeterministicEvaluator(pass_threshold=1.0))
    diagnoser = FailureDiagnoser()
    baseline = AgentVersion("support-agent", "1.0.0", {"responses": BASELINE_RESPONSES})
    baseline_result = runner.run(tasks, baseline)
    baseline_diagnoses = diagnoser.diagnose(tasks, baseline_result)
    improver = OllamaCandidateImprover(
        model, endpoint=endpoint, timeout=timeout, transport=transport
    )
    candidate = improver.improve(baseline, tasks, baseline_result, baseline_diagnoses)
    candidate_result = runner.run(tasks, candidate)
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
        "generation_failures": improver.generation_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Ollama model (or AGENTFORGE_OLLAMA_MODEL)")
    parser.add_argument("--endpoint", default="http://localhost:11434/api/generate")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    run = run_demo(model=args.model, endpoint=args.endpoint, timeout=args.timeout)
    baseline = run["baseline_result"]
    candidate = run["candidate_result"]
    comparison = run["comparison"]
    decision = run["decision"]

    print(f"V1: score={baseline.aggregate_score:.4f} passed/failed={baseline.passed}/{baseline.failed}")
    print("failure diagnoses:")
    for diagnosis in run["baseline_diagnoses"]:
        print(f"- {diagnosis.task_id}: {diagnosis.explanation}")
    for failure in run["generation_failures"]:
        print(f"Ollama generation retained baseline: {failure}")
    print(f"V2: score={candidate.aggregate_score:.4f} passed/failed={candidate.passed}/{candidate.failed}")
    print(f"improved: {', '.join(comparison.improved_tasks) or 'none'}")
    print(f"unchanged: {', '.join(comparison.unchanged_tasks) or 'none'}")
    print(f"regressed: {', '.join(comparison.regressed_tasks) or 'none'}")
    print("PROMOTE" if decision.promoted else "REJECT")
    print(f"reason: {decision.reason}")


if __name__ == "__main__":
    main()
