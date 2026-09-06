"""Run the AgentForge governance loop for structured business extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.extraction_baseline import extraction_baseline
from agentforge.improvement import FailureDiagnoser, PromotionPolicy, compare_suites
from agentforge.ollama import OllamaCandidateImprover, OllamaTransport


def run_demo(
    path: str | Path | None = None,
    *,
    model: str | None = None,
    endpoint: str = "http://localhost:11434/api/generate",
    timeout: float = 30.0,
    transport: OllamaTransport | None = None,
) -> dict[str, object]:
    benchmark_path = path or (
        Path(__file__).resolve().parents[2] / "benchmarks" / "structured-extraction.json"
    )
    tasks = load_benchmark(benchmark_path)
    suite_runner = SuiteRunner(
        DeterministicAgentRunner(), DeterministicEvaluator(pass_threshold=1.0)
    )
    diagnoser = FailureDiagnoser()
    baseline = extraction_baseline()
    baseline_result = suite_runner.run(tasks, baseline)
    baseline_diagnoses = diagnoser.diagnose(tasks, baseline_result)
    improver = OllamaCandidateImprover(
        model, endpoint=endpoint, timeout=timeout, transport=transport
    )
    candidate = improver.improve(
        baseline, tasks, baseline_result, baseline_diagnoses
    )
    candidate_result = suite_runner.run(tasks, candidate)
    candidate_diagnoses = diagnoser.diagnose(tasks, candidate_result)
    comparison = compare_suites(baseline_result, candidate_result)
    decision = PromotionPolicy().decide(
        baseline_result, candidate_result, comparison, candidate_diagnoses
    )
    return {
        "model": improver.model,
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

    print("SECOND DOMAIN: STRUCTURED EXTRACTION")
    print(f"V1 score: {baseline.aggregate_score:.4f}")
    print(f"V1 passed/failed: {baseline.passed}/{baseline.failed}")
    print("failed fields by task:")
    for diagnosis in run["baseline_diagnoses"]:
        print(f"- {diagnosis.task_id}: {', '.join(diagnosis.missing_requirements)}")
    print("\nLOCAL MODEL:")
    print(run["model"])
    for failure in run["generation_failures"]:
        print(f"candidate generation retained V1: {failure}")
    print(f"\nV2 score: {candidate.aggregate_score:.4f}")
    print(f"V2 passed/failed: {candidate.passed}/{candidate.failed}")
    print(f"improved tasks: {', '.join(comparison.improved_tasks) or 'none'}")
    print(f"unchanged tasks: {', '.join(comparison.unchanged_tasks) or 'none'}")
    print(f"regressed tasks: {', '.join(comparison.regressed_tasks) or 'none'}")
    print(f"\n{'PROMOTE' if decision.promoted else 'REJECT'}")
    print(f"reason: {decision.reason}")


if __name__ == "__main__":
    main()
