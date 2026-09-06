"""Reproducible local execution of the bundled technical-support baseline."""

from __future__ import annotations

import json
from pathlib import Path

from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.models import AgentVersion


BASELINE_RESPONSES = {
    "http-bearer-auth": "Inspect the Authorization header for the Bearer scheme and verify token expiry.",
    "sqlite-database-locked": "Shorten transactions, set a busy timeout, and use WAL for concurrent readers.",
    "python-none-attribute": "Trace where None originates, validate the value, then call get only when valid.",
    "dns-resolution-failure": "This is DNS: inspect resolv.conf and use dig against the configured nameserver.",
    "git-merge-conflict": "Resolve conflict markers, test, stage with git add, and finish with git commit.",
    "missing-environment-variable": "The production environment is missing the DATABASE_URL configuration.",
    # Intentionally incomplete so the bundled suite demonstrates preserved failure detail.
    "http-timeout-debugging": "Check service health, routing, proxy configuration, and timeout logs.",
}


def run_baseline(path: str | Path) -> dict[str, object]:
    agent = AgentVersion("support-baseline", "1.0.0", {"responses": BASELINE_RESPONSES})
    report = SuiteRunner(
        DeterministicAgentRunner(), DeterministicEvaluator(pass_threshold=1.0)
    ).run(load_benchmark(path), agent)
    return {
        "agent_version": report.agent_version,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "aggregate_score": report.aggregate_score,
        "results": [
            {
                "task_id": result.task_id,
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
            }
            for result in report.results
        ],
    }


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    report = run_baseline(repository_root / "benchmarks" / "technical-support.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
