# AgentForge Mini

AgentForge Mini is a small, deterministic system for benchmarking versioned
agents during the Syndicate by Maximor hackathon. It runs configured responses
without network calls and scores them with transparent, repeatable policies.

## Architecture

- `src/agentforge/models.py` contains immutable benchmark, agent version,
  evaluation result, and promotion decision value objects.
- `src/agentforge/benchmarks.py` loads and validates task collections from JSON.
- `src/agentforge/interfaces.py` defines abstract runner and evaluator contracts.
- `src/agentforge/execution.py` provides the deterministic runner, evaluator,
  and suite orchestration layer.
- `src/agentforge/baseline.py` runs the bundled support-agent baseline.
- `benchmarks/sample.json` is a one-task example fixture.
- `benchmarks/technical-support.json` contains seven realistic debugging cases.
- `tests/` verifies models, loading, scoring, aggregation, and repeatability.

The benchmark format is a JSON object with a `tasks` array. Each task requires
string fields named `id`, `prompt`, and `expected_output`; optional `metadata`
must be a JSON object. Task IDs must be unique within a file.

Evaluation defaults to `exact_match`, which compares stripped, case-sensitive
text. A task can instead set `metadata.evaluation` to `required_keywords` and
provide a non-empty `metadata.required_keywords` string list. Keyword matching
is case-insensitive substring matching, and its normalized score is the matched
keyword count divided by the required keyword count. `DeterministicEvaluator`
uses an explicit pass threshold (default `0.8`), with scores equal to the
threshold passing.

`DeterministicAgentRunner` reads outputs from the `responses` mapping in an
`AgentVersion` configuration. `SuiteRunner` preserves each `EvaluationResult`
in benchmark order and reports total, passed, failed, and mean aggregate score.
Failures retain expected/actual values or missing keyword details.

## Local setup

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m agentforge.baseline
```

The baseline command prints a deterministic JSON report for the technical
support suite. The runtime package has no third-party dependencies. Pytest is
used only for development and tests.

## Current scope

This repository provides deterministic domain models, JSON fixture loading,
configured execution, exact/keyword evaluation, and suite aggregation. It does
not implement agent self-improvement, candidate generation, external LLM calls,
API services, web interfaces, dashboards, or persistence.
