# AgentForge Mini

AgentForge Mini is a small, deterministic foundation for benchmarking versioned
agents during the Syndicate by Maximor hackathon. It defines the data and
extension boundaries needed to run repeatable benchmark experiments without
choosing an agent implementation, evaluation policy, service, or user interface.

## Architecture

- `src/agentforge/models.py` contains immutable benchmark, agent version,
  evaluation result, and promotion decision value objects.
- `src/agentforge/benchmarks.py` loads and validates task collections from JSON.
- `src/agentforge/interfaces.py` defines abstract runner and evaluator contracts.
- `benchmarks/sample.json` is a one-task example fixture.
- `tests/` verifies model validation, immutability, and benchmark loading.

The benchmark format is a JSON object with a `tasks` array. Each task requires
string fields named `id`, `prompt`, and `expected_output`; optional `metadata`
must be a JSON object. Task IDs must be unique within a file.

## Local setup

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

The runtime package has no third-party dependencies. Pytest is used only for
development and tests.

## Current scope

This repository currently provides deterministic domain models, JSON fixture
loading, and interfaces for future runner and evaluator implementations. It does
not implement agent self-improvement, external LLM calls, API services, web
interfaces, dashboards, persistence, or a promotion policy.
