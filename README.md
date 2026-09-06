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
- `src/agentforge/improvement.py` diagnoses failures, creates candidates,
  compares suite runs, and applies the promotion policy.
- `src/agentforge/improvement_demo.py` runs the complete V1-to-V2 loop.
- `src/agentforge/ollama.py` optionally proposes failed-response repairs with a
  configurable local Ollama model and no additional runtime dependency.
- `src/agentforge/ollama_demo.py` runs the same benchmark and deterministic gate
  around the optional model proposal step.
- `src/agentforge/extraction_demo.py` applies that same loop to structured
  business-document extraction.
- `benchmarks/sample.json` is a one-task example fixture.
- `benchmarks/technical-support.json` contains seven realistic debugging cases.
- `benchmarks/structured-extraction.json` contains four invoice, order, support,
  and shipment extraction cases.
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

The `json_fields` policy parses both outputs as JSON objects and scores the
field names listed in `metadata.required_fields` individually using exact JSON
value equality. Failure evidence contains only unmatched field names—never the
canonical values stored in `expected_output`. Malformed model JSON safely
scores zero rather than interrupting the suite.

`DeterministicAgentRunner` reads outputs from the `responses` mapping in an
`AgentVersion` configuration. `SuiteRunner` preserves each `EvaluationResult`
in benchmark order and reports total, passed, failed, and mean aggregate score.
Exact-match failures retain expected/actual details, keyword failures retain
missing keywords, and JSON-field failures retain field names only.

## Local setup

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
python -m agentforge.baseline
python -m agentforge.improvement_demo
# Requires a local Ollama server and model:
AGENTFORGE_OLLAMA_MODEL=llama3.2 python -m agentforge.ollama_demo
# Same architecture, second domain; requires the named local model:
python -m agentforge.extraction_demo --model llama3.1-8b:latest
```

The baseline command prints a deterministic JSON report for the technical
support suite. The runtime package has no third-party dependencies. Pytest is
used only for development and tests.

## Second-domain generality

AgentForge Mini governs both technical-support answers and structured
business-document extraction with the same sequence: deterministic V1
evaluation, field-safe failure diagnosis, optional local Ollama proposal, the
same benchmark rerun, regression analysis, and the existing deterministic
PROMOTE/REJECT policy. Passing responses remain untouched. For extraction,
Ollama receives the original document prompt, score, category, critical flag,
and missing field names; it does not receive the baseline object or canonical
`expected_output` answer object. Live model improvement is environment-dependent
and is not claimed here without a local run.

For `json_fields` failures, the model receives the original prompt and missing
field names and returns only a repair object. AgentForge ignores keys outside
that diagnosed set and overlays accepted fields on the parsed V1 object, so
already-correct fields cannot regress. Malformed proposals retain V1 unchanged.
Local HTTP generation uses temperature zero for more reproducible proposals.

## Current scope

This repository provides deterministic domain models, JSON fixture loading,
configured execution, exact/keyword/JSON-field evaluation, suite aggregation, structured
failure diagnosis, deterministic candidate creation from missing keyword
evidence, regression analysis, and promotion gating. Candidate creation does
not consume benchmark expected outputs; unsupported exact-match failures remain
unchanged. The deterministic improver remains the reproducible reference
strategy. An optional local Ollama strategy may propose repairs for diagnosed
keyword failures, while passing responses and unsafe exact-match failures remain
unchanged. Both strategies use the same deterministic evaluator, regression
analysis, and PROMOTE/REJECT policy: AI proposes, deterministic evidence
decides. Model success is environment-dependent and is not assumed. The project
does not provide API services, web interfaces, dashboards, or persistence.
