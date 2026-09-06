"""Optional local Ollama-backed candidate generation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Protocol

from agentforge.interfaces import CandidateImprover
from agentforge.models import AgentVersion, BenchmarkTask, FailureDiagnosis, SuiteResult


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


class OllamaTransport(Protocol):
    """Small injectable boundary used to generate one response."""

    def generate(
        self, model: str, prompt: str, timeout: float, *, json_mode: bool = False
    ) -> str:
        """Return the generated text or raise when generation fails."""


class UrllibOllamaTransport:
    """Call Ollama's generate endpoint using only the Python standard library."""

    def __init__(self, endpoint: str = DEFAULT_OLLAMA_ENDPOINT) -> None:
        self.endpoint = endpoint

    def generate(
        self, model: str, prompt: str, timeout: float, *, json_mode: bool = False
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        generated = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(generated, str) or not generated.strip():
            raise RuntimeError("Ollama returned no non-empty response")
        return generated


class OllamaCandidateImprover(CandidateImprover):
    """Ask a local model to repair eligible failed responses, then stop."""

    def __init__(
        self,
        model: str | None = None,
        *,
        endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
        timeout: float = 30.0,
        candidate_version: str = "2.0.0-ollama",
        transport: OllamaTransport | None = None,
    ) -> None:
        self.model = model or os.environ.get("AGENTFORGE_OLLAMA_MODEL", "llama3.2")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Ollama model must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.candidate_version = candidate_version
        self.transport = transport or UrllibOllamaTransport(endpoint)
        self.generation_failures: tuple[str, ...] = ()

    def improve(
        self,
        baseline: AgentVersion,
        tasks: Sequence[BenchmarkTask],
        baseline_result: SuiteResult,
        diagnoses: Sequence[FailureDiagnosis],
    ) -> AgentVersion:
        if self.candidate_version == baseline.version:
            raise ValueError("candidate version must differ from baseline version")
        responses = baseline.configuration.get("responses")
        if not isinstance(responses, Mapping):
            raise ValueError("baseline configuration must contain a responses object")

        failed_ids = {result.task_id for result in baseline_result.results if not result.passed}
        diagnosed_ids = {diagnosis.task_id for diagnosis in diagnoses}
        if failed_ids != diagnosed_ids:
            raise ValueError("diagnoses must correspond exactly to baseline failures")
        if len(diagnosed_ids) != len(diagnoses):
            raise ValueError("diagnoses must contain each baseline failure exactly once")

        task_by_id = {task.id: task for task in tasks}
        if any(task_id not in task_by_id for task_id in diagnosed_ids):
            raise ValueError("diagnosis references an unknown benchmark task")

        candidate_responses = dict(responses)
        failures: list[str] = []
        for diagnosis in diagnoses:
            task = task_by_id[diagnosis.task_id]
            baseline_response = responses.get(task.id)
            if not isinstance(baseline_response, str):
                raise ValueError(f"baseline response for task {task.id!r} must be a string")

            # Exact-match evaluator evidence contains the expected answer. It must
            # never be sent to the model. Missing evidence is also insufficient.
            if task.metadata.get("evaluation", "exact_match") == "exact_match":
                continue
            if not diagnosis.missing_requirements:
                continue

            evaluation_policy = task.metadata.get("evaluation", "exact_match")
            baseline_object = None
            if evaluation_policy == "json_fields":
                try:
                    baseline_object = json.loads(baseline_response)
                except json.JSONDecodeError:
                    failures.append(f"{task.id}: baseline response is not valid JSON")
                    continue
                if not isinstance(baseline_object, dict):
                    failures.append(f"{task.id}: baseline response is not a JSON object")
                    continue

            prompt = self._build_prompt(task, baseline_response, diagnosis)
            try:
                generated = self.transport.generate(
                    self.model,
                    prompt,
                    self.timeout,
                    json_mode=evaluation_policy == "json_fields",
                )
                if not isinstance(generated, str) or not generated.strip():
                    raise RuntimeError("model returned no non-empty response")
            except Exception as exc:  # A failed proposal must leave V1 intact.
                failures.append(f"{task.id}: {exc}")
                continue
            if evaluation_policy == "json_fields":
                try:
                    proposed_fields = self._parse_json_object(generated)
                except ValueError as exc:
                    failures.append(f"{task.id}: {exc}")
                    continue
                missing_fields = set(diagnosis.missing_requirements)
                repaired = dict(baseline_object)
                repaired.update(
                    (key, value)
                    for key, value in proposed_fields.items()
                    if key in missing_fields
                )
                candidate_responses[task.id] = json.dumps(repaired)
            else:
                candidate_responses[task.id] = generated

        self.generation_failures = tuple(failures)
        configuration = dict(baseline.configuration)
        configuration["responses"] = candidate_responses
        return AgentVersion(baseline.name, self.candidate_version, configuration)

    @staticmethod
    def _parse_json_object(generated: str) -> dict[str, object]:
        """Parse a complete JSON object, optionally inside one exact JSON fence."""

        candidate = generated.strip()
        if candidate.startswith("```json"):
            lines = candidate.splitlines()
            if (
                len(lines) < 3
                or lines[0].strip() != "```json"
                or lines[-1].strip() != "```"
                or any("```" in line for line in lines[1:-1])
            ):
                raise ValueError("model response has an invalid JSON code fence")
            candidate = "\n".join(lines[1:-1]).strip()
        elif candidate.startswith("```") or candidate.endswith("```"):
            raise ValueError("model response has an invalid JSON code fence")

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("model response is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("model response is not a JSON object")
        return parsed

    def _build_prompt(
        self,
        task: BenchmarkTask,
        baseline_response: str,
        diagnosis: FailureDiagnosis,
    ) -> str:
        evidence: dict[str, object] = {
            "task_prompt": task.prompt,
            "failure_diagnosis": {
                "task_id": diagnosis.task_id,
                "category": diagnosis.category,
                "baseline_score": diagnosis.baseline_score,
                "missing_requirements": list(diagnosis.missing_requirements),
                "critical": diagnosis.critical,
                "explanation": diagnosis.explanation,
            },
            "missing_requirements": list(diagnosis.missing_requirements),
            "category": diagnosis.category,
            "critical": diagnosis.critical,
            # No free-form metadata is forwarded by default. Category and
            # criticality are the validated, safe metadata fields above.
            "safe_metadata": {},
        }
        if task.metadata.get("evaluation", "exact_match") == "json_fields":
            output_instruction = (
                "Return ONLY a valid JSON object containing proposed values for the "
                "diagnosed missing_requirements field names. Do not include any other "
                "keys. Infer values from task_prompt. Use no Markdown fences, preamble, "
                "or commentary. "
            )
        else:
            evidence["baseline_response"] = baseline_response
            output_instruction = (
                "Return ONLY the improved response, with no preamble or commentary. "
            )
        return (
            f"{output_instruction}Address the diagnosed missing requirements. "
            "Do not invent unrelated information. Produce concise, task-appropriate output.\n\n"
            f"INPUT:\n{json.dumps(evidence, sort_keys=True)}"
        )
