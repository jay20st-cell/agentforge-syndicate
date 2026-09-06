"""Purposefully incomplete deterministic baseline for structured extraction."""

from __future__ import annotations

import json
from pathlib import Path

from agentforge.benchmarks import load_benchmark
from agentforge.execution import DeterministicAgentRunner, DeterministicEvaluator, SuiteRunner
from agentforge.models import AgentVersion


EXTRACTION_BASELINE_RESPONSES = {
    "invoice-northwind": json.dumps(
        {
            "invoice_id": "INV-2048",
            "customer": "Northwind Garden Supply",
            "amount": "1842.75 USD",
            "due_date": "2026-10-15",
        }
    ),
    "purchase-order-copperline": json.dumps(
        {
            "po_number": "PO-7719",
            "supplier": "Copperline Office Works",
        }
    ),
    "support-ticket-access": json.dumps(
        {
            "ticket_id": "SR-55301",
            "priority": "high",
        }
    ),
    "shipment-record-lumina": json.dumps(
        {
            "order_id": "SO-88014",
            "carrier": "SwiftRoute Freight",
            "tracking_number": "SRF-9021176",
        }
    ),
}


def extraction_baseline() -> AgentVersion:
    return AgentVersion(
        "structured-extraction-agent",
        "1.0.0",
        {"responses": EXTRACTION_BASELINE_RESPONSES},
    )


def run_baseline(path: str | Path) -> dict[str, object]:
    report = SuiteRunner(
        DeterministicAgentRunner(), DeterministicEvaluator(pass_threshold=1.0)
    ).run(load_benchmark(path), extraction_baseline())
    return {
        "agent_version": report.agent_version,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "aggregate_score": report.aggregate_score,
        "results": report.results,
    }
