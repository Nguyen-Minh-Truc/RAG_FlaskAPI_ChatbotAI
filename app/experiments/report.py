"""Reporting helpers for chunk parameter experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_comparison_report(run_output: dict) -> dict:
    """Build sorted comparison payload for all chunk configurations."""
    results = list(run_output.get("results", []))
    ranked = sorted(results, key=lambda item: item.get("avg_accuracy_proxy", 0.0), reverse=True)

    ranking = []
    for idx, item in enumerate(ranked, start=1):
        ranking.append(
            {
                "rank": idx,
                "chunk_size": item.get("chunk_size"),
                "chunk_overlap": item.get("chunk_overlap"),
                "avg_accuracy_proxy": item.get("avg_accuracy_proxy"),
                "avg_groundedness": item.get("avg_groundedness"),
                "avg_relevance": item.get("avg_relevance"),
                "avg_retrieval_score": item.get("avg_retrieval_score"),
                "avg_latency_sec": item.get("avg_latency_sec"),
                "chunks": item.get("chunks"),
                "ingest_time_sec": item.get("ingest_time_sec"),
            }
        )

    best = ranking[0] if ranking else None
    worst = ranking[-1] if ranking else None

    return {
        "generated_at": _utc_timestamp(),
        "source_file": run_output.get("source_file"),
        "top_k": run_output.get("top_k"),
        "question_count": run_output.get("question_count"),
        "ranking": ranking,
        "best_configuration": best,
        "worst_configuration": worst,
        "raw_results": results,
        "notes": {
            "accuracy_metric": "answer-based proxy (groundedness/relevance/coverage/clarity)",
            "limitations": "No ground-truth labels; use ranking comparatively and validate with manual spot-check.",
        },
    }


def write_comparison_report(run_output: dict, output_dir: str | None = None) -> str:
    """Write report JSON file and return output path."""
    report = build_comparison_report(run_output)

    report_dir = Path(output_dir or "storage/experiments")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"chunk_grid_report_{report['generated_at']}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(report_path)
