"""Run chunk-size/chunk-overlap grid experiments and export comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import config
from app.experiments.report import write_comparison_report
from app.experiments.runner import run_chunk_grid_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunk parameter experiments")
    parser.add_argument("--file", required=True, help="Path to source PDF/DOC/DOCX file")
    parser.add_argument("--max-questions", type=int, default=20, help="Max historical questions to evaluate")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Retriever top-k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_output = run_chunk_grid_experiment(
        source_file_path=args.file,
        max_questions=args.max_questions,
        top_k=args.top_k,
    )
    report_path = write_comparison_report(run_output)

    ranked = sorted(
        run_output["results"],
        key=lambda item: item.get("avg_accuracy_proxy", 0.0),
        reverse=True,
    )

    print("\n=== Chunk Tuning Summary ===")
    for idx, item in enumerate(ranked[:5], start=1):
        print(
            f"{idx}. chunk_size={item['chunk_size']}, overlap={item['chunk_overlap']} "
            f"| accuracy={item['avg_accuracy_proxy']:.4f} "
            f"| latency={item['avg_latency_sec']:.4f}s"
        )

    print(f"\nReport saved to: {report_path}")
    print(json.dumps({"report_path": report_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
