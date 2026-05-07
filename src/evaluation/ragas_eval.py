"""
ragas_eval.py — Evaluates the RAG pipeline using RAGAS metrics.

Metrics computed
----------------
- Faithfulness        : Is the answer grounded in the retrieved context?
- Answer Relevance    : Does the answer address the question?
- Context Precision   : Are the retrieved chunks relevant to the question?
- Context Recall      : Does the retrieved context cover the ground-truth answer?

Usage
-----
    python -m src.evaluation.ragas_eval --dataset_path data/eval_dataset.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from configs.embeddings_factory import get_embeddings
from configs.llm_factory import get_llm

from src.agents.graph import run_agent
from configs.loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

cfg = load_config()


def build_ragas_dataset(eval_questions: list[dict[str, str]]) -> Dataset:
    """
    Run the agent on each evaluation question and collect data for RAGAS.

    Each item in ``eval_questions`` must have:
    - ``question``       : str
    - ``ground_truth``   : str   (the reference answer)

    Returns a HuggingFace Dataset compatible with RAGAS.
    """
    records: list[dict[str, Any]] = []

    for i, item in enumerate(eval_questions):
        question = item["question"]
        ground_truth = item["ground_truth"]
        logger.info("[%d/%d] Running agent for: '%s'", i + 1, len(eval_questions), question)

        state = run_agent(question)

        answer = state.get("final_answer", "")
        contexts = [doc["content"] for doc in state.get("retrieved_docs", [])]

        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return Dataset.from_list(records)


def run_evaluation(dataset_path: str, output_path: str = "data/eval_results.csv") -> pd.DataFrame:
    """
    Load an evaluation dataset, run RAGAS, and save results to CSV.

    Parameters
    ----------
    dataset_path : str
        Path to a JSON file with a list of {question, ground_truth} dicts.
    output_path : str
        Where to save the results CSV.

    Returns
    -------
    pd.DataFrame
        Per-sample RAGAS scores.
    """
    with open(dataset_path, encoding="utf-8") as f:
        eval_questions: list[dict] = json.load(f)

    logger.info("Loaded %d evaluation questions.", len(eval_questions))

    dataset = build_ragas_dataset(eval_questions)

    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_llm(),
        embeddings=get_embeddings(),
    )

    scores_df: pd.DataFrame = result.to_pandas()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RAGAS Evaluation Summary")
    print("=" * 55)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in scores_df.columns:
            mean_score = scores_df[metric].mean()
            print(f"  {metric:<25} {mean_score:.4f}")
    print("=" * 55)
    print(f"  Results saved to: {output_path}")
    print("=" * 55 + "\n")

    return scores_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the agent.")
    parser.add_argument("--dataset_path", default="data/eval_dataset.json",
                        help="Path to evaluation Q&A JSON file.")
    parser.add_argument("--output_path", default="data/eval_results.csv",
                        help="Path to save evaluation results CSV.")
    args = parser.parse_args()

    run_evaluation(args.dataset_path, args.output_path)
