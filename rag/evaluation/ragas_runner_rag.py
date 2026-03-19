"""
rag/evaluation/ragas_runner_rag.py

RAGAS evaluation runner for CiteRagLab.

Metrics evaluated
─────────────────
    faithfulness       — does the answer stay grounded in the context?
    answer_relevancy   — how relevant is the answer to the question?
    context_precision  — are the retrieved chunks actually useful?
    context_recall     — does the context cover the ground-truth answer?

Install deps:
    pip install ragas datasets
"""

import logging

logger = logging.getLogger("rag.evaluation.ragas_runner_rag")


def run_ragas_evaluation(
    questions:    list[str],
    answers:      list[str],
    contexts:     list[list[str]],   # one list of chunk strings per question
    ground_truths: list[str],
) -> dict:
    """
    Run RAGAS evaluation on a dataset and return metric scores.

    Parameters
    ──────────
    questions     : list of user questions
    answers       : corresponding LLM answers (same order)
    contexts      : list of context-string lists — one per question
    ground_truths : expected answers / reference answers

    Returns
    ───────
    On success:
        {faithfulness, answer_relevancy, context_precision, context_recall}
        all values are floats rounded to 4 decimal places.

    On failure (missing deps or RAGAS error):
        {"error": str}
    """
    logger.info(
        "📊 run_ragas_evaluation — %d question(s) to evaluate",
        len(questions),
    )

    if not questions:
        logger.warning("   ⚠️  run_ragas_evaluation: empty question list — aborting")
        return {"error": "Empty question list supplied"}

    # ── Import RAGAS dependencies ─────────────────────────────────────────────
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        logger.info("   ✅ RAGAS and datasets imported successfully")
    except ImportError as err:
        logger.error(
            "   ❌ RAGAS or datasets not installed: %s — "
            "run `pip install ragas datasets` to enable evaluation",
            err,
        )
        return {"error": str(err)}

    # ── Build HuggingFace Dataset ─────────────────────────────────────────────
    logger.info("   📋 Building evaluation dataset…")
    data = {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    }
    try:
        dataset = Dataset.from_dict(data)
        logger.info(
            "   ✅ Dataset built — %d rows, columns: %s",
            len(dataset), list(dataset.column_names),
        )
    except Exception as err:
        logger.error("   ❌ Failed to build dataset: %s", err)
        return {"error": str(err)}

    # ── Run RAGAS ────────────────────────────────────────────────────────────
    logger.info("   🔬 Running RAGAS evaluation (faithfulness, relevancy, precision, recall)…")
    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        scores = {
            "faithfulness":      round(float(result["faithfulness"]),      4),
            "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
            "context_precision": round(float(result["context_precision"]), 4),
            "context_recall":    round(float(result["context_recall"]),    4),
        }
        logger.info(
            "   ✅ RAGAS evaluation complete — "
            "faithfulness=%.4f  relevancy=%.4f  precision=%.4f  recall=%.4f",
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_precision"],
            scores["context_recall"],
        )
        return scores

    except Exception as err:
        logger.error("   ❌ RAGAS evaluation failed: %s", err)
        return {"error": str(err)}