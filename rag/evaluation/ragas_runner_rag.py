"""
rag/evaluation/ragas_runner_rag.py

RAGAS evaluation runner for CiteRagLab.

RAGAS by default tries to use OpenAI directly. We configure it to use
Azure OpenAI instead — the same deployments used by the rest of the stack:
    LLM judge  : AZURE_OPENAI_LLM_KEY  + AZURE_LLM_DEPLOYMENT_41_MINI
    Embeddings : AZURE_OPENAI_EMB_KEY  + AZURE_EMB_DEPLOYMENT

Metrics evaluated
─────────────────
    faithfulness       — does the answer stay grounded in the context?
    answer_relevancy   — how relevant is the answer to the question?
    context_precision  — are the retrieved chunks actually useful?
    context_recall     — does the context cover the ground-truth answer?

Install deps:
    pip install ragas datasets
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.evaluation.ragas_runner_rag")


def _build_azure_llm():
    """Build a LangChain AzureChatOpenAI instance for RAGAS to use as its judge."""
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
        azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_LLM_KEY", ""),
        api_version=os.getenv("AZURE_LLM_API_VERSION", "2024-02-01"),
        temperature=0,
    )


def _build_azure_embeddings():
    """Build a LangChain AzureOpenAIEmbeddings instance for RAGAS to use."""
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large"),
        azure_endpoint=os.getenv("AZURE_EMB_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_EMB_KEY", ""),
        api_version=os.getenv("AZURE_EMB_API_VERSION", "2024-12-01-preview"),
    )


def run_ragas_evaluation(
    questions:     list[str],
    answers:       list[str],
    contexts:      list[list[str]],   # one list of chunk strings per question
    ground_truths: list[str],
) -> dict:
    """
    Run RAGAS evaluation using Azure OpenAI as the judge.

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

    On failure:
        {"error": str}
    """
    logger.info(
        "📊 run_ragas_evaluation — %d question(s) to evaluate",
        len(questions),
    )

    if not questions:
        logger.warning("   ⚠️  run_ragas_evaluation: empty question list — aborting")
        return {"error": "Empty question list supplied"}

    # ── Import RAGAS ──────────────────────────────────────────────────────────
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

    # ── Configure Azure OpenAI as the RAGAS judge ─────────────────────────────
    # RAGAS uses LangChain under the hood — we pass Azure LLM + embeddings
    # explicitly so it never falls back to looking for OPENAI_API_KEY.
    try:
        azure_llm        = _build_azure_llm()
        azure_embeddings = _build_azure_embeddings()
        logger.info(
            "   ✅ Azure OpenAI configured for RAGAS "
            "(llm=%s, embeddings=%s)",
            os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
            os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large"),
        )
    except Exception as err:
        logger.error("   ❌ Failed to build Azure clients for RAGAS: %s", err)
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

    # ── Run RAGAS with Azure OpenAI ───────────────────────────────────────────
    logger.info(
        "   🔬 Running RAGAS evaluation "
        "(faithfulness, relevancy, precision, recall)…"
    )
    try:
        import traceback
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=azure_llm,
            embeddings=azure_embeddings,
        )
        logger.info("   🔬 Raw RAGAS result type: %s", type(result))

        # RAGAS returns an EvaluationResult object — convert to DataFrame to
        # extract per-metric averages reliably regardless of RAGAS version.
        df = result.to_pandas()
        logger.info("   🔬 RAGAS result columns: %s", list(df.columns))

        def _col_avg(col: str) -> float:
            if col not in df.columns:
                return 0.0
            return round(float(df[col].dropna().mean()), 4)

        scores = {
            "faithfulness":      _col_avg("faithfulness"),
            "answer_relevancy":  _col_avg("answer_relevancy"),
            "context_precision": _col_avg("context_precision"),
            "context_recall":    _col_avg("context_recall"),
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
        logger.error("   🔬 Full traceback:\n%s", traceback.format_exc())
        return {"error": str(err)}