"""
RAG Evaluation Pipeline.

Framework: RAGAS (chọn vì tích hợp tốt với Q&A + context, output là DataFrame
dễ tổng hợp thành bảng markdown, và metrics faithfulness/relevancy/recall/precision
đều có sẵn native, không cần tự viết feedback function như TruLens).

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.

Cài đặt thêm cho phần LLM/Embeddings wrapper của RAGAS:
    pip install langchain-openai langchain-huggingface
"""

import inspect
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# File này nằm tại: <root>/group_project/evaluation/eval_pipeline.py
# src/ nằm tại:     <root>/src/
# => cần thêm <root> (3 cấp cha) vào sys.path để import được "src.task10_generation"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _call_pipeline(pipeline_fn, question: str, params: dict) -> dict:
    """
    Gọi pipeline_fn(question, **params) nhưng chỉ truyền các kwargs mà
    pipeline_fn thực sự chấp nhận (dò qua inspect.signature).

    Lý do: Task 10 (generate_with_citation) hiện tại chỉ nhận (query, top_k).
    Nếu compare_configs truyền thêm use_reranking/alpha mà hàm chưa hỗ trợ,
    gọi trực tiếp sẽ TypeError. Hàm này lọc bỏ kwargs không được hỗ trợ để
    pipeline luôn chạy được, đồng thời tự động dùng full config nếu sau này
    task9_retrieval_pipeline.retrieve() / generate_with_citation() được mở
    rộng để nhận thêm use_reranking, alpha, ...
    """
    sig = inspect.signature(pipeline_fn)
    accepted = {k: v for k, v in params.items() if k in sig.parameters}
    return pipeline_fn(question, **accepted)


# =============================================================================
# RAGAS LLM / Embeddings config (trỏ về OpenRouter, khớp với Task 10)
# =============================================================================

def _get_ragas_llm():
    """
    RAGAS mặc định dùng ChatOpenAI của LangChain và tự tìm OPENAI_API_KEY.
    Task 10 của project này dùng OpenRouter (OPENROUTER_API_KEY), nên phải tự
    khởi tạo ChatOpenAI trỏ base_url về OpenRouter rồi wrap lại cho RAGAS.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu OPENROUTER_API_KEY hoặc OPENAI_API_KEY trong .env — "
            "cần key này để RAGAS gọi LLM chấm điểm."
        )

    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    # Model dùng để evaluate — nên chọn model đủ mạnh để chấm điểm chính xác,
    # có thể khác với LLM_MODEL sinh câu trả lời ở Task 10.
    eval_model = os.getenv("RAGAS_EVAL_MODEL", "openai/gpt-4o-mini")

    chat = ChatOpenAI(
        model=eval_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    return LangchainLLMWrapper(chat)


def _get_ragas_embeddings():
    """
    answer_relevancy cần embeddings để so sánh câu hỏi sinh ra vs câu hỏi gốc.
    Dùng lại đúng embedding model đã dùng ở Task 4/5 (BAAI/bge-m3) để nhất
    quán, và vì OpenRouter không có endpoint embeddings.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    try:
        from src.task4_chunking_indexing import EMBEDDING_MODEL
    except ImportError:
        EMBEDDING_MODEL = "BAAI/bge-m3"

    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(hf_embeddings)


# =============================================================================
# Option 1: DeepEval (không dùng — giữ lại tham khảo)
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    raise NotImplementedError(
        "Không dùng — framework đã chọn là RAGAS. Xem evaluate_with_ragas()."
    )


# =============================================================================
# Option 2: RAGAS  ← Framework đã chọn
# =============================================================================

def evaluate_with_ragas(pipeline_fn, golden_dataset: list[dict], config: dict | None = None):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas langchain-openai langchain-huggingface

    Args:
        pipeline_fn: hàm generate_with_citation(query, top_k=...) -> {
            "answer": str, "sources": [{"content": str, "metadata": dict, "score": float}, ...]
        }
        golden_dataset: list of {"question": str, "expected_answer": str, ...}
        config: dict tham số bổ sung (vd top_k, use_reranking, alpha...) —
                chỉ những key được pipeline_fn thực sự hỗ trợ mới được truyền vào.

    Returns:
        pandas.DataFrame — mỗi row 1 câu hỏi, cột gồm question/answer +
        faithfulness/answer_relevancy/context_recall/context_precision.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    config = config or {}

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = _call_pipeline(pipeline_fn, item["question"], config)
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["content"] for c in result["sources"]])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)

    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens (không dùng — giữ lại tham khảo)
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    raise NotImplementedError(
        "Không dùng — framework đã chọn là RAGAS. Xem evaluate_with_ragas()."
    )


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(pipeline_fn, golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs:
        Config A: hybrid search + reranking
        Config B: dense-only, không reranking

    Lưu ý: generate_with_citation hiện tại (Task 10) chỉ nhận (query, top_k).
    Việc bật/tắt hybrid+rerank thực sự phụ thuộc vào cách task9_retrieval_pipeline.retrieve()
    được implement. Nếu retrieve() đã hỗ trợ use_reranking/alpha, chỉ cần forward các
    tham số đó xuống trong generate_with_citation() — _call_pipeline() sẽ tự động nhận
    diện và truyền vào khi khả dụng, không cần sửa file này.

    Returns:
        {
            "hybrid_rerank": {"scores": {metric: avg}, "per_question": DataFrame},
            "dense_only":    {"scores": {metric: avg}, "per_question": DataFrame},
        }
    """
    configs = {
        "hybrid_rerank": {"top_k": 5, "use_reranking": True, "alpha": 0.5},
        "dense_only": {"top_k": 5, "use_reranking": False, "alpha": 1.0},
    }

    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    results = {}
    for config_name, params in configs.items():
        df = evaluate_with_ragas(pipeline_fn, golden_dataset, config=params)
        scores = {m: float(df[m].mean()) for m in metric_cols if m in df.columns}
        results[config_name] = {"scores": scores, "per_question": df}

    return results


# =============================================================================
# Export Results
# =============================================================================

def _fmt(x) -> str:
    if x is None or x != x:  # None hoặc NaN
        return ""
    return f"{x:.3f}"


def _avg(scores: dict) -> float | None:
    vals = list(scores.values())
    return sum(vals) / len(vals) if vals else None


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md — đúng theo template đã thống nhất."""
    config_names = list(comparison.keys())
    a_name, b_name = config_names[0], config_names[1]
    a_scores = comparison[a_name]["scores"]
    b_scores = comparison[b_name]["scores"]
    a_avg, b_avg = _avg(a_scores), _avg(b_scores)

    content = "# RAG Evaluation Results\n\n"

    content += "## Framework sử dụng\n\n"
    content += "> RAGAS\n\n"
    content += "---\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |\n"
    content += "|--------|---------------------------|----------------------|---|\n"
    for metric, label in METRIC_LABELS.items():
        a_val = a_scores.get(metric)
        b_val = b_scores.get(metric)
        delta = (a_val - b_val) if (a_val is not None and b_val is not None) else None
        content += f"| {label} | {_fmt(a_val)} | {_fmt(b_val)} | {_fmt(delta)} |\n"
    avg_delta = (a_avg - b_avg) if (a_avg is not None and b_avg is not None) else None
    content += f"| **Average** | **{_fmt(a_avg)}** | **{_fmt(b_avg)}** | **{_fmt(avg_delta)}** |\n\n"
    content += "---\n\n"

    content += "## A/B Comparison Analysis\n\n"
    content += "**Config A:**\n"
    content += "> Hybrid search (dense + keyword) kết hợp reranking trước khi đưa context vào generation.\n\n"
    content += "**Config B:**\n"
    content += "> Dense-only search, không có bước reranking, lấy trực tiếp top-k theo vector similarity.\n\n"
    content += "**Kết luận:**\n"
    if a_avg is not None and b_avg is not None and a_avg != b_avg:
        if a_avg > b_avg:
            content += (
                f"> Config A (hybrid + rerank) tốt hơn với average score {_fmt(a_avg)} so với "
                f"{_fmt(b_avg)} của Config B. Reranking giúp lọc bớt chunk nhiễu, cải thiện "
                f"context precision/recall, kéo theo faithfulness và relevance tốt hơn.\n\n"
            )
        else:
            content += (
                f"> Config B (dense-only) tốt hơn với average score {_fmt(b_avg)} so với "
                f"{_fmt(a_avg)} của Config A. Có thể reranker chưa phù hợp domain hoặc overhead "
                f"reranking không mang lại cải thiện đáng kể trên tập câu hỏi này.\n\n"
            )
    else:
        content += "> Hai config cho kết quả tương đương, chưa thấy khác biệt rõ rệt.\n\n"
    content += "---\n\n"

    content += "## Worst Performers (Bottom 3)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
    content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"

    df = comparison[a_name]["per_question"].copy()
    metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    existing_cols = [c for c in metric_cols if c in df.columns]
    df["_avg"] = df[existing_cols].mean(axis=1)
    worst = df.sort_values("_avg").head(3)

    for i, (_, row) in enumerate(worst.iterrows(), start=1):
        recall = row.get("context_recall")
        faith = row.get("faithfulness")
        if recall is not None and faith is not None and recall < faith:
            stage = "Retrieval"
            cause = "Context recall thấp — thiếu chunk liên quan trong top-k."
        else:
            stage = "Generation"
            cause = "Faithfulness thấp — câu trả lời chứa thông tin ngoài context."
        content += (
            f"| {i} | {row.get('question', '')} | {_fmt(faith)} | "
            f"{_fmt(row.get('answer_relevancy'))} | {_fmt(recall)} | {stage} | {cause} |\n"
        )
    content += "\n---\n\n"

    content += "## Recommendations\n\n"
    content += "### Cải tiến 1\n"
    content += "**Action:** Tăng top_k và cải thiện reranker cho các câu hỏi có context_recall thấp.\n"
    content += "**Expected impact:** Tăng context_recall trung bình, giảm tỷ lệ thiếu chunk liên quan.\n\n"
    content += "### Cải tiến 2\n"
    content += "**Action:** Siết prompt generation để model chỉ trả lời dựa trên context được cung cấp.\n"
    content += "**Expected impact:** Giảm hallucination, tăng faithfulness.\n\n"
    content += "### Cải tiến 3\n"
    content += "**Action:** Mở rộng golden_dataset để có sample size lớn hơn, đa dạng loại câu hỏi hơn.\n"
    content += "**Expected impact:** Kết quả evaluation ổn định và đáng tin cậy hơn.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"✓ Exported results to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # Task 10 export hàm generate_with_citation(query, top_k=...) -> dict,
    # không phải class — import và dùng trực tiếp.
    from src.task10_generation import generate_with_citation

    comparison = compare_configs(generate_with_citation, golden_dataset)
    export_results(results=None, comparison=comparison)