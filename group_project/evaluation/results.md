# RAG Evaluation Results

## Framework sử dụng

> RAGAS

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.742 | 0.609 | 0.133 |
| Answer Relevance | 0.481 | 0.546 | -0.065 |
| Context Recall | 0.867 | 1.000 | -0.133 |
| Context Precision | 0.933 | 0.933 | 0.000 |
| **Average** | **0.756** | **0.772** | **-0.016** |

---

## A/B Comparison Analysis

**Config A:**
> Hybrid search (dense + keyword) kết hợp reranking trước khi đưa context vào generation.

**Config B:**
> Dense-only search, không có bước reranking, lấy trực tiếp top-k theo vector similarity.

**Kết luận:**
> Config B (dense-only) tốt hơn với average score 0.772 so với 0.756 của Config A. Có thể reranker chưa phù hợp domain hoặc overhead reranking không mang lại cải thiện đáng kể trên tập câu hỏi này.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | COD là gì? | 0.000 | 0.000 | 1.000 | Generation | Faithfulness thấp — câu trả lời chứa thông tin ngoài context. |
| 2 | Shopee xử lý dữ liệu cá nhân của người dùng nhằm mục đích gì? | 1.000 | 0.000 | 0.000 | Retrieval | Context recall thấp — thiếu chunk liên quan trong top-k. |
| 3 | Người bán có trách nhiệm gì đối với thông tin sản phẩm? | 0.000 | 0.000 | 1.000 | Generation | Faithfulness thấp — câu trả lời chứa thông tin ngoài context. |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng top_k và cải thiện reranker cho các câu hỏi có context_recall thấp.
**Expected impact:** Tăng context_recall trung bình, giảm tỷ lệ thiếu chunk liên quan.

### Cải tiến 2
**Action:** Siết prompt generation để model chỉ trả lời dựa trên context được cung cấp.
**Expected impact:** Giảm hallucination, tăng faithfulness.

### Cải tiến 3
**Action:** Mở rộng golden_dataset để có sample size lớn hơn, đa dạng loại câu hỏi hơn.
**Expected impact:** Kết quả evaluation ổn định và đáng tin cậy hơn.
