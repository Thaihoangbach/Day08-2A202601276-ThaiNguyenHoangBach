# RAG Evaluation Results

# Thành viên:
Nguyễn Thành Công - 2A202601396
Lâm Vũ - 2A202601914
Nguyễn Thành Công - 2A202601396
Thái Nguyễn Hoàng Bách - 2A202601276

## Framework sử dụng

**Framework:** DeepEval

Sử dụng DeepEval để đánh giá chất lượng hệ thống RAG trên bốn tiêu chí chính:
- **Faithfulness**: Mức độ trung thực của câu trả lời so với context được truy xuất.
- **Answer Relevance**: Mức độ liên quan giữa câu trả lời và câu hỏi.
- **Context Recall**: Khả năng retrieval lấy được đầy đủ thông tin cần thiết.
- **Context Precision**: Tỷ lệ context được truy xuất thực sự hữu ích cho việc trả lời.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------:|----------------------:|---:|
| Faithfulness | 0.93 | 0.86 | +0.07 |
| Answer Relevance | 0.91 | 0.84 | +0.07 |
| Context Recall | 0.95 | 0.81 | +0.14 |
| Context Precision | 0.89 | 0.80 | +0.09 |
| **Average** | **0.92** | **0.83** | **+0.09** |

---

## A/B Comparison Analysis

### Config A
- Hybrid Retrieval (Semantic Search + BM25)
- Reciprocal Rank Fusion (RRF)
- Reranking
- PageIndex fallback khi semantic score dưới ngưỡng
- Top-k = 5

### Config B
- Chỉ sử dụng Semantic Search (Dense Retrieval)
- Không BM25
- Không reranking
- Không fallback
- Top-k = 5

### Kết luận

Config A đạt điểm cao hơn ở tất cả các metric. Việc kết hợp semantic search với BM25 giúp tăng khả năng truy xuất đúng tài liệu, trong khi reranking và fallback giúp loại bỏ các kết quả ít liên quan, từ đó cải thiện đáng kể chất lượng câu trả lời.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------:|----------:|--------:|---------------|------------|
| 1 | Shopee có hỗ trợ thanh toán bằng Apple Pay không? | 0.68 | 0.71 | 0.63 | Retrieval | Không có tài liệu đề cập trực tiếp đến Apple Pay nên retrieval trả về context chưa đủ. |
| 2 | Người bán có được bán sản phẩm NFT trên Shopee không? | 0.70 | 0.73 | 0.65 | Retrieval | Corpus không chứa chính sách liên quan đến NFT, dẫn đến thiếu evidence. |
| 3 | Shopee xử lý tranh chấp quốc tế như thế nào? | 0.74 | 0.76 | 0.69 | Generation | Context chỉ đề cập quy trình giải quyết tranh chấp chung, chưa đủ thông tin để trả lời đầy đủ. |

---

## Recommendations

### Cải tiến 1

**Action:**
Mở rộng corpus bằng cách bổ sung thêm các văn bản pháp luật, chính sách Shopee và tài liệu FAQ mới nhất.

**Expected impact:**
Tăng Context Recall, giảm số lượng câu hỏi không có đủ evidence.

---

### Cải tiến 2

**Action:**
Sử dụng Cross-Encoder Reranker (ví dụ: BAAI/bge-reranker-base hoặc bge-reranker-v2-m3) thay cho RRF-only.

**Expected impact:**
Cải thiện Context Precision và giảm các chunk không liên quan trong top-k.

---

### Cải tiến 3

**Action:**
Tinh chỉnh chunking (chunk size và overlap) kết hợp metadata filtering theo loại tài liệu và chủ đề.

**Expected impact:**
Nâng cao độ chính xác của retrieval, tăng Faithfulness và Answer Relevance của câu trả lời cuối cùng.