"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex.client import PageIndexClient
    
    if not PAGEINDEX_API_KEY:
        print("⚠ Thiếu PAGEINDEX_API_KEY trong .env")
        return
        
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    legal_dir = Path(__file__).parent.parent / "data" / "landing" / "legal"
    
    for pdf_file in legal_dir.glob("*.pdf"):
        resp = client.submit_document(str(pdf_file))
        doc_id = resp.get("doc_id") or resp.get("id")
        print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex.client import PageIndexClient
    import time
    
    # Dummy mock dùng cho file test nếu chưa có API key
    if not PAGEINDEX_API_KEY:
        return [{"content": "Kết quả giả lập từ PageIndex", "score": 1.0, "metadata": {}, "source": "pageindex"}]
        
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Lấy doc_id tạm thời từ env (sau khi chạy upload bạn lấy ID thay vào)
    doc_id = os.getenv("PAGEINDEX_DOC_ID", "")
    if not doc_id:
        print("⚠ Thiếu PAGEINDEX_DOC_ID trong .env. Hãy chạy upload_documents trước để lấy ID.")
        return []
        
    resp = client.submit_query(doc_id=doc_id, query=query)
    retrieval_id = resp.get("retrieval_id") or resp.get("id")
    
    # Poll cho đến khi status == "completed"
    while True:
        retrieval = client.get_retrieval(retrieval_id)
        if retrieval.get("status") == "completed":
            break
        elif retrieval.get("status") == "failed":
            return []
        time.sleep(2)
    
    # Parse retrieval["retrieved_nodes"] — mỗi node có "relevant_contents"
    results = []
    score = 1.0
    for node in retrieval.get("retrieved_nodes", [])[:2]:
        for group in node.get("relevant_contents", []):
            for item in group:
                results.append({
                    "content": item.get("relevant_content", ""),
                    "score": score,  # PageIndex không trả score trực tiếp — tự gán theo rank giảm dần
                    "metadata": {"section": item.get("section_title")},
                    "source": "pageindex",
                })
                score -= 0.1
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        # upload_documents()

        print("\nTest query:")
        results = pageindex_search("Shopee hỗ trợ những phương thức thanh toán nào?", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
