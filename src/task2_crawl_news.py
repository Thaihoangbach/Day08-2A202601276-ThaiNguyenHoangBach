"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    # Ví dụ (trang công khai Shopee Vietnam):
    # "https://help.shopee.vn/portal/4/article/...",
    "https://help.shopee.vn/portal/4/article/79488-%5BVoucher%2FM%C3%A3-gi%E1%BA%A3m-gi%C3%A1%5D-Voucher%2FM%C3%A3-gi%E1%BA%A3m-gi%C3%A1-tr%C3%AAn-Shopee-l%C3%A0-g%C3%AC?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79551-%5BH%E1%BB%A7y-%C4%91%C6%A1n%5D-N%E1%BA%BFu-%C4%91%C6%A1n-h%C3%A0ng-c%E1%BB%A7a-t%C3%B4i-b%E1%BB%8B-h%E1%BB%A7y-b%E1%BB%9Fi-Ng%C6%B0%E1%BB%9Di-b%C3%A1n-th%C3%AC-sao?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79532-%5BMi%E1%BB%85n-ph%C3%AD-v%E1%BA%ADn-chuy%E1%BB%83n%5D-N%E1%BA%BFu-t%E1%BB%95ng-ti%E1%BB%81n-%C4%91%C6%A1n-h%C3%A0ng-c%E1%BB%A7a-t%C3%B4i-nh%E1%BB%8F-h%C6%A1n-m%E1%BB%A9c-%C6%B0u-%C4%91%C3%A3i-t%E1%BB%91i-thi%E1%BB%83u-c%E1%BB%A7a-m%C3%A3-mi%E1%BB%85n-ph%C3%AD-v%E1%BA%ADn-chuy%E1%BB%83n-th%C3%AC-sao?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79556-%5B%C4%90%C6%A1n-h%C3%A0ng-Qu%E1%BB%91c-t%E1%BA%BF%5D-T%C3%B4i-c%E1%BA%A7n-ch%E1%BB%9D-bao-l%C3%A2u-%C4%91%E1%BB%83-nh%E1%BA%ADn-%C4%91%C6%B0%E1%BB%A3c-%C4%91%C6%A1n-h%C3%A0ng-Qu%E1%BB%91c-t%E1%BA%BF?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79475-%5BMua-h%C3%A0ng%5D-Shopee-c%C3%B3-cung-c%E1%BA%A5p-b%E1%BA%A3o-hi%E1%BB%83m-h%C3%A0ng-h%C3%B3a-cho-%C4%91%C6%A1n-h%C3%A0ng-c%E1%BB%A7a-t%C3%B4i-kh%C3%B4ng?previousPage=secondary%20category",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    # TODO: Implement crawling logic
    # async with AsyncWebCrawler() as crawler:
    #     result = await crawler.arun(url=url)
    #     return {
    #         "url": url,
    #         "title": result.metadata.get("title", "Unknown"),
    #         "date_crawled": datetime.now().isoformat(),
    #         "content_markdown": result.markdown,
    #     }

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)

        title = "Unknown"
        if result.metadata:
            title = result.metadata.get("title", "Unknown")

        content = result.markdown if result.markdown else ""

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
