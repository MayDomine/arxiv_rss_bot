#!/usr/bin/env python3
"""
Test script to download arXiv PDF from RSS and summarize it using Qwen API (ChatPDF).
Downloads a single paper PDF, uploads it to DashScope, and asks Qwen to summarize core conclusions and experimental results.
"""

import feedparser
import requests
import logging
import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_openai_client(base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> Optional[OpenAI]:
    """Get OpenAI client configured for DashScope API.
    
    Args:
        base_url: API base URL (default: regular API, use batch URL for batch operations)
    
    Returns:
        OpenAI client instance or None if API key is not set
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY environment variable not set!")
        logger.info("Please set it with: export DASHSCOPE_API_KEY='your-api-key'")
        return None
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client


def get_arxiv_pdf_url(paper_link: str) -> str:
    """Convert arXiv paper link to PDF download URL.
    
    Args:
        paper_link: arXiv paper link (e.g., https://arxiv.org/abs/1234.5678)
        
    Returns:
        PDF download URL (e.g., https://arxiv.org/pdf/1234.5678.pdf)
    """
    if "/abs/" in paper_link:
        pdf_url = paper_link.replace("/abs/", "/pdf/") + ".pdf"
    elif "/pdf/" in paper_link:
        pdf_url = paper_link if paper_link.endswith(".pdf") else paper_link + ".pdf"
    else:
        arxiv_id = paper_link.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    return pdf_url


def fetch_papers_from_rss(category: str = "cs.AI", max_papers: int = 5) -> List[Dict[str, Any]]:
    """Fetch papers from arXiv RSS feed.
    
    Args:
        category: arXiv category (default: cs.AI)
        max_papers: Maximum number of papers to fetch
        
    Returns:
        List of paper dictionaries with title, link, etc.
    """
    rss_url = f"http://export.arxiv.org/rss/{category}"
    logger.info(f"Fetching papers from {rss_url}")
    
    try:
        feed = feedparser.parse(rss_url)
        
        if feed.bozo:
            logger.warning(f"RSS feed has issues: {feed.bozo_exception}")
        
        if not feed.entries:
            logger.warning("No entries found in RSS feed")
            return []
        
        papers = []
        for entry in feed.entries[:max_papers]:
            paper = {
                "title": entry.title,
                "link": entry.link,
                "authors": [author.name for author in entry.authors] if hasattr(entry, "authors") else [],
                "published": entry.published if hasattr(entry, "published") else None,
                "summary": entry.summary if hasattr(entry, "summary") else "",
            }
            papers.append(paper)
            logger.info(f"Found paper: {paper['title']}")
        
        return papers
        
    except Exception as e:
        logger.error(f"Error fetching from RSS: {e}")
        return []


def fetch_first_paper_from_rss(category: str = "cs.AI") -> Optional[dict]:
    """Fetch the first paper from arXiv RSS feed.
    
    Args:
        category: arXiv category (default: cs.AI)
        
    Returns:
        Paper dictionary with title, link, etc., or None if no papers found
    """
    papers = fetch_papers_from_rss(category, max_papers=1)
    return papers[0] if papers else None


def download_pdf(pdf_url: str, output_path: Path) -> bool:
    """Download PDF from URL to local file.
    
    Args:
        pdf_url: URL of the PDF file
        output_path: Path where to save the PDF
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading PDF from {pdf_url}")
        logger.info(f"Saving to {output_path}")
        
        response = requests.get(pdf_url, stream=True, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            logger.warning(f"Content-Type is {content_type}, might not be a PDF")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = output_path.stat().st_size
        logger.info(f"Successfully downloaded PDF ({file_size:,} bytes)")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def prepare_batch_requests(papers: List[Dict[str, Any]], pdf_paths: List[Path], client: OpenAI) -> Optional[str]:
    """Prepare batch requests JSONL file for multiple papers.
    
    Args:
        papers: List of paper dictionaries
        pdf_paths: List of PDF file paths corresponding to papers
        client: OpenAI client for uploading files
        
    Returns:
        Path to the created JSONL file, or None if failed
    """
    try:
        # Upload all PDFs first and collect file IDs
        file_ids = []
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"Uploading file {i+1}/{len(pdf_paths)}: {pdf_path.name}")
            file_object = client.files.create(
                file=pdf_path,
                purpose="file-extract"
            )
            file_ids.append(file_object.id)
            logger.info(f"File uploaded: {file_object.id}")
        
        # Create JSONL file with batch requests
        jsonl_path = Path("test_downloads") / "batch_requests.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, (paper, file_id) in enumerate(zip(papers, file_ids)):
                prompt = """请总结这篇论文的核心结论和实验结果。请包括：
1. 论文的主要贡献和创新点
2. 核心实验方法和设置
3. 主要实验结果和性能指标
4. 关键结论和发现

请用中文回答，结构清晰，重点突出。"""
                
                if paper.get("title"):
                    prompt = f"论文标题：{paper['title']}\n\n{prompt}"
                
                request_data = {
                    "custom_id": str(i + 1),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "qwen-long",
                        "messages": [
                            {
                                "role": "system",
                                "content": f"fileid://{file_id}"
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.3
                    }
                }
                f.write(json.dumps(request_data, ensure_ascii=False) + "\n")
        
        logger.info(f"Created batch requests file: {jsonl_path}")
        return str(jsonl_path)
        
    except Exception as e:
        logger.error(f"Error preparing batch requests: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_batch(client: OpenAI, jsonl_path: str) -> Optional[str]:
    """Create a batch job from JSONL file.
    
    Args:
        client: OpenAI client configured for DashScope API (used for both file upload and batch creation)
        jsonl_path: Path to JSONL file
        
    Returns:
        Batch ID if successful, None otherwise
    """
    try:
        # Upload the JSONL file
        logger.info(f"Uploading batch requests file: {jsonl_path}")
        file_object = client.files.create(
            file=Path(jsonl_path),
            purpose="batch"
        )
        logger.info(f"Batch file uploaded: {file_object.id}")
        
        # Create batch job (using the same client)
        logger.info("Creating batch job...")
        batch = client.batches.create(
            input_file_id=file_object.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        logger.info(f"Batch job created: {batch.id}")
        logger.info(f"Batch status: {batch.status}")
        return batch.id
        
    except Exception as e:
        logger.error(f"Error creating batch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def wait_for_batch_completion(client: OpenAI, batch_id: str, check_interval: int = 10, max_retries: int = 3) -> Optional[Any]:
    """Wait for batch job to complete and return the batch object.
    
    Args:
        client: OpenAI client configured for DashScope API
        batch_id: Batch job ID
        check_interval: Seconds to wait between status checks
        max_retries: Maximum number of retries for failed requests
        
    Returns:
        Batch object if successful, None otherwise
    """
    logger.info(f"Waiting for batch {batch_id} to complete...")
    retry_count = 0
    
    while True:
        try:
            batch = client.batches.retrieve(batch_id=batch_id)
            logger.info(f"Batch status: {batch.status}")
            
            if batch.status == "completed":
                logger.info("Batch completed successfully!")
                return batch
            elif batch.status in ["failed", "expired", "cancelled"]:
                logger.error(f"Batch {batch.status}")
                if batch.status == "failed" and hasattr(batch, 'errors'):
                    logger.error(f"Batch errors: {batch.errors}")
                return None
            
            # Reset retry count on successful request
            retry_count = 0
            time.sleep(check_interval)
            
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            # Check if it's a timeout or connection error
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                if retry_count <= max_retries:
                    wait_time = check_interval * retry_count
                    logger.warning(f"Request timed out (attempt {retry_count}/{max_retries}). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request timed out after {max_retries} retries. Batch may still be processing.")
                    logger.info(f"You can check batch status later using batch_id: {batch_id}")
                    return None
            else:
                # For other errors, log and return None
                logger.error(f"Error waiting for batch: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None


def download_batch_results(client: OpenAI, batch: Any, output_dir: Path) -> Optional[str]:
    """Download batch results file.
    
    Args:
        client: OpenAI client configured for batch API
        batch: Batch object with output_file_id
        output_dir: Directory to save results
        
    Returns:
        Path to results file if successful, None otherwise
    """
    try:
        if not batch.output_file_id:
            logger.error("Batch has no output file ID")
            return None
        
        logger.info(f"Downloading batch results file: {batch.output_file_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "batch_results.jsonl"
        
        # Download file content
        file_content = client.files.content(file_id=batch.output_file_id).text
        
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        logger.info(f"Results saved to: {results_path}")
        return str(results_path)
        
    except Exception as e:
        logger.error(f"Error downloading batch results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_batch_results(results_path: str, papers: List[Dict[str, Any]]) -> Dict[int, str]:
    """Process batch results and extract summaries.
    
    Args:
        results_path: Path to batch results JSONL file
        papers: List of paper dictionaries
        
    Returns:
        Dictionary mapping paper index to summary
    """
    summaries = {}
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = json.loads(line)
                custom_id = int(result.get("custom_id", 0))
                
                if result.get("response", {}).get("status_code") == 200:
                    body = result.get("response", {}).get("body", {})
                    if "choices" in body and len(body["choices"]) > 0:
                        summary = body["choices"][0].get("message", {}).get("content", "")
                        summaries[custom_id] = summary
                        logger.info(f"Got summary for paper {custom_id}")
                    else:
                        logger.warning(f"No choices in response for paper {custom_id}")
                else:
                    error = result.get("response", {}).get("body", {}).get("error", {})
                    logger.error(f"Error for paper {custom_id}: {error}")
        
        return summaries
        
    except Exception as e:
        logger.error(f"Error processing batch results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def upload_and_summarize_pdf(client: OpenAI, file_path: Path, paper_title: str = "") -> Optional[str]:
    """Upload PDF file to DashScope and get summary using OpenAI SDK.
    
    Args:
        client: OpenAI client configured for DashScope API
        file_path: Path to the PDF file
        paper_title: Optional paper title for context
        
    Returns:
        Summary text if successful, None otherwise
    """
    try:
        logger.info(f"Uploading file to DashScope: {file_path.name}")
        
        # Upload file using OpenAI SDK
        file_object = client.files.create(
            file=file_path,
            purpose="file-extract"
        )
        
        logger.info(f"File uploaded successfully. File ID: {file_object.id}")
        
        # Construct the prompt
        prompt = """请总结这篇论文的核心结论和实验结果。请包括：
1. 论文的主要贡献和创新点
2. 核心实验方法和设置
3. 主要实验结果和性能指标
4. 关键结论和发现

请用中文回答，结构清晰，重点突出。"""
        
        if paper_title:
            prompt = f"论文标题：{paper_title}\n\n{prompt}"
        
        # Prepare messages with file reference (DashScope/Qwen uses fileid:// format)
        messages = [
            {
                "role": "system",
                "content": f"fileid://{file_object.id}",
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        logger.info("Requesting summary from Qwen...")
        
        # Call chat completion
        completion = client.chat.completions.create(
            model="qwen-long",  # Use qwen-long model for long context
            messages=messages,
            temperature=0.3,  # Lower temperature for more focused responses
        )
        
        summary = completion.choices[0].message.content
        logger.info("Summary received successfully!")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test arXiv PDF download and Qwen summary")
    parser.add_argument("--batch", action="store_true", help="Use batch API for multiple papers")
    parser.add_argument("--count", type=int, default=3, help="Number of papers to process (default: 3)")
    parser.add_argument("--category", type=str, default="cs.AI", help="arXiv category (default: cs.AI)")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    if args.batch:
        logger.info("Testing Batch Processing: arXiv PDF Download and Qwen ChatPDF Summary")
    else:
        logger.info("Testing arXiv PDF Download and Qwen ChatPDF Summary")
    logger.info("=" * 70)
    
    # Step 1: Initialize OpenAI client
    client = get_openai_client()
    if not client:
        return
    
    if args.batch:
        # Batch processing mode
        logger.info(f"Fetching {args.count} papers from RSS...")
        papers = fetch_papers_from_rss(category=args.category, max_papers=args.count)
        
        if not papers:
            logger.error("Failed to fetch papers from RSS")
            return
        
        logger.info(f"Found {len(papers)} papers")
        
        # Step 2: Download all PDFs
        pdf_paths = []
        for i, paper in enumerate(papers, 1):
            logger.info(f"\n[{i}/{len(papers)}] Processing: {paper['title']}")
            pdf_url = get_arxiv_pdf_url(paper["link"])
            arxiv_id = paper["link"].split("/")[-1]
            output_path = Path("test_downloads") / f"{arxiv_id}.pdf"
            
            if not output_path.exists():
                success = download_pdf(pdf_url, output_path)
                if not success:
                    logger.error(f"Failed to download PDF for {paper['title']}")
                    continue
            else:
                logger.info(f"PDF already exists: {output_path}")
            
            pdf_paths.append(output_path)
        
        if not pdf_paths:
            logger.error("No PDFs downloaded")
            return
        
        logger.info(f"\nPreparing batch requests for {len(pdf_paths)} papers...")
        
        # Step 3: Prepare batch requests
        jsonl_path = prepare_batch_requests(papers[:len(pdf_paths)], pdf_paths, client)
        if not jsonl_path:
            logger.error("Failed to prepare batch requests")
            return
        
        # Step 4: Create batch job using batch API client
        batch_client = get_openai_client(base_url="https://batch.dashscope.aliyuncs.com/compatible-mode/v1")
        if not batch_client:
            logger.error("Failed to create batch API client")
            return
        
        batch_id = create_batch(client, batch_client, jsonl_path)
        if not batch_id:
            logger.error("Failed to create batch job")
            return
        
        # Step 5: Wait for batch completion
        batch = wait_for_batch_completion(client, batch_id)
        if not batch:
            logger.error("Batch job failed")
            return
        
        # Step 6: Download results
        results_path = download_batch_results(client, batch, Path("test_downloads"))
        if not results_path:
            logger.error("Failed to download batch results")
            return
        
        # Step 7: Process results and save summaries
        summaries = process_batch_results(results_path, papers[:len(pdf_paths)])
        
        logger.info("\n" + "=" * 70)
        logger.info("BATCH PROCESSING RESULTS")
        logger.info("=" * 70)
        
        for i, paper in enumerate(papers[:len(pdf_paths)], 1):
            arxiv_id = paper["link"].split("/")[-1]
            summary = summaries.get(i)
            
            if summary:
                logger.info(f"\n[{i}] {paper['title']}")
                logger.info(f"Summary length: {len(summary)} characters")
                
                # Save individual summary
                summary_path = Path("test_downloads") / f"{arxiv_id}_summary.txt"
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(f"Paper: {paper['title']}\n")
                    f.write(f"Link: {paper['link']}\n")
                    f.write(f"\n{'='*70}\n")
                    f.write("SUMMARY FROM QWEN (BATCH)\n")
                    f.write(f"{'='*70}\n\n")
                    f.write(summary)
                
                logger.info(f"Summary saved to: {summary_path}")
            else:
                logger.warning(f"No summary for paper {i}: {paper['title']}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"Batch processing completed: {len(summaries)}/{len(papers[:len(pdf_paths)])} summaries")
        logger.info("=" * 70)
        
    else:
        # Single paper mode (original behavior)
        paper = fetch_first_paper_from_rss(category=args.category)
        
        if not paper:
            logger.error("Failed to fetch paper from RSS")
            return
        
        # Step 2: Get PDF URL and download
        pdf_url = get_arxiv_pdf_url(paper["link"])
        logger.info(f"PDF URL: {pdf_url}")
        
        arxiv_id = paper["link"].split("/")[-1]
        output_path = Path("test_downloads") / f"{arxiv_id}.pdf"
        
        if not output_path.exists():
            success = download_pdf(pdf_url, output_path)
            if not success:
                logger.error("Failed to download PDF")
                return
        else:
            logger.info(f"PDF already exists: {output_path}")
        
        # Step 3: Upload file and get summary from Qwen
        summary = upload_and_summarize_pdf(client, output_path, paper["title"])
        
        if summary:
            logger.info("=" * 70)
            logger.info("SUMMARY FROM QWEN")
            logger.info("=" * 70)
            print("\n" + summary + "\n")
            logger.info("=" * 70)
            
            # Save summary to file
            summary_path = Path("test_downloads") / f"{arxiv_id}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"Paper: {paper['title']}\n")
                f.write(f"Link: {paper['link']}\n")
                f.write(f"\n{'='*70}\n")
                f.write("SUMMARY FROM QWEN\n")
                f.write(f"{'='*70}\n\n")
                f.write(summary)
            
            logger.info(f"Summary saved to: {summary_path.absolute()}")
            logger.info("=" * 70)
            logger.info("Test completed successfully!")
        else:
            logger.error("Failed to get summary from Qwen")


if __name__ == "__main__":
    main()

