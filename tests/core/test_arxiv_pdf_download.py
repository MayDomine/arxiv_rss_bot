#!/usr/bin/env python3
"""
Test script to download arXiv PDF from RSS feed.
Downloads a single paper PDF for testing purposes.
"""

import feedparser
import requests
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_arxiv_pdf_url(paper_link: str) -> str:
    """Convert arXiv paper link to PDF download URL.
    
    Args:
        paper_link: arXiv paper link (e.g., https://arxiv.org/abs/1234.5678)
        
    Returns:
        PDF download URL (e.g., https://arxiv.org/pdf/1234.5678.pdf)
    """
    # Replace /abs/ with /pdf/ and add .pdf extension
    if "/abs/" in paper_link:
        pdf_url = paper_link.replace("/abs/", "/pdf/") + ".pdf"
    elif "/pdf/" in paper_link:
        # Already a PDF link, just ensure .pdf extension
        pdf_url = paper_link if paper_link.endswith(".pdf") else paper_link + ".pdf"
    else:
        # Extract arxiv ID and construct PDF URL
        arxiv_id = paper_link.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    return pdf_url


def fetch_first_paper_from_rss(category: str = "cs.AI") -> Optional[dict]:
    """Fetch the first paper from arXiv RSS feed.
    
    Args:
        category: arXiv category (default: cs.AI)
        
    Returns:
        Paper dictionary with title, link, etc., or None if no papers found
    """
    rss_url = f"http://export.arxiv.org/rss/{category}"
    logger.info(f"Fetching papers from {rss_url}")
    
    try:
        feed = feedparser.parse(rss_url)
        
        if feed.bozo:
            logger.warning(f"RSS feed has issues: {feed.bozo_exception}")
        
        if not feed.entries:
            logger.warning("No entries found in RSS feed")
            return None
        
        # Get the first entry
        entry = feed.entries[0]
        
        paper = {
            "title": entry.title,
            "link": entry.link,
            "authors": [author.name for author in entry.authors] if hasattr(entry, "authors") else [],
            "published": entry.published if hasattr(entry, "published") else None,
            "summary": entry.summary if hasattr(entry, "summary") else "",
        }
        
        logger.info(f"Found paper: {paper['title']}")
        logger.info(f"Link: {paper['link']}")
        
        return paper
        
    except Exception as e:
        logger.error(f"Error fetching from RSS: {e}")
        return None


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
        
        # Check if it's actually a PDF
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            logger.warning(f"Content-Type is {content_type}, might not be a PDF")
        
        # Write to file
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


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("Testing arXiv PDF Download from RSS")
    logger.info("=" * 60)
    
    # Step 1: Fetch first paper from RSS
    paper = fetch_first_paper_from_rss(category="cs.AI")
    
    if not paper:
        logger.error("Failed to fetch paper from RSS")
        return
    
    # Step 2: Get PDF URL
    pdf_url = get_arxiv_pdf_url(paper["link"])
    logger.info(f"PDF URL: {pdf_url}")
    
    # Step 3: Download PDF
    # Extract arxiv ID from link for filename
    arxiv_id = paper["link"].split("/")[-1]
    output_path = Path("test_downloads") / f"{arxiv_id}.pdf"
    
    success = download_pdf(pdf_url, output_path)
    
    if success:
        logger.info("=" * 60)
        logger.info("Test completed successfully!")
        logger.info(f"PDF saved to: {output_path.absolute()}")
        logger.info("=" * 60)
    else:
        logger.error("Test failed: Could not download PDF")


if __name__ == "__main__":
    main()

