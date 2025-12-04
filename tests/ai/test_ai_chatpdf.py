#!/usr/bin/env python3
"""
Test script for PDF download and AI chat functionality.
Tests the download_arxiv_pdf and summarize_pdf_with_ai methods.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import arxiv_bot
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arxiv_bot import ArxivBot

def test_pdf_download():
    """Test PDF download functionality."""
    print("\n" + "=" * 70)
    print("Testing PDF Download")
    print("=" * 70)
    
    bot = ArxivBot()
    
    # Fetch a real paper from RSS
    print("Fetching a paper from arXiv RSS feed...")
    try:
        papers = bot.fetch_arxiv_papers()
        if not papers:
            print("❌ No papers found in RSS feed")
            return None
        
        # Use the first paper
        test_paper = papers[0]
        print(f"✅ Found paper: {test_paper['title']}")
        print(f"   Link: {test_paper['link']}")
        print(f"   Category: {test_paper['category']}")
        print(f"   Published: {test_paper['published_date']}")
        
    except Exception as e:
        print(f"❌ Error fetching papers from RSS: {e}")
        return None
    
    # Download PDF
    print(f"\nDownloading PDF...")
    pdf_path = bot.download_arxiv_pdf(test_paper)
    
    if pdf_path and pdf_path.exists():
        file_size = pdf_path.stat().st_size
        print(f"✅ PDF downloaded successfully!")
        print(f"   Path: {pdf_path}")
        print(f"   Size: {file_size:,} bytes")
        return pdf_path
    else:
        print("❌ PDF download failed")
        return None


def test_ai_summary(pdf_path: Path):
    """Test AI summary functionality."""
    print("\n" + "=" * 70)
    print("Testing AI Summary")
    print("=" * 70)
    
    if not pdf_path or not pdf_path.exists():
        print("❌ PDF file not found, skipping AI summary test")
        return None
    
    bot = ArxivBot()
    
    # Check if AI summary is enabled
    ai_config = bot.config.get("ai_summary", {})
    if not ai_config.get("enabled", False):
        print("⚠️  AI summary is disabled in config.json")
        print("   Set 'ai_summary.enabled' to true to test this feature")
        return None
    
    # Check API key
    api_key_env = ai_config.get("api_key_env", "DASHSCOPE_API_KEY")
    api_key = os.environ.get(api_key_env)
    
    if not api_key:
        print(f"⚠️  API key not found in environment variable: {api_key_env}")
        print(f"   Set it with: export {api_key_env}='your-api-key'")
        return None
    
    print(f"Using API: {ai_config.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}")
    print(f"Using Model: {ai_config.get('model', 'qwen-long')}")
    print(f"PDF: {pdf_path.name}")
    print("\nGenerating AI summary (this may take a while)...")
    
    # Fetch paper info for title
    papers = bot.fetch_arxiv_papers()
    paper_title = papers[0]['title'] if papers else "Unknown Paper"
    
    # Get AI summary
    summary = bot.summarize_pdf_with_ai(
        pdf_path, 
        paper_title=paper_title
    )
    
    if summary:
        print("\n✅ AI summary generated successfully!")
        print("\n" + "-" * 70)
        print("AI SUMMARY:")
        print("-" * 70)
        print(summary)
        print("-" * 70)
        print(f"\nSummary length: {len(summary)} characters")
        return summary
    else:
        print("❌ AI summary generation failed")
        return None


def test_parallel_processing():
    """Test parallel processing of multiple papers using ThreadPoolExecutor."""
    print("\n" + "=" * 70)
    print("Testing Parallel Processing (Multi-threaded)")
    print("=" * 70)
    
    bot = ArxivBot()
    
    # Check if AI summary is enabled
    ai_config = bot.config.get("ai_summary", {})
    if not ai_config.get("enabled", False):
        print("⚠️  AI summary is disabled, skipping parallel processing test")
        return
    
    api_key_env = ai_config.get("api_key_env", "DASHSCOPE_API_KEY")
    api_key = os.environ.get(api_key_env)
    
    if not api_key:
        print(f"⚠️  API key not found, skipping parallel processing test")
        print(f"   Set it with: export {api_key_env}='your-api-key'")
        return
    
    # Fetch real papers from RSS
    print("Fetching papers from RSS feed...")
    try:
        papers = bot.fetch_arxiv_papers()
        if not papers:
            print("❌ No papers found in RSS feed")
            return
        
        # Use first 3 papers for testing
        test_papers = papers[:3]
        print(f"✅ Found {len(papers)} paper(s), testing with first {len(test_papers)}")
        for i, paper in enumerate(test_papers, 1):
            print(f"   {i}. {paper['title'][:60]}...")
        
    except Exception as e:
        print(f"❌ Error fetching papers: {e}")
        return
    
    print(f"\nTesting parallel processing with {len(test_papers)} paper(s)")
    print("Note: Using ThreadPoolExecutor for concurrent processing.")
    
    # Initialize client
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    base_url = ai_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    ai_client = OpenAI(api_key=api_key, base_url=base_url)
    
    max_workers = ai_config.get("max_workers", 3)
    print(f"Using {max_workers} worker threads for parallel processing")
    
    # Process papers in parallel
    ai_summaries = {}
    print(f"\n🚀 Starting parallel processing...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(bot._process_paper_with_ai, paper, i+1, ai_config, ai_client): i+1
            for i, paper in enumerate(test_papers)
        }
        
        # Process with progress bar
        with tqdm(total=len(test_papers), desc="Processing papers", unit="paper") as pbar:
            for future in as_completed(future_to_index):
                index, ai_summary = future.result()
                ai_summaries[index] = ai_summary
                pbar.update(1)
                if ai_summary:
                    pbar.set_postfix({"status": f"✓ {index}/{len(test_papers)}"})
    
    print(f"\n✅ Parallel processing test completed!")
    print(f"   Processed {len(ai_summaries)}/{len(test_papers)} summaries successfully")
    
    # Display summaries
    for i, (index, summary) in enumerate(sorted(ai_summaries.items()), 1):
        if summary and index <= len(test_papers):
            paper = test_papers[index - 1]
            print(f"\n[{i}] {paper['title'][:60]}...")
            print(f"    Summary length: {len(summary)} characters")
            print(f"    Preview: {summary[:100]}...")
        elif not summary:
            print(f"\n[{i}] Paper {index}: ❌ Failed to generate summary")


def main():
    """Run all tests."""
    print("=" * 70)
    print("PDF Download and AI Chat Test Suite")
    print("=" * 70)
    
    bot = ArxivBot()
    
    # Test 1: PDF Download
    pdf_path = test_pdf_download()
    
    # Test 2: AI Summary
    if pdf_path:
        summary = test_ai_summary(pdf_path)
        
        # Save summary to file if successful
        if summary:
            # Get paper info for filename
            papers = bot.fetch_arxiv_papers()
            paper = papers[0] if papers else None
            
            if paper:
                paper_id = paper.get("id", "unknown")
                summary_path = Path("test_downloads") / f"{paper_id}_summary.txt"
            else:
                summary_path = Path("test_downloads") / "test_summary.txt"
            
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                if paper:
                    f.write(f"Paper: {paper['title']}\n")
                    f.write(f"Link: {paper['link']}\n")
                    f.write(f"Category: {paper['category']}\n")
                    f.write(f"Published: {paper['published_date']}\n")
                    f.write("=" * 70 + "\n\n")
                f.write("AI Summary\n")
                f.write("=" * 70 + "\n\n")
                f.write(summary)
            print(f"\n💾 Summary saved to: {summary_path}")
    
    # Test 3: Parallel Processing (optional, requires API key)
    try:
        test_parallel_processing()
    except Exception as e:
        print(f"\n⚠️  Parallel processing test skipped: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test Suite Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()

