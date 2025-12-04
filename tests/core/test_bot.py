#!/usr/bin/env python3
"""
Test script for arXiv Bot
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import arxiv_bot
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from arxiv_bot import ArxivBot


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    bot = ArxivBot()
    print(f"Categories: {bot.config['categories']}")
    print(f"Keywords: {bot.config['keywords']}")
    print(f"Max papers: {bot.config['max_papers']}")
    print("✅ Configuration loaded successfully")


def test_paper_fetching():
    """Test paper fetching (limited to avoid rate limiting)."""
    print("\nTesting paper fetching...")
    bot = ArxivBot()

    # Use a smaller config for testing
    test_config = {
        "categories": ["cs.AI"],
        "keywords": ["machine learning"],
        "max_papers": 5,
        "days_back": 3,
        "exclude_keywords": [],
        "min_score": 0.0,
    }
    bot.config = test_config

    try:
        papers = bot.fetch_arxiv_papers()
        print(f"✅ Fetched {len(papers)} papers")

        if papers:
            print(f"Sample paper: {papers[0]['title'][:50]}...")

        return papers
    except Exception as e:
        print(f"❌ Error fetching papers: {e}")
        return []


def test_filtering(papers):
    """Test paper filtering."""
    if not papers:
        print("❌ No papers to filter")
        return []

    print("\nTesting paper filtering...")
    bot = ArxivBot()
    filtered_papers = bot.filter_papers(papers)
    print(f"✅ Filtered to {len(filtered_papers)} papers")

    if filtered_papers:
        print(f"Top paper score: {filtered_papers[0]['score']}")

    return filtered_papers


def test_readme_generation(papers):
    """Test README generation."""
    print("\nTesting README generation...")
    bot = ArxivBot()

    # Generate README
    readme_content = bot.render_readme(papers)

    # Save to test file
    with open("test_readme.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✅ Generated README with {len(readme_content)} characters")
    print("📄 Check test_readme.md for output")


def main():
    """Run all tests."""
    print("🧪 Starting arXiv Bot Tests\n")

    # Test configuration
    test_config_loading()

    # Test paper fetching
    papers = test_paper_fetching()

    # Test filtering
    filtered_papers = test_filtering(papers)

    # Test README generation
    test_readme_generation(filtered_papers)

    print("\n🎉 All tests completed!")
    print("📝 To run the full bot: python arxiv_bot.py")


if __name__ == "__main__":
    main()
