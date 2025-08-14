#!/usr/bin/env python3
"""
Test script for deduplication functionality
"""

from arxiv_bot import ArxivBot
import json

def create_test_papers():
    """Create test papers with duplicates."""
    papers = [
        {
            "id": "2401.00123",
            "title": "Test Paper 1",
            "authors": ["Author A", "Author B"],
            "summary": "This is a test paper about machine learning.",
            "category": "cs.AI",
            "published_date": "2024-01-01",
            "link": "https://arxiv.org/abs/2401.00123",
            "score": 2.0
        },
        {
            "id": "2401.00123",  # Duplicate ID
            "title": "Test Paper 1",  # Duplicate title
            "authors": ["Author A", "Author B"],
            "summary": "This is a test paper about machine learning.",
            "category": "cs.LG",  # Different category
            "published_date": "2024-01-01",
            "link": "https://arxiv.org/abs/2401.00123",  # Duplicate link
            "score": 1.5
        },
        {
            "id": "2401.00456",
            "title": "Test Paper 2",
            "authors": ["Author C"],
            "summary": "This is another test paper.",
            "category": "cs.CL",
            "published_date": "2024-01-02",
            "link": "https://arxiv.org/abs/2401.00456",
            "score": 3.0
        },
        {
            "id": "2401.00789",
            "title": "Test Paper 2",  # Duplicate title but different ID
            "authors": ["Author D"],
            "summary": "This is a different paper with same title.",
            "category": "cs.CV",
            "published_date": "2024-01-03",
            "link": "https://arxiv.org/abs/2401.00789",
            "score": 2.5
        },
        {
            "id": "2401.00123",  # Another duplicate ID
            "title": "Different Title",
            "authors": ["Author E"],
            "summary": "This has a different title but same ID.",
            "category": "cs.NE",
            "published_date": "2024-01-04",
            "link": "https://arxiv.org/abs/2401.00123",
            "score": 1.0
        }
    ]
    return papers

def test_deduplication():
    """Test the deduplication functionality."""
    print("🧪 Testing Deduplication Functionality")
    print("=" * 50)
    
    # Create bot instance
    bot = ArxivBot()
    
    # Create test papers with duplicates
    test_papers = create_test_papers()
    print(f"📄 Created {len(test_papers)} test papers (including duplicates)")
    
    # Show original papers
    print("\n📋 Original papers:")
    for i, paper in enumerate(test_papers, 1):
        print(f"  {i}. ID: {paper['id']}, Title: {paper['title']}, Category: {paper['category']}")
    
    # Test deduplication
    print("\n🔄 Applying deduplication...")
    unique_papers = bot.deduplicate_papers(test_papers)
    
    print(f"\n✅ Deduplication completed!")
    print(f"📊 Original papers: {len(test_papers)}")
    print(f"📊 Unique papers: {len(unique_papers)}")
    print(f"🗑️  Removed duplicates: {len(test_papers) - len(unique_papers)}")
    
    # Show unique papers
    print("\n📋 Unique papers after deduplication:")
    for i, paper in enumerate(unique_papers, 1):
        print(f"  {i}. ID: {paper['id']}, Title: {paper['title']}, Category: {paper['category']}")
    
    # Test with empty list
    print("\n🧪 Testing with empty list...")
    empty_result = bot.deduplicate_papers([])
    print(f"Empty list result: {len(empty_result)} papers")
    
    # Test with None
    print("\n🧪 Testing with None...")
    try:
        none_result = bot.deduplicate_papers(None)
        print(f"None result: {none_result}")
    except Exception as e:
        print(f"Error with None: {e}")
    
    print("\n🎉 Deduplication test completed!")

def test_filter_with_deduplication():
    """Test filtering with deduplication."""
    print("\n🧪 Testing Filtering with Deduplication")
    print("=" * 50)
    
    bot = ArxivBot()
    test_papers = create_test_papers()
    
    print(f"📄 Testing with {len(test_papers)} papers (including duplicates)")
    
    # Test filtering without deduplication
    print("\n🔄 Testing filtering without deduplication...")
    filtered_without_dedup = bot.filter_papers(test_papers)
    print(f"Filtered papers (without dedup): {len(filtered_without_dedup)}")
    
    # Test filtering with deduplication
    print("\n🔄 Testing filtering with deduplication...")
    deduped_papers = bot.deduplicate_papers(test_papers)
    filtered_with_dedup = bot.filter_papers(deduped_papers)
    print(f"Filtered papers (with dedup): {len(filtered_with_dedup)}")
    
    print("\n✅ Filtering test completed!")

def main():
    """Run all tests."""
    test_deduplication()
    test_filter_with_deduplication()
    
    print("\n📝 Summary:")
    print("- Deduplication removes papers with duplicate IDs, titles, or links")
    print("- Deduplication happens before filtering")
    print("- You can disable deduplication by setting 'enable_deduplication': false in config.json")
    print("- Check the logs for detailed information about removed duplicates")

if __name__ == "__main__":
    main() 