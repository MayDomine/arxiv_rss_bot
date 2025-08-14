#!/usr/bin/env python3
"""
Example Custom Rules for arXiv Bot
Modify this file to create your own filtering rules.
"""

from typing import Dict, Any, List

def custom_paper_filter(paper: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Custom function to filter papers.
    Return True to include the paper, False to exclude it.
    
    Args:
        paper: Dictionary containing paper information
        config: Configuration dictionary
    
    Returns:
        bool: True if paper should be included, False otherwise
    """
    # Example 1: Only include papers with specific authors
    target_authors = ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio"]
    paper_authors = " ".join(paper.get("authors", [])).lower()
    
    for author in target_authors:
        if author.lower() in paper_authors:
            return True
    
    # Example 2: Exclude papers with certain words in title
    title = paper.get("title", "").lower()
    exclude_words = ["preliminary", "draft", "work in progress", "extended abstract"]
    
    for word in exclude_words:
        if word in title:
            return False
    
    # Example 3: Only include papers with sufficient abstract length
    abstract = paper.get("summary", "")
    if len(abstract) < 200:
        return False
    
    # Example 4: Check for required keywords
    required_keywords = config.get("required_keywords", [])
    if required_keywords:
        text_to_check = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
        found_required = False
        for keyword in required_keywords:
            if keyword.lower() in text_to_check:
                found_required = True
                break
        if not found_required:
            return False
    
    return True

def custom_paper_score(paper: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Custom function to score papers.
    Return a float score (higher = more relevant).
    
    Args:
        paper: Dictionary containing paper information
        config: Configuration dictionary
    
    Returns:
        float: Relevance score
    """
    score = 0.0
    title = paper.get("title", "").lower()
    abstract = paper.get("summary", "").lower()
    
    # Score based on keyword matches
    keywords = config.get("keywords", [])
    for keyword in keywords:
        if keyword.lower() in title:
            score += 2.0  # Title matches worth more
        if keyword.lower() in abstract:
            score += 1.0
    
    # Score based on author reputation
    authors = paper.get("authors", [])
    famous_authors = [
        "geoffrey hinton", "yann lecun", "yoshua bengio", 
        "andrew ng", "fei-fei li", "ilya sutskever",
        "david silver", "richard sutton", "peter norvig"
    ]
    for author in authors:
        if author.lower() in famous_authors:
            score += 1.0
    
    # Score based on abstract quality indicators
    quality_indicators = [
        "novel", "improved", "state-of-the-art", "benchmark",
        "significant", "outperform", "advance", "breakthrough"
    ]
    for indicator in quality_indicators:
        if indicator in abstract:
            score += 0.5
    
    # Score based on abstract length (longer abstracts often more detailed)
    abstract_length = len(abstract)
    if abstract_length > 500:
        score += 0.5
    elif abstract_length < 100:
        score -= 1.0
    
    return score

# Example usage in your config.json:
"""
{
    "categories": ["cs.AI", "cs.LG", "cs.CL"],
    "keywords": ["machine learning", "deep learning", "transformer"],
    "exclude_keywords": ["survey", "review", "tutorial"],
    "required_keywords": ["neural", "network"],
    "max_papers": 30,
    "days_back": 7,
    "min_score": 1.0
}
""" 