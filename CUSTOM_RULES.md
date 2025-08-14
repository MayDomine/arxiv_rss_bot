# 🎯 Custom Rules Configuration Guide

Learn how to customize your arXiv spider to filter papers exactly the way you want. This guide shows you how to define your own rules, scoring algorithms, and filtering criteria.

## 🔧 Basic Configuration

### 1. Keywords and Categories

Edit `config.json` to define your basic filtering criteria:

```json
{
    "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"],
    "keywords": ["machine learning", "deep learning", "transformer"],
    "exclude_keywords": ["survey", "review", "tutorial"],
    "max_papers": 30,
    "days_back": 7,
    "min_score": 1.0
}
```

### 2. Advanced Filtering Options

Add these advanced options to your `config.json`:

```json
{
    "categories": ["cs.AI", "cs.LG"],
    "keywords": ["machine learning", "deep learning"],
    "exclude_keywords": ["survey", "review"],
    "max_papers": 30,
    "days_back": 7,
    "min_score": 1.0,
    
    "advanced_filtering": {
        "require_all_keywords": false,
        "case_sensitive": false,
        "regex_patterns": [],
        "author_whitelist": [],
        "author_blacklist": [],
        "institution_whitelist": [],
        "institution_blacklist": [],
        "min_abstract_length": 100,
        "max_abstract_length": 2000,
        "required_fields": ["title", "abstract"],
        "excluded_fields": []
    },
    
    "scoring": {
        "title_match_weight": 2.0,
        "abstract_match_weight": 1.0,
        "keyword_density_weight": 0.5,
        "recency_weight": 0.3,
        "author_reputation_weight": 0.2,
        "institution_weight": 0.1
    }
}
```

## 🎨 Custom Filtering Functions

### Option 1: Simple Custom Rules

Create a `custom_rules.py` file:

```python
# custom_rules.py
from typing import Dict, Any, List

def custom_paper_filter(paper: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Custom function to filter papers.
    Return True to include the paper, False to exclude it.
    """
    # Example: Only include papers with specific authors
    target_authors = ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio"]
    paper_authors = " ".join(paper.get("authors", [])).lower()
    
    for author in target_authors:
        if author.lower() in paper_authors:
            return True
    
    # Example: Exclude papers with certain words in title
    title = paper.get("title", "").lower()
    exclude_words = ["preliminary", "draft", "work in progress"]
    
    for word in exclude_words:
        if word in title:
            return False
    
    # Example: Only include papers with sufficient abstract length
    abstract = paper.get("summary", "")
    if len(abstract) < 200:
        return False
    
    return True

def custom_paper_score(paper: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Custom function to score papers.
    Return a float score (higher = more relevant).
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
    
    # Score based on recency (newer papers get higher scores)
    days_back = config.get("days_back", 7)
    # You can implement recency scoring here
    
    # Score based on author reputation
    authors = paper.get("authors", [])
    famous_authors = ["geoffrey hinton", "yann lecun", "yoshua bengio"]
    for author in authors:
        if author.lower() in famous_authors:
            score += 1.0
    
    return score
```

### Option 2: Advanced Custom Rules

Create a more sophisticated filtering system:

```python
# advanced_rules.py
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

class AdvancedPaperFilter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_rules()
    
    def setup_rules(self):
        """Setup custom filtering rules."""
        self.rules = {
            "keyword_rules": self.config.get("keyword_rules", {}),
            "author_rules": self.config.get("author_rules", {}),
            "institution_rules": self.config.get("institution_rules", {}),
            "content_rules": self.config.get("content_rules", {}),
            "quality_rules": self.config.get("quality_rules", {})
        }
    
    def filter_paper(self, paper: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Filter a paper and return (include, score, reason).
        """
        score = 0.0
        reasons = []
        
        # Apply keyword filtering
        keyword_result = self.apply_keyword_rules(paper)
        if not keyword_result[0]:
            return False, 0.0, f"Keyword filter: {keyword_result[2]}"
        score += keyword_result[1]
        reasons.append(keyword_result[2])
        
        # Apply author filtering
        author_result = self.apply_author_rules(paper)
        if not author_result[0]:
            return False, 0.0, f"Author filter: {author_result[2]}"
        score += author_result[1]
        reasons.append(author_result[2])
        
        # Apply content quality filtering
        content_result = self.apply_content_rules(paper)
        if not content_result[0]:
            return False, 0.0, f"Content filter: {content_result[2]}"
        score += content_result[1]
        reasons.append(content_result[2])
        
        return True, score, "; ".join(reasons)
    
    def apply_keyword_rules(self, paper: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Apply keyword-based filtering rules."""
        title = paper.get("title", "").lower()
        abstract = paper.get("summary", "").lower()
        score = 0.0
        
        # Required keywords
        required_keywords = self.rules["keyword_rules"].get("required", [])
        if required_keywords:
            found_required = False
            for keyword in required_keywords:
                if keyword.lower() in title or keyword.lower() in abstract:
                    found_required = True
                    score += 2.0
            if not found_required:
                return False, 0.0, "Missing required keywords"
        
        # Bonus keywords
        bonus_keywords = self.rules["keyword_rules"].get("bonus", [])
        for keyword in bonus_keywords:
            if keyword.lower() in title:
                score += 1.5
            elif keyword.lower() in abstract:
                score += 0.5
        
        # Excluded keywords
        excluded_keywords = self.rules["keyword_rules"].get("excluded", [])
        for keyword in excluded_keywords:
            if keyword.lower() in title or keyword.lower() in abstract:
                return False, 0.0, f"Contains excluded keyword: {keyword}"
        
        return True, score, f"Keyword score: {score}"
    
    def apply_author_rules(self, paper: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Apply author-based filtering rules."""
        authors = [author.lower() for author in paper.get("authors", [])]
        score = 0.0
        
        # Required authors
        required_authors = self.rules["author_rules"].get("required", [])
        if required_authors:
            found_required = False
            for author in required_authors:
                if author.lower() in authors:
                    found_required = True
                    score += 3.0
            if not found_required:
                return False, 0.0, "Missing required authors"
        
        # Preferred authors
        preferred_authors = self.rules["author_rules"].get("preferred", [])
        for author in preferred_authors:
            if author.lower() in authors:
                score += 2.0
        
        # Excluded authors
        excluded_authors = self.rules["author_rules"].get("excluded", [])
        for author in excluded_authors:
            if author.lower() in authors:
                return False, 0.0, f"Contains excluded author: {author}"
        
        return True, score, f"Author score: {score}"
    
    def apply_content_rules(self, paper: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Apply content quality filtering rules."""
        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        score = 0.0
        
        # Abstract length requirements
        min_length = self.rules["content_rules"].get("min_abstract_length", 100)
        max_length = self.rules["content_rules"].get("max_abstract_length", 2000)
        
        if len(abstract) < min_length:
            return False, 0.0, f"Abstract too short: {len(abstract)} chars"
        if len(abstract) > max_length:
            return False, 0.0, f"Abstract too long: {len(abstract)} chars"
        
        # Title quality checks
        if len(title) < 10:
            return False, 0.0, "Title too short"
        
        # Content quality indicators
        quality_indicators = self.rules["content_rules"].get("quality_indicators", [])
        for indicator in quality_indicators:
            if indicator.lower() in abstract.lower():
                score += 0.5
        
        return True, score, f"Content score: {score}"
```

## 🔧 Integration with Main Bot

### Step 1: Update arxiv_bot.py

Add this to your `arxiv_bot.py`:

```python
# Add at the top of the file
try:
    from custom_rules import custom_paper_filter, custom_paper_score
    CUSTOM_RULES_AVAILABLE = True
except ImportError:
    CUSTOM_RULES_AVAILABLE = False

# Update the ArxivBot class
class ArxivBot:
    def __init__(self, config_file: str = "config.json"):
        self.config = self.load_config(config_file)
        self.papers = []
        self.custom_filter = None
        
        # Initialize custom filter if available
        if CUSTOM_RULES_AVAILABLE:
            try:
                from advanced_rules import AdvancedPaperFilter
                self.custom_filter = AdvancedPaperFilter(self.config)
            except ImportError:
                pass
    
    def matches_criteria(self, paper: Dict[str, Any]) -> bool:
        """Check if paper matches user-defined criteria."""
        
        # Use custom filter if available
        if self.custom_filter:
            include, score, reason = self.custom_filter.filter_paper(paper)
            paper["score"] = score
            paper["filter_reason"] = reason
            return include
        
        # Fall back to simple filtering
        text_to_check = f"{paper['title']} {paper['summary']}".lower()
        
        # Check for required keywords
        keywords = self.config.get("keywords", [])
        if keywords:
            keyword_matches = any(keyword.lower() in text_to_check for keyword in keywords)
            if not keyword_matches:
                return False
        
        # Check for excluded keywords
        exclude_keywords = self.config.get("exclude_keywords", [])
        if exclude_keywords:
            exclude_matches = any(keyword.lower() in text_to_check for keyword in exclude_keywords)
            if exclude_matches:
                return False
        
        return True
    
    def calculate_score(self, paper: Dict[str, Any]) -> float:
        """Calculate relevance score for a paper."""
        
        # Use custom scoring if available
        if CUSTOM_RULES_AVAILABLE:
            try:
                return custom_paper_score(paper, self.config)
            except Exception as e:
                logger.warning(f"Custom scoring failed: {e}")
        
        # Fall back to default scoring
        score = 0.0
        text_to_check = f"{paper['title']} {paper['summary']}".lower()
        
        keywords = self.config.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_to_check:
                score += 1.0
        
        title_lower = paper['title'].lower()
        for keyword in keywords:
            if keyword.lower() in title_lower:
                score += 0.5
        
        return score
```

### Step 2: Create Custom Configuration

Create a `custom_config.json` with your specific rules:

```json
{
    "categories": ["cs.AI", "cs.LG", "cs.CL"],
    "keywords": ["machine learning", "deep learning"],
    "exclude_keywords": ["survey", "review"],
    "max_papers": 30,
    "days_back": 7,
    "min_score": 1.0,
    
    "keyword_rules": {
        "required": ["transformer", "attention"],
        "bonus": ["bert", "gpt", "llm"],
        "excluded": ["preliminary", "draft"]
    },
    
    "author_rules": {
        "preferred": ["Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio"],
        "excluded": []
    },
    
    "content_rules": {
        "min_abstract_length": 200,
        "max_abstract_length": 1500,
        "quality_indicators": ["novel", "improved", "state-of-the-art"]
    },
    
    "scoring": {
        "title_match_weight": 3.0,
        "abstract_match_weight": 1.0,
        "author_reputation_weight": 2.0,
        "recency_weight": 0.5
    }
}
```

## 🌐 HTTPS and Nginx Configuration

### 1. SSL Certificate Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 2. Nginx Configuration with HTTPS

Create `/etc/nginx/sites-available/arxiv-rss`:

```nginx
# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=rss:10m rate=10r/s;
    
    # Main location
    location / {
        limit_req zone=rss burst=20 nodelay;
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # RSS feeds with caching
    location ~* \.(xml|rss)$ {
        limit_req zone=rss burst=20 nodelay;
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        expires 1h;
        add_header Cache-Control "public, max-age=3600";
        add_header Content-Type "application/rss+xml; charset=utf-8";
    }
    
    # Health check
    location /health {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. Update RSS Service for HTTPS

Update your `rss_service.py` to handle HTTPS:

```python
# Add to rss_service.py
import os

def get_base_url():
    """Get the base URL for the service."""
    if os.environ.get('HTTPS_ENABLED', 'false').lower() == 'true':
        return f"https://{os.environ.get('DOMAIN', 'localhost')}"
    else:
        return f"http://localhost:{os.environ.get('PORT', '5000')}"

# Update the generate_rss_feed function
def generate_rss_feed(category=None):
    # ... existing code ...
    
    # Use HTTPS-aware base URL
    base_url = get_base_url()
    feed_url = f"{base_url}/rss/{category}" if category else f"{base_url}/rss"
    
    # ... rest of the function ...
```

## 📋 Example Custom Rules

### Example 1: Computer Vision Focus

```json
{
    "categories": ["cs.CV", "cs.AI"],
    "keyword_rules": {
        "required": ["computer vision", "image"],
        "bonus": ["detection", "segmentation", "classification"],
        "excluded": ["survey", "review"]
    },
    "author_rules": {
        "preferred": ["Fei-Fei Li", "Andrew Ng", "Yann LeCun"]
    },
    "content_rules": {
        "min_abstract_length": 150,
        "quality_indicators": ["novel", "improved", "benchmark"]
    }
}
```

### Example 2: NLP Research

```json
{
    "categories": ["cs.CL", "cs.AI"],
    "keyword_rules": {
        "required": ["natural language", "nlp"],
        "bonus": ["transformer", "bert", "gpt", "attention"],
        "excluded": ["preliminary", "draft"]
    },
    "content_rules": {
        "min_abstract_length": 200,
        "quality_indicators": ["state-of-the-art", "novel", "improved"]
    }
}
```

### Example 3: Machine Learning Papers

```json
{
    "categories": ["cs.LG", "stat.ML"],
    "keyword_rules": {
        "required": ["machine learning", "deep learning"],
        "bonus": ["neural network", "optimization", "gradient"],
        "excluded": ["tutorial", "introduction"]
    },
    "author_rules": {
        "preferred": ["Geoffrey Hinton", "Yoshua Bengio", "Yann LeCun"]
    }
}
```

## 🎯 Testing Your Custom Rules

```bash
# Test your custom rules
python -c "
from arxiv_bot import ArxivBot
bot = ArxivBot('custom_config.json')
papers = bot.fetch_arxiv_papers()
filtered = bot.filter_papers(papers)
print(f'Found {len(filtered)} papers matching your criteria')
for paper in filtered[:3]:
    print(f'- {paper[\"title\"]} (Score: {paper[\"score\"]})')
"
```

## ✅ Custom Rules Checklist

- [ ] Created `custom_rules.py` or `advanced_rules.py`
- [ ] Defined your filtering criteria in `custom_config.json`
- [ ] Updated `arxiv_bot.py` to use custom rules
- [ ] Tested your rules with sample papers
- [ ] Configured HTTPS and Nginx (if self-hosting)
- [ ] Updated RSS service for HTTPS support
- [ ] Verified RSS feeds work with your custom rules

Your arXiv spider is now fully customizable! You can create sophisticated filtering rules that match your exact research interests and requirements. 