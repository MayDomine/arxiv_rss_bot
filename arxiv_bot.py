#!/usr/bin/env python3
"""
arXiv Bot - Fetches papers from arXiv RSS feed and filters them based on user-defined conditions.
Updates README.md with matching papers.
"""

import feedparser
import json
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ArxivBot:
    def __init__(self, config_file: str = "config.json"):
        """Initialize the arXiv bot with configuration."""
        self.config = self.load_config(config_file)
        self.papers = []

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Config file {config_file} not found, using default configuration"
            )
            return self.get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
            "keywords": [
                "machine learning",
                "deep learning",
                "neural network",
                "transformer",
            ],
            "max_papers": 50,
            "days_back": 7,
            "exclude_keywords": [],
            "min_score": 0.0,
            "ai_summary": {
                "enabled": False,
                "api_key_env": "DASHSCOPE_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-long",
                "prompt_file": "ai_summary_prompt.txt",
                "max_papers_to_summarize": 5,
                "max_workers": 5,  # Number of concurrent requests
            },
        }

    def parse_paper_summary(self, summary: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse the summary of a paperself. 
        arXiv:${arxiv_number} Announce Type: ${Type} Abstract: ${abstract}
        """
        pattern = r"arXiv:(\S+)\s+Announce\s+Type:\s+(\S+)\s+Abstract:\s+(.+)"
        match = re.match(pattern, summary)
        
        if match:
            try:
                arxiv_number = match.group(1)
                announce_type = match.group(2)
                abstract = match.group(3)
                return arxiv_number, announce_type, abstract
            except Exception as e:
                return None, None, summary
        else:
            return None, None, summary


    def parse_paper_entry(self, entry: Any, category: str) -> Optional[Dict[str, Any]]:
        """Parse a single paper entry from RSS feed."""
        try:
            # Extract paper ID from link
            paper_id = entry.link.split("/")[-1]

            # Parse publication date
            pub_date = datetime(*entry.published_parsed[:6])

            # Check if paper is within the specified time range
            days_back = self.config.get("days_back", 7)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            arxiv_id, pub_type, summary = self.parse_paper_summary(entry.summary)

            if pub_date < cutoff_date:
                return None

            paper = {
                "id": paper_id,
                "title": entry.title,
                "authors": (
                    [author.name for author in entry.authors]
                    if hasattr(entry, "authors")
                    else []
                ),
                "type": pub_type,
                "arxiv_id": arxiv_id,
                "summary": summary,
                "category": category,
                "published_date": pub_date.strftime("%Y-%m-%d"),
                "link": entry.link,
                "score": 0.0,
            }

            return paper

        except Exception as e:
            logger.error(f"Error parsing paper entry: {e}")
            return None
    def fetch_arxiv_papers(self) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv RSS feeds for specified categories."""
        all_papers = []
        seen_papers = set()  # Track seen paper IDs to avoid duplicates during fetching

        for category in self.config["categories"]:
            try:
                # Construct arXiv RSS URL
                rss_url = f"http://export.arxiv.org/rss/{category}"
                logger.info(f"Fetching papers from {rss_url}")

                # Parse RSS feed
                feed = feedparser.parse(rss_url)

                if feed.bozo:
                    logger.warning(
                        f"RSS feed for {category} has issues: {feed.bozo_exception}"
                    )

                # Process entries
                for entry in feed.entries:
                    paper = self.parse_paper_entry(entry, category)
                    if paper:
                        # Check for duplicates using paper ID
                        paper_id = paper.get("id", "")
                        if paper_id and paper_id in seen_papers:
                            logger.debug(
                                f"Skipping duplicate paper during fetch: {paper_id}"
                            )
                            continue

                        # Check for duplicates using title (fallback)
                        title_normalized = paper.get("title", "").lower().strip()
                        if title_normalized in seen_papers:
                            logger.debug(
                                f"Skipping duplicate paper by title during fetch: {title_normalized[:50]}..."
                            )
                            continue

                        # Add to seen set and papers list
                        seen_papers.add(paper_id)
                        seen_papers.add(title_normalized)
                        all_papers.append(paper)

            except Exception as e:
                logger.error(f"Error fetching papers from {category}: {e}")
                continue

        logger.info(f"Fetched {len(all_papers)} unique papers total")
        return all_papers


    def filter_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter papers based on user-defined conditions and remove duplicates."""
        filtered_papers = []
        seen_papers = set()  # Track seen paper IDs to avoid duplicates

        for paper in papers:
            if self.matches_criteria(paper):
                # Check for duplicates using paper ID
                paper_id = paper.get("id", "")
                if paper_id and paper_id in seen_papers:
                    logger.debug(f"Skipping duplicate paper: {paper_id}")
                    continue

                # Check for duplicates using title (fallback)
                title_normalized = paper.get("title", "").lower().strip()
                if title_normalized in seen_papers:
                    logger.debug(
                        f"Skipping duplicate paper by title: {title_normalized[:50]}..."
                    )
                    continue

                # Add to seen set
                seen_papers.add(paper_id)
                seen_papers.add(title_normalized)

                # Calculate score and add to filtered papers
                paper["score"] = self.calculate_score(paper)
                filtered_papers.append(paper)

        # Sort by score (highest first)
        filtered_papers.sort(key=lambda x: x["score"], reverse=True)

        # Limit to max_papers
        max_papers = self.config.get("max_papers", 50)
        filtered_papers = filtered_papers[:max_papers]

        logger.info(
            f"Filtered to {len(filtered_papers)} unique papers (removed {len(papers) - len(filtered_papers)} duplicates)"
        )
        return filtered_papers

    def matches_criteria(self, paper: Dict[str, Any]) -> bool:
        """Check if paper matches user-defined criteria."""
        text_to_check = f"{paper['title']} {paper['summary']}".lower()

        # Check for required keywords
        keywords = self.config.get("keywords", [])

        if paper.get("type", "").lower() != "new" :
            return False

        if keywords:
            keyword_matches = any(
                keyword.lower() in text_to_check for keyword in keywords
            )
            if not keyword_matches:
                return False

        # Check for excluded keywords
        exclude_keywords = self.config.get("exclude_keywords", [])
        if exclude_keywords:
            exclude_matches = any(
                keyword.lower() in text_to_check for keyword in exclude_keywords
            )
            if exclude_matches:
                return False

        return True

    def calculate_score(self, paper: Dict[str, Any]) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0
        text_to_check = f"{paper['title']} {paper['summary']}".lower()

        # Score based on keyword matches
        keywords = self.config.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_to_check:
                score += 1.0

        # Bonus for title matches
        title_lower = paper["title"].lower()
        for keyword in keywords:
            if keyword.lower() in title_lower:
                score += 0.5

        return score

    def download_arxiv_pdf(self, paper: Dict[str, Any], output_dir: Path = Path("pdf_cache")) -> Optional[Path]:
        """Download arXiv PDF for a paper.
        
        Args:
            paper: Paper dictionary with 'link' and 'id' fields
            output_dir: Directory to save PDF files
            
        Returns:
            Path to downloaded PDF file, or None if download failed
        """
        try:
            # Get PDF URL from paper link
            pdf_url = paper["link"].replace("/abs/", "/pdf/") + ".pdf"
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use paper ID as filename
            paper_id = paper.get("id", paper["link"].split("/")[-1])
            pdf_path = output_dir / f"{paper_id}.pdf"
            
            # Skip if already downloaded
            if pdf_path.exists():
                logger.debug(f"PDF already exists: {pdf_path}")
                return pdf_path
            
            logger.info(f"Downloading PDF for {paper['title'][:50]}...")
            
            # Download PDF
            response = requests.get(pdf_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                logger.warning(f"Content-Type is {content_type}, might not be a PDF")
            
            # Save to file
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = pdf_path.stat().st_size
            logger.info(f"Downloaded PDF: {pdf_path.name} ({file_size:,} bytes)")
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF for {paper.get('title', 'unknown')}: {e}")
            return None

    def summarize_pdf_with_ai(self, pdf_path: Path, paper_title: str = "", client: Optional[OpenAI] = None) -> Optional[str]:
        """Summarize PDF using AI (Qwen/DashScope or other OpenAI-compatible API).
        
        Args:
            pdf_path: Path to the PDF file
            paper_title: Optional paper title for context
            client: Optional OpenAI client (if None, will create one)
            
        Returns:
            Summary text if successful, None otherwise
        """
        ai_config = self.config.get("ai_summary", {})
        
        if not ai_config.get("enabled", False):
            return None
        
        try:
            # Use provided client or create new one
            if client is None:
                # Get API key from environment
                api_key_env = ai_config.get("api_key_env", "DASHSCOPE_API_KEY")
                api_key = os.environ.get(api_key_env)
                
                if not api_key:
                    logger.warning(f"AI summary enabled but {api_key_env} not set, skipping")
                    return None
                
                # Initialize OpenAI client with configurable base URL
                base_url = ai_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
            
            # Upload file
            file_object = client.files.create(
                file=pdf_path,
                purpose="file-extract"
            )
            
            # Load prompt from file
            prompt_file = ai_config.get("prompt_file", "ai_summary_prompt.txt")
            prompt = self._load_ai_prompt(prompt_file)
            
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
            
            # Get model from config
            model = ai_config.get("model", "qwen-long")
            
            # Call chat completion
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
            )
            
            summary = completion.choices[0].message.content
            return summary
            
        except Exception as e:
            logger.debug(f"Error summarizing PDF {pdf_path.name}: {e}")
            return None

    def _load_ai_prompt(self, prompt_file: str) -> str:
        """Load AI prompt from file.
        
        Args:
            prompt_file: Path to prompt file
            
        Returns:
            Prompt text, or default prompt if file not found
        """
        try:
            prompt_path = Path(prompt_file)
            if prompt_path.exists():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Error loading prompt file {prompt_file}: {e}")
        
        # Return default prompt
        return """请总结这篇论文的核心结论和实验结果。请包括：
1. 论文的主要贡献和创新点
2. 核心实验方法和设置
3. 主要实验结果和性能指标
4. 关键结论和发现

请用中文回答，结构清晰，重点突出。"""

    def render_readme(self, papers: List[Dict[str, Any]]) -> str:
        """Render papers to README.md format."""
        if not papers:
            return self.get_empty_readme()

        # Read template if exists
        template = self.get_readme_template()

        # Generate papers section
        papers_section = self.generate_papers_section(papers)

        # Replace placeholder in template
        readme_content = template.replace("{{PAPERS_SECTION}}", papers_section)

        # Add last updated timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        readme_content = readme_content.replace("{{LAST_UPDATED}}", timestamp)

        return readme_content

    def get_readme_template(self) -> str:
        """Get README template."""
        template_path = "README.template.md"
        try:
            with open(template_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return self.get_default_template()

    def get_default_template(self) -> str:
        """Get default README template."""
        return """# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)

## 📊 Statistics

- **Last Updated**: {{LAST_UPDATED}}
- **Total Papers Found**: {{PAPER_COUNT}}
- **Categories Monitored**: {{CATEGORIES}}

## 📚 Recent Papers

{{PAPERS_SECTION}}

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- {{KEYWORDS}}

## 📅 Schedule

The bot runs daily at 12:00 UTC via GitHub Actions to fetch the latest papers.

---
*Generated automatically by arXiv Bot*
"""

    def get_empty_readme(self) -> str:
        """Get README content when no papers are found."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""# arXiv Papers Bot 🤖

## 📊 Statistics

- **Last Updated**: {timestamp}
- **Total Papers Found**: 0
- **Categories Monitored**: {', '.join(self.config.get('categories', []))}

## 📚 Recent Papers

No papers matching the criteria were found in the last {self.config.get('days_back', 7)} days.

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- {', '.join(self.config.get('keywords', []))}

## 📅 Schedule

The bot runs daily at 12:00 UTC via GitHub Actions to fetch the latest papers.

---
*Generated automatically by arXiv Bot*
"""

    def _process_paper_with_ai(self, paper: Dict[str, Any], index: int, ai_config: Dict[str, Any], client: Optional[OpenAI]) -> Tuple[int, Optional[str]]:
        """Process a single paper: download PDF and get AI summary.
        
        Args:
            paper: Paper dictionary
            index: Paper index (1-based)
            ai_config: AI configuration
            client: OpenAI client instance
            
        Returns:
            Tuple of (index, ai_summary)
        """
        try:
            # Download PDF
            pdf_path = self.download_arxiv_pdf(paper)
            if pdf_path:
                # Get AI summary
                ai_summary = self.summarize_pdf_with_ai(pdf_path, paper['title'], client)
                return (index, ai_summary)
        except Exception as e:
            logger.debug(f"Failed to get AI summary for paper {index}: {e}")
        
        return (index, None)

    def generate_papers_section(self, papers: List[Dict[str, Any]]) -> str:
        """Generate the papers section for README."""
        if not papers:
            return "No papers found matching the criteria."

        ai_config = self.config.get("ai_summary", {})
        ai_enabled = ai_config.get("enabled", False)
        max_summarize = ai_config.get("max_papers_to_summarize", 5)
        
        # Prepare papers that need AI summary
        papers_to_summarize = []
        if ai_enabled:
            papers_to_summarize = papers[:max_summarize]
        
        # Initialize OpenAI client once if needed
        ai_client = None
        if ai_enabled and papers_to_summarize:
            api_key_env = ai_config.get("api_key_env", "DASHSCOPE_API_KEY")
            api_key = os.environ.get(api_key_env)
            if api_key:
                base_url = ai_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                ai_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )
        
        # Process papers in parallel with progress bar
        ai_summaries = {}
        if papers_to_summarize:
            max_workers = ai_config.get("max_workers", 5)  # Configurable concurrency
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(self._process_paper_with_ai, paper, i+1, ai_config, ai_client): i+1
                    for i, paper in enumerate(papers_to_summarize)
                }
                
                # Process with progress bar
                with tqdm(total=len(papers_to_summarize), desc="Processing papers with AI", unit="paper") as pbar:
                    for future in as_completed(future_to_index):
                        index, ai_summary = future.result()
                        ai_summaries[index] = ai_summary
                        pbar.update(1)
                        if ai_summary:
                            pbar.set_postfix({"status": f"✓ {index}/{len(papers_to_summarize)}"})
        
        # Generate papers section
        papers_text = []
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
            ai_summary = ai_summaries.get(i)

            paper_entry = f"""### {i}. [{paper['title']}]({paper['link']})

**Authors**: {authors_str}  
**Category**: {paper['category']}  
**Published**: {paper['published_date']}  
**Score**: {paper['score']:.1f}  
**Type**: {paper['type']}  
**ArXiv ID**: {paper['arxiv_id']}  

#### Abstract
{paper['summary'][:300]}{'...' if len(paper['summary']) > 300 else ''}"""

            # Add AI summary if available (wrapped in collapsible details block)
            if ai_summary:
                model_name = ai_config.get('model', 'AI')
                paper_entry += f"""

<details>
<summary><strong>🤖 AI Summary (by {model_name})</strong> - Click to expand</summary>

{ai_summary}

</details>"""

            paper_entry += "\n\n---"
            papers_text.append(paper_entry)

        return "\n\n".join(papers_text)

    def update_readme(self, papers: List[Dict[str, Any]]) -> None:
        """Update README.md file with filtered papers."""
        readme_content = self.render_readme(papers)

        # Update placeholders with actual values
        readme_content = readme_content.replace("{{PAPER_COUNT}}", str(len(papers)))
        readme_content = readme_content.replace(
            "{{CATEGORIES}}", ", ".join(self.config.get("categories", []))
        )
        readme_content = readme_content.replace(
            "{{KEYWORDS}}", ", ".join(self.config.get("keywords", []))
        )

        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        logger.info(f"Updated README.md with {len(papers)} papers")

    def run(self) -> None:
        """Main execution method."""
        logger.info("Starting arXiv Bot...")

        # Fetch papers
        papers = self.fetch_arxiv_papers()

        # Apply additional deduplication if enabled
        if self.config.get("enable_deduplication", True):
            papers = self.deduplicate_papers(papers)

        # Filter papers
        filtered_papers = self.filter_papers(papers)

        # Update README
        if filtered_papers :
            self.update_readme(filtered_papers)

        logger.info("arXiv Bot completed successfully!")

    def deduplicate_papers(self, papers: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """
        Remove duplicate papers based on multiple criteria.

        Args:
            papers: List of paper dictionaries

        Returns:
            List of unique papers
        """
        if not papers:
            return papers

        unique_papers = []
        seen_ids = set()
        seen_titles = set()
        seen_links = set()

        for paper in papers:
            paper_id = paper.get("id", "").strip()
            title = paper.get("title", "").lower().strip()
            link = paper.get("link", "").strip()

            # Skip if we've seen this paper before
            is_duplicate = False

            # Check by ID (most reliable)
            if paper_id and paper_id in seen_ids:
                logger.debug(f"Skipping duplicate by ID: {paper_id}")
                is_duplicate = True

            # Check by title (normalized)
            elif title and title in seen_titles:
                logger.debug(f"Skipping duplicate by title: {title[:50]}...")
                is_duplicate = True

            # Check by link
            elif link and link in seen_links:
                logger.debug(f"Skipping duplicate by link: {link}")
                is_duplicate = True

            if not is_duplicate:
                # Add to tracking sets
                if paper_id:
                    seen_ids.add(paper_id)
                if title:
                    seen_titles.add(title)
                if link:
                    seen_links.add(link)

                unique_papers.append(paper)

        removed_count = len(papers) - len(unique_papers)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate papers")

        return unique_papers


def main():
    """Main entry point."""
    bot = ArxivBot()
    bot.run()


if __name__ == "__main__":
    main()
