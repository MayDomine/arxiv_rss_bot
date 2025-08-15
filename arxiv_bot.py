#!/usr/bin/env python3
"""
arXiv Bot - Fetches papers from arXiv RSS feed and filters them based on user-defined conditions.
Updates README.md with matching papers.
"""

import feedparser
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

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
        }

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
                "summary": entry.summary,
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

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv-rss-equ1ow2e6-maydomines-projects.vercel.app/)

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

    def generate_papers_section(self, papers: List[Dict[str, Any]]) -> str:
        """Generate the papers section for README."""
        if not papers:
            return "No papers found matching the criteria."

        papers_text = []
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"

            paper_entry = f"""### {i}. [{paper['title']}]({paper['link']})

**Authors**: {authors_str}  
**Category**: {paper['category']}  
**Published**: {paper['published_date']}  
**Score**: {paper['score']:.1f}

{paper['summary'][:300]}{'...' if len(paper['summary']) > 300 else ''}

---"""
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
        self.update_readme(filtered_papers)

        logger.info("arXiv Bot completed successfully!")

    def deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
