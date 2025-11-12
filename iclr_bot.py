#!/usr/bin/env python3
"""
ICLR Bot - Fetches accepted/reviewed papers for ICLR 2026 from OpenReview.
Generates a cached JSON file and a README with quick links to the papers.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from tqdm import tqdm

import requests  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
high_score_keywords = ["Sparse Attention", "Speculative", "Reinforcement"]


OPENREVIEW_API = os.environ.get("OPENREVIEW_API", "https://api2.openreview.net")

def _ensure_directory(path: Path) -> None:
    """Create parent directories for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _timestamp_ms_to_datetime(timestamp_ms: Optional[int]) -> Optional[datetime]:
    """Convert milliseconds timestamp to timezone-aware datetime."""
    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    except Exception:
        return None


def _extract_value(field: Any) -> Any:
    """Extract value from OpenReview v2 content field structure."""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def _extract_numeric_rating(raw_rating: Any) -> Optional[float]:
    """Attempt to extract a numeric rating from the OpenReview rating field."""
    rating_value = _extract_value(raw_rating)
    if rating_value is None:
        return None

    if isinstance(rating_value, (int, float)):
        return float(rating_value)

    if isinstance(rating_value, str):
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", rating_value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


@dataclass
class ICLRPaper:
    """Container representing an ICLR paper with review stats."""

    forum_id: str
    paper_number: int
    title: str
    authors: List[str]
    keywords: List[str]
    abstract: str
    pdf_link: Optional[str]
    forum_link: str
    submission_date: Optional[datetime]
    decision: Optional[str]
    decision_comment: Optional[str]
    average_rating: Optional[float]
    rating_count: int
    ratings: List[float] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the paper into a JSON-compatible dictionary."""
        return {
            "forum_id": self.forum_id,
            "paper_number": self.paper_number,
            "title": self.title,
            "authors": self.authors,
            "keywords": self.keywords,
            "abstract": self.abstract,
            "pdf_link": self.pdf_link,
            "forum_link": self.forum_link,
            "submission_date": self.submission_date.isoformat()
            if self.submission_date
            else None,
            "decision": self.decision,
            "decision_comment": self.decision_comment,
            "average_rating": self.average_rating,
            "rating_count": self.rating_count,
            "ratings": self.ratings,
            "updated_at": self.updated_at.isoformat(),
        }


class ICLRBot:
    """Bot responsible for fetching ICLR papers and producing artifacts."""

    def __init__(
        self,
        year: int = 2026,
        output_dir: Optional[Path] = None,
        max_papers: Optional[int] = None,
        display_limit: int = 100,
        session: Optional[requests.Session] = None,
        config_file: str = "config.json",
    ) -> None:
        self.year = year
        self.output_dir = output_dir or Path("iclr")
        self.max_papers = max_papers
        self.display_limit = display_limit
        self.session = session or requests.Session()
        self.config_file = config_file

        self.cache_path = self.output_dir / "iclr_cache.json"
        self.readme_path = self.output_dir / "README.md"

        # Load configuration
        self.config = self.load_config()

        # Override display_limit from config if not explicitly set
        if display_limit == 100:  # Default value, so check config
            self.display_limit = self.config.get("display_limit", 100)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded ICLR configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            logger.warning(
                f"Config file {self.config_file} not found, using default configuration"
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
                "LLM",
                "RL",
                "RLHF",
                "Inference",
                "Training",
                "Attention",
                "Transformer",
                "MOE",
                "Sparse",
                "Quantization",
                "Speculative",
                "Efficient",
                "Efficiency",
                "Framework",
                "Parallel",
                "Distributed",
                "Kernel",
                "Decode",
                "Decoding",
                "Prefill",
                "Throughput",
                "Fast",
                "Network",
                "Hardware",
                "Cluster",
                "FP8",
                "FP4",
                "Optimization",
                "Scalable",
                "Communication"
            ],
            "max_papers": 50,
            "days_back": 7,
            "exclude_keywords": [
                "survey",
                "review"
            ],
            "min_score": 0.0,
            "fetch_ratings": True,  # Whether to fetch review ratings
            "display_limit": 100,  # Number of papers to display in README
        }

    def matches_criteria(self, paper: ICLRPaper) -> bool:
        """Check if paper matches user-defined criteria."""
        text_to_check = f"{paper.title} {paper.abstract}".lower()

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

    def calculate_score(self, paper: ICLRPaper) -> float:
        """Calculate relevance score for a paper based on keyword matches."""
        score = 0.0
        text_to_check = f"{paper.title} {paper.abstract}".lower()

        # Score based on keyword matches
        keywords = self.config.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_to_check:
                score += 1.0

        for keyword in high_score_keywords:
            if keyword.lower() in text_to_check:
                score += 10


        # Bonus for title matches
        title_lower = paper.title.lower()
        for keyword in keywords:
            if keyword.lower() in title_lower:
                score += 0.5

        for keyword in high_score_keywords:
            if keyword.lower() in title_lower:
                score += 20

        return score

    @property
    def conference_domain(self) -> str:
        return f"ICLR.cc/{self.year}/Conference"

    def _search_notes(self, term: str, offset: int, limit: int) -> Dict[str, Any]:
        """Use the OpenReview search endpoint to retrieve notes."""
        params = {
            "term": term,
            "offset": offset,
            "limit": limit,
        }
        try:
            url = f"{OPENREVIEW_API}/notes/search"
            logger.debug("GET %s params=%s", url, params)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            logger.error("OpenReview search failed for term=%s: %s", term, exc)
            raise

    def _fetch_notes(
        self, domain: str, forum: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch all notes for a specific domain."""
        notes: List[Dict[str, Any]] = []
        page_size = 10000 # Use larger page size to minimize requests

        try:
            url = f"{OPENREVIEW_API}/notes"

            for current_offset in tqdm(range(0, self.max_papers, page_size)):
                params = {
                    "content.venueid": f"{domain}/Submission",
                    "domain": domain,
                    "details": "replyCount,presentation,writable",
                    "offset": current_offset,
                    "limit": page_size,
                }

                logger.debug("GET %s params=%s", url, params)
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                chunk = data.get("notes") or []
                if not chunk:
                    break

                # Filter by forum if specified
                if forum:
                    chunk = [note for note in chunk if note.get("forum") == forum]

                notes.extend(chunk)

                # Check if we've fetched all available notes
                if len(chunk) < page_size:
                    # This was the last page
                    break

                # Respect max_papers limit
                if self.max_papers and len(notes) >= self.max_papers:
                    notes = notes[: self.max_papers]
                    break

        except requests.HTTPError as exc:
            logger.error("Failed to fetch notes for domain=%s: %s", domain, exc)
            raise

        return notes

    def fetch_submissions(self) -> List[Dict[str, Any]]:
        """Fetch all submissions for the configured ICLR year."""
        # Check if cache exists and is recent (less than 1 hour old)
        import os
        from datetime import datetime, timedelta

        if os.path.exists(self.cache_path):
            try:
                cache_mtime = datetime.fromtimestamp(os.path.getmtime(self.cache_path))
                if datetime.now() - cache_mtime < timedelta(hours=1):
                    logger.info("Using cached ICLR submissions (less than 1 hour old)")
                    cached = self.load_cached_papers()
                    if cached:
                        logger.info("Loaded %d submissions from cache", len(cached))
                        return [{"content": {
                            "title": {"value": p["title"]},
                            "authors": {"value": p["authors"]},
                            "keywords": {"value": p["keywords"]},
                            "abstract": {"value": p["abstract"]},
                            "pdf": p.get("pdf_link")
                        }, "forum": p["forum_id"], "id": p["forum_id"], "cdate": int(datetime.fromisoformat(p["submission_date"].replace("Z", "+00:00")).timestamp() * 1000) if p.get("submission_date") else None} for p in cached]
            except Exception as e:
                logger.warning("Failed to load cache: %s", e)

        logger.info("Fetching ICLR %s submissions from OpenReview...", self.year)
        # For testing, limit to first 2000 papers to avoid timeout
        self.max_papers = 20000
        submissions = self._fetch_notes(self.conference_domain)

        logger.info("Fetched %d submissions", len(submissions))
        return submissions

    def _fetch_ratings(self, paper_number: int, forum_id: str) -> Dict[str, Any]:
        """Fetch review ratings for a specific paper."""
        try:
            # Get all replies/reviews for this paper's forum
            reply_params = {
                "forum": forum_id,
                "limit": 20,  # Reduce limit to speed up requests
            }

            url = f"{OPENREVIEW_API}/notes"
            response = self.session.get(url, params=reply_params, timeout=10)  # Shorter timeout
            response.raise_for_status()
            data = response.json()

            reviews = data.get("notes", [])

            ratings: List[float] = []
            for review in reviews:
                content = review.get("content", {})
                rating_value = _extract_value(content.get("rating"))
                parsed = _extract_numeric_rating(rating_value)
                if parsed is not None:
                    ratings.append(parsed)

            if not ratings:
                return {"ratings": [], "average": None}

            return {
                "ratings": ratings,
                "average": sum(ratings) / len(ratings),
            }

        except requests.exceptions.Timeout:
            # Don't log timeout warnings to avoid spam
            return {"ratings": [], "average": None}
        except requests.HTTPError as exc:
            # Only log serious errors
            if exc.response.status_code not in [404, 429]:
                logger.warning(
                    "Failed to fetch reviews for paper %s (%s): %s",
                    paper_number,
                    forum_id,
                    exc,
                )
            return {"ratings": [], "average": None}
        except Exception as exc:
            # Log other exceptions but continue
            logger.debug(
                "Error fetching reviews for paper %s (%s): %s",
                paper_number,
                forum_id,
                exc,
            )
            return {"ratings": [], "average": None}

    def _fetch_ratings_batch(self, paper_data_list: List[Tuple[int, str]]) -> Dict[str, Dict[str, Any]]:
        """Fetch ratings for multiple papers using batch API."""
        if not paper_data_list:
            return {}

        # Group papers into smaller batches to avoid issues
        batch_size = 20  # Smaller batches
        results = {}

        for i in tqdm(range(0, len(paper_data_list), batch_size)):
            batch = paper_data_list[i:i + batch_size]
            forum_ids = [forum_id for _, forum_id in batch]

            try:
                # Use batch API to get reviews for multiple forums at once
                params = {
                    "forum": forum_ids,
                    "limit": 100,  # Fixed limit, should be enough for all reviews
                }

                url = f"{OPENREVIEW_API}/notes"
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                reviews = data.get("notes", [])

                # Group reviews by forum
                forum_reviews = {}
                for review in reviews:
                    forum = review.get("forum")
                    if forum not in forum_reviews:
                        forum_reviews[forum] = []
                    forum_reviews[forum].append(review)

                # Extract ratings for each forum
                for _, forum_id in batch:
                    forum_reviews_list = forum_reviews.get(forum_id, [])
                    ratings = []

                    for review in forum_reviews_list:
                        content = review.get("content", {})
                        rating_value = _extract_value(content.get("rating"))
                        parsed = _extract_numeric_rating(rating_value)
                        if parsed is not None:
                            ratings.append(parsed)

                    if ratings:
                        results[forum_id] = {
                            "ratings": ratings,
                            "average": sum(ratings) / len(ratings),
                        }
                    else:
                        results[forum_id] = {"ratings": [], "average": None}

                logger.debug(f"Processed batch {i//batch_size + 1}: {len(batch)} papers, {len(reviews)} reviews")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout fetching batch {i//batch_size + 1}, marking as no ratings")
                for _, forum_id in batch:
                    results[forum_id] = {"ratings": [], "average": None}
            except Exception as exc:
                logger.warning(f"Failed to fetch ratings batch {i//batch_size + 1}: {exc}")
                for _, forum_id in batch:
                    results[forum_id] = {"ratings": [], "average": None}

        return results

    def _fetch_decision(self, paper_number: int, forum_id: str) -> Dict[str, Optional[str]]:
        """Fetch the decision/meta review for a paper."""
        # For now, return no decision since decisions aren't available yet in ICLR 2026
        # When decisions become available, we can search for notes with the paper's forum
        # that contain decision content
        logger.debug("Decisions not yet available for ICLR %s", self.year)
        return {"decision": None, "comment": None}

        # Future implementation when decisions are available:
        # try:
        #     # Search for all notes in the conference domain with this forum
        #     all_notes = self._fetch_notes(self.conference_domain, forum=forum_id)
        #     decisions = [note for note in all_notes if "decision" in (note.get("invitation", "").lower()) or "meta" in (note.get("invitation", "").lower())]
        # except requests.HTTPError:
        #     decisions = []

        # if decisions:
        #     note = decisions[0]
        #     content = note.get("content", {})
        #     decision = _extract_value(content.get("decision") or content.get("recommendation"))
        #     comment = _extract_value(
        #         content.get("comment")
        #         or content.get("justification")
        #         or content.get("metareview")
        #     )
        #     return {"decision": decision, "comment": comment}

        # return {"decision": None, "comment": None}

    def _build_paper(self, submission: Dict[str, Any], fetch_ratings_now: bool = True) -> ICLRPaper:
        """Convert a submission note into an ICLRPaper dataclass."""
        content = submission.get("content", {})

        forum_id = submission.get("forum") or submission.get("id")
        number = submission.get("number") or _extract_value(content.get("number"))
        title = _extract_value(content.get("title", "Untitled"))
        authors = _extract_value(content.get("authors", []))
        keywords = _extract_value(content.get("keywords", []))
        abstract = _extract_value(content.get("abstract", "")).strip()
        pdf_link = _extract_value(content.get("pdf"))
        submission_date = _timestamp_ms_to_datetime(submission.get("cdate"))

        # Only fetch ratings if enabled in config and requested
        if self.config.get("fetch_ratings", True) and fetch_ratings_now:
            ratings_info = self._fetch_ratings(number, forum_id)
        else:
            ratings_info = {"ratings": [], "average": None}

        decision_info = self._fetch_decision(number, forum_id)

        forum_link = f"https://openreview.net/forum?id={forum_id}"

        return ICLRPaper(
            forum_id=forum_id,
            paper_number=number or -1,
            title=title,
            authors=authors,
            keywords=keywords,
            abstract=abstract,
            pdf_link=pdf_link,
            forum_link=forum_link,
            submission_date=submission_date,
            decision=decision_info.get("decision"),
            decision_comment=decision_info.get("comment"),
            average_rating=ratings_info.get("average"),
            rating_count=len(ratings_info.get("ratings", [])),
            ratings=ratings_info.get("ratings", []),
        )

    def fetch_papers(self) -> List[ICLRPaper]:
        """Fetch and enrich papers with review decisions and ratings."""
        submissions = self.fetch_submissions()
        papers: List[ICLRPaper] = []

        # First pass: build papers without ratings for speed
        for submission in submissions:
            try:
                paper = self._build_paper(submission, fetch_ratings_now=False)
                papers.append(paper)
            except Exception as exc:
                forum_id = submission.get("forum") or submission.get("id")
                logger.error("Failed to process submission %s: %s", forum_id, exc)

        # Filter papers by keywords and calculate scores
        filtered_papers = []
        for paper in papers:
            if self.matches_criteria(paper):
                # Calculate relevance score for sorting
                paper.score = self.calculate_score(paper)
                filtered_papers.append(paper)

        logger.info("Filtered %d papers matching keywords out of %d total", len(filtered_papers), len(papers))

        # Only fetch ratings for top 100 filtered papers (for performance)
        if self.config.get("fetch_ratings", True) and filtered_papers:
            # Sort by some criteria (we'll use paper number as proxy for submission order)
            # Papers with higher numbers might be more recent and potentially rated
            sorted_papers = sorted(filtered_papers, key=lambda p: getattr(p, 'score', 0), reverse=True)
            top_papers = sorted_papers[:self.display_limit]

            logger.info(f"Fetching ratings for top {self.display_limit} filtered papers...")

            # Prepare batch data
            paper_data = [(i+1, paper.forum_id) for i, paper in enumerate(top_papers)]

            ratings_batch = self._fetch_ratings_batch(paper_data)

            # Update papers with fetched ratings
            for i, paper in enumerate(top_papers):
                _, forum_id = paper_data[i]
                ratings_info = ratings_batch.get(forum_id, {"ratings": [], "average": None})
                paper.average_rating = ratings_info["average"]
                paper.rating_count = len(ratings_info["ratings"])
                paper.ratings = ratings_info["ratings"]

            rated_count = sum(1 for p in top_papers if p.average_rating is not None)
            logger.info(f"Found ratings for %d out of {self.display_limit} top papers", rated_count)

        logger.info("Processed %d papers with review data", len(filtered_papers))
        return filtered_papers

    def save_cache(self, papers: Iterable[ICLRPaper]) -> None:
        """Persist papers to a JSON cache file."""
        papers_list = [paper.to_dict() for paper in papers]
        _ensure_directory(self.cache_path)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "year": self.year,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "paper_count": len(papers_list),
                    "papers": papers_list,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info("Saved %d papers to cache %s", len(papers_list), self.cache_path)

    def load_cached_papers(self) -> List[Dict[str, Any]]:
        """Load cached papers if available."""
        if not self.cache_path.exists():
            return []
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("papers", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cached papers: %s", exc)
            return []

    def render_readme(self, papers: List[ICLRPaper]) -> str:
        """Render the README markdown content for ICLR papers."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Papers should already be filtered by keywords in fetch_papers
        total_filtered = len(papers)

        # Sort papers by keyword relevance score (descending)
        sorted_papers = sorted(papers, key=lambda p: getattr(p, 'score', 0), reverse=True)
        display_papers = sorted_papers[:self.display_limit]

        # Count rated papers
        rated_count = sum(1 for p in papers if p.average_rating is not None)

        header = [
            "# ICLR 2026 Papers 📚",
            "",
            f"- **Last Updated**: {timestamp}",
            f"- **Total Filtered Papers**: {total_filtered}",
            f"- **Papers with Ratings**: {rated_count}",
            f"- **Total Submissions**: (cached)",
            "",
            "Click any title to view the full discussion on OpenReview.",
            "",
            f"**Configuration**: Filtered by keywords {self.config.get('keywords', [])}",
            "",
        ]

        if not display_papers:
            return "\n".join(
                header
                + [
                    "> No papers matched the current keyword filters. "
                    "Try updating keywords in config.json or run `/update-iclr-cache`."
                ]
            )

        if rated_count > 0:
            header.append(f"> Showing top {len(display_papers)} papers (out of {total_filtered} matching), sorted by keyword relevance score. {rated_count} papers have review ratings.")
        else:
            header.append(f"> Showing top {len(display_papers)} papers (out of {total_filtered} matching), sorted by keyword relevance score (no ratings available yet).")
        header.append("")

        table_header = "| # | Title | Ratings |Avg Rating | Reviews | Decision |\n"
        table_divider = "| --- | --- | --- | --- | --- | --- |\n"

        rows: List[str] = []
        for idx, paper in enumerate(display_papers, 1):
            avg_rating = (
                f"{paper.average_rating:.2f}" if paper.average_rating is not None else "N/A"
            )
            reviews = str(paper.rating_count) if paper.rating_count else "0"
            decision = paper.decision or "Pending"
            title = paper.title.replace("\n", " ").strip()
            rate_str = ",".join(str(round(r)) for r in paper.ratings) if paper.ratings else "N/A"

            rows.append(
                f"| {idx} | [{title}]({paper.forum_link}) | **{rate_str}** | **{avg_rating}** | {reviews} | {decision} | "
            )

        content = header + [table_header + table_divider + "\n".join(rows)]
        return "\n".join(content)

    def update_readme(self, papers: List[ICLRPaper]) -> None:
        """Write the README file with the latest papers."""
        readme_content = self.render_readme(papers)
        _ensure_directory(self.readme_path)
        with self.readme_path.open("w", encoding="utf-8") as f:
            f.write(readme_content)
        logger.info("Updated %s", self.readme_path)

    def run(self) -> List[ICLRPaper]:
        """Execute the full workflow: fetch, cache, and render README."""
        papers = self.fetch_papers()
        self.save_cache(papers)
        self.update_readme(papers)
        return papers


def main() -> None:
    """CLI entrypoint for manual execution."""
    bot = ICLRBot()
    try:
        papers = bot.run()
        logger.info("ICLR Bot completed successfully with %d papers", len(papers))
    except Exception as exc:
        logger.error("ICLR Bot failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

