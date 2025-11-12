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

import requests  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        session: Optional[requests.Session] = None,
    ) -> None:
        self.year = year
        self.output_dir = output_dir or Path("iclr")
        self.max_papers = max_papers
        self.session = session or requests.Session()

        self.cache_path = self.output_dir / "iclr_cache.json"
        self.readme_path = self.output_dir / "README.md"

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
        offset = 0
        page_size = 1000  # Use larger page size to minimize requests

        try:
            url = f"{OPENREVIEW_API}/notes"

            while True:
                params = {
                    "content.venueid": f"{domain}/Submission",
                    "domain": domain,
                    "details": "replyCount,presentation,writable",
                    "offset": offset,
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

                offset += page_size

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
        logger.info("Fetching ICLR %s submissions from OpenReview...", self.year)
        submissions = self._fetch_notes(self.conference_domain)

        logger.info("Fetched %d submissions", len(submissions))
        return submissions

    def _fetch_ratings(self, paper_number: int, forum_id: str) -> Dict[str, Any]:
        """Fetch review ratings for a specific paper."""
        # For now, return empty ratings since reviews aren't available yet in ICLR 2026
        # When reviews become available, we can search for notes with the paper's forum
        # that contain review content
        logger.debug("Reviews not yet available for ICLR %s", self.year)
        return {"ratings": [], "average": None}

        # Future implementation when reviews are available:
        # try:
        #     # Search for all notes in the conference domain with this forum
        #     all_notes = self._fetch_notes(self.conference_domain, forum=forum_id)
        #     reviews = [note for note in all_notes if "review" in (note.get("invitation", "").lower())]
        # except requests.HTTPError as exc:
        #     logger.warning(
        #         "Failed to fetch reviews for paper %s (%s): %s",
        #         paper_number,
        #         forum_id,
        #         exc,
        #     )
        #     reviews = []

        # ratings: List[float] = []
        # for review in reviews:
        #     rating_value = _extract_value(review.get("content", {}).get("rating"))
        #     parsed = _extract_numeric_rating(rating_value)
        #     if parsed is not None:
        #         ratings.append(parsed)

        # if not ratings:
        #     return {"ratings": [], "average": None}

        # return {
        #     "ratings": ratings,
        #     "average": sum(ratings) / len(ratings),
        # }

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

    def _build_paper(self, submission: Dict[str, Any]) -> ICLRPaper:
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

        ratings_info = self._fetch_ratings(number, forum_id)
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

        for submission in submissions:
            try:
                paper = self._build_paper(submission)
                papers.append(paper)
            except Exception as exc:
                forum_id = submission.get("forum") or submission.get("id")
                logger.error("Failed to process submission %s: %s", forum_id, exc)

        logger.info("Processed %d papers with review data", len(papers))
        return papers

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
        total_papers = len(papers)

        # Limit display to top 100 papers for readability
        display_limit = 100
        display_papers = papers[:display_limit] if len(papers) > display_limit else papers

        header = [
            "# ICLR 2026 Papers 📚",
            "",
            f"- **Last Updated**: {timestamp}",
            f"- **Total Papers**: {total_papers}",
            "",
            "Click any title to view the full discussion on OpenReview.",
            "",
        ]

        if not display_papers:
            return "\n".join(
                header
                + [
                    "> No papers were retrieved from OpenReview. "
                    "Try updating again later via `/update-iclr-cache`."
                ]
            )

        if total_papers > display_limit:
            header.append(f"> Showing top {display_limit} papers out of {total_papers} total submissions.")
            header.append("")

        table_header = "| # | Title | Avg Rating | Reviews | Decision | OpenReview |\n"
        table_divider = "| --- | --- | --- | --- | --- | --- |\n"

        rows: List[str] = []
        for idx, paper in enumerate(sorted(display_papers, key=lambda p: (p.average_rating or 0), reverse=True), 1):
            avg_rating = (
                f"{paper.average_rating:.2f}" if paper.average_rating is not None else "N/A"
            )
            reviews = str(paper.rating_count) if paper.rating_count else "0"
            decision = paper.decision or "Pending"
            title = paper.title.replace("\n", " ").strip()

            rows.append(
                f"| {idx} | [{title}]({paper.forum_link}) | {avg_rating} | {reviews} | {decision} | [Link]({paper.forum_link}) |"
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

