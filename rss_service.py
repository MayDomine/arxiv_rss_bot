#!/usr/bin/env python3
"""
RSS Service for arXiv Bot
Provides an RSS feed endpoint that can be subscribed to in RSS readers.
Implements a static cache that updates daily at 12:00.
"""

from flask import Flask, Response
from arxiv_bot import ArxivBot
from feedgen.feed import FeedGenerator
import json
from datetime import datetime, timedelta, timezone
import logging
import os
import time
import threading
import schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global cache for RSS feeds
# Structure: {category: {"papers": [paper_dict, ...], "last_update": datetime}}
rss_cache = {}
# Last update timestamp
last_update_time = None
# Maximum number of papers to keep in cache per feed (0 = unlimited)
MAX_PAPERS_PER_FEED = 100
# Maximum days to keep papers in cache (0 = unlimited)
MAX_DAYS_TO_KEEP = 30



@app.route("/")
def index():
    """Home page with information about the RSS service."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>arXiv RSS Service</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .rss-link { background: #ff6600; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
            .info { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🤖 arXiv RSS Service</h1>
        <p>This service provides RSS feeds for filtered arXiv papers based on your configuration.</p>
        
        <div class="info">
            <h3>📡 Available RSS Feeds</h3>
            <p><a href="/rss" class="rss-link">📊 Main Feed</a> - All filtered papers</p>
            <p><a href="/rss/cs.AI" class="rss-link">🤖 AI Papers</a> - Computer Science AI papers</p>
            <p><a href="/rss/cs.LG" class="rss-link">🧠 ML Papers</a> - Machine Learning papers</p>
            <p><a href="/rss/cs.CL" class="rss-link">💬 NLP Papers</a> - Natural Language Processing papers</p>
            <p><a href="/rss/cs.CV" class="rss-link">👁️ CV Papers</a> - Computer Vision papers</p>
        </div>
        
        <div class="info">
            <h3>🔧 How to Subscribe</h3>
            <p>Copy any of the RSS feed URLs above and add them to your RSS reader:</p>
            <ul>
                <li><strong>Feedly:</strong> Click the "+" button and paste the RSS URL</li>
                <li><strong>Inoreader:</strong> Go to "Subscriptions" → "Add New" → paste the URL</li>
                <li><strong>RSS Reader apps:</strong> Use the "Add Feed" option with the URL</li>
            </ul>
        </div>
        
        <div class="info">
            <h3>⚙️ Configuration</h3>
            <p>The feeds are filtered based on your <code>config.json</code> settings:</p>
            <ul>
                <li>Keywords: <code id="keywords"></code></li>
                <li>Max papers: <code id="max_papers"></code></li>
                <li>Days back: <code id="days_back"></code></li>
            </ul>
        </div>
        
        <script>
            // Load and display config
            fetch('/config')
                .then(response => response.json())
                .then(config => {
                    document.getElementById('keywords').textContent = config.keywords.join(', ');
                    document.getElementById('max_papers').textContent = config.max_papers;
                    document.getElementById('days_back').textContent = config.days_back;
                });
        </script>
    </body>
    </html>
    """


@app.route("/rss")
def rss_feed():
    """Main RSS feed endpoint."""
    return generate_rss_feed()


@app.route("/rss/<category>")
def category_rss_feed(category):
    """Category-specific RSS feed endpoint."""
    return generate_rss_feed(category=category)


@app.route("/config")
def get_config():
    """Return the current configuration as JSON."""
    try:
        bot = ArxivBot()
        return Response(json.dumps(bot.config, indent=2), mimetype="application/json")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return Response(
            json.dumps({"error": "Failed to load configuration"}),
            status=500,
            mimetype="application/json",
        )


def _ensure_timezone(dt):
    """Ensure a datetime object has timezone info. Returns UTC datetime if no timezone."""
    if not isinstance(dt, datetime):
        # If it's a string, try to parse it
        if isinstance(dt, str):
            try:
                dt = datetime.strptime(dt, "%Y-%m-%d")
            except:
                return datetime.now(timezone.utc)
        else:
            return datetime.now(timezone.utc)
    
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _cleanup_old_papers(papers):
    """Remove papers older than MAX_DAYS_TO_KEEP days."""
    if MAX_DAYS_TO_KEEP <= 0:
        return papers
    
    # Ensure all papers have timezone info (fix for old cached data)
    for paper in papers:
        paper['pub_date'] = _ensure_timezone(paper.get('pub_date'))
    
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=MAX_DAYS_TO_KEEP)
    return [p for p in papers if p['pub_date'] >= cutoff_date]


def _limit_papers(papers):
    """Limit the number of papers to MAX_PAPERS_PER_FEED."""
    if MAX_PAPERS_PER_FEED <= 0:
        return papers
    return papers[:MAX_PAPERS_PER_FEED]


def _merge_papers(existing_papers, new_papers):
    """Merge new papers with existing ones, removing duplicates by link."""
    # Ensure all existing papers have timezone info (fix for old cached data)
    for paper in existing_papers:
        paper['pub_date'] = _ensure_timezone(paper.get('pub_date'))
    
    # Create a dict with link as key for fast lookup
    papers_dict = {p['link']: p for p in existing_papers}
    
    # Add new papers (will overwrite if duplicate)
    for paper in new_papers:
        papers_dict[paper['link']] = paper
    
    # Convert back to list and sort by publication date (newest first)
    merged = list(papers_dict.values())
    merged.sort(key=lambda x: x['pub_date'], reverse=True)
    
    return merged


def update_rss_cache():
    """Update the RSS cache with fresh data, preserving historical papers."""
    global rss_cache, last_update_time
    
    try:
        logger.info("Updating RSS cache...")
        # Initialize bot
        bot = ArxivBot()
        
        # Fetch and filter papers once
        papers = bot.fetch_arxiv_papers()
        filtered_papers = bot.filter_papers(papers)

        # Base URL for feed links
        base_url = os.environ.get('BASE_URL', 'http://localhost:1999').strip()
        
        # Generate cache for main feed and each category
        categories = bot.config.get("categories", [])
        categories_to_cache = [None] + categories  # None represents the main feed
        
        for category in categories_to_cache:
            # Prepare category-specific data
            if category:
                category_papers = [p for p in filtered_papers if p["category"] == category]
            else:
                category_papers = filtered_papers
            
            # Prepare new papers for RSS
            new_rss_papers = []
            for paper in category_papers:
                # Format authors
                authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
                
                # Format date for RSS (with timezone info)
                try:
                    pub_date = datetime.strptime(paper["published_date"], "%Y-%m-%d")
                    pub_date = _ensure_timezone(pub_date)
                except:
                    pub_date = datetime.now(timezone.utc)
                
                new_rss_papers.append(
                    {
                        "title": paper["title"],
                        "link": paper["link"],
                        "authors_str": authors_str,
                        "category": paper["category"],
                        "score": paper["score"],
                        "summary": paper["summary"],
                        "pub_date": pub_date,
                    }
                )
            
            # Get existing papers from cache
            cache_key = category if category else "main"
            existing_papers = []
            if cache_key in rss_cache and "papers" in rss_cache[cache_key]:
                existing_papers = rss_cache[cache_key]["papers"]
            
            # Merge with existing papers (removes duplicates by link)
            merged_papers = _merge_papers(existing_papers, new_rss_papers)
            
            # Clean up old papers
            merged_papers = _cleanup_old_papers(merged_papers)
            
            # Limit number of papers
            merged_papers = _limit_papers(merged_papers)
            
            # Store papers in cache (we'll generate RSS XML on-demand)
            rss_cache[cache_key] = {
                "papers": merged_papers,
                "last_update": datetime.now(timezone.utc)
            }
        
        # Update timestamp
        last_update_time = datetime.now(timezone.utc)
        new_count = len(filtered_papers) if filtered_papers else 0
        total_count = sum(len(feed['papers']) for feed in rss_cache.values())
        logger.info(f"RSS cache updated at {last_update_time.isoformat()}")
        logger.info(f"Added {new_count} new papers. Total cached: {total_count} papers across {len(rss_cache)} feeds")
        
        return True
    except Exception as e:
        logger.error(f"Error updating RSS cache: {e}")
        return False


def _generate_rss_xml(papers, feed_title, feed_description, feed_url):
    """Generate RSS XML from paper data using feedgen.
    
    Generates a valid RSS 2.0 compliant feed with all required and recommended fields.
    """
    fg = FeedGenerator()
    
    # Required RSS 2.0 channel elements
    fg.title(feed_title)
    fg.link(href=feed_url, rel='self')
    fg.link(href="https://arxiv.org", rel='alternate')
    fg.description(feed_description)
    
    # Recommended RSS 2.0 channel elements
    fg.language('en-us')
    fg.lastBuildDate(datetime.now(timezone.utc))
    fg.generator('arXiv RSS Bot v1.0')
    # TTL (Time To Live) in minutes - suggests cache refresh interval (24 hours = 1440 minutes)
    fg.ttl(1440)
    
    # Add items (papers)
    for idx, paper in enumerate(papers):
        try:
            fe = fg.add_entry()
            
            # Required RSS 2.0 item elements
            fe.title(paper['title'])
            fe.link(href=paper['link'])
            
            # Recommended RSS 2.0 item elements
            fe.guid(paper['link'], permalink=True)
            
            # Ensure pub_date has timezone info (fix for cached data without timezone)
            pub_date_raw = paper.get('pub_date')
            logger.debug(f"Paper {idx}: raw pub_date type={type(pub_date_raw)}, value={pub_date_raw}")
            pub_date = _ensure_timezone(pub_date_raw)
            logger.debug(f"Paper {idx}: processed pub_date type={type(pub_date)}, value={pub_date}, tzinfo={pub_date.tzinfo}")
            fe.pubDate(pub_date)
        except Exception as e:
            logger.error(f"Error processing paper {idx} (title: {paper.get('title', 'Unknown')[:50]}): {e}")
            logger.error(f"Paper data: {paper}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Build description with paper details (required for item)
        description = f"<strong>Authors:</strong> {paper['authors_str']}<br/>"
        description += f"<strong>Category:</strong> {paper['category']}<br/>"
        description += f"<strong>Score:</strong> {paper['score']}<br/>"
        if 'type' in paper:
            description += f"<strong>Type:</strong> {paper['type']}<br/>"
        if 'arxiv_id' in paper:
            description += f"<strong>Arxiv ID:</strong> {paper['arxiv_id']}<br/>"
        description += f"<strong>Abstract:</strong> {paper['summary']}"
        fe.description(description)
    
    return fg.rss_str(pretty=True).decode('utf-8')


def generate_rss_feed(category=None):
    """Generate RSS feed from cached data or update if needed."""
    global rss_cache, last_update_time
    
    try:
        # Determine cache key
        cache_key = category if category else "main"
        
        # Check if we need to update the cache (first request or cache is empty)
        if last_update_time is None or not rss_cache:
            logger.info("Initial cache update required")
            update_rss_cache()
        
        # Get from cache
        if cache_key in rss_cache and "papers" in rss_cache[cache_key]:
            papers = rss_cache[cache_key]["papers"]
            base_url = os.environ.get('BASE_URL', 'http://localhost:1999').strip()
            
            # Prepare feed metadata
            if category:
                feed_title = f"arXiv {category} Papers"
                feed_description = f"Filtered arXiv papers from {category} category"
                feed_url = f"{base_url}/rss/{category}"
            else:
                feed_title = "arXiv Filtered Papers"
                feed_description = "Papers filtered based on configured keywords and categories"
                feed_url = f"{base_url}/rss"
            
            # Generate RSS XML from cached papers
            rss_content = _generate_rss_xml(papers, feed_title, feed_description, feed_url)
            
            logger.info(f"Serving {cache_key} feed from cache with {len(papers)} papers (last updated: {rss_cache[cache_key].get('last_update', 'unknown')})")
            return Response(
                rss_content,
                mimetype="application/rss+xml",
                headers={"Content-Type": "application/rss+xml; charset=utf-8"},
            )
        else:
            logger.warning(f"Cache miss for {cache_key}, generating on-demand")
            # If category not in cache, generate it on-demand (fallback)
            bot = ArxivBot()
            
            if category:
                # Filter to specific category
                bot.config["categories"] = [category]
                feed_title = f"arXiv {category} Papers"
                feed_description = f"Filtered arXiv papers from {category} category"
            else:
                feed_title = "arXiv Filtered Papers"
                feed_description = "Papers filtered based on configured keywords and categories"
            
            # Fetch and filter papers
            papers = bot.fetch_arxiv_papers()
            filtered_papers = bot.filter_papers(papers)
            
            # Prepare papers for RSS
            rss_papers = []
            for paper in filtered_papers:
                # Format authors
                authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
                
                # Format date for RSS (with timezone info)
                try:
                    pub_date = datetime.strptime(paper["published_date"], "%Y-%m-%d")
                    pub_date = _ensure_timezone(pub_date)
                except:
                    pub_date = datetime.now(timezone.utc)
                
                rss_papers.append(
                    {
                        "title": paper["title"],
                        "link": paper["link"],
                        "authors_str": authors_str,
                        "category": paper["category"],
                        "score": paper["score"],
                        "summary": paper["summary"],
                        "pub_date": pub_date,
                    }
                )
            
            # Sort by publication date (newest first)
            rss_papers = sorted(rss_papers, key=lambda x: x['pub_date'], reverse=True)
            base_url = os.environ.get('BASE_URL', 'http://localhost:1999').strip()
            
            # Generate RSS XML using feedgen
            feed_url = f"{base_url}/rss/{category}" if category else f"{base_url}/rss"
            
            rss_content = _generate_rss_xml(rss_papers, feed_title, feed_description, feed_url)
            
            logger.info(f"Generated on-demand RSS feed with {len(rss_papers)} papers")
            
            return Response(
                rss_content,
                mimetype="application/rss+xml",
                headers={"Content-Type": "application/rss+xml; charset=utf-8"},
            )
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating RSS feed: {e}"
        logger.error(error_msg)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return Response(
            f"Error generating RSS feed: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}", 
            status=500, 
            mimetype="text/plain"
        )


@app.route("/health")
def health_check():
    """Health check endpoint."""
    global last_update_time, rss_cache
    total_papers = sum(len(feed.get("papers", [])) for feed in rss_cache.values())
    return Response(
        json.dumps({
            "status": "healthy", 
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_cache_update": last_update_time.isoformat() if last_update_time else None,
            "cache_status": "active" if rss_cache else "empty",
            "total_feeds": len(rss_cache),
            "total_papers_cached": total_papers,
            "max_papers_per_feed": MAX_PAPERS_PER_FEED,
            "max_days_to_keep": MAX_DAYS_TO_KEEP
        }),
        mimetype="application/json",
    )


@app.route("/update-cache")
def manual_update_cache():
    """Manually trigger a cache update."""
    success = update_rss_cache()
    return Response(
        json.dumps({
            "status": "success" if success else "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Cache updated successfully" if success else "Failed to update cache"
        }),
        mimetype="application/json",
    )


def schedule_cache_updates():
    """Schedule daily cache updates at 12:00."""
    def run_scheduler():
        # Initial update on startup
        update_rss_cache()
        
        # Schedule daily update at 12:00
        schedule.every().day.at("12:00").do(update_rss_cache)
        
        logger.info("Scheduled daily cache updates at 12:00")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Run scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Cache update scheduler started")


if __name__ == "__main__":
    logger.info("Starting arXiv RSS Service...")
    # Start the scheduler
    schedule_cache_updates()
    # Run the Flask app
    app.run(host="0.0.0.0", port=1999)
