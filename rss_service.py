#!/usr/bin/env python3
"""
RSS Service for arXiv Bot
Provides an RSS feed endpoint that can be subscribed to in RSS readers.
Implements a static cache that updates daily at 12:00.
"""

from flask import Flask, Response, render_template_string
from arxiv_bot import ArxivBot
from iclr_bot import ICLRBot
import json
from datetime import datetime
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
rss_cache = {}
# Last update timestamp
last_update_time = None

# ICLR cache
iclr_cache = {
    "content": None,
    "papers": [],
    "last_update": None,
}

# RSS template
RSS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
    <channel>
        <title>{{feed_title}}</title>
        <link>{{feed_link}}</link>
        <description>{{feed_description}}</description>
        <language>en-us</language>
        <lastBuildDate>{{last_build_date}}</lastBuildDate>
        <atom:link href="{{feed_url}}" rel="self" type="application/rss+xml" />
        {% for paper in papers %}
        <item>
            <title>{{paper.title}}</title>
            <link>{{paper.link}}</link>
            <guid>{{paper.link}}</guid>
            <pubDate>{{paper.pub_date_rss}}</pubDate>
            <description><![CDATA[
                <strong>Authors:</strong> {{paper.authors_str}}<br/>
                <strong>Category:</strong> {{paper.category}}<br/>
                <strong>Score:</strong> {{paper.score}}<br/>
                <strong>Type:</strong> {{paper.type}}<br/>
                <strong>Arxiv ID:</strong> {{paper.arxiv_id}}<br/>
                <strong>Abstract:</strong> {{paper.summary}}
            ]]></description>
        </item>
        {% endfor %}
    </channel>
</rss>"""

ICLR_RSS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
    <channel>
        <title>ICLR {{year}} Highlighted Papers</title>
        <link>https://openreview.net</link>
        <description>ICLR {{year}} papers aggregated from OpenReview with review statistics.</description>
        <language>en-us</language>
        <lastBuildDate>{{last_build_date}}</lastBuildDate>
        <atom:link href="{{feed_url}}" rel="self" type="application/rss+xml" />
        {% for paper in papers %}
        <item>
            <title>{{paper.title}}</title>
            <link>{{paper.link}}</link>
            <guid>{{paper.link}}</guid>
            <pubDate>{{paper.pub_date}}</pubDate>
            <description><![CDATA[
                <strong>Authors:</strong> {{paper.authors}}<br/>
                <strong>Average Rating:</strong> {{paper.average_rating}}<br/>
                <strong>Review Count:</strong> {{paper.rating_count}}<br/>
                <strong>Decision:</strong> {{paper.decision}}<br/>
                <strong>Abstract:</strong> {{paper.abstract}}
            ]]></description>
        </item>
        {% endfor %}
    </channel>
</rss>"""


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
            <p><a href="/rss/iclr" class="rss-link">📑 ICLR 2026 Papers</a> - Highlighted OpenReview papers</p>
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


def update_rss_cache():
    """Update the RSS cache with fresh data."""
    global rss_cache, last_update_time
    
    try:
        logger.info("Updating RSS cache...")
        # Initialize bot
        bot = ArxivBot()
        
        # Fetch and filter papers once
        papers = bot.fetch_arxiv_papers()
        filtered_papers = bot.filter_papers(papers)

        # If no new papers, don't update the cache
        if not filtered_papers:
            logger.info("No new papers found. RSS cache not updated.")
            return True
        
        # Base URL for feed links
        base_url = os.environ.get('BASE_URL', 'http://localhost:5000').strip()
        
        # Clear existing cache
        rss_cache = {}
        
        # Generate cache for main feed and each category
        categories = bot.config.get("categories", [])
        categories_to_cache = [None] + categories  # None represents the main feed
        
        for category in categories_to_cache:
            # Prepare category-specific data
            if category:
                category_papers = [p for p in filtered_papers if p["category"] == category]
                feed_title = f"arXiv {category} Papers"
                feed_description = f"Filtered arXiv papers from {category} category"
                feed_url = f"{base_url}/rss/{category}"
            else:
                category_papers = filtered_papers
                feed_title = "arXiv Filtered Papers"
                feed_description = "Papers filtered based on configured keywords and categories"
                feed_url = f"{base_url}/rss"
            
            # Prepare papers for RSS
            rss_papers = []
            for paper in category_papers:
                # Format authors
                authors_str = ", ".join(paper["authors"]) if paper["authors"] else "Unknown"
                
                # Format date for RSS
                try:
                    pub_date = datetime.strptime(paper["published_date"], "%Y-%m-%d")
                    pub_date_rss = pub_date.strftime("%a, %d %b %Y %H:%M:%S +0000")
                except:
                    pub_date_rss = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
                
                rss_papers.append(
                    {
                        "title": paper["title"],
                        "link": paper["link"],
                        "authors_str": authors_str,
                        "category": paper["category"],
                        "score": paper["score"],
                        "summary": paper["summary"],
                        "pub_date_rss": pub_date_rss,
                    }
                )
            
            # Sort papers by score
            rss_papers = sorted(rss_papers, key=lambda x: x['score'], reverse=True)
            
            # Generate RSS XML
            rss_content = render_template_string(
                RSS_TEMPLATE,
                feed_title=feed_title,
                feed_link="https://arxiv.org",
                feed_description=feed_description,
                feed_url=feed_url,
                last_build_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000"),
                papers=rss_papers,
            )
            
            # Store in cache
            cache_key = category if category else "main"
            rss_cache[cache_key] = {
                "content": rss_content,
                "paper_count": len(rss_papers)
            }
        
        # Update timestamp
        last_update_time = datetime.now()
        logger.info(f"RSS cache updated at {last_update_time.isoformat()}")
        logger.info(f"Cached {len(rss_cache)} feeds with {sum(feed['paper_count'] for feed in rss_cache.values())} total papers")
        
        return True
    except Exception as e:
        logger.error(f"Error updating RSS cache: {e}")
        return False


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
        if cache_key in rss_cache:
            logger.info(f"Serving {cache_key} feed from cache (last updated: {last_update_time.isoformat() if last_update_time else 'never'})")
            return Response(
                rss_cache[cache_key]["content"],
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
                
                # Format date for RSS
                try:
                    pub_date = datetime.strptime(paper["published_date"], "%Y-%m-%d")
                    pub_date_rss = pub_date.strftime("%a, %d %b %Y %H:%M:%S +0000")
                except:
                    pub_date_rss = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
                
                rss_papers.append(
                    {
                        "title": paper["title"],
                        "link": paper["link"],
                        "authors_str": authors_str,
                        "category": paper["category"],
                        "score": paper["score"],
                        "summary": paper["summary"],
                        "pub_date_rss": pub_date_rss,
                    }
                )
            
            rss_papers = sorted(rss_papers, key=lambda x: x['score'], reverse=True)
            base_url = os.environ.get('BASE_URL', 'http://localhost:5000').strip()
            
            # Generate RSS XML
            feed_url = f"{base_url}/rss/{category}" if category else f"{base_url}/rss"
            
            rss_content = render_template_string(
                RSS_TEMPLATE,
                feed_title=feed_title,
                feed_link="https://arxiv.org",
                feed_description=feed_description,
                feed_url=feed_url,
                last_build_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000"),
                papers=rss_papers,
            )
            
            logger.info(f"Generated on-demand RSS feed with {len(rss_papers)} papers")
            
            return Response(
                rss_content,
                mimetype="application/rss+xml",
                headers={"Content-Type": "application/rss+xml; charset=utf-8"},
            )
    
    except Exception as e:
        logger.error(f"Error generating RSS feed: {e}")
        return Response(
            f"Error generating RSS feed: {str(e)}", status=500, mimetype="text/plain"
        )


@app.route("/health")
def health_check():
    """Health check endpoint."""
    global last_update_time
    return Response(
        json.dumps({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "last_cache_update": last_update_time.isoformat() if last_update_time else None,
            "cache_status": "active" if rss_cache else "empty"
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
            "timestamp": datetime.now().isoformat(),
            "message": "Cache updated successfully" if success else "Failed to update cache"
        }),
        mimetype="application/json",
    )


def build_iclr_rss_payload(papers, year: int, base_url: str):
    """Prepare structured data for the ICLR RSS template."""
    rss_papers = []
    for paper in papers:
        pub_date = paper.get("submission_date")
        if pub_date:
            try:
                pub_date_dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                pub_date_rss = pub_date_dt.strftime("%a, %d %b %Y %H:%M:%S +0000")
            except ValueError:
                pub_date_rss = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")
        else:
            pub_date_rss = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")

        rss_papers.append(
            {
                "title": paper.get("title"),
                "link": paper.get("forum_link"),
                "authors": ", ".join(paper.get("authors", [])) or "Unknown",
                "average_rating": (
                    f"{paper.get('average_rating'):.2f}"
                    if isinstance(paper.get("average_rating"), (int, float))
                    else "N/A"
                ),
                "rating_count": paper.get("rating_count", 0),
                "decision": paper.get("decision") or "Pending",
                "abstract": (paper.get("abstract") or "")[:500],
                "pub_date": pub_date_rss,
            }
        )

    return {
        "feed_url": f"{base_url}/rss/iclr",
        "papers": rss_papers,
        "year": year,
    }


def update_iclr_cache():
    """Fetch ICLR papers and regenerate cache/README."""
    global iclr_cache

    try:
        logger.info("Updating ICLR cache...")
        bot = ICLRBot()
        papers = bot.run()
        cache_payload = bot.load_cached_papers()

        if not cache_payload and papers:
            cache_payload = [paper.to_dict() for paper in papers]

        if not cache_payload:
            logger.warning("ICLR cache could not be populated.")
            return False

        base_url = os.environ.get("BASE_URL", "http://localhost:5000").strip()
        year = bot.year
        rss_payload = build_iclr_rss_payload(cache_payload, year, base_url)

        iclr_cache["content"] = render_template_string(
            ICLR_RSS_TEMPLATE,
            feed_title=f"ICLR {year} Highlighted Papers",
            feed_link="https://openreview.net",
            feed_description=f"ICLR {year} papers aggregated from OpenReview with review statistics.",
            feed_url=f"{base_url}/rss/iclr",
            last_build_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000"),
            papers=rss_payload,
            year=year,
        )
        iclr_cache["papers"] = cache_payload
        iclr_cache["last_update"] = datetime.now()

        logger.info("ICLR cache updated with %d papers", len(cache_payload))
        return True
    except Exception as exc:
        logger.error("Failed to update ICLR cache: %s", exc)
        return False


@app.route("/rss/iclr")
def iclr_rss_feed():
    """Serve the ICLR RSS feed."""
    global iclr_cache

    if not iclr_cache.get("content"):
        logger.info("ICLR cache empty, triggering update.")
        update_iclr_cache()

    if not iclr_cache.get("content"):
        return Response(
            json.dumps(
                {
                    "status": "error",
                    "message": "ICLR cache is empty. Trigger /update-iclr-cache first.",
                }
            ),
            status=503,
            mimetype="application/json",
        )

    return Response(
        iclr_cache["content"],
        mimetype="application/rss+xml",
        headers={"Content-Type": "application/rss+xml; charset=utf-8"},
    )


@app.route("/update-iclr-cache")
def manual_update_iclr_cache():
    """Manually trigger the ICLR cache and README refresh."""
    success = update_iclr_cache()
    status = "success" if success else "error"
    return Response(
        json.dumps(
            {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "message": "ICLR cache updated successfully"
                if success
                else "Failed to update ICLR cache",
                "last_update": iclr_cache.get("last_update").isoformat()
                if iclr_cache.get("last_update")
                else None,
            }
        ),
        mimetype="application/json",
        status=200 if success else 500,
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
    app.run(host="0.0.0.0", port=1999, debug=True)
