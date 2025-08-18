#!/usr/bin/env python3
"""
RSS Service for arXiv Bot
Provides an RSS feed endpoint that can be subscribed to in RSS readers.
"""

from flask import Flask, Response, render_template_string
from arxiv_bot import ArxivBot
import json
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
                <strong>Abstract:</strong> {{paper.summary}}
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


def generate_rss_feed(category=None):
    """Generate RSS feed from filtered papers."""
    try:
        # Initialize bot and fetch papers
        bot = ArxivBot()

        if category:
            # Filter to specific category
            bot.config["categories"] = [category]
            feed_title = f"arXiv {category} Papers"
            feed_description = f"Filtered arXiv papers from {category} category"
        else:
            feed_title = "arXiv Filtered Papers"
            feed_description = (
                "Papers filtered based on configured keywords and categories"
            )

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
        rss_papers = sorted(rss_papers, key = lambda: x: x['score'], reverse=True)
        # Generate RSS XML
        feed_url = (
            f"{os.environ.get('BASE_URL', 'http://localhost:5000')}/rss/{category}"
            if category
            else f"{os.environ.get('BASE_URL', 'http://localhost:5000')}/rss"
        )

        rss_content = render_template_string(
            RSS_TEMPLATE,
            feed_title=feed_title,
            feed_link="https://arxiv.org",
            feed_description=feed_description,
            feed_url=feed_url,
            last_build_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000"),
            papers=rss_papers,
        )

        logger.info(f"Generated RSS feed with {len(rss_papers)} papers")

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
    return Response(
        json.dumps({"status": "healthy", "timestamp": datetime.now().isoformat()}),
        mimetype="application/json",
    )


if __name__ == "__main__":
    logger.info("Starting arXiv RSS Service...")
    app.run(host="0.0.0.0", port=1999, debug=True)
