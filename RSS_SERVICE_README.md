# 📡 arXiv RSS Service

Transform your arXiv bot into a web service that provides RSS feeds for your filtered papers. This allows you to subscribe to your personalized arXiv papers in any RSS reader.

## ✨ Features

- **RSS Feed Generation**: Convert filtered papers into RSS feeds
- **Multiple Feed Types**: Main feed and category-specific feeds
- **Web Interface**: Beautiful web page showing available feeds
- **Easy Deployment**: Support for multiple hosting platforms
- **Static Cache**: Feeds are cached and updated daily at 12:00
- **Manual Cache Updates**: Ability to manually trigger cache updates

## 🚀 Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the RSS service
python rss_service.py

# Open in browser
open http://localhost:1999
```

### 2. Test the Service

```bash
# Test all endpoints
python test_rss_service.py
```

## 📡 Available RSS Feeds

Once running, you'll have access to these RSS feeds:

- **Main Feed**: `http://localhost:1999/rss` - All filtered papers
- **AI Papers**: `http://localhost:1999/rss/cs.AI` - Computer Science AI papers
- **ML Papers**: `http://localhost:1999/rss/cs.LG` - Machine Learning papers
- **NLP Papers**: `http://localhost:1999/rss/cs.CL` - Natural Language Processing papers
- **CV Papers**: `http://localhost:1999/rss/cs.CV` - Computer Vision papers
- **ICLR 2026 Papers**: `http://localhost:1999/rss/iclr` - Highlighted ICLR submissions from OpenReview

## 🔄 Cache System

The RSS service now uses a static cache system that updates daily:

- **Automatic Updates**: Cache refreshes automatically at 12:00 every day
- **Manual Updates**: Trigger a cache update by visiting `http://localhost:1999/update-cache`
- **ICLR Refresh**: Pull the latest ICLR data via `http://localhost:1999/update-iclr-cache`
- **Cache Status**: Check cache status at `http://localhost:1999/health`

This improves performance and reduces load on the arXiv API.

## 🔧 How to Subscribe

### Popular RSS Readers

1. **Feedly**
   - Go to [feedly.com](https://feedly.com)
   - Click the "+" button
   - Paste your RSS feed URL
   - Click "Follow"

2. **Inoreader**
   - Go to [inoreader.com](https://inoreader.com)
   - Click "Subscriptions" → "Add New"
   - Paste your RSS feed URL
   - Click "Add"

3. **RSS Reader Apps**
   - **iOS**: Reeder, Feedly, Newsify
   - **Android**: Feedly, Inoreader, FeedMe
   - **Desktop**: Feedly, Inoreader, Thunderbird

### Example RSS Feed URLs

```
# Main feed (all categories)
https://your-app.herokuapp.com/rss

# Category-specific feeds
https://your-app.herokuapp.com/rss/cs.AI
https://your-app.herokuapp.com/rss/cs.LG
https://your-app.herokuapp.com/rss/cs.CL
```

## 🌐 Deployment Options

### Option 1: Heroku (Recommended for beginners)

```bash
# Run the deployment script
python deploy_rss_service.py

# Choose option 1 (Heroku)
# Follow the instructions provided
```

### Option 2: Railway (Free tier available)

```bash
# Run the deployment script
python deploy_rss_service.py

# Choose option 4 (Railway)
# Connect your GitHub repository
```

### Option 3: Vercel (Fast deployment)

```bash
# Run the deployment script
python deploy_rss_service.py

# Choose option 3 (Vercel)
# Deploy with Vercel CLI
```

### Option 4: Docker

```bash
# Run the deployment script
python deploy_rss_service.py

# Choose option 2 (Docker)
# Build and run the container
docker-compose up
```

## 📊 RSS Feed Structure

Each RSS feed contains:

```xml
<rss version="2.0">
  <channel>
    <title>arXiv Filtered Papers</title>
    <description>Papers filtered based on configured keywords</description>
    <item>
      <title>Paper Title</title>
      <link>https://arxiv.org/abs/paper-id</link>
      <description>
        <strong>Authors:</strong> Author 1, Author 2<br/>
        <strong>Category:</strong> cs.AI<br/>
        <strong>Score:</strong> 3.5<br/>
        <strong>Abstract:</strong> Paper abstract...
      </description>
    </item>
  </channel>
</rss>
```

## 🔧 Configuration

The RSS service uses the same `config.json` as your arXiv bot:

```json
{
  "categories": ["cs.AI", "cs.LG", "cs.CL"],
  "keywords": ["machine learning", "deep learning"],
  "max_papers": 30,
  "days_back": 7
}
```

## 🛠️ API Endpoints

- `GET /` - Web interface with feed links
- `GET /rss` - Main RSS feed
- `GET /rss/{category}` - Category-specific RSS feed
- `GET /rss/iclr` - ICLR 2026 feed with OpenReview ratings
- `GET /config` - Current configuration (JSON)
- `GET /health` - Health check endpoint
- `GET /update-iclr-cache` - Manually refresh the cached ICLR data and README

## 🔄 Integration with GitHub Actions

You can integrate the RSS service with your existing GitHub Actions workflow:

```yaml
# Add to your .github/workflows/arxiv_bot.yml
- name: Deploy RSS Service
  run: |
    # Your deployment commands here
    # This will update your RSS feeds when new papers are found
```

## 📱 Mobile RSS Readers

### iOS Apps
- **Reeder** - Premium RSS reader with excellent UI
- **Feedly** - Popular with good free tier
- **Newsify** - Clean interface with offline reading

### Android Apps
- **Feedly** - Cross-platform consistency
- **Inoreader** - Feature-rich with good free tier
- **FeedMe** - Highly customizable

## 🔍 Troubleshooting

### Common Issues

1. **Feed not updating**
   - Check if your hosting service is running
   - Verify the RSS feed URL is correct
   - Check the service logs for errors

2. **No papers in feed**
   - Verify your `config.json` has the right keywords
   - Check if papers were published in the last `days_back` days
   - Test the main arXiv bot first

3. **RSS reader can't parse feed**
   - Check the feed URL returns valid XML
   - Verify the Content-Type header is `application/rss+xml`
   - Test with a different RSS reader

### Testing Your Feed

```bash
# Test locally
python test_rss_service.py

# Test with curl
curl -H "Accept: application/rss+xml" http://localhost:1999/rss

# Validate RSS format
curl http://localhost:1999/rss | xmllint --format -
```

## 🎯 Advanced Usage

### Custom RSS Templates

You can modify the RSS template in `rss_service.py`:

```python
RSS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{{feed_title}}</title>
    <!-- Your custom template here -->
  </channel>
</rss>"""
```

### Adding New Categories

1. Add the category to your `config.json`
2. The RSS service will automatically create a feed at `/rss/{category}`
3. Update the web interface in `rss_service.py` if needed

### Caching

For better performance, you can add caching:

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/rss')
@cache.cached(timeout=3600)  # Cache for 1 hour
def rss_feed():
    return generate_rss_feed()
```

## 📈 Monitoring

### Health Checks

```bash
# Check service health
curl http://your-app.herokuapp.com/health

# Expected response
{"status": "healthy", "timestamp": "2024-01-15T12:00:00"}
```

### Logs

Monitor your service logs:

```bash
# Heroku
heroku logs --tail

# Railway
railway logs

# Docker
docker logs arxiv-rss-service
```

## 🎉 Success!

Once deployed, you'll have:

1. ✅ A web service providing RSS feeds
2. ✅ Personalized arXiv papers in your RSS reader
3. ✅ Automatic updates when new papers are published
4. ✅ Category-specific feeds for focused reading

Your RSS reader will now show you the latest relevant papers from arXiv, filtered according to your preferences!

---

*Happy reading! 📚*