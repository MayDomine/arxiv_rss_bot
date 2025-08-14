# 🖥️ Server Deployment Guide

Complete guide for hosting your arXiv RSS service on your own server.

## 🚀 Quick Start

### Prerequisites
- Linux server (Ubuntu 20.04+ recommended)
- Python 3.8+
- Git
- Docker (optional, for containerized deployment)

### Option 1: Docker Deployment (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/arxiv_rss_bot.git
cd arxiv_rss_bot

# 2. Create deployment files
python deploy_rss_service.py
# Choose option 2 (Docker)

# 3. Build and run
docker build -t arxiv-rss-service .
docker run -d -p 5000:5000 --name arxiv-rss arxiv-rss-service

# 4. Check if it's running
docker ps
curl http://localhost:5000/health
```

### Option 2: Direct Python Deployment

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/arxiv_rss_bot.git
cd arxiv_rss_bot
pip3 install -r requirements.txt

# 2. Test locally
python3 rss_service.py

# 3. Set up as system service
sudo nano /etc/systemd/system/arxiv-rss.service
```

## 🔧 Systemd Service Setup

Create `/etc/systemd/system/arxiv-rss.service`:

```ini
[Unit]
Description=arXiv RSS Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/arxiv_rss_bot
ExecStart=/usr/bin/python3 rss_service.py
Restart=always
RestartSec=10
Environment=FLASK_ENV=production
Environment=PORT=5000

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable arxiv-rss
sudo systemctl start arxiv-rss
sudo systemctl status arxiv-rss
```

## 🌐 Nginx Configuration

### Basic Nginx Setup

```bash
# Install nginx
sudo apt update
sudo apt install nginx

# Create site configuration
sudo nano /etc/nginx/sites-available/arxiv-rss
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Cache RSS feeds for 1 hour
    location ~* \.(xml|rss)$ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        expires 1h;
        add_header Cache-Control "public, max-age=3600";
        add_header Content-Type "application/rss+xml; charset=utf-8";
    }
}
```

### HTTPS Configuration (Recommended)

For production use, you should enable HTTPS:

```bash
# Install certbot for SSL certificates
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

Or manually configure HTTPS:

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
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/arxiv-rss /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 🔒 Security Setup

### Firewall Configuration

```bash
# Install ufw
sudo apt install ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### Rate Limiting

Add to your Nginx configuration:

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=rss:10m rate=10r/s;

server {
    # ... existing config ...
    
    location /rss {
        limit_req zone=rss burst=20 nodelay;
        proxy_pass http://127.0.0.1:5000;
        # ... proxy headers ...
    }
}
```

## 📊 Monitoring & Logs

### Log Management

```bash
# View service logs
sudo journalctl -u arxiv-rss -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# Log rotation
sudo nano /etc/logrotate.d/arxiv-rss
```

Add log rotation configuration:

```
/var/log/arxiv-rss/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload arxiv-rss
    endscript
}
```

### Health Monitoring

Create a simple health check script:

```bash
#!/bin/bash
# /usr/local/bin/check-arxiv-rss.sh

HEALTH_URL="http://localhost:5000/health"
LOG_FILE="/var/log/arxiv-rss/health.log"

response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $response -eq 200 ]; then
    echo "$(date): Service healthy" >> $LOG_FILE
else
    echo "$(date): Service unhealthy (HTTP $response)" >> $LOG_FILE
    systemctl restart arxiv-rss
fi
```

Add to crontab:

```bash
sudo crontab -e
# Add: */5 * * * * /usr/local/bin/check-arxiv-rss.sh
```

## 🔄 Auto-Deployment

### GitHub Webhook Setup

Create a webhook endpoint:

```bash
# Install webhook handler
pip3 install flask-webhook

# Create webhook script
nano /usr/local/bin/arxiv-webhook.py
```

```python
#!/usr/bin/env python3
import subprocess
import os
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.headers.get('X-GitHub-Event') == 'push':
        # Pull latest changes
        os.chdir('/var/www/arxiv_rss_bot')
        subprocess.run(['git', 'pull'])
        
        # Restart service
        subprocess.run(['systemctl', 'restart', 'arxiv-rss'])
        
        return 'OK', 200
    return 'Ignored', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
```

### GitHub Actions Integration

Add to your `.github/workflows/arxiv_bot.yml`:

```yaml
- name: Deploy to Server
  if: github.ref == 'refs/heads/main'
  run: |
    curl -X POST https://your-domain.com/webhook \
      -H "Content-Type: application/json" \
      -H "X-GitHub-Event: push"
```

## 🐳 Docker Compose (Production)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  arxiv-rss:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - PORT=5000
    restart: unless-stopped
    volumes:
      - ./config.json:/app/config.json:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - arxiv-rss
    restart: unless-stopped
```

## 📈 Performance Optimization

### Gunicorn Setup

```bash
# Install gunicorn
pip3 install gunicorn

# Create gunicorn config
nano gunicorn.conf.py
```

```python
bind = "127.0.0.1:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

Update systemd service:

```ini
[Service]
ExecStart=/usr/local/bin/gunicorn -c gunicorn.conf.py rss_service:app
```

### Caching

Add Redis caching:

```bash
# Install Redis
sudo apt install redis-server

# Install Python Redis client
pip3 install redis flask-caching
```

Update `rss_service.py`:

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@app.route('/rss')
@cache.cached(timeout=3600)  # Cache for 1 hour
def rss_feed():
    return generate_rss_feed()
```

## 🔍 Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   sudo journalctl -u arxiv-rss -n 50
   sudo systemctl status arxiv-rss
   ```

2. **Port already in use**
   ```bash
   sudo netstat -tlnp | grep :5000
   sudo lsof -i :5000
   ```

3. **Permission issues**
   ```bash
   sudo chown -R www-data:www-data /var/www/arxiv_rss_bot
   sudo chmod -R 755 /var/www/arxiv_rss_bot
   ```

4. **Nginx errors**
   ```bash
   sudo nginx -t
   sudo tail -f /var/log/nginx/error.log
   ```

### Performance Monitoring

```bash
# Monitor resource usage
htop
iotop
nethogs

# Monitor application
curl -s http://localhost:5000/health | jq
ab -n 1000 -c 10 http://localhost:5000/rss
```

## ✅ Deployment Checklist

- [ ] Server prepared with Python 3.8+
- [ ] Repository cloned and dependencies installed
- [ ] Service configured and running
- [ ] Nginx configured and SSL certificate obtained
- [ ] Firewall configured
- [ ] Monitoring and logging set up
- [ ] Auto-deployment configured
- [ ] Performance optimized
- [ ] RSS feeds tested and working

## 🎉 Success!

Your RSS service is now running on your server! Your feeds are available at:

- `https://your-domain.com/rss`
- `https://your-domain.com/rss/cs.AI`
- `https://your-domain.com/rss/cs.LG`
- `https://your-domain.com/rss/cs.CL`
- `https://your-domain.com/rss/cs.CV`

Add these URLs to your RSS reader and enjoy your personalized arXiv papers! 