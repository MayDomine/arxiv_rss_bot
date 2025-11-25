#!/usr/bin/env python3
"""
Deployment script for arXiv RSS Service
Provides options for hosting the RSS service on different platforms.
"""

import os
import json
import subprocess
import sys
from pathlib import Path


def create_heroku_files():
    """Create Heroku deployment files."""
    print("🚀 Creating Heroku deployment files...")

    # Create Procfile
    with open("Procfile", "w") as f:
        f.write("web: python rss_service.py\n")

    # Create runtime.txt
    with open("runtime.txt", "w") as f:
        f.write("python-3.9.18\n")

    # Update requirements.txt for Heroku
    with open("requirements.txt", "a") as f:
        f.write("gunicorn==21.2.0\n")

    # Create app.json for Heroku
    app_config = {
        "name": "arxiv-rss-service",
        "description": "RSS service for filtered arXiv papers",
        "repository": "https://github.com/yourusername/arxiv_rss_bot",
        "keywords": ["arxiv", "rss", "papers", "machine-learning"],
        "env": {
            "FLASK_ENV": {"description": "Flask environment", "value": "production"}
        },
        "buildpacks": [{"url": "heroku/python"}],
    }

    with open("app.json", "w") as f:
        json.dump(app_config, f, indent=2)

    print("✅ Heroku files created!")


def create_docker_files():
    """Create Docker deployment files."""
    print("🐳 Creating Docker deployment files...")

    # Create Dockerfile
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 1999

CMD ["python", "rss_service.py"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    # Create docker-compose.yml
    compose_content = """version: '3.8'

services:
  arxiv-rss-service:
    build: .
    ports:
      - "1999:1999"
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    volumes:
      - ./config.json:/app/config.json:ro
"""

    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)

    print("✅ Docker files created!")


def create_vercel_files():
    """Create Vercel deployment files."""
    print("⚡ Creating Vercel deployment files...")

    # Create vercel.json
    vercel_config = {
        "version": 2,
        "builds": [{"src": "rss_service.py", "use": "@vercel/python"}],
        "routes": [{"src": "/(.*)", "dest": "rss_service.py"}],
    }

    with open("vercel.json", "w") as f:
        json.dump(vercel_config, f, indent=2)

    print("✅ Vercel files created!")


def create_railway_files():
    """Create Railway deployment files."""
    print("🚂 Creating Railway deployment files...")

    # Create railway.json
    railway_config = {
        "$schema": "https://railway.app/railway.schema.json",
        "build": {"builder": "NIXPACKS"},
        "deploy": {
            "startCommand": "python rss_service.py",
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10,
        },
    }

    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)

    print("✅ Railway files created!")


def update_rss_service_for_production():
    """Update RSS service for production deployment."""
    print("🔧 Updating RSS service for production...")

    # Read the current rss_service.py
    with open("rss_service.py", "r") as f:
        content = f.read()

    # Replace localhost with environment variable
    content = content.replace(
        'feed_url = f"http://localhost:1999/rss/{category}" if category else "http://localhost:1999/rss"',
        "feed_url = f\"{os.environ.get('BASE_URL', 'http://localhost:1999')}/rss/{category}\" if category else f\"{os.environ.get('BASE_URL', 'http://localhost:1999')}/rss\"",
    )

    # Update the run command for production
    content = content.replace(
        "app.run(host='0.0.0.0', port=1999, debug=True)",
        "port = int(os.environ.get('PORT', 1999))\n    app.run(host='0.0.0.0', port=port, debug=False)",
    )

    # Write back the updated content
    with open("rss_service.py", "w") as f:
        f.write(content)

    print("✅ RSS service updated for production!")


def show_deployment_instructions(platform):
    """Show deployment instructions for the selected platform."""
    print(f"\n📋 Deployment Instructions for {platform.upper()}")
    print("=" * 50)

    if platform == "heroku":
        print(
            """
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Login to Heroku: heroku login
3. Create a new Heroku app: heroku create your-app-name
4. Deploy: git push heroku main
5. Open the app: heroku open

Your RSS feed will be available at: https://your-app-name.herokuapp.com/rss
        """
        )

    elif platform == "docker":
        print(
            """
1. Build the Docker image: docker build -t arxiv-rss-service .
2. Run the container: docker run -p 1999:1999 arxiv-rss-service
3. Or use docker-compose: docker-compose up

Your RSS feed will be available at: http://localhost:1999/rss
        """
        )

    elif platform == "vercel":
        print(
            """
1. Install Vercel CLI: npm i -g vercel
2. Deploy: vercel --prod
3. Follow the prompts to connect your GitHub repository

Your RSS feed will be available at: https://your-app-name.vercel.app/rss
        """
        )

    elif platform == "railway":
        print(
            """
1. Go to https://railway.app
2. Connect your GitHub repository
3. Railway will automatically detect and deploy your app
4. Set environment variables if needed

Your RSS feed will be available at: https://your-app-name.railway.app/rss
        """
        )

    elif platform == "local":
        print(
            """
1. Install dependencies: pip install -r requirements.txt
2. Run the service: python rss_service.py
3. Open http://localhost:1999 in your browser

Your RSS feed will be available at: http://localhost:1999/rss
        """
        )


def main():
    """Main deployment script."""
    print("🚀 arXiv RSS Service Deployment")
    print("=" * 40)

    platforms = {
        "1": ("heroku", "Heroku"),
        "2": ("docker", "Docker"),
        "3": ("vercel", "Vercel"),
        "4": ("railway", "Railway"),
        "5": ("local", "Local Development"),
    }

    print("\nChoose a deployment platform:")
    for key, (platform, name) in platforms.items():
        print(f"{key}. {name}")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice not in platforms:
        print("❌ Invalid choice!")
        return

    platform, name = platforms[choice]

    print(f"\n🎯 Setting up deployment for {name}...")

    # Create deployment files
    if platform == "heroku":
        create_heroku_files()
        update_rss_service_for_production()
    elif platform == "docker":
        create_docker_files()
    elif platform == "vercel":
        create_vercel_files()
        update_rss_service_for_production()
    elif platform == "railway":
        create_railway_files()
        update_rss_service_for_production()
    elif platform == "local":
        print("✅ No additional files needed for local development!")

    # Show deployment instructions
    show_deployment_instructions(platform)

    print("\n🎉 Setup complete!")
    print("📝 Don't forget to:")
    print("   - Customize your config.json with your preferred keywords")
    print("   - Test the RSS feed in your RSS reader")
    print("   - Set up automatic updates if needed")


if __name__ == "__main__":
    main()
