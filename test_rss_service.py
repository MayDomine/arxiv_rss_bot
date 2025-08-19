#!/usr/bin/env python3
"""
Test script for RSS Service
"""

import requests
import xml.etree.ElementTree as ET
import json


def test_rss_feed(url):
    """Test an RSS feed endpoint."""
    print(f"🔍 Testing RSS feed: {url}")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Content-Type: {response.headers.get('content-type', 'Unknown')}")

            # Parse RSS XML
            try:
                root = ET.fromstring(response.content)

                # Get channel info
                channel = root.find("channel")
                if channel is not None:
                    title = channel.find("title")
                    description = channel.find("description")

                    print(f"📰 Title: {title.text if title is not None else 'N/A'}")
                    print(
                        f"📝 Description: {description.text if description is not None else 'N/A'}"
                    )

                # Count items
                items = root.findall(".//item")
                print(f"📊 Number of papers: {len(items)}")

                # Show first few items
                for i, item in enumerate(items[:3]):
                    item_title = item.find("title")
                    item_link = item.find("link")
                    print(
                        f"  {i+1}. {item_title.text if item_title is not None else 'N/A'}"
                    )
                    print(
                        f"     Link: {item_link.text if item_link is not None else 'N/A'}"
                    )

                return True

            except ET.ParseError as e:
                print(f"❌ Error parsing RSS XML: {e}")
                return False

        else:
            print(f"❌ Status: {response.status_code}")
            print(f"❌ Response: {response.text[:200]}...")
            return False

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_config_endpoint(base_url):
    """Test the config endpoint."""
    print(f"\n🔧 Testing config endpoint: {base_url}/config")

    try:
        response = requests.get(f"{base_url}/config", timeout=10)

        if response.status_code == 200:
            config = response.json()
            print("✅ Config loaded successfully")
            print(f"📋 Categories: {config.get('categories', [])}")
            print(f"🔑 Keywords: {config.get('keywords', [])}")
            print(f"📊 Max papers: {config.get('max_papers', 'N/A')}")
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            return False

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print(f"\n🏥 Testing health endpoint: {base_url}/health")

    try:
        response = requests.get(f"{base_url}/health", timeout=10)

        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed: {health.get('status', 'Unknown')}")
            print(f"📅 Last cache update: {health.get('last_cache_update', 'Never')}")
            print(f"💾 Cache status: {health.get('cache_status', 'Unknown')}")
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            return False

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_cache_update_endpoint(base_url):
    """Test the cache update endpoint."""
    print(f"\n🔄 Testing cache update endpoint: {base_url}/update-cache")

    try:
        response = requests.get(f"{base_url}/update-cache", timeout=30)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cache update status: {result.get('status', 'Unknown')}")
            print(f"📝 Message: {result.get('message', 'No message')}")
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            return False

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing arXiv RSS Service")
    print("=" * 40)

    base_url = "http://localhost:1999"

    # Test health endpoint
    test_health_endpoint(base_url)
    
    # Test cache update endpoint
    test_cache_update_endpoint(base_url)

    # Test config endpoint
    test_config_endpoint(base_url)

    # Test main RSS feed
    print(f"\n📡 Testing RSS feeds...")
    test_rss_feed(f"{base_url}/rss")

    # Test category-specific feeds
    categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]
    for category in categories:
        test_rss_feed(f"{base_url}/rss/{category}")

    print(f"\n🎉 RSS service testing completed!")
    print(f"📝 Your RSS feeds are ready to use in your RSS reader!")
    print(f"🔗 Main feed: {base_url}/rss")


if __name__ == "__main__":
    main()
