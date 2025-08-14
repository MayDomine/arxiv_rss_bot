#!/usr/bin/env python3
"""
Test script to verify workflow configuration
"""

import os
import json
import yaml

def test_config_files():
    """Test that all configuration files are valid."""
    print("🔍 Testing configuration files...")
    
    # Test config.json
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        print("✅ config.json is valid JSON")
        
        required_keys = ["categories", "keywords", "max_papers", "days_back"]
        for key in required_keys:
            if key in config:
                print(f"✅ config.json contains {key}")
            else:
                print(f"❌ config.json missing {key}")
                
    except Exception as e:
        print(f"❌ config.json error: {e}")
    
    # Test workflow files
    workflow_files = [
        ".github/workflows/arxiv_bot.yml",
        ".github/workflows/rss-service.yml", 
        ".github/workflows/config-update.yml"
    ]
    
    for workflow_file in workflow_files:
        try:
            with open(workflow_file, "r") as f:
                workflow = yaml.safe_load(f)
            print(f"✅ {workflow_file} is valid YAML")
            
            # Check for required workflow fields
            if "on" in workflow:
                triggers = workflow["on"]
                if isinstance(triggers, dict):
                    trigger_types = list(triggers.keys())
                    print(f"✅ {workflow_file} has triggers: {', '.join(trigger_types)}")
                else:
                    print(f"✅ {workflow_file} has trigger configuration")
            else:
                print(f"❌ {workflow_file} missing trigger configuration")
                
        except Exception as e:
            print(f"❌ {workflow_file} error: {e}")

def test_bot_functionality():
    """Test that the bot can run successfully."""
    print("\n🤖 Testing bot functionality...")
    
    try:
        from arxiv_bot import ArxivBot
        
        # Test with minimal config
        bot = ArxivBot()
        print("✅ ArxivBot imported successfully")
        
        # Test config loading
        if hasattr(bot, 'config') and bot.config:
            print("✅ Configuration loaded successfully")
        else:
            print("❌ Configuration loading failed")
            
    except Exception as e:
        print(f"❌ Bot functionality error: {e}")

def test_rss_service():
    """Test RSS service functionality."""
    print("\n📡 Testing RSS service...")
    
    try:
        from rss_service import app
        print("✅ RSS service imported successfully")
        
        # Test basic functionality
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint returned {response.status_code}")
                
    except Exception as e:
        print(f"❌ RSS service error: {e}")

def main():
    """Run all tests."""
    print("🧪 Testing arXiv RSS Bot Workflow Configuration")
    print("=" * 50)
    
    test_config_files()
    test_bot_functionality()
    test_rss_service()
    
    print("\n🎉 Workflow configuration test completed!")
    print("\n📋 Next steps:")
    print("1. Push to GitHub: git push origin master")
    print("2. Check Actions tab in your repository")
    print("3. Verify workflows are triggered on push")
    print("4. Monitor README.md for updates")

if __name__ == "__main__":
    main() 