#!/usr/bin/env python3
"""
Setup script for arXiv Bot
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="arxiv-rss-bot",
    version="1.0.0",
    author="arXiv Bot",
    description="An automated bot that fetches papers from arXiv RSS feeds and updates README.md",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arxiv_rss_bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "arxiv-bot=arxiv_bot:main",
        ],
    },
) 