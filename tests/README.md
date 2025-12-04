# Tests Directory

This directory contains test scripts organized by category.

## Directory Structure

- `ai/` - AI-related tests (Qwen API, batch processing, PDF summarization)
- `core/` - Core functionality tests (bot, PDF download, deduplication, workflow)
- `services/` - Service-related tests (RSS service, scheduling)

## Running Tests

### AI Tests
```bash
# Test AI chat PDF functionality (single paper)
python tests/ai/test_ai_chatpdf.py

# Test Qwen batch API
python tests/ai/test_qwen_batch.py --batch --count 3
```

### Core Tests
```bash
# Test bot functionality
python tests/core/test_bot.py

# Test PDF download
python tests/core/test_arxiv_pdf_download.py

# Test deduplication
python tests/core/test_deduplication.py
```

### Service Tests
```bash
# Test RSS service
python tests/services/test_rss_service.py

# Test scheduling
python tests/services/test_schedule.py
```

## Environment Variables

For AI tests, you need to set:
```bash
export DASHSCOPE_API_KEY='your-api-key'
```

