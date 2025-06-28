# LLM Screening Tool - Research

A Flask-based dual-LLM screening tool for systematic literature reviews that automates academic paper screening using both OpenAI and Anthropic models.

## Features
- Dual-LLM validation with OpenAI and Anthropic models
- Support for latest models (GPT-4o, Claude-3.5-Sonnet, o1-preview)
- Multiple file format support (RIS, BibTeX, CSV, XML, PMID)
- Real-time screening with structured outputs
- Cost tracking and error handling

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
3. Initialize database: `python init_database.py`
4. Run application: `python run.py`

---
*Verification testing completed on 2025-06-28 - Repository access, linting, and PR workflow confirmed.*
