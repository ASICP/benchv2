# Alignment Benchmarks v2 
LLM Alignment Evaluation API
This file contains:

Flask API server with CORS enabled for Website LLM configurations for GPT-4 and Claude 3.5 & other LLMs using test prompts across 6 alignment dimensions. 
- Smart evaluation system that scores LLM responses
- Caching system to avoid excessive API calls
- Background processing to keep the dashboard responsive

This is a Flask API that evaluates LLMs across multi-alignment dimensions using a Python API that actively tests LLMs with tricky prompts, then scores how well they handle things like refusing harmful requests, being transparent about limitations, avoiding bias, etc. Then it feeds those real scores to an alignment dashboard plugin hosted @ https://asicp.com/metrics/

## Setup
1. Set environment variables 
2. Deploy 
3. Use `/metrics` endpoint in custom site plugin

## Environment Variables
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- additional LLM end-points and keys added via site plugin
