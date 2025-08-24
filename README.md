# Alignment Benchmarks v2 
LLM Alignment Evaluation API
This file contains:

Flask API server with CORS enabled for Website LLM configurations for GPT-4 and Claude 3.5 & other test prompts across 6 alignment dimensions. 
Smart evaluation system that scores LLM responses
Caching system to avoid excessive API calls
Background processing to keep your dashboard responsive

Flask API that evaluates LLMs across alignment dimensions.

## Setup
1. Set environment variables 
2. Deploy 
3. Use `/metrics` endpoint in custom site plugin

## Environment Variables
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
