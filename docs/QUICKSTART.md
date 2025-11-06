# Quick Start Guide - VLM Fraud Detection

## Prerequisites

1. **Python 3.11+** installed
2. **API Key** from one of:
   - [OpenRouter](https://openrouter.ai/) (recommended - access to all models)
   - [OpenAI](https://platform.openai.com/)
   - [Anthropic](https://www.anthropic.com/)
   - [Google AI](https://makersuite.google.com/)

## Installation

### 1. Clone and Setup

```bash
cd ../Documents/isitreal-pablo
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file or export environment variables:

```bash
# Option 1: Use OpenRouter (recommended)
export OPENROUTER_API_KEY="your-key-here"

# Option 2: Use native APIs
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

