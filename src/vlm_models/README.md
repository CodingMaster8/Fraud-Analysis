# VLM-Based Fraud Detection

This module uses Vision Language Models (VLMs) to detect fraudulent alterations in bank check images.

## Features

- ✅ **Multiple VLM Support**: GPT-4o, Claude, Gemini, Llama Vision, Pixtral
- ✅ **OpenRouter Integration**: Access to multiple models through a single API
- ✅ **Pydantic Validation**: Structured output with full validation
- ✅ **LangChain Integration**: Consistent interface across different providers
- ✅ **Retry Logic**: Automatic retry with exponential backoff
- ✅ **Comprehensive Analysis**: Detailed fraud detection with specific regions

## Architecture

```
vlm_models/
├── __init__.py          # Module exports
├── config.py            # VLM configuration with model selection
├── schemas.py           # Pydantic models for output validation
├── prompts.py           # Prompt templates for fraud detection
├── analyzer.py          # Main VLM analyzer with LangChain
└── README.md           # This file
```

## Configuration

### Model Selection

Choose from multiple VLM providers:

```python
from src.vlm_models import VLMConfig, VLMAnalyzer

config = VLMConfig(
    model_provider="gpt-4o",  # or "claude-3-5-sonnet", "gemini-2.0-flash-exp", etc.
    temperature=0.1,
    max_tokens=4000
)

analyzer = VLMAnalyzer(config)
```

### Available Models

- **OpenAI**: `gpt-4o`, `gpt-5`
- **Anthropic**: `claude-4-5-sonnet`, `claude-4-5-haiku`
- **Google**: `gemini-2.5-flash`, `gemini-2.5-pro`
- **Meta**: `llama-3.2-90b-vision`
- **Mistral**: `pixtral-large`

### API Keys

Set environment variables:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # Recommended for all models
# OR use native APIs:
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```
