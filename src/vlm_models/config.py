"""
Configuration for Vision Language Model fraud detection
"""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, SecretStr
import os

# Model provider options
ModelProvider = Literal[
    "gpt-4o",
    "gpt-5-chat",
    "gpt-5",
    "claude-4-5-sonnet",
    "claude-4-5-haiku",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "qwen-3-vl",
   # "pixtral-large",
]

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "vlm_analysis"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class VLMConfig(BaseModel):
    """Configuration for VLM-based fraud detection"""
    
    # Model selection
    model_provider: ModelProvider = Field(
        default="gpt-4o",
        description="VLM model to use for analysis"
    )
    
    # API Configuration
    openrouter_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENROUTER_API_KEY", "")),
        description="OpenRouter API key for accessing models"
    )
    
    openai_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", "")),
        description="OpenAI API key for GPT models"
    )
    
    anthropic_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        description="Anthropic API key for Claude models"
    )
    
    google_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY", "")),
        description="Google API key for Gemini models"
    )
    
    # Model parameters
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Model temperature (lower = more deterministic)"
    )
    
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum tokens in response"
    )
    
    # Analysis configuration
    save_results: bool = Field(
        default=True,
        description="Save analysis results to disk"
    )
    
    save_visualizations: bool = Field(
        default=False,
        description="Save annotated images with detected regions"
    )
    
    # Retry configuration
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of API retries"
    )
    
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay between retries in seconds"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def get_model_mapping(self) -> dict:
        """Get the OpenRouter model identifier for each provider"""
        return {
            "gpt-4o": "openai/gpt-4o",
            "gpt-5-chat": "openai/gpt-5-chat",
            "gpt-5": "openai/gpt-5",
            "claude-4-5-sonnet": "anthropic/claude-sonnet-4.5",
            "claude-4-5-haiku": "anthropic/claude-haiku-4.5",
            "gemini-2.5-flash": "google/gemini-2.5-flash",
            "gemini-2.5-pro": "google/gemini-2.5-pro",
            "qwen-3-vl": "qwen/qwen3-vl-235b-a22b-instruct",
            #"pixtral-large": "mistralai/pixtral-large-latest",
        }
    
    def get_openrouter_model_id(self) -> str:
        """Get the OpenRouter model identifier for the selected provider"""
        return self.get_model_mapping()[self.model_provider]
    
    def uses_native_api(self) -> bool:
        """Check if model should use native API instead of OpenRouter"""
        return False
