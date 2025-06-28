"""
Model Registry for Latest OpenAI and Anthropic Models
Centralized configuration for all supported models with their capabilities and settings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
from enum import Enum

class ModelTier(Enum):
    """Model performance and cost tiers."""
    PREMIUM = "premium"      # Highest performance, highest cost (o1, Claude-3.5-Sonnet)
    STANDARD = "standard"    # Balanced performance/cost (GPT-4o, Claude-3.5-Haiku)
    EFFICIENT = "efficient"  # Fast and cost-effective (GPT-4o-mini, Claude-3-Haiku)

class ModelCapability(Enum):
    """Model-specific capabilities."""
    STRUCTURED_OUTPUT = "structured_output"     # Native structured output support
    TOOL_CALLING = "tool_calling"              # Function/tool calling support
    REASONING = "reasoning"                     # Advanced reasoning capabilities
    FAST_RESPONSE = "fast_response"            # Optimized for speed
    LONG_CONTEXT = "long_context"              # Extended context window

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: Literal["openai", "anthropic"]
    tier: ModelTier
    capabilities: List[ModelCapability]
    max_tokens: int
    context_window: int
    recommended_timeout: int
    supports_system_message: bool
    pricing_per_1k_input: float
    pricing_per_1k_output: float
    description: str

class ModelRegistry:
    """Registry of all supported models with their configurations."""
    
    # OpenAI Models
    OPENAI_MODELS = {
        # Latest GPT-4o models
        "gpt-4o-2024-11-20": ModelConfig(
            name="gpt-4o-2024-11-20",
            provider="openai",
            tier=ModelTier.STANDARD,
            capabilities=[ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING],
            max_tokens=16384,
            context_window=128000,
            recommended_timeout=120,
            supports_system_message=True,
            pricing_per_1k_input=2.50,
            pricing_per_1k_output=10.00,
            description="Latest GPT-4o with structured outputs and improved reasoning"
        ),
        
        "gpt-4o-2024-08-06": ModelConfig(
            name="gpt-4o-2024-08-06",
            provider="openai",
            tier=ModelTier.STANDARD,
            capabilities=[ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING],
            max_tokens=16384,
            context_window=128000,
            recommended_timeout=120,
            supports_system_message=True,
            pricing_per_1k_input=2.50,
            pricing_per_1k_output=10.00,
            description="Stable GPT-4o with structured outputs"
        ),
        
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            tier=ModelTier.EFFICIENT,
            capabilities=[ModelCapability.STRUCTURED_OUTPUT, ModelCapability.TOOL_CALLING, ModelCapability.FAST_RESPONSE],
            max_tokens=16384,
            context_window=128000,
            recommended_timeout=60,
            supports_system_message=True,
            pricing_per_1k_input=0.15,
            pricing_per_1k_output=0.60,
            description="Fast and cost-effective GPT-4o variant"
        ),
        
        # Reasoning models
        "o1-preview": ModelConfig(
            name="o1-preview",
            provider="openai",
            tier=ModelTier.PREMIUM,
            capabilities=[ModelCapability.REASONING],
            max_tokens=32768,
            context_window=128000,
            recommended_timeout=300,
            supports_system_message=False,
            pricing_per_1k_input=15.00,
            pricing_per_1k_output=60.00,
            description="Advanced reasoning model with chain-of-thought capabilities"
        ),
        
        "o1-mini": ModelConfig(
            name="o1-mini",
            provider="openai",
            tier=ModelTier.STANDARD,
            capabilities=[ModelCapability.REASONING, ModelCapability.FAST_RESPONSE],
            max_tokens=65536,
            context_window=128000,
            recommended_timeout=180,
            supports_system_message=False,
            pricing_per_1k_input=3.00,
            pricing_per_1k_output=12.00,
            description="Faster reasoning model for complex analysis"
        ),
        
        # Legacy models still supported
        "gpt-4-turbo": ModelConfig(
            name="gpt-4-turbo",
            provider="openai",
            tier=ModelTier.STANDARD,
            capabilities=[ModelCapability.TOOL_CALLING],
            max_tokens=4096,
            context_window=128000,
            recommended_timeout=120,
            supports_system_message=True,
            pricing_per_1k_input=10.00,
            pricing_per_1k_output=30.00,
            description="Previous generation GPT-4 Turbo"
        ),
    }
    
    # Anthropic Models
    ANTHROPIC_MODELS = {
        # Claude 3.5 family
        "claude-3-5-sonnet-20241022": ModelConfig(
            name="claude-3-5-sonnet-20241022",
            provider="anthropic",
            tier=ModelTier.PREMIUM,
            capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.LONG_CONTEXT],
            max_tokens=8192,
            context_window=200000,
            recommended_timeout=120,
            supports_system_message=True,
            pricing_per_1k_input=3.00,
            pricing_per_1k_output=15.00,
            description="Latest Claude 3.5 Sonnet with enhanced reasoning and tool use"
        ),
        
        "claude-3-5-sonnet-20240620": ModelConfig(
            name="claude-3-5-sonnet-20240620",
            provider="anthropic",
            tier=ModelTier.PREMIUM,
            capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.LONG_CONTEXT],
            max_tokens=8192,
            context_window=200000,
            recommended_timeout=120,
            supports_system_message=True,
            pricing_per_1k_input=3.00,
            pricing_per_1k_output=15.00,
            description="Stable Claude 3.5 Sonnet"
        ),
        
        "claude-3-5-haiku-20241022": ModelConfig(
            name="claude-3-5-haiku-20241022",
            provider="anthropic",
            tier=ModelTier.EFFICIENT,
            capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.FAST_RESPONSE],
            max_tokens=8192,
            context_window=200000,
            recommended_timeout=60,
            supports_system_message=True,
            pricing_per_1k_input=1.00,
            pricing_per_1k_output=5.00,
            description="Fast and efficient Claude 3.5 Haiku for quick analysis"
        ),
        
        # Claude 3 family
        "claude-3-opus-20240229": ModelConfig(
            name="claude-3-opus-20240229",
            provider="anthropic",
            tier=ModelTier.PREMIUM,
            capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.LONG_CONTEXT],
            max_tokens=4096,
            context_window=200000,
            recommended_timeout=180,
            supports_system_message=True,
            pricing_per_1k_input=15.00,
            pricing_per_1k_output=75.00,
            description="Most capable Claude 3 model for complex reasoning"
        ),
        
        "claude-3-sonnet-20240229": ModelConfig(
            name="claude-3-sonnet-20240229",
            provider="anthropic",
            tier=ModelTier.STANDARD,
            capabilities=[ModelCapability.TOOL_CALLING],
            max_tokens=4096,
            context_window=200000,
            recommended_timeout=120,
            supports_system_message=True,
            pricing_per_1k_input=3.00,
            pricing_per_1k_output=15.00,
            description="Balanced Claude 3 model"
        ),
        
        "claude-3-haiku-20240307": ModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic",
            tier=ModelTier.EFFICIENT,
            capabilities=[ModelCapability.FAST_RESPONSE],
            max_tokens=4096,
            context_window=200000,
            recommended_timeout=60,
            supports_system_message=True,
            pricing_per_1k_input=0.25,
            pricing_per_1k_output=1.25,
            description="Fast and cost-effective Claude 3 model"
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelConfig]:
        """Get all available models."""
        return {**cls.OPENAI_MODELS, **cls.ANTHROPIC_MODELS}
    
    @classmethod
    def get_models_by_provider(cls, provider: str) -> Dict[str, ModelConfig]:
        """Get models for a specific provider."""
        if provider == "openai":
            return cls.OPENAI_MODELS
        elif provider == "anthropic":
            return cls.ANTHROPIC_MODELS
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @classmethod
    def get_models_by_tier(cls, tier: ModelTier) -> Dict[str, ModelConfig]:
        """Get models by performance tier."""
        all_models = cls.get_all_models()
        return {name: config for name, config in all_models.items() if config.tier == tier}
    
    @classmethod
    def get_models_by_capability(cls, capability: ModelCapability) -> Dict[str, ModelConfig]:
        """Get models that support a specific capability."""
        all_models = cls.get_all_models()
        return {name: config for name, config in all_models.items() if capability in config.capabilities}
    
    @classmethod
    def get_recommended_pairs(cls) -> List[Dict[str, str]]:
        """Get recommended model pairs for dual-provider screening."""
        return [
            {
                "name": "Premium Performance",
                "openai": "gpt-4o-2024-11-20",
                "anthropic": "claude-3-5-sonnet-20241022",
                "description": "Best quality and reasoning capabilities"
            },
            {
                "name": "Balanced Performance",
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-haiku-20241022",
                "description": "Good balance of speed, quality, and cost"
            },
            {
                "name": "Reasoning Focus",
                "openai": "o1-mini",
                "anthropic": "claude-3-5-sonnet-20241022",
                "description": "Advanced reasoning with structured analysis"
            },
            {
                "name": "Speed Optimized",
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-haiku-20241022",
                "description": "Fastest processing with good quality"
            },
            {
                "name": "Cost Effective",
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-haiku-20240307",
                "description": "Lowest cost while maintaining quality"
            }
        ]
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        all_models = cls.get_all_models()
        return all_models.get(model_name)
    
    @classmethod
    def is_reasoning_model(cls, model_name: str) -> bool:
        """Check if a model is a reasoning model (o1, o3 series)."""
        return "o1" in model_name.lower() or "o3" in model_name.lower()
    
    @classmethod
    def supports_structured_output(cls, model_name: str) -> bool:
        """Check if a model supports native structured output."""
        config = cls.get_model_config(model_name)
        return config and ModelCapability.STRUCTURED_OUTPUT in config.capabilities
    
    @classmethod
    def supports_tool_calling(cls, model_name: str) -> bool:
        """Check if a model supports tool calling."""
        config = cls.get_model_config(model_name)
        return config and ModelCapability.TOOL_CALLING in config.capabilities
    
    @classmethod
    def get_recommended_timeout(cls, model_name: str) -> int:
        """Get recommended timeout for a model."""
        config = cls.get_model_config(model_name)
        return config.recommended_timeout if config else 120
    
    @classmethod
    def estimate_cost(cls, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a model based on token usage."""
        config = cls.get_model_config(model_name)
        if not config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * config.pricing_per_1k_input
        output_cost = (output_tokens / 1000) * config.pricing_per_1k_output
        
        return input_cost + output_cost