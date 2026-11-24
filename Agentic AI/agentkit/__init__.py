"""agentkit: provider-independent LLM layer shared by the Agentic AI projects.

Supports OpenAI (and OpenAI-compatible servers), Google Gemini, Ollama, and local Hugging Face models,
selected automatically from environment variables, with no code changes needed to switch.
"""
from .config import Settings, resolve, LOCAL_HF_MODELS, DEFAULT_API_MODELS
from .llm import LLM, AgentRun, AgentStep, CacheMiss, StructuredOutputError, configure_logging
from .providers import Completion, ProviderError
from .tools import tool_schema, execute_tool

__all__ = ["LLM", "Settings", "resolve", "configure_logging", "AgentRun", "AgentStep", "Completion", "ProviderError",
           "StructuredOutputError", "CacheMiss", "tool_schema", "execute_tool", "LOCAL_HF_MODELS", "DEFAULT_API_MODELS"]
__version__ = "1.0.0"
