"""Unified interface for AI agent orchestration."""

from typing import Optional, List
from .registry import registry
from .core.agent import AgentBackend


# Factory functions that return AgentBackend instances directly
def cligent(agent_name: str = "claude-code", location: Optional[str] = None, 
           api_key: Optional[str] = None, use_mock: bool = True) -> AgentBackend:
    """Create an agent instance.
    
    Args:
        agent_name: Agent name or alias (default: "claude-code")
        location: Optional workspace location
        api_key: API key for the agent (uses env vars if not provided)
        use_mock: If True, use mock executor (default: True for safety)
        
    Returns:
        AgentBackend instance
    """
    agent = registry.create_agent(agent_name, location, api_key=api_key, use_mock=use_mock)
    if not agent:
        available = [config.name for config in registry.list_agents()]
        raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
    return agent


def claude(location: Optional[str] = None, api_key: Optional[str] = None, use_mock: bool = True) -> AgentBackend:
    """Create a Claude Code agent.
    
    Args:
        location: Optional workspace location
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        use_mock: If True, use mock executor instead of real Claude Code SDK
    """
    return cligent("claude-code", location, api_key, use_mock)


def gemini(location: Optional[str] = None, api_key: Optional[str] = None, use_mock: bool = True) -> AgentBackend:
    """Create a Gemini CLI agent.
    
    Args:
        location: Optional workspace location
        api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        use_mock: If True, use mock executor instead of real Gemini API
    """
    return cligent("gemini-cli", location, api_key, use_mock)


def qwen(location: Optional[str] = None, api_key: Optional[str] = None, use_mock: bool = True) -> AgentBackend:
    """Create a Qwen Code agent.
    
    Args:
        location: Optional workspace location
        api_key: Qwen API key (uses DASHSCOPE_API_KEY env var if not provided)
        use_mock: If True, use mock executor instead of real Qwen API
    """
    return cligent("qwen-code", location, api_key, use_mock)


def list_available_agents() -> List[str]:
    """List all available agents."""
    return [config.name for config in registry.list_agents()]