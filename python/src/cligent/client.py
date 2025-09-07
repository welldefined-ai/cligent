"""Unified interface for AI agent orchestration."""

from typing import Optional, List
from .registry import registry
from .core.agent import AgentBackend


# Factory functions that return AgentBackend instances directly
def cligent(agent_name: str = "claude-code", location: Optional[str] = None) -> AgentBackend:
    """Create an agent instance.
    
    Args:
        agent_name: Agent name or alias (default: "claude-code")
        location: Optional workspace location
        
    Returns:
        AgentBackend instance
    """
    agent = registry.create_agent(agent_name, location)
    if not agent:
        available = [config.name for config in registry.list_agents()]
        raise ValueError(f"Unknown agent: {agent_name}. Available: {available}")
    return agent


def claude(location: Optional[str] = None) -> AgentBackend:
    """Create a Claude Code agent."""
    return cligent("claude-code", location)


def gemini(location: Optional[str] = None) -> AgentBackend:
    """Create a Gemini CLI agent."""
    return cligent("gemini-cli", location)


def qwen(location: Optional[str] = None) -> AgentBackend:
    """Create a Qwen Code agent."""
    return cligent("qwen-code", location)


def list_available_agents() -> List[str]:
    """List all available agents."""
    return [config.name for config in registry.list_agents()]