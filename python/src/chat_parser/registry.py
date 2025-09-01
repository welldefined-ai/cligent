from typing import Dict, Type, Optional, List
from .agent import AgentBackend, AgentConfig

class AgentRegistry:
    """Registry for managing available agent backends."""

    def __init__(self):
        self._agents: Dict[str, Type[AgentBackend]] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, agent_class: Type[AgentBackend], aliases: List[str] = None):
        """Register an agent backend."""
        config = agent_class().config
        self._agents[config.name] = agent_class

          # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = config.name

    def get_agent(self, name: str) -> Optional[Type[AgentBackend]]:
        """Get agent by name or alias."""
        # Check direct name
        if name in self._agents:
            return self._agents[name]

          # Check aliases
        if name in self._aliases:
            return self._agents[self._aliases[name]]

        return None

    def list_agents(self) -> List[AgentConfig]:
        """List all registered agents."""
        return [agent_class().config for agent_class in self._agents.values()]

    def auto_detect(self, log_path: Path) -> Optional[str]:
        """Auto-detect agent from log file."""
        for name, agent_class in self._agents.items():
            agent = agent_class()
            if agent.detect_agent(log_path):
                return name
        return None

# Global registry instance
registry = AgentRegistry()