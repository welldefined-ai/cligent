"""Log Store interface for managing agent logs."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple


class LogStore(ABC):
    """Manager for an agent's logs."""
    
    def __init__(self, agent: str, location: str = None):
        """Initialize log store for a specific agent.
        
        Args:
            agent: Name of the agent (e.g., "claude-code")
            location: Location of the agent's logs (optional, implementation-specific)
        """
        self.agent = agent
        self.location = location
    
    @abstractmethod
    def list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Show available logs for the agent.
        
        Returns:
            List of (log_uri, metadata) tuples with implementation-specific metadata.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get(self, log_uri: str) -> str:
        """Retrieve raw content of a specific log.
        
        Args:
            log_uri: Log URI
            
        Returns:
            Raw log content as string
        """
        raise NotImplementedError
    
    @abstractmethod
    def live(self) -> Optional[str]:
        """Get URI of currently active log.
        
        Returns:
            Log URI for active log or None if no active log
        """
        raise NotImplementedError


