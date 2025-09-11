"""Cligent - Unified SDK for CLI agent orchestration.

This is a minimal package wrapper that provides the public API
while the actual implementation is in the flattened core/ and agents/ structure.
"""

__version__ = "0.2.0"

# Re-export everything from core for the public API
from core import *

# Make sure the main factory function is available 
from core import cligent

# Backwards compatibility
ChatParser = cligent