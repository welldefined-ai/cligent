"""Cligent - Unified SDK for CLI agent orchestration.

This is the main package that provides the public API
while the actual implementation is in the flattened core/ and agents/ structure.
"""

# Re-export everything from core for the public API
from .core import *

# Import and expose agent framework from cligent.py
from .cligent import Cligent, create

# Backwards compatibility
ChatParser = create