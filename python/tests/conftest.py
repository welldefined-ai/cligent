"""Test configuration and fixtures."""

import sys
from pathlib import Path

# Add src to Python path so tests can import the package
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))