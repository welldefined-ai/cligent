#!/usr/bin/env python3
"""Backwards-compatible launcher that invokes the packaged CLI.

Use `cligent` after installing the package. This file exists to ease
trial runs from the repository root.
"""

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["interactive"]))
