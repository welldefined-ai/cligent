"""Cligent trial/debugging CLI.

This provides a lightweight command-line interface for exploring and
debugging agent chat parsing across supported providers.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import create  # re-exported factory from package


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cligent",
        description="Trial/debug CLI for Cligent chat parsing",
    )
    parser.add_argument(
        "--agent",
        choices=["claude", "gemini", "qwen", "codex"],
        default="claude",
        help="Agent provider to use (default: claude)",
    )

    sub = parser.add_subparsers(dest="cmd", required=False)
    sub.add_parser("interactive", help="Run interactive REPL (default)")
    sub.add_parser("list", help="List available logs for the agent")

    p_parse = sub.add_parser("parse", help="Parse a specific log URI/ID")
    p_parse.add_argument("log_uri", help="Log URI/ID to parse")

    sub.add_parser("live", help="Parse most recent (live) log")

    return parser


def _cmd_list(agent) -> int:
    logs = agent.list_logs()
    if not logs:
        print("No logs found", file=sys.stderr)
        return 1
    for uri, meta in sorted(logs, key=lambda x: x[1].get("modified", ""), reverse=True):
        print(uri)
    return 0


def _cmd_parse(agent, log_uri: str) -> int:
    chat = agent.parse(log_uri)
    if chat is None:
        print("No chat data", file=sys.stderr)
        return 1
    print(f"Parsed {len(chat.messages)} messages")
    return 0


def _cmd_live(agent) -> int:
    chat = agent.parse()
    if chat is None:
        print("No live log found", file=sys.stderr)
        return 1
    print(f"Parsed {len(chat.messages)} messages (live)")
    return 0


def _interactive(agent) -> int:
    print("Cligent Chat Parser - Interactive CLI")
    print("Type 'list', 'parse <uri>', 'live', or 'quit'.")
    while True:
        try:
            raw = input(f"[{agent.name}]> ").strip()
            if not raw:
                continue
            if raw in {"quit", "exit", "q"}:
                return 0
            if raw == "list":
                _cmd_list(agent)
                continue
            if raw == "live":
                _cmd_live(agent)
                continue
            if raw.startswith("parse "):
                _, uri = raw.split(" ", 1)
                _cmd_parse(agent, uri)
                continue
            print("Unknown command. Try: list | parse <uri> | live | quit")
        except KeyboardInterrupt:
            print()
            return 130


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    agent = create(args.agent)

    # Default to interactive if no subcommand
    cmd = args.cmd or "interactive"

    if cmd == "list":
        return _cmd_list(agent)
    if cmd == "parse":
        return _cmd_parse(agent, args.log_uri)
    if cmd == "live":
        return _cmd_live(agent)

    return _interactive(agent)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

