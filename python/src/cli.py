"""Cligent trial/debugging CLI.

This provides a lightweight command-line interface for exploring and
debugging agent chat parsing across supported providers.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import create  # re-exported factory from package
from .core.models import Chat


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
    sub.add_parser("list", help="List available logs for the agent")

    p_parse = sub.add_parser("parse", help="Parse a specific log URI/ID")
    p_parse.add_argument("log_uri", help="Log URI/ID to parse")

    p_live = sub.add_parser("live", help="Parse most recent (live) log")
    p_live.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        help="Number of latest messages to show (default: 10)",
    )

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


def _cmd_live(agent, num: int) -> int:
    chat = agent.parse()
    if chat is None:
        print("No live log found", file=sys.stderr)
        return 1
    total = len(chat.messages)
    k = max(0, min(num, total))
    # Build a Chat and use compose() for YAML formatting
    recent = Chat(messages=chat.messages[-k:])
    yaml_text = agent.compose(recent)
    print(yaml_text)
    return 0


    # Interactive mode removed to keep CLI simple and explicit


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    agent = create(args.agent)

    # If no subcommand provided, show help and exit
    cmd = args.cmd

    if cmd == "list":
        return _cmd_list(agent)
    if cmd == "parse":
        return _cmd_parse(agent, args.log_uri)
    if cmd == "live":
        return _cmd_live(agent, getattr(args, "num", 10))

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
