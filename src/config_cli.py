"""CLI for managing doc-rag configuration.

Usage:
    doc-rag-config init      Create default config at ~/.config/doc-rag/config.json
    doc-rag-config show      Show current effective config
    doc-rag-config path      Print config file path
"""

import argparse
import json
import sys

from src.config import CONFIG_FILE, generate_config, settings


def cmd_init(_args):
    """Create default config file."""
    if CONFIG_FILE.exists():
        print(f"Config already exists at {CONFIG_FILE}")
        print("Use --force to overwrite.")
        return

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(generate_config(), f, indent=2)
    print(f"Config created at {CONFIG_FILE}")
    print("Edit this file to customize your settings.")


def cmd_show(_args):
    """Show current effective configuration."""
    from dataclasses import asdict

    print(json.dumps(asdict(settings), indent=2))


def cmd_path(_args):
    """Print config file path."""
    print(CONFIG_FILE)


def main():
    parser = argparse.ArgumentParser(description="doc-rag configuration manager")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Create default config file")
    sub.add_parser("show", help="Show current effective config")
    sub.add_parser("path", help="Print config file path")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "path":
        cmd_path(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
