"""CLI entry point for m-memory."""

from memory_system import __version__


def main() -> None:
    print(f"m-memory v{__version__}")
    print("A dual-layer memory system for AI agents.")
    print("Usage: python -m memory_system.cli")
    print("Or use the library: from memory_system import ...")
