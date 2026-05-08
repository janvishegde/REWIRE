"""Batch simulation helpers.

This module exists to provide a stable interface for demo scripts:
- list_completed_drugs(output_dir)
- load_graph(path)

It intentionally centralizes pickle I/O so downstream modules can keep their
imports minimal.
"""

from __future__ import annotations

from pathlib import Path
import pickle


def list_completed_drugs(output_dir: str | Path) -> list[str]:
    """Return completed drug names (one per line) if completed.txt exists.

    Falls back to inferring names from *.pkl stems if completed.txt is missing.
    """

    out = Path(output_dir)
    completed = out / "completed.txt"

    if completed.exists():
        try:
            lines = completed.read_text(encoding="utf-8").splitlines()
            return [ln.strip() for ln in lines if ln.strip()]
        except OSError:
            return []

    # Fallback: use filenames.
    pkls = sorted(out.glob("*.pkl"))
    # Best-effort: reverse the simple sanitize (underscores -> spaces)
    return [p.stem.replace("_", " ") for p in pkls]


def load_graph(path: str | Path):
    """Load a pickled NetworkX graph from disk."""

    p = Path(path)
    with p.open("rb") as handle:
        return pickle.load(handle)
