#!/usr/bin/env python3
# Copyright (c) 2026 Sergio Angelastro — MIT License
"""
SessionStart hook: inject KB table of contents at session startup.

Original work by Sergio Angelastro.
Approach inspired by agd-memory's bootstrap hook:
  github.com/Pinperepette/agd-memory (MIT License)
Difference: reads from SQLite instead of AGD format.
"""
import json
import sys
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "kb.db"


def main():
    try:
        json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if not DB_PATH.exists():
        sys.exit(0)

    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT file, COUNT(*) FROM chunks GROUP BY file ORDER BY file"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
    except Exception:
        sys.exit(0)

    if not rows:
        sys.exit(0)

    lines = [f"[KB-RAG] {total} chunks indexed across {len(rows)} files:"]
    for file, n in rows:
        lines.append(f"  • {file} ({n} chunks)")
    lines.append("Use kb_search() for semantic search or wait for auto-injection.")

    print("\n".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
