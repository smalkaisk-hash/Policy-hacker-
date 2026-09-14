"""Tiny local dedupe store so re-running the digest only surfaces new items.

Just a flat JSON file of item URLs seen on any previous run. Good enough for
a weekly, single-machine prototype — no need for a real database yet.
"""

import json
import os
import tempfile
from pathlib import Path


def load_seen(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        with state_path.open(encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        # This file is a best-effort dedupe cache, not the source of truth for anything —
        # losing it just means some already-seen items get refetched/reclassified once.
        # A run that got killed mid-write (Ctrl+C, OOM, crash) could leave it truncated;
        # crashing every future run on startup because of that would be far worse than
        # just starting over. (save_seen writes atomically now to avoid causing this, but
        # this file is easy to edit or move by hand too, so still don't trust it blindly.)
        print(f"  ! state file {state_path} is unreadable/corrupt ({exc}) — starting with empty state")
        return set()


def save_seen(state_path: Path, seen_urls: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename into place, so a run that gets killed mid-write
    # (Ctrl+C, OOM, crash) can never leave a half-written/corrupt state.json behind —
    # the rename is atomic, so the file on disk is always either the old or new complete
    # version, never a partial one.
    fd, tmp_path = tempfile.mkstemp(dir=state_path.parent, prefix=state_path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sorted(seen_urls), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, state_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
