"""Tiny local dedupe store so re-running the digest only surfaces new items.

Just a flat JSON file of item URLs seen on any previous run. Good enough for
a weekly, single-machine prototype — no need for a real database yet.
"""

import json
from pathlib import Path


def load_seen(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    with state_path.open(encoding="utf-8") as f:
        return set(json.load(f))


def save_seen(state_path: Path, seen_urls: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(seen_urls), f, ensure_ascii=False, indent=2)
