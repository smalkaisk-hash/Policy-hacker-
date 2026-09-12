"""Merges items that are the same underlying article/document published on more than
one source — e.g. EM and LIAA sometimes syndicate the identical press release verbatim,
which otherwise shows up twice in the digest (once per source).

Keyed on (title, date), not title alone: several sources have *recurring* items that
legitimately share an identical title on different dates (e.g. a standing Saeima
committee's sitting is always named "Budžeta un finanšu (nodokļu) komisijas sēde"
regardless of that day's actual agenda) — those must NOT be collapsed together.
"""

from dataclasses import replace

from .sources.base import Item


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def dedupe_by_title(items: list[Item]) -> list[Item]:
    groups: dict[tuple[str, str], list[Item]] = {}
    for item in items:
        groups.setdefault((_normalize_title(item.title), item.date), []).append(item)

    merged: list[Item] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Keep whichever copy has the most complete scraped text as the base, but credit
        # every source it was found on.
        base = max(group, key=lambda it: len(it.raw_text))
        sources = []
        for it in group:
            if it.source not in sources:
                sources.append(it.source)
        merged.append(replace(base, source=" + ".join(sources)))

    return merged
