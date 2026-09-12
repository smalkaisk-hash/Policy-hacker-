"""Merges items that are the same underlying article/document published on more than
one source — e.g. EM and LIAA sometimes syndicate the identical press release, whether
verbatim or with a slightly reworded headline.

Matching is done on the actual scraped article/document body (not the title), so a
reworded headline no longer slips through. Comparison is scoped to same-day items: real
duplicates are the same press release published the same day, and several sources have
*recurring* items that legitimately share a name on different days (e.g. a standing
Saeima committee's sitting is always called "Budžeta un finanšu (nodokļu) komisijas
sēde" regardless of that day's actual agenda) — those have genuinely different content
and must stay separate, which content-based matching naturally respects.
"""

import difflib
from dataclasses import replace

from .sources.base import Item

SIMILARITY_THRESHOLD = 0.82
COMPARE_CHARS = 800  # lead paragraphs are the most distinctive part; capped for speed


def _content_signature(item: Item) -> str:
    # raw_text is built as "{title}\n\n{body...}" by every source module — compare on the
    # body, since two genuinely different items can share a title (see module docstring).
    split_at = item.raw_text.find("\n\n")
    body = item.raw_text[split_at + 2 :] if split_at != -1 else item.raw_text
    return " ".join(body.lower().split())[:COMPARE_CHARS]


def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    matcher = difflib.SequenceMatcher(None, a, b)
    if matcher.quick_ratio() < SIMILARITY_THRESHOLD:
        return False  # cheap upper-bound check before the more expensive real ratio
    return matcher.ratio() >= SIMILARITY_THRESHOLD


def _merge_cluster(cluster: list[Item]) -> Item:
    base = max(cluster, key=lambda it: len(it.raw_text))
    sources = []
    for it in cluster:
        if it.source not in sources:
            sources.append(it.source)
    return replace(base, source=" + ".join(sources))


class _Cluster:
    def __init__(self, item: Item, signature: str):
        self.items = [item]
        self.signatures = [signature]
        self.sources = {item.source}

    def accepts(self, item: Item, signature: str) -> bool:
        # Never merge two items that came from the same source — a source's own items are
        # already unique by construction (distinct URLs from that source's own scrape), so
        # a same-source "match" here is a false positive (typically shared boilerplate:
        # several agenda items from one meeting all carrying the same "(No sēdes: ...)"
        # suffix), not a real duplicate. Duplicates only happen *across* sources.
        if item.source in self.sources:
            return False
        return any(_similar(signature, s) for s in self.signatures)

    def add(self, item: Item, signature: str) -> None:
        self.items.append(item)
        self.signatures.append(signature)
        self.sources.add(item.source)


def dedupe_items(items: list[Item]) -> list[Item]:
    by_date: dict[str, list[Item]] = {}
    for item in items:
        by_date.setdefault(item.date, []).append(item)

    merged: list[Item] = []
    for same_day_items in by_date.values():
        clusters: list[_Cluster] = []

        for item in same_day_items:
            sig = _content_signature(item)
            for cluster in clusters:
                if cluster.accepts(item, sig):
                    cluster.add(item, sig)
                    break
            else:
                clusters.append(_Cluster(item, sig))

        for cluster in clusters:
            merged.append(cluster.items[0] if len(cluster.items) == 1 else _merge_cluster(cluster.items))

    return merged
