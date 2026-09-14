"""Merges items that are the same underlying news story published on more than one
source, in two stages:

1. `dedupe_items` — free, text-similarity based (difflib). Catches the same press
   release republished verbatim or near-verbatim (e.g. a syndicated article with a
   reworded headline).
2. `dedupe_semantic` — LLM based, only runs if ANTHROPIC_API_KEY is set. Catches the
   same underlying event covered by two *independently written* articles — e.g. EM's
   own short ministry note about a company's €10M investment vs. LIAA's much longer
   piece on the same investment with different quotes and a different angle. These
   share almost no overlapping wording, so no text-similarity threshold would catch
   them without also producing false positives elsewhere — recognizing "same real-world
   event, different write-up" needs actual reading comprehension.

Both stages compare same-day items only (real duplicates are the same news published
the same day), and both stages only ever merge items across *different* sources — a
single source's own items are already unique by construction (distinct URLs from that
source's own scrape), so a same-source "match" is necessarily a false positive (e.g.
several agenda items from one meeting sharing boilerplate text), never a real
duplicate. Several sources also have *recurring* items that legitimately share a name
on different days (a standing Saeima committee's sitting is always called "Budžeta un
finanšu (nodokļu) komisijas sēde" regardless of that day's actual agenda) — scoping to
same-day items, plus both stages judging on actual content, keeps those correctly
separate.
"""

import difflib
import os
from dataclasses import replace

from .sources.base import Item

SIMILARITY_THRESHOLD = 0.82
COMPARE_CHARS = 800  # lead paragraphs are the most distinctive part; capped for speed
SEMANTIC_MODEL = "claude-haiku-4-5-20251001"
SEMANTIC_COMPARE_CHARS = 600

DEDUPE_TOOL = {
    "name": "group_duplicates",
    "description": "Group items that describe the exact same real-world news event or announcement.",
    "input_schema": {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "description": (
                    "Each inner array is the indices of items that all describe the same "
                    "specific event. Every input index must appear in exactly one group — "
                    "items with no duplicate go in their own single-item group."
                ),
            }
        },
        "required": ["groups"],
    },
}

DEDUPE_SYSTEM_PROMPT = """You spot duplicate news coverage: multiple articles, from different
government/agency sources, describing the exact same specific real-world event or announcement
(e.g. the same company's investment, the same program's launch), even when the wording, quotes,
and framing differ between the write-ups.

Only group items together if they describe the SAME specific event/decision/announcement — not
merely the same general topic, sector, or theme. When in doubt, keep items separate."""


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


def _llm_duplicate_groups(client, items: list[Item]) -> list[list[int]]:
    prompt_items = "\n\n".join(
        f"[{i}] Source: {it.source}\nTitle: {it.title}\nText: {it.raw_text[:SEMANTIC_COMPARE_CHARS]}"
        for i, it in enumerate(items)
    )
    try:
        message = client.messages.create(
            model=SEMANTIC_MODEL,
            max_tokens=500,
            system=DEDUPE_SYSTEM_PROMPT,
            tools=[DEDUPE_TOOL],
            tool_choice={"type": "tool", "name": "group_duplicates"},
            messages=[{"role": "user", "content": f"Group these {len(items)} items:\n\n{prompt_items}"}],
        )
    except Exception as exc:
        # A transient API failure (rate limit, timeout, overload) here must not take down
        # the whole run — every requests.get() call elsewhere in this codebase already
        # degrades gracefully on failure; this LLM call should too. Worst case, this
        # day's near-duplicates go unmerged instead of the entire digest being lost.
        print(f"  ! semantic dedupe call failed ({exc}) — leaving these {len(items)} item(s) unmerged")
        return [[i] for i in range(len(items))]
    for block in message.content:
        if block.type == "tool_use":
            groups = block.input.get("groups")
            if groups:
                return groups
    return [[i] for i in range(len(items))]  # fall back to "nothing is a duplicate"


def dedupe_semantic(items: list[Item]) -> list[Item]:
    """A second pass over `dedupe_items`'s output that catches same-event items which
    text-similarity missed because they're independently written. No-ops (returns items
    unchanged) if ANTHROPIC_API_KEY isn't set — this needs real reading comprehension.
    """
    # .strip(): see the matching comment in classify.py — a stray trailing newline/space
    # turns this into an illegal HTTP header value instead of a clean "no key" no-op.
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return items

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    by_date: dict[str, list[Item]] = {}
    for item in items:
        by_date.setdefault(item.date, []).append(item)

    merged: list[Item] = []
    for same_day_items in by_date.values():
        distinct_sources = {it.source for it in same_day_items}
        if len(same_day_items) < 2 or len(distinct_sources) < 2:
            merged.extend(same_day_items)  # nothing to compare against
            continue

        groups = _llm_duplicate_groups(client, same_day_items)
        placed = set()
        for group in groups:
            indices = [i for i in group if 0 <= i < len(same_day_items) and i not in placed]
            if not indices:
                continue
            placed.update(indices)
            cluster_items = [same_day_items[i] for i in indices]
            if len(cluster_items) > 1 and len({it.source for it in cluster_items}) > 1:
                merged.append(_merge_cluster(cluster_items))
            else:
                merged.extend(cluster_items)  # never merge a same-source "group"

        for i, item in enumerate(same_day_items):  # safety net: model must not drop items
            if i not in placed:
                merged.append(item)

    return merged
