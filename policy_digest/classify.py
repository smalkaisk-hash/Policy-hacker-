"""Decides which fetched items are "startup-relevant".

Two stages:
1. A free keyword pre-filter cuts obviously unrelated items (routine
   administrative/ceremonial notices, sectors with no startup angle) before
   any API call is made.
2. If an ANTHROPIC_API_KEY is available, the surviving candidates are sent to
   Claude Haiku in small batches for a real judgment call + a one-line reason,
   using tool-use so the response is structured JSON, not free text to parse.

Without an API key, stage 1's matches are used directly (relevant=True,
reason="keyword match: <keyword>") — cruder, but keeps the prototype runnable
with zero external dependencies.
"""

import json
import os
from dataclasses import dataclass

from .sources.base import Item

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 10

# Definition of "startup-relevant" for this digest (see README for the full writeup):
# funding & support programs, regulatory/legal/tax changes affecting startups or SMEs,
# and government initiatives on innovation, digitalization or entrepreneurship.
KEYWORDS = [
    "jaunuzņēm", "start-up", "startup", "riska kapitāl", "venture", "inovāc",
    "inkubat", "akselerat", "atbalsta programm", "atbalsts uzņēmēj", "grant",
    "granti", "subsīdij", "es fond", "eiropas savienības fond", "digitalizāc",
    "digitālā transformācij", "mākslīgais intelekts", "nodokļ", "uin", "iin",
    "darba tiesīb", "opciju līgum", "kapitāla daļu opcij", "publiskais iepirkum",
    "iepirkum", "uzņēmējdarbīb", "mazie un vidējie uzņēmum", "mvu", "eksport",
    "investīcij", "finansējum", "darba atļauj", "talantu piesaist", "it nozar",
    "tehnoloģij", "komercdarbīb",
]

CLASSIFY_TOOL = {
    "name": "classify_items",
    "description": "Classify each policy/news item for relevance to Latvian tech startups.",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "relevant": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {
                            "type": "string",
                            "description": "One short sentence, same language as the item title, explaining the verdict.",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["funding", "regulation", "tax_labor", "digitalization_innovation", "other"],
                        },
                    },
                    "required": ["index", "relevant", "confidence", "reason", "category"],
                },
            }
        },
        "required": ["results"],
    },
}

SYSTEM_PROMPT = """You are a policy analyst for startin.lv, a Latvian startup accelerator.

You review Latvian government documents (draft legislation, cabinet/state-secretary meeting
agenda items, ministry and agency news) and decide which are relevant to startups.

Mark an item RELEVANT if it involves any of:
- funding or support programs for startups/SMEs (grants, EU funds, accelerator/incubator
  programs, investment/venture capital initiatives, LIAA/Altum programs)
- legal or regulatory changes affecting startups, SMEs, or tech companies (company law, tax
  treatment incl. reinvested profit or employee stock options, labor law, digital services or
  AI regulation, public procurement rules relevant to tech vendors)
- draft legislation or government initiatives on innovation, digitalization, or entrepreneurship

Mark it NOT relevant if it's routine administrative/personnel/ceremonial business, or concerns a
sector with no plausible startup angle (e.g. agricultural subsidies unrelated to agtech,
healthcare staffing, road maintenance).

Be decisive. Write the one-line reason in the same language as the item's title."""


@dataclass
class Classification:
    item: Item
    relevant: bool
    confidence: float
    reason: str
    category: str


def _keyword_match(item: Item) -> str | None:
    text = f"{item.title}\n{item.raw_text}".lower()
    for kw in KEYWORDS:
        if kw in text:
            return kw
    return None


def _classify_batch_with_llm(client, batch: list[Item]) -> list[Classification]:
    prompt_items = "\n\n".join(
        f"[{i}] Source: {it.source}\nTitle: {it.title}\nDetails: {it.raw_text[:2500]}"
        for i, it in enumerate(batch)
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "classify_items"},
        messages=[{"role": "user", "content": f"Classify these {len(batch)} items:\n\n{prompt_items}"}],
    )

    for block in message.content:
        if block.type == "tool_use":
            results = block.input.get("results", [])
            by_index = {r["index"]: r for r in results}
            out = []
            for i, it in enumerate(batch):
                r = by_index.get(i)
                if r is None:
                    continue
                out.append(
                    Classification(
                        item=it,
                        relevant=r["relevant"],
                        confidence=r.get("confidence", 0.5),
                        reason=r.get("reason", ""),
                        category=r.get("category", "other"),
                    )
                )
            return out
    return []


def classify_items(items: list[Item]) -> list[Classification]:
    candidates = []
    for item in items:
        kw = _keyword_match(item)
        if kw:
            candidates.append((item, kw))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return [
            Classification(
                item=item,
                relevant=True,
                confidence=0.5,
                reason=f'Keyword match: "{kw}" (no ANTHROPIC_API_KEY set — keyword-only mode)',
                category="other",
            )
            for item, kw in candidates
        ]

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    results: list[Classification] = []
    candidate_items = [item for item, _ in candidates]
    for start in range(0, len(candidate_items), BATCH_SIZE):
        batch = candidate_items[start : start + BATCH_SIZE]
        results.extend(_classify_batch_with_llm(client, batch))

    return [r for r in results if r.relevant]
