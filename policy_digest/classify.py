"""Decides which fetched items are "startup-relevant".

Two stages:
1. A free keyword pre-filter cuts obviously unrelated items (routine
   administrative/ceremonial notices, sectors with no startup angle) before
   any API call is made.
2. If an ANTHROPIC_API_KEY is available, the surviving candidates are sent to
   Claude Haiku in small batches for a real judgment call + a one-line reason,
   using tool-use so the response is structured JSON, not free text to parse.

Without an API key, stage 1's matches are used directly (relevant=True, reason=the actual
sentence the keyword was found in, quoted from the article/document body) — cruder than a
real judgment call, but still grounded in the text itself rather than a bare keyword label,
and keeps the prototype runnable with zero external dependencies.
"""

import json
import os
import re
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
                            "description": "One short sentence IN LATVIAN explaining the verdict.",
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

Be decisive. Always write the one-line reason in Latvian, regardless of what language the
source item is in — the digest this feeds is Latvian-only. Use the proper Latvian term
"jaunuzņēmums/jaunuzņēmumi" (in whatever case the sentence needs) for "startup" — never the
English loanword "startaps/startups"."""


@dataclass
class Classification:
    item: Item
    relevant: bool
    confidence: float
    reason: str
    category: str


# Splits on sentence-ending punctuation followed by a capital letter (Latvian included) —
# deliberately does NOT split on "Nr.1449", "1.lasījums", etc. (no space, or no capital
# after), but does split on numbered-list markers like "8. Likumprojekts" which is exactly
# the boundary we want in this domain's agenda/protocol text.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZĀČĒĢĪĶĻŅŠŪŽ"„])')
_SNIPPET_MAX_LEN = 320


def _snippet_around(text: str, keyword: str) -> str:
    """Returns the sentence the keyword appears in (widened to a neighbor sentence if
    that one alone is too short to be useful) instead of a fixed-radius, often mid-word
    slice — reads as an actual excerpt, not a jagged fragment."""
    idx = text.lower().find(keyword)
    if idx == -1:
        return text[:_SNIPPET_MAX_LEN].strip()

    sentences = _SENTENCE_SPLIT_RE.split(text)
    pos = 0
    for i, sentence in enumerate(sentences):
        end_pos = pos + len(sentence)
        if pos <= idx < end_pos + 1:
            snippet = sentence.strip()
            if len(snippet) < 40 and i + 1 < len(sentences):
                snippet = f"{snippet} {sentences[i + 1].strip()}"
            if len(snippet) < 40 and i > 0:
                snippet = f"{sentences[i - 1].strip()} {snippet}"
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) > _SNIPPET_MAX_LEN:
                snippet = snippet[:_SNIPPET_MAX_LEN].rsplit(" ", 1)[0] + "…"
            return snippet
        pos = end_pos + 1

    return text[:_SNIPPET_MAX_LEN].strip()


def _keyword_match(item: Item) -> tuple[str, str] | None:
    """Returns (keyword, context_snippet) — the snippet is quoted straight from the
    fetched article/document body so the reason reflects what the text actually says,
    not just that a word appeared somewhere. Body content is preferred over the title:
    e.g. a standing committee named "...(nodokļu)..." would otherwise "match" on every
    single sitting regardless of that day's actual agenda.
    """
    # raw_text is built as "{title}\n\n{body...}" by every source module.
    split_at = item.raw_text.find("\n\n")
    body = item.raw_text[split_at + 2 :] if split_at != -1 else ""

    for kw in KEYWORDS:
        if kw in body.lower():
            return kw, _snippet_around(body, kw)

    for kw in KEYWORDS:
        if kw in item.raw_text.lower():
            return kw, _snippet_around(item.raw_text, kw)

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
        match = _keyword_match(item)
        if match:
            candidates.append((item, *match))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return [
            Classification(
                item=item,
                relevant=True,
                confidence=0.5,
                reason=f'"{snippet}" — atrasta atslēgvārda "{kw}" sakarā (bez ANTHROPIC_API_KEY, atslēgvārdu režīms)',
                category="other",
            )
            for item, kw, snippet in candidates
        ]

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    results: list[Classification] = []
    candidate_items = [item for item, _, _ in candidates]
    for start in range(0, len(candidate_items), BATCH_SIZE):
        batch = candidate_items[start : start + BATCH_SIZE]
        results.extend(_classify_batch_with_llm(client, batch))

    return [r for r in results if r.relevant]
