"""Decides which fetched items are "startup-relevant".

Every fetched item is sent to Claude Haiku, in small batches, so relevance is judged from
what the item's actual text says — not from whether it happens to contain a keyword. A
keyword hit (e.g. "finansējums", "es fondi") is common in routine, unrelated administrative
and social-policy items too, so gating on keywords either drops genuinely relevant items
phrased differently, or lets clearly unrelated ones through because of an incidental,
boilerplate-sounding mention. The system prompt requires the model's one-line reason to be
grounded in what the item's own text actually says, not an assumed or generic connection.

Without an ANTHROPIC_API_KEY, there's no model to make that judgment call, so this falls back
to a crude keyword pre-filter (relevant=True, reason=the actual sentence the keyword was found
in, quoted from the article/document body) — good enough to keep the prototype runnable with
zero external dependencies, but not a substitute for the real classification above.
"""

import json
import os
import re
from dataclasses import dataclass

from .sources.base import Item

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 10
# Matches the char cap sources already truncate full document/article text to (see
# tap_legal_acts.FULL_TEXT_CHAR_LIMIT) — high enough that the model sees the actual
# substance of an item, not just its title.
DETAIL_CHAR_LIMIT = 4000
# Some sources (esp. mk_meetings' protocol pages) sometimes only expose a short procedural
# line ("accepted the draft, forwarded for signing") rather than the regulation's actual
# content — below this length there's nothing substantive to judge from, so we flag it as
# such rather than let the model quietly guess from the title alone.
MIN_SUBSTANTIVE_BODY_LEN = 150
NO_BODY_MARKER = (
    "[SATURS NAV PIEEJAMS ŠIM IERAKSTAM — pieejams tikai nosaukums (un, iespējams, īss "
    "procesuāls ieraksts, piem. 'pieņemts, virzīt parakstīšanai', kas neko neatklāj par "
    "satura būtību). NEIZDOMĀ un NEPIEŅEM, ko šis akts varētu regulēt — ja nosaukums pats "
    "par sevi nepārprotami nenorāda uz jaunuzņēmumu/MVU atbalstu vai regulējumu, atzīmē "
    "NOT relevant.]"
)

# Definition of "startup-relevant" for this digest (see README for the full writeup):
# funding & support programs, regulatory/legal/tax changes affecting startups or SMEs,
# and government initiatives on innovation, digitalization or entrepreneurship.
#
# Used only by the no-API-key fallback path below — the real classification path sends
# every item to the LLM and doesn't consult this list.
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
healthcare staffing, road maintenance). This includes general labor-market/social programs —
e.g. requalification or activation measures for the unemployed, youth not in employment, or
economically inactive people — even when the source text tacks on a generic line about
supporting "entrepreneurship" or the "startup ecosystem": that connection only counts if the
item's actual substance (funding mechanism, eligibility, regulatory change) is about startups
or SMEs specifically, not general jobseekers or self-employment in general.

Base your verdict strictly on what the item's own text says, not on what a program like this
could plausibly also help with. If you have to reach or infer a startup connection the text
itself doesn't make, mark it NOT relevant.

Reject category-level truisms as a basis for relevance — "startups are companies, and this
affects companies" or "startups need X, and this concerns X in general" is not a real
connection. A law, fee, or duty that applies identically to every company regardless of size
or stage (e.g. a universal per-employee levy, a standard filing fee) is NOT relevant just
because startups happen to be companies too — it would need to say something specific about
early-stage/small companies, not merely apply to "uzņēmumi" as a category. Likewise, a
government agency's routine budget approval is NOT relevant merely because the agency's
general mandate (e.g. patents, standards, licensing) is something startups also rely on —
approving next year's line-item budget doesn't change what startups can or can't do. Watch
for hedge words in your own reasoning ("could potentially", "may affect", "is relevant to
X in general") — if that's the strongest connection you can state, the honest verdict is NOT
relevant, not relevant-with-caveats.

Your one-line reason must point to something specific and concrete in the item's own text (the
actual mechanism, amount, eligibility criterion, or clause) — never a generic assertion like
"this is important for the startup ecosystem" with nothing in the item to back it up.

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


def _body_text(item: Item) -> str:
    """Everything in raw_text after the title — raw_text is built as "{title}\n\n{body...}"
    by every source module."""
    split_at = item.raw_text.find("\n\n")
    return item.raw_text[split_at + 2 :] if split_at != -1 else ""


def _keyword_match(item: Item) -> tuple[str, str] | None:
    """Returns (keyword, context_snippet) — the snippet is quoted straight from the
    fetched article/document body so the reason reflects what the text actually says,
    not just that a word appeared somewhere. Body content is preferred over the title:
    e.g. a standing committee named "...(nodokļu)..." would otherwise "match" on every
    single sitting regardless of that day's actual agenda.
    """
    body = _body_text(item)

    for kw in KEYWORDS:
        if kw in body.lower():
            return kw, _snippet_around(body, kw)

    for kw in KEYWORDS:
        if kw in item.raw_text.lower():
            return kw, _snippet_around(item.raw_text, kw)

    return None


def _item_detail_text(item: Item) -> str:
    body = _body_text(item)
    if len(body) < MIN_SUBSTANTIVE_BODY_LEN:
        return NO_BODY_MARKER
    return item.raw_text[:DETAIL_CHAR_LIMIT]


def _classify_batch_with_llm(client, batch: list[Item]) -> list[Classification]:
    prompt_items = "\n\n".join(
        f"[{i}] Source: {it.source}\nTitle: {it.title}\nDetails: {_item_detail_text(it)}"
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
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # No model available to judge relevance from context, so fall back to a keyword
        # pre-filter — crude, but keeps the prototype runnable with zero external deps.
        candidates = [(item, *m) for item in items if (m := _keyword_match(item))]
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
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        results.extend(_classify_batch_with_llm(client, batch))

    return [r for r in results if r.relevant]
