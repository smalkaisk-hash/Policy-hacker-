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
                    # Field order here is deliberate and load-bearing, not cosmetic: Claude
                    # fills tool-call fields in the order they're declared, so "reason" must
                    # come before "relevant"/"confidence" — otherwise the model commits to a
                    # verdict first and writes "reason" afterward as a post-hoc rationalization
                    # for whatever it already decided, instead of the verdict actually
                    # following from the stated evidence. "category" comes first too, as a
                    # cheap warm-up classification before the harder relevance call.
                    "properties": {
                        "index": {"type": "integer"},
                        "category": {
                            "type": "string",
                            "enum": ["funding", "regulation", "tax_labor", "digitalization_innovation", "other"],
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "One short sentence IN LATVIAN, written BEFORE you decide "
                                "'relevant' below, not after. Must quote or closely paraphrase "
                                "the specific fact in the item's own text (the eligibility "
                                "criterion, size/stage restriction, amount, or mechanism — or "
                                "the lack of one) that your verdict follows from. If you can't "
                                "point to such a fact, that itself is the reason: say so, and "
                                "the verdict below must be NOT relevant."
                            ),
                        },
                        "relevant": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["index", "category", "reason", "relevant", "confidence"],
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
  programs, investment/venture capital initiatives, LIAA/Altum programs) — but only when the
  program's own eligibility is actually scoped to startups/SMEs/early-stage companies (a size,
  turnover, or company-age cap; a jaunuzņēmumi/MVU-branded program). An Altum or LIAA loan or
  grant open to "Latvijas uzņēmumi" in general, with no size or stage restriction, is NOT
  relevant merely because Altum/LIAA are institutions that also run SME programs elsewhere —
  and a news item about one specific company receiving such general-eligibility financing is
  NOT relevant unless the item's own text says that company is a startup/SME (its size, age,
  or explicit SME/jaunuzņēmums framing), not just that it's "a Latvian company"
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

Write the reason first and let the verdict follow from it — never decide "relevant" first and
then compose a reason to justify what you already decided. If, while writing the reason, you
find you're describing the institution/program/sector in general rather than something this
specific item's own text establishes, that's a sign to reconsider before answering.

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
    if len(item.raw_text) > DETAIL_CHAR_LIMIT:
        # Silent truncation here is dangerous specifically because eligibility/scope wording
        # ("paredzēts maziem un vidējiem uzņēmumiem", size caps, etc.) is often in the last
        # paragraph of a press release — cut it and the model has no way to know it's missing,
        # so it can't even hedge. Printing this makes a wrong verdict traceable to "didn't see
        # the whole thing" instead of looking identical to "saw it all and judged wrong".
        print(
            f"  ! {item.source}: article text truncated to {DETAIL_CHAR_LIMIT} chars for "
            f"classification ({item.title[:60]!r}) — content past this point wasn't seen"
        )
    return item.raw_text[:DETAIL_CHAR_LIMIT]


def _classify_batch_with_llm(client, batch: list[Item]) -> tuple[list[Classification], list[Item]]:
    """Returns (classifications, unclassified_items) — `unclassified_items` is every item
    in `batch` that got no real verdict (API call failed outright, or the model's response
    didn't include a usable result for it). Callers must not treat those as "processed":
    see the `seen`-tracking note in classify_items and run_digest.py's main()."""
    prompt_items = "\n\n".join(
        f"[{i}] Source: {it.source}\nTitle: {it.title}\nDetails: {_item_detail_text(it)}"
        for i, it in enumerate(batch)
    )
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_items"},
            messages=[{"role": "user", "content": f"Classify these {len(batch)} items:\n\n{prompt_items}"}],
        )
    except Exception as exc:
        # A transient API failure (rate limit, timeout, overload) here must not take down
        # the whole run — every requests.get() call elsewhere in this codebase already
        # degrades gracefully on failure; this LLM call should too. Worst case, this batch
        # of items is missing from this run's digest instead of the entire digest being
        # lost (including every already-classified batch before this one).
        print(f"  ! classification call failed for a batch of {len(batch)} item(s) ({exc}) — skipped, will retry next run")
        return [], list(batch)

    for block in message.content:
        if block.type == "tool_use":
            # .get(..., []) only substitutes the default when the key is MISSING — a
            # response with the key present but explicitly null ("results": null) still
            # comes back as None here and crashes enumerate() below. `or []` covers both.
            results = block.input.get("results", []) or []
            # Defensive: don't let a malformed batch (seen in practice: the whole batch's
            # "index" fields came back as numeric strings, e.g. "0" instead of 0, so a
            # strict isinstance(..., int) check silently dropped all 10 items) crash the
            # run or vanish items with no trace. Prefer the model's own "index" (coerced
            # from a numeric string if needed). Order isn't guaranteed to match the
            # request, so an entry with a missing/unusable index is skipped rather than
            # guessed from array position — a wrong guess would silently overwrite a
            # different item's real result with no trace at all.
            by_index: dict[int, dict] = {}
            for pos, r in enumerate(results):
                if not isinstance(r, dict):
                    print(f"  ! classification result at position {pos} is not an object ({r!r}) — skipped")
                    continue
                idx = r.get("index")
                if isinstance(idx, str) and idx.strip().lstrip("-").isdigit():
                    idx = int(idx)
                if not isinstance(idx, int) or isinstance(idx, bool):
                    print(f"  ! classification result at position {pos} has missing/invalid 'index' ({r.get('index')!r}) — skipped")
                    continue
                if idx in by_index:
                    print(f"  ! classification result at position {pos} has duplicate 'index' {idx} — overwriting the earlier result for it")
                by_index[idx] = r
            out = []
            unclassified = []
            for i, it in enumerate(batch):
                r = by_index.get(i)
                if r is None:
                    print(f"  ! no classification result for batch item {i} ({it.title[:60]!r}) — skipped")
                    unclassified.append(it)
                    continue
                # `r.get(key, default)` only substitutes the default for a MISSING key —
                # "reason": null (key present, value explicitly null) still comes back as
                # bare None, which crashes digest.py's rendering (str.split() on None) at
                # the very last step of the whole pipeline, after every other stage already
                # succeeded. Treat "present but null" the same as "missing": fall back to
                # the same default, and warn either way.
                if r.get("relevant") is None:
                    print(f"  ! classification result for batch item {i} ({it.title[:60]!r}) missing/null 'relevant' — treating as NOT relevant")
                if r.get("reason") is None:
                    print(f"  ! classification result for batch item {i} ({it.title[:60]!r}) missing/null 'reason'")
                out.append(
                    Classification(
                        item=it,
                        relevant=r.get("relevant") if r.get("relevant") is not None else False,
                        confidence=r.get("confidence") if r.get("confidence") is not None else 0.5,
                        reason=r.get("reason") if r.get("reason") is not None else "",
                        category=r.get("category") if r.get("category") is not None else "other",
                    )
                )
            return out, unclassified
    return [], list(batch)


def classify_items(items: list[Item]) -> tuple[list[Classification], list[Item]]:
    """Returns (relevant_classifications, unclassified_items). `unclassified_items` is
    every item that got no real verdict this run (a batch's API call failed, or the
    model's response didn't cover it) — callers must NOT mark those as "seen"/processed,
    or a transient API hiccup would silently and permanently drop them from all future
    runs instead of just this one.
    """
    # .strip(): a stray trailing newline/space from copy-pasting the key (e.g. into a
    # GitHub Actions secret) makes it an illegal HTTP header value and breaks every API
    # call with a confusing httpcore/httpx error — not what "no key set" should mean.
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        # No model available to judge relevance from context, so fall back to a keyword
        # pre-filter — crude, but keeps the prototype runnable with zero external deps.
        # Every item is actually examined here (unlike an LLM batch failure), so nothing
        # is "unclassified".
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
        ], []

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    results: list[Classification] = []
    unclassified: list[Item] = []
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        batch_results, batch_unclassified = _classify_batch_with_llm(client, batch)
        results.extend(batch_results)
        unclassified.extend(batch_unclassified)

    return [r for r in results if r.relevant], unclassified
