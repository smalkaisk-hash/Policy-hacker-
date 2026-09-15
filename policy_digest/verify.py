"""Deep-verifies legislative items against their actual primary legal text.

Motivated by a real false positive (2026-09-16, see CLAUDE.md): the classifier's only view
of a Saeima committee agenda item is whatever the scraper already fetched — sometimes just a
bare routing list of bill titles ("Grozījumi Elektronisko sakaru likumā", a document number,
a committee name), with no sentence anywhere describing what the bill actually does. A
cybercrime-convention ratification bill got a fabricated "this affects startup-relevant
digital services" verdict from exactly that gap — the model reached for background knowledge
of what a law with that name generically might cover, instead of what the text said, because
there was nothing else to go on.

This module gives the model live `web_search` + `web_fetch` tools, restricted to official
Latvian government/legal domains, so it can go find and read the ACTUAL bill/law text (likumi.lv,
Saeima's likumprojekti pages, the TAP portal) using the title and document/bill number already
scraped — not just re-read the same shallow agenda blurb that caused the original failure — and
confirm or reject the startup/SME scope claim against that real text. Only runs for items
classify.py already marked relevant=True, from the three government-decision/law sources (TAP
portāls, Saeima committees, the two MK/VSS meeting feeds) — the news/funding sources (Ekonomikas
ministrija, LIAA, Altum) are already full-article press-release text with no "was this actually
enacted with this scope" ambiguity for a primary-source lookup to resolve.

Server tools (web_search/web_fetch) run inside Anthropic's infrastructure — no client-side
tool-execution loop needed here, same shape as classify.py's single-call batches: one
`client.messages.create()` per item, the model does its own search/fetch rounds internally,
and the final content block is the structured `record_verification` tool call this module reads.

Cost/latency note (this is the expensive path in the pipeline, by design — see the
conversation this module was built from): each call involves real network round-trips on
Anthropic's side, so budget several seconds to a low tens-of-seconds per item, not the
sub-second batched classify.py call. Only relevant=True legislative items go through it —
typically a small fraction of a run's total candidates — so this stays bounded even though
each individual call is much pricier than a classify.py batch entry.
"""

import os
from dataclasses import replace

from .classify import Classification
from .sources.base import Item

MODEL = "claude-sonnet-5"

# mk_meetings.py takes its display name as a caller-supplied parameter (see run_digest.py's
# SOURCES list) rather than exporting a constant like the other two sources do, so these two
# strings are duplicated from there — keep in sync if either changes.
LEGISLATIVE_SOURCES = {
    "TAP portāls",
    "Saeimas komisiju darba kārtības",
    "Valsts sekretāru sanāksme",
    "Ministru kabineta protokoli",
}

# Official Latvian government/legal sources only — deliberately excludes news sites, law
# firm blogs, etc. A "primary source" that isn't actually the primary source defeats the
# entire point of this module.
ALLOWED_DOMAINS = [
    "likumi.lv",
    "saeima.lv",
    "titania.saeima.lv",
    "tapportals.mk.gov.lv",
    "data.gov.lv",
    "mk.gov.lv",
    "likumprojekti.lv",
    "vestnesis.lv",
]
MAX_TOOL_USES = 4

VERIFY_TOOL = {
    "name": "record_verification",
    "description": (
        "Record the result of verifying a legislative item's startup/SME relevance "
        "against its actual primary legal text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            # Field order is deliberate, same reasoning as classify.CLASSIFY_TOOL: the
            # model should commit to what it found and quote it BEFORE stating the
            # verdict, not decide first and rationalize after.
            "found_primary_source": {
                "type": "boolean",
                "description": "Whether you located and actually read the bill/law's own operative text (not just an agenda listing or a news summary of it).",
            },
            "source_url": {
                "type": "string",
                "description": "The URL of the primary source you read. Omit if found_primary_source is false.",
            },
            "evidence_quote": {
                "type": "string",
                "description": (
                    "IN LATVIAN. If found_primary_source is true: the exact clause from "
                    "the primary text stating (or ruling out) startup/SME-specific scope — "
                    "an eligibility criterion, size/turnover/age cap, or explicit "
                    "jaunuzņēmums/MVU framing. If found_primary_source is false: a short "
                    "note on what you searched and why you couldn't find or access the "
                    "actual text."
                ),
            },
            "verified_relevant": {
                "type": "boolean",
                "description": (
                    "True only if the primary text itself confirms startup/SME-specific "
                    "scope. If you could not find/read the primary text, or the primary "
                    "text doesn't confirm such scope, this must be false — never guess in "
                    "the item's favor."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One short sentence in Latvian summarizing the verdict, suitable for showing directly in the digest.",
            },
        },
        "required": ["found_primary_source", "verified_relevant", "reason"],
    },
}

SYSTEM_PROMPT = """You are fact-checking a Latvian government policy digest for startin.lv, a
startup accelerator. You are given one item that has already been provisionally marked
"relevant to startups" from a scraped agenda/document snippet — your job is to find that
item's ACTUAL primary legal text (not a news summary, not the agenda listing you're given —
the bill or law's own operative text) and confirm whether it really does what the provisional
reason claims.

Use web_search (restricted to official Latvian government/legal domains) to find the primary
source — try likumi.lv for an already-enacted law, the Saeima's own likumprojekti pages or
titania.saeima.lv for a bill in progress (search using the exact bill/document number if one
is given, e.g. "1477/Lp14" or "Dok. nr. 5169" — it's the most reliable search key), or
tapportals.mk.gov.lv for a TAP-stage draft. Then use web_fetch to actually read it.

A verdict of verified_relevant=true requires the primary text ITSELF to state a startup/SME-
specific scope: an eligibility criterion, a size/turnover/company-age cap, explicit
jaunuzņēmums/MVU/start-up framing, or (for legislation, not funding) a regulatory change whose
own text is specifically about startups/SMEs/tech companies rather than companies in general.
A law that applies identically to every company regardless of size is NOT relevant merely
because startups are companies too — same standard classify.py's SYSTEM_PROMPT already uses,
just checked here against the real text instead of a possibly-incomplete scraped snippet.

If you cannot find the primary source after a reasonable search, or you find it but it doesn't
confirm startup/SME-specific scope, verified_relevant MUST be false — never let an unconfirmed
claim through just because the provisional reason sounded plausible. The whole point of this
check is that a plausible-sounding claim was already shown to be wrong once (see the
cybercrime-convention case this module exists because of): only what the primary text actually
says counts as evidence, not what a law with this name would typically be expected to cover.

When you're done researching, call record_verification exactly once with your findings."""


def _verify_one(client, c: Classification) -> Classification:
    item: Item = c.item
    prompt = (
        f"Source: {item.source}\nTitle: {item.title}\nDate: {item.date}\n"
        f"Already-scraped URL (may itself be the primary source, or may just be an agenda "
        f"page — check): {item.url}\n"
        f"Already-scraped text:\n{item.raw_text[:2000]}\n\n"
        f"Provisional classification reason (from a possibly-incomplete scraped snippet, "
        f"NOT yet confirmed against the real text): {c.reason}\n\n"
        "Find and read this item's actual primary legal text, then call record_verification."
    )
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": MAX_TOOL_USES,
                    "allowed_domains": ALLOWED_DOMAINS,
                },
                {
                    "type": "web_fetch_20260209",
                    "name": "web_fetch",
                    "max_uses": MAX_TOOL_USES,
                    "allowed_domains": ALLOWED_DOMAINS,
                },
                VERIFY_TOOL,
            ],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        # Same graceful-degradation rule as everywhere else in this codebase (classify.py's
        # API-failure handling, every sources/*.py fetch): a transient failure on this one
        # item must not take down the run or the other items' verification. Fail CLOSED
        # (hold back), not open — an item this module couldn't check gets no benefit of the
        # doubt, since the entire purpose of this pass is not letting an unconfirmed claim
        # reach the digest. See classify.MIN_CONFIDENCE_TO_INCLUDE for the same philosophy.
        print(f"  ! verification call failed for {item.title[:60]!r} ({exc}) — held back")
        return replace(c, relevant=False, reason=f'[Atcelts — padziļinātā pārbaude neizdevās: "{c.reason}"]')

    result = None
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "record_verification":
            result = block.input
            break

    if not isinstance(result, dict):
        print(f"  ! verification call for {item.title[:60]!r} produced no usable result — held back")
        return replace(c, relevant=False, reason=f'[Atcelts — padziļinātā pārbaude neizdevās: "{c.reason}"]')

    verified = result.get("verified_relevant") is True
    verify_reason = result.get("reason") if isinstance(result.get("reason"), str) else c.reason
    source_url = result.get("source_url") if isinstance(result.get("source_url"), str) else None
    quote = result.get("evidence_quote") if isinstance(result.get("evidence_quote"), str) else None

    if not verified:
        print(f"  ! overriding relevant=True to False for {item.title[:60]!r} — deep verification against primary source did not confirm startup/SME scope")
        return replace(
            c,
            relevant=False,
            reason=f'[Atcelts — padziļinātā pārbaude neapstiprināja jaunuzņēmumu/MVU tvērumu: "{verify_reason}"]',
        )

    return replace(c, reason=verify_reason, verification_url=source_url, verification_quote=quote)


def verify_legislative_items(classifications: list[Classification]) -> list[Classification]:
    """Runs the deep-verification pass over every relevant=True item from a legislative
    source, leaving every other item (non-legislative sources, or already relevant=False)
    untouched. Degrades gracefully with no ANTHROPIC_API_KEY — same as classify.py's
    keyword-fallback mode, this feature is simply skipped rather than crashing the run,
    since there's no model available to do the research.
    """
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        print("  ! ANTHROPIC_API_KEY not set — skipping deep verification of legislative items")
        return classifications

    to_verify = [c for c in classifications if c.relevant and c.item.source in LEGISLATIVE_SOURCES]
    if not to_verify:
        return classifications

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    verified_by_url = {}
    for c in to_verify:
        verified_by_url[c.item.url] = _verify_one(client, c)

    return [verified_by_url.get(c.item.url, c) for c in classifications]
