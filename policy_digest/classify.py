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
from datetime import date

from .sources.base import Item

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 10
# Caught in production 2026-09-15: even after tightening SYSTEM_PROMPT with explicit
# negative examples, a hedge-word/category-truism failure (e.g. "var ietekmēt...
# jaunuzņēmumus" with no concrete mechanism) kept recurring on items the examples didn't
# cover verbatim — LLM prompt instructions are probabilistic, not a guarantee. `confidence`
# correlates well with exactly these cases in practice (0.60-0.75 on confirmed-bad items
# vs. 0.92-0.98 on confirmed-good ones — see evals/classification_evals.py). Rather than
# surface a "verify manually" flag and lean on a human to catch what the classifier
# missed, an item below this bar is held back from the digest entirely — every item that
# DOES appear needs no second-guessing. The cost is recall (an occasional true positive
# with thin evidence goes unreported), which is the cheaper failure mode for this tool
# than a low-confidence claim reaching the digest and eroding trust in all of it. Only
# applies to the real LLM path below — the no-API-key keyword fallback's confidence=0.5
# is a flat placeholder, not a calibrated score, so filtering on it there would silence
# that mode entirely.
MIN_CONFIDENCE_TO_INCLUDE = 0.75
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

# Product decision 2026-09-16: relevance requires the literal word "startup" (or its
# Latvian form/synonym) somewhere in the item's OWN text — no SME/MVU carve-out, no
# VC/incubator/accelerator carve-out. Deliberately narrower than earlier revisions of this
# list: a program scoped to "mazie un vidējie uzņēmumi" that never says "jaunuzņēmums"
# does NOT count, and neither does a venture-capital-named program that never uses the
# word either — see the matching SYSTEM_PROMPT rewrite below. Applied globally to every
# classified item (any category, any source), not just funding items — see
# _has_explicit_startup_signal below.
EXPLICIT_STARTUP_KEYWORDS = ["jaunuzņēm", "starta uzņēm", "startup", "start-up"]


def _has_explicit_startup_signal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in EXPLICIT_STARTUP_KEYWORDS)

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
                        "deadline": {
                            "type": "string",
                            "description": (
                                "An explicit application/submission/public-consultation "
                                "deadline date stated in the item's own text, as ISO 8601 "
                                "(YYYY-MM-DD) — e.g. 'līdz 2026. gada 30. septembrim' becomes "
                                "'2026-09-30'. Only set this when the text states an actual "
                                "calendar date; never infer or guess one from context. Omit "
                                "this field entirely if no explicit deadline is stated."
                            ),
                        },
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

Mark an item RELEVANT only if it involves any of:
- funding or support programs FOR STARTUPS SPECIFICALLY (grants, EU funds,
  accelerator/incubator programs, investment/venture capital initiatives, LIAA/Altum
  programs) — the item's own text must itself use the word "jaunuzņēmums"/"jaunuzņēmumi"
  (or "starta uzņēmums"/the English "startup") to describe who the program/funding is for.
  Being scoped to SMEs ("mazie un vidējie uzņēmumi"/MVU) is NOT enough on its own — an
  SME-wide program that never says "jaunuzņēmums" is NOT relevant. Being a venture/risk
  capital ("riska kapitāls"/"iespējkapitāls") program is NOT enough on its own either — a VC
  fund that never says "jaunuzņēmums"/"startup" anywhere in its own text is NOT relevant,
  even though VC is usually startup-oriented in practice. A funding-discovery tool or
  wizard open to "companies of any size" is NOT relevant unless it too uses the word
  somewhere. A news item about one specific company receiving financing (e.g. an Altum/LIAA
  loan, an investment) is NOT relevant unless the item's own text calls that company a
  "jaunuzņēmums"/"startup" — "SME", "innovative company", "tech company", or "Latvian
  company" is not the same word and does not count
- legal or regulatory changes affecting startups specifically (company law, tax treatment
  incl. reinvested profit or employee stock options, labor law, digital services or AI
  regulation, public procurement rules relevant to tech vendors) — again, the item's own
  text must itself say "jaunuzņēmums"/"jaunuzņēmumi"/"startup" somewhere in connection with
  the change; a change that applies to companies or SMEs in general without that word is
  NOT relevant
- draft legislation or government initiatives on innovation, digitalization, or
  entrepreneurship that explicitly names startups ("jaunuzņēmums"/"jaunuzņēmumi"/"startup")
  as who it's for or about

Mark it NOT relevant if it's routine administrative/personnel/ceremonial business, or concerns a
sector with no plausible startup angle (e.g. agricultural subsidies unrelated to agtech,
healthcare staffing, road maintenance). This includes general labor-market/social programs —
e.g. requalification or activation measures for the unemployed, youth not in employment, or
economically inactive people — even when the source text tacks on a generic line about
supporting "entrepreneurship" or the "startup ecosystem": that connection only counts if the
item's actual substance (funding mechanism, eligibility, regulatory change) is about startups
specifically (using that literal word — see the hard requirement below), not general
jobseekers or self-employment in general.

Also NOT relevant: a delegation, trade mission, conference, or ceremonial event announcement
that merely mentions startups as one invited/attending category, with no funding mechanism,
eligibility criterion, or regulatory change of its own — "startups get to attend/participate"
is not the same as "this affects startups". And a youth/school entrepreneurship education
program (e.g. teaching schoolchildren to run a mock "skolēnu mācību uzņēmums") is NOT relevant
regardless of how much its own materials use the word "uzņēmējdarbība" — this digest is about
companies, not students role-playing as companies.

Events are not policy — this digest tracks government funding programs and regulatory
changes, not the calendar of contests, courses, and volunteer drives that agencies also run.
Mark these NOT relevant even when real money or a formal application is involved, because the
item itself is a one-off event, not a funding mechanism or regulatory change:
- a business-idea/innovation contest or award announcement (e.g. "Ideju Kauss", "Eksporta un
  inovācijas balva") — prize money for winning a competition is not a funding program with
  ongoing eligibility criteria, and a follow-up item reporting how many people/ideas registered
  for one is even further from being policy
- a call recruiting mentors, judges, or volunteers to support founders (e.g. "LIAA aicina
  pieredzējušus uzņēmējus kļūt par mentoriem") — this is staffing an advisory pool, not funding
  or regulating startups themselves
- a training/course cohort launch, enrollment milestone, or "programmas kārtas sākums"
  announcement for an education/mentorship offering (e.g. a Mini MBA cohort, a mentorship
  program's next phase starting) — a course has a start date and syllabus, not eligibility
  criteria for ongoing funding, even when it's EU-funded or free
This is distinct from an actual incubation/acceleration or financing program that provides
direct financial support with a stated amount/percentage and a formal application deadline
(e.g. "finansiāls atbalsts līdz 70%", a grant, a loan, a venture-capital investment) — that
remains relevant funding, not an event, even if it also includes mentorship as part of the
package.

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

Worked examples of this hedge-word failure — all three are NOT relevant, even though each is
real government business genuinely touching companies in general: a minister's meeting summary
noting an EU regulation "could affect SMEs and startups" with no bill text yet; a cybercrime
convention ratification bill justified only as "affecting digital service providers"; a media
law amendment justified only as "relevant to the digital services sector, including tech
startups". Each names a broad sector, not a mechanism specific to startups/SMEs — that is the
failure this section describes, not a real connection, no matter how plausible it sounds.

Be very specific for startups, not just "business" in general — this is the single most common
way an item wrongly passes. A funding program, grant competition, or investment scheme that is
open to companies of any size/age/turnover is NOT relevant just because startups are technically
eligible to apply like everyone else, and reasoning like "the program is oriented toward
companies with mature/high-readiness projects" or "this supports business development" is
describing ordinary commercial activity, not a startup-specific mechanism — do not let it read
as a size/stage cap when it isn't one. More worked examples of this exact failure, all NOT
relevant: an open project competition funding "high-readiness" dual-use/industrial R&D projects
for companies in general, with no startup/SME/early-stage eligibility criterion stated; a
regional development measure funding "private investment growth" or "territory revitalization"
based on local business needs, with no size/age scoping; a portal or system modernization for
submitting loan applications that explicitly serves "all [Altum/LIAA] clients", not a
startup-specific product; a news item about a specific company (however innovative-sounding,
e.g. an AI or defense-tech company) receiving investment or opening an R&D center, where the
item's own text never states that company's size, age, or an explicit startup/SME/jaunuzņēmums
label — "international company" or "tech company" is not "startup", and a vague closing line
like "this strengthens the innovation ecosystem" is the ecosystem-truism failure, not a real
connection to this specific item.

Hard requirement, no exceptions: if the item's own text never uses the word
"jaunuzņēmums"/"jaunuzņēmumi", "starta uzņēmums", or the English "startup"/"start-up"
anywhere in connection with the funding/regulation itself, the item is NOT relevant — full
stop, regardless of how startup-adjacent the program, institution, or sector sounds. There
are no carve-outs for SME scoping, VC/risk-capital framing, funding-discovery tools, or any
other inferred connection: this literal word is the only signal that counts. (This is
enforced in code as well as here, so guessing around it doesn't help — see
`classify._has_explicit_startup_signal`.)

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
    # ISO 8601 date string (YYYY-MM-DD), or None if the item states no explicit deadline
    # (application/submission/consultation-comment-period) — see _validate_deadline and
    # _extract_deadline_fallback below for how each classification path fills this in.
    deadline: str | None = None


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


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_deadline(raw, item: Item) -> str | None:
    """The LLM's 'deadline' field is untrusted input just like every other tool-use
    field in this file (see module docstring / CLAUDE.md) — it could come back in a
    format that isn't the ISO date we asked for, or not be a string at all. Never let
    a malformed deadline crash the run; just drop it and say why.
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or not _ISO_DATE_RE.match(raw):
        print(f"  ! classification result for {item.title[:60]!r} has malformed 'deadline' ({raw!r}) — ignored")
        return None
    try:
        date.fromisoformat(raw)
    except ValueError:
        print(f"  ! classification result for {item.title[:60]!r} has invalid 'deadline' date ({raw!r}) — ignored")
        return None
    return raw


# Latvian month names in the genitive case, as they appear in "līdz <year>. gada <day>.
# <month>" phrasing (the standard way a Latvian government text states a deadline date).
_LV_MONTHS_GENITIVE = {
    "janvārim": 1, "februārim": 2, "martam": 3, "aprīlim": 4, "maijam": 5,
    "jūnijam": 6, "jūlijam": 7, "augustam": 8, "septembrim": 9,
    "oktobrim": 10, "novembrim": 11, "decembrim": 12,
}
_DEADLINE_LONG_RE = re.compile(
    r"līdz\s+(\d{4})\.\s*gada\s+(\d{1,2})\.\s*(" + "|".join(_LV_MONTHS_GENITIVE) + r")",
    re.IGNORECASE,
)
_DEADLINE_NUMERIC_RE = re.compile(r"līdz\s+(\d{1,2})\.(\d{1,2})\.(\d{4})")
_DEADLINE_ISO_RE = re.compile(r"līdz\s+(\d{4})-(\d{2})-(\d{2})")


def _extract_deadline_fallback(text: str) -> str | None:
    """No-API-key path's equivalent of the LLM's 'deadline' extraction: a few common
    Latvian phrasings for "apply/comment by <date>". Best-effort only — unlike the LLM
    path this can't understand phrasing it wasn't explicitly written for, but it's free
    and catches the standard government-text date formats.
    """
    m = _DEADLINE_LONG_RE.search(text.lower())
    if m:
        year, day, month_name = m.group(1), m.group(2), m.group(3)
        month = _LV_MONTHS_GENITIVE.get(month_name)
        if month:
            try:
                return date(int(year), month, int(day)).isoformat()
            except ValueError:
                return None

    m = _DEADLINE_NUMERIC_RE.search(text)
    if m:
        day, month, year = m.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None

    m = _DEADLINE_ISO_RE.search(text)
    if m:
        year, month, day = m.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None

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


def _classify_batch_with_llm(
    client, batch: list[Item], _retry: bool = True
) -> tuple[list[Classification], list[Item]]:
    """Returns (classifications, unclassified_items) — `unclassified_items` is every item
    in `batch` that got no real verdict (API call failed outright, or the model's response
    didn't include a usable result for it). Callers must not treat those as "processed":
    see the `seen`-tracking note in classify_items and run_digest.py's main().

    `_retry` is internal (see the isinstance(results, str) handling below): a single
    automatic retry of the whole batch when the model's response is unusable, since that
    has been observed to be a one-off stochastic formatting quirk, not a deterministic
    bug — a fresh attempt at the same batch typically just works. Never recurses more
    than once (the recursive call itself passes _retry=False), so a batch that keeps
    failing costs at most 2 API calls, not an unbounded loop.
    """
    detail_texts = [_item_detail_text(it) for it in batch]
    prompt_items = "\n\n".join(
        f"[{i}] Source: {it.source}\nTitle: {it.title}\nDetails: {detail}"
        for i, (it, detail) in enumerate(zip(batch, detail_texts))
    )
    try:
        message = client.messages.create(
            model=MODEL,
            # 8192 (not the original 4096): cheap insurance against genuinely running out
            # of output budget on a large/verbose batch, though see the retry logic below
            # for the failure mode actually observed in practice (a formatting quirk, not
            # truncation) — 10 items x (category + one-sentence reason + relevant +
            # confidence + optional deadline) comfortably fits under either limit normally.
            max_tokens=8192,
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
            # Seen in practice on real batches of Latvian government titles (which are
            # themselves full of embedded quote marks, e.g. 'Noteikumu projekts
            # "Grozījumi..."'): the model sometimes double-encodes the whole "results"
            # array as a JSON *string* instead of a native array, and in doing so breaks
            # its own JSON by mishandling those embedded quotes — so the string doesn't
            # even parse back to valid JSON. Confirmed NOT a max_tokens budget problem
            # (raising max_tokens to 8192 above didn't stop it, and it happens well under
            # that limit) — it's a stochastic model formatting quirk, so retry the whole
            # batch once rather than just giving up on ~10 items at a time. Without this
            # recovery at all, enumerate() below would iterate the string
            # character-by-character — thousands of spurious "not an object" warnings.
            if isinstance(results, str):
                try:
                    parsed = json.loads(results)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, list):
                    results = parsed
                elif _retry:
                    print(
                        f"  ! batch's 'results' came back unusable ({len(results)} chars, "
                        f"stop_reason={message.stop_reason}) — retrying this batch once"
                    )
                    return _classify_batch_with_llm(client, batch, _retry=False)
                else:
                    print(
                        f"  ! batch's 'results' still unusable after retry ({len(results)} "
                        "chars) — skipped, will retry next run"
                    )
                    results = []
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
                relevant = r.get("relevant") if r.get("relevant") is not None else False
                reason = r.get("reason") if r.get("reason") is not None else ""
                # Deterministic backstop — product decision 2026-09-16: relevance requires
                # the literal word "jaunuzņēmums"/"starta uzņēmums"/"startup" somewhere in
                # the item's own text (title + body), no exceptions for SME scoping, VC
                # framing, or any other inferred connection. Checked against the full item
                # text so a real body's wording counts too, not just the title — this
                # supersedes and subsumes the old no-body-only check (a no-body item's
                # "text" is just its title anyway, so the outcome there is unchanged).
                # Applied to every category/source, not just funding items.
                if relevant and not _has_explicit_startup_signal(it.raw_text):
                    print(
                        f"  ! overriding relevant=True to False for {it.title[:60]!r} — "
                        "item's own text never uses the word jaunuzņēmums/starta uzņēmums/"
                        "startup (model's claim wasn't grounded in that literal signal)"
                    )
                    reason = (
                        f'[Atcelts — trūkst vārda "jaunuzņēmums"/"starta uzņēmums"/"startup": '
                        f'"{reason}"]'
                    )
                    relevant = False
                out.append(
                    Classification(
                        item=it,
                        relevant=relevant,
                        confidence=r.get("confidence") if r.get("confidence") is not None else 0.5,
                        reason=reason,
                        category=r.get("category") if r.get("category") is not None else "other",
                        deadline=_validate_deadline(r.get("deadline"), it),
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
                deadline=_extract_deadline_fallback(item.raw_text),
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

    relevant = [r for r in results if r.relevant]
    confident_enough = [r for r in relevant if r.confidence >= MIN_CONFIDENCE_TO_INCLUDE]
    held_back = len(relevant) - len(confident_enough)
    if held_back:
        print(
            f"  ! {held_back} item(s) marked relevant but held back — confidence below "
            f"{MIN_CONFIDENCE_TO_INCLUDE:.0%} (not shown without a stronger signal)"
        )
    return confident_enough, unclassified
