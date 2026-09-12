# Policy Hacker — Task 1: Policy Monitoring Digest

A prototype that replaces manual checking of Latvian government/policy sources with a
weekly digest of only the items that are actually relevant to startups.

## Sources covered

**All 7 listed sources are implemented** — well past the task's minimum bar (≥3 sources, TAP
portāls included). Every source scrapes the actual article/document body, not just the
headline — see "How it works" below for what's fetched per source.

| Source | How it's fetched |
|---|---|
| **TAP portāls** (mandatory) | Metadata from the official [open dataset on data.gov.lv](https://data.gov.lv/dati/lv/dataset/tap-publicetie-tiesibu-akti) (clean, no-auth, CC0 JSON, updated daily), plus the full draft decision/annotation text pulled from TAP's public "structuralizer" preview endpoint for each act. |
| Valsts sekretāru sanāksme | Scrapes `tapportals.mk.gov.lv/meetings/state_secretaries`, following each meeting's protocol page to list the individual agenda items *and* their full decision text. |
| Ministru kabineta protokoli | Same mechanism, `tapportals.mk.gov.lv/meetings/cabinet_ministers`. |
| Ekonomikas ministrija | Scrapes the `em.gov.lv/lv/jaunumi` listing, then fetches each article's own page for the full body. |
| LIAA | Same mechanism as EM (same underlying CMS template), `liaa.gov.lv/lv/jaunumi`. |
| Altum | Scrapes `altum.lv/par-altum/aktualitates/` (a different CMS than EM/LIAA, so it has its own scraper) and fetches each article's full body. Note: this listing page isn't paginated, so a lookback window longer than ~1 month may miss older items. |
| Saeimas komisiju darba kārtības | No unified feed exists on saeima.lv itself (each committee runs its own mini-site) — but its "Komisiju sēžu darba kārtības" link forwards to an internal Domino system (`titania.saeima.lv`) whose public view lists *all* committees' sittings for a given day, each with the full agenda text. Queried once per day in the lookback window. |

## What counts as "startup-relevant"

An item is flagged if it involves:
- **Funding & support programs** — grants, EU funds, accelerator/incubator programs, LIAA/Altum
  initiatives, investment/venture capital programs.
- **Regulatory or legal changes** affecting startups, SMEs, or tech companies — company law, tax
  treatment (incl. reinvested profit, employee stock options), labor law, digital services/AI
  regulation, public procurement rules relevant to tech vendors.
- **Draft legislation or government initiatives** on innovation, digitalization, or
  entrepreneurship.

Routine administrative/personnel/ceremonial items and unrelated sector regulation (e.g.
agriculture subsidies with no tech angle) are excluded. This definition is encoded directly in
[`policy_digest/classify.py`](policy_digest/classify.py) (`SYSTEM_PROMPT` and `KEYWORDS`).

## How it works

```
fetch (7 sources, full text) → dedupe against local state → classify for relevance → render digest
```

1. **Fetch** — one module per source under `policy_digest/sources/`, each returning a normalized
   `Item(source, title, url, date, summary, raw_text)`. `raw_text` is the actual article/document
   body where one exists (fetched from the item's own detail page, or already present on the page
   already being scraped — e.g. MK/VSS agenda item text and Saeima's full agenda sit right on the
   listing/detail pages already being requested, so no extra request is needed for those). This
   matters because classification is only as good as what it can see: a title alone is often too
   vague to judge.
2. **Dedupe** — `policy_digest/state.py` keeps `output/state.json`, a flat list of previously seen
   item URLs, so re-running only surfaces genuinely new items (this is what actually saves the
   "5 hours/week of manual checking"). This dedupe check happens *inside* the fetch step, not just
   after it: each fetcher is passed the seen-URL set and skips re-downloading an already-seen
   item's full article/document body (the expensive part), since something already published
   isn't going to change. Only the cheap listing/index page for each source is always re-checked,
   to discover what's actually new. One consequence: for VSS/MK protocols specifically, an
   already-fully-scraped meeting is skipped outright rather than re-parsed, so the "items
   examined" count printed for those two sources reflects work done *this run*, not a stable
   total — the dedupe/relevance results themselves aren't affected by this.
3. **Classify** — `policy_digest/classify.py`:
   - a free keyword pre-filter cuts obviously unrelated items first;
   - if `ANTHROPIC_API_KEY` is set, surviving candidates go to **Claude Haiku** in small
     batches (structured tool-use call) for a real relevance judgment, confidence, one-line
     reason, and category;
   - without a key, the keyword-filtered items are used directly, with the reason built from
     the actual matched text — the full *sentence* the keyword appears in (sentence-boundary
     aware, not a fixed character radius, so it reads cleanly instead of cutting off mid-word),
     quoted from the article/document body, preferring the body over the title (a committee
     literally named "...(nodokļu)..." would otherwise "match" on every single sitting
     regardless of that day's real agenda) — rather than a bare "keyword found" label. This is
     the ceiling for what's possible without an LLM: a real one-line "why this matters to
     startups" summary needs actual reading comprehension, which is exactly what the Claude
     Haiku stage below does once a key is supplied.
4. **Render** — `policy_digest/digest.py` writes a digest grouped by source, both as
   `output/digest_<date>.md` and a standalone `output/digest_<date>.html`. The HTML version is
   startin.lv-branded (logo from `assets/logobig.png`, embedded inline as base64 so the file
   stays self-contained/emailable) with color-coded category badges (funding, regulation,
   tax & labor, digital & innovation) for quick scanning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY (optional — see below)
```

Requires Python 3.10+.

## Run it

```bash
python run_digest.py                # last 7 days, updates dedupe state
python run_digest.py --days 14       # wider window
python run_digest.py --no-state      # don't read/write dedupe state (repeatable demo runs)
```

Set `ANTHROPIC_API_KEY` in `.env` (auto-loaded via `python-dotenv`) or your environment to get
real LLM classification; omit it to run in free keyword-only mode. Output lands in
`output/digest_<today>.md` and `output/digest_<today>.html` (gitignored — a sample run is
checked into [`sample_digest/`](sample_digest/) instead — that sample was generated **with**
`ANTHROPIC_API_KEY` set, i.e. the real Claude Haiku classification path, not the keyword
fallback).

## Known limitations

- **TAP portāls**: only the document version rendered by TAP's inline "structuralizer" preview is
  fetched; a version that's a plain file attachment (.docx) is skipped rather than downloaded and
  parsed. Most acts have at least one structuralizer-rendered version, but not all.
- **Altum**: its news listing page isn't paginated, so it only sees the ~12 most recent items —
  fine for a weekly run, not for a lookback window beyond about a month.
- **EM / LIAA syndication**: the same article is sometimes published on both sites verbatim and
  currently shows up twice (once per source) rather than being deduplicated across sources.
- **Saeima**: pulled from an internal-looking Domino endpoint reached only via a redirect from the
  public site — undocumented, so it could change without notice; no official API was found.

## What's next (week 1 → month 3)

- **Week 1**: run this as-is via a scheduled GitHub Actions workflow (free) that posts the
  digest to a Slack channel via an incoming webhook — lowest friction, no server to maintain.
- **Month 3**: dedupe near-identical items across sources (e.g. EM/LIAA syndication); move
  dedupe state from a JSON file to SQLite; add email delivery (Resend/Postmark/SMTP) alongside
  Slack for non-technical stakeholders; parse TAP's .docx attachments for the acts that don't
  have a structuralizer preview; tune the keyword list and classifier prompt against a few weeks
  of real flagged/skipped items.
