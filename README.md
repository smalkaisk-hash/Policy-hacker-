# Policy Hacker — Task 1: Policy Monitoring Digest

A prototype that replaces manual checking of Latvian government/policy sources with a
weekly digest of only the items that are actually relevant to startups.

## Sources covered

| Source | Required? | How it's fetched |
|---|---|---|
| **TAP portāls** | Mandatory | Official [open dataset on data.gov.lv](https://data.gov.lv/dati/lv/dataset/tap-publicetie-tiesibu-akti) — clean, no-auth, CC0 JSON, updated daily. No scraping needed. |
| Valsts sekretāru sanāksme | ✓ | Scrapes `tapportals.mk.gov.lv/meetings/state_secretaries`, following each meeting's protocol to list the actual agenda items decided. |
| Ministru kabineta protokoli | ✓ | Same mechanism as above, `tapportals.mk.gov.lv/meetings/cabinet_ministers`. |
| Ekonomikas ministrija | ✓ | Scrapes the `em.gov.lv/lv/jaunumi` news listing. |
| LIAA | ✓ | Scrapes the `liaa.gov.lv/lv/jaunumi` news listing (same CMS template as EM). |
| Saeimas komisiju darba kārtības | Not implemented | No unified feed found — each committee runs its own mini-site, and the one
  lead that looked like a combined listing (`titania.saeima.lv`, an old IBM Domino system) didn't
  return parseable content on a first attempt. Skipped given the time budget; see "What's next" below. |
| Altum | Not implemented | Not selected as one of the 5 target sources for this prototype (same news-listing
  pattern as EM/LIAA would very likely work — straightforward to add). |

This clears the task's minimum bar (≥3 sources, TAP portāls included) with room to spare —
**5 of 7 sources implemented**.

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
fetch (5 sources) → dedupe against local state → classify for relevance → render digest
```

1. **Fetch** — one module per source under `policy_digest/sources/`, each returning a normalized
   `Item(source, title, url, date, summary, raw_text)`.
2. **Dedupe** — `policy_digest/state.py` keeps `output/state.json`, a flat list of previously seen
   item URLs, so re-running only surfaces genuinely new items (this is what actually saves the
   "5 hours/week of manual checking").
3. **Classify** — `policy_digest/classify.py`:
   - a free keyword pre-filter cuts obviously unrelated items first;
   - if `ANTHROPIC_API_KEY` is set, surviving candidates go to **Claude Haiku** in small
     batches (structured tool-use call) for a real relevance judgment, confidence, one-line
     reason, and category;
   - without a key, the keyword-filtered items are used directly (cruder, but keeps the tool
     runnable with zero paid dependencies).
4. **Render** — `policy_digest/digest.py` writes a digest grouped by source, both as
   `output/digest_<date>.md` and a standalone `output/digest_<date>.html`.

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

Set `ANTHROPIC_API_KEY` in your environment (or `.env`, loaded by whatever runs the script) to
get real LLM classification; omit it to run in free keyword-only mode. Output lands in
`output/digest_<today>.md` and `output/digest_<today>.html` (gitignored — a sample run is
checked into [`sample_digest/`](sample_digest/) instead).

## What's next (week 1 → month 3)

- **Week 1**: run this as-is via a scheduled GitHub Actions workflow (free) that posts the
  digest to a Slack channel via an incoming webhook — lowest friction, no server to maintain.
- **Month 3**: add Saeima (once a real per-committee or Domino-based feed is worked out) and
  Altum (same scraper as EM/LIAA); move dedupe state from a JSON file to SQLite; add email
  delivery (Resend/Postmark/SMTP) alongside Slack for non-technical stakeholders; tune the
  keyword list and classifier prompt against a few weeks of real flagged/skipped items.
