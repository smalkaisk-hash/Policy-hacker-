# Policy Hacker — Task 1: Policy Monitoring Digest

A prototype that replaces manual checking of Latvian government/policy sources with a
weekly digest of only the items that are actually relevant to startups.

## Sources covered

All 7 listed sources are implemented (past the task's minimum of 3, TAP portāls included).
Every source pulls the full article/document body, not just the headline.

| Source | How it's fetched |
|---|---|
| **TAP portāls** (mandatory) | [Open dataset on data.gov.lv](https://data.gov.lv/dati/lv/dataset/tap-publicetie-tiesibu-akti) for metadata, plus full act text via TAP's public "structuralizer" preview endpoint. |
| Valsts sekretāru sanāksme | `tapportals.mk.gov.lv/meetings/state_secretaries` — full agenda item text. |
| Ministru kabineta protokoli | Same mechanism, `tapportals.mk.gov.lv/meetings/cabinet_ministers`. |
| Ekonomikas ministrija | `em.gov.lv/lv/jaunumi` listing + each article's full body. |
| LIAA | Same CMS as EM, `liaa.gov.lv/lv/jaunumi`. |
| Altum | `altum.lv/par-altum/aktualitates/` + each article's full body. |
| Saeimas komisiju darba kārtības | Public agenda feed on `titania.saeima.lv` (reached via saeima.lv's own committee-agenda link), all committees' sittings per day. |

## What counts as "startup-relevant"

An item is flagged if it involves:
- **Funding & support programs** — grants, EU funds, accelerator/incubator programs,
  LIAA/Altum initiatives, investment/VC programs.
- **Regulatory or legal changes** affecting startups, SMEs, or tech companies — company law,
  tax treatment, labor law, digital services/AI regulation, procurement rules for tech vendors.
- **Draft legislation or government initiatives** on innovation, digitalization, or
  entrepreneurship.

Routine administrative/personnel/ceremonial items and unrelated sector regulation are
excluded. Definition lives in [`policy_digest/classify.py`](policy_digest/classify.py)
(`SYSTEM_PROMPT` and `KEYWORDS`).

## How it works

```
fetch (7 sources, full text) → dedupe against local state → classify for relevance → render digest
```

1. **Fetch** — one module per source under `policy_digest/sources/`, returning normalized
   items with the real article/document body attached, not just a title.
2. **Dedupe (seen-before)** — `policy_digest/state.py` tracks previously seen item URLs in
   `output/state.json`, so re-runs only surface genuinely new items.
3. **Classify** — `policy_digest/classify.py` sends every item to **Claude Haiku** for a
   relevance judgment, confidence, one-line reason, and category (falls back to a free
   keyword pre-filter if no `ANTHROPIC_API_KEY` is set).
4. **Dedupe (cross-source)** — `policy_digest/dedupe.py` merges the same underlying
   article/document when it's published on more than one source, first by text similarity
   then (with an API key) an LLM pass for independently-written pieces about the same event.
5. **Render** — `policy_digest/digest.py` writes a Latvian-language, startin.lv-branded
   digest as `output/digest_<date>.md` and `.html`.

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

Set `ANTHROPIC_API_KEY` in `.env` to get real LLM classification; omit it to run in free
keyword-only mode. Output lands in `output/digest_<today>.md` and `.html` (gitignored — a
sample run generated **with** an API key is checked into
[`sample_digest/`](sample_digest/)).

## Hosting a live version (GitHub Pages)

[`.github/workflows/digest.yml`](.github/workflows/digest.yml) runs the digest and publishes
it to GitHub Pages — free, no server. One-time setup in the repo's GitHub web UI:

1. **Add the API key as a secret**: Settings → Secrets and variables → Actions → New
   repository secret → name `ANTHROPIC_API_KEY`.
2. **Turn on Pages**: Settings → Pages → Build and deployment → Source: **GitHub Actions**.

After that, it publishes automatically every Monday, or on demand via the Actions tab →
"Publish policy digest" → Run workflow.
