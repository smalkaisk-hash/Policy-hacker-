#!/usr/bin/env python
"""Policy monitoring digest — CLI entrypoint.

Fetches recent items from TAP portāls, VSS, MK protocols, Ekonomikas
ministrija, LIAA, Altum, and Saeima committee sittings, classifies them for
startup relevance, and writes a Markdown + HTML digest to output/.

Usage:
    python run_digest.py [--days 7] [--no-state]
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env file, if present

from policy_digest.classify import classify_items
from policy_digest.dedupe import dedupe_items, dedupe_semantic
from policy_digest.digest import render_html, render_markdown
from policy_digest.sources import (
    fetch_altum_news,
    fetch_mk_meetings,
    fetch_news_listing,
    fetch_saeima_committees,
    fetch_tap_legal_acts,
)
from policy_digest.state import load_seen, save_seen

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
STATE_PATH = OUTPUT_DIR / "state.json"

SOURCES = [
    ("TAP portāls", lambda since, seen: fetch_tap_legal_acts(since, seen=seen)),
    (
        "Valsts sekretāru sanāksme",
        lambda since, seen: fetch_mk_meetings(
            "state_secretaries", "Valsts sekretāru sanāksme", since, seen=seen
        ),
    ),
    (
        "Ministru kabineta protokoli",
        lambda since, seen: fetch_mk_meetings(
            "cabinet_ministers", "Ministru kabineta protokoli", since, seen=seen
        ),
    ),
    (
        "Ekonomikas ministrija",
        lambda since, seen: fetch_news_listing(
            "https://www.em.gov.lv", "Ekonomikas ministrija", since, seen=seen
        ),
    ),
    (
        "LIAA",
        lambda since, seen: fetch_news_listing("https://www.liaa.gov.lv", "LIAA", since, seen=seen),
    ),
    ("Altum", lambda since, seen: fetch_altum_news(since, seen=seen)),
    (
        "Saeimas komisiju darba kārtības",
        lambda since, seen: fetch_saeima_committees(since, seen=seen),
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="Fetch, classify, and digest Latvian policy sources for startup relevance."
    )
    parser.add_argument("--days", type=int, default=7, help="How many days back to look (default: 7).")
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Ignore and don't update the dedupe state file (useful for demos/repeated testing).",
    )
    args = parser.parse_args()

    today = date.today()
    since = today - timedelta(days=args.days)

    seen = set() if args.no_state else load_seen(STATE_PATH)

    all_new_items = []
    for name, fetch_fn in SOURCES:
        print(f"Fetching {name}...")
        try:
            items = fetch_fn(since, seen)
        except Exception as exc:  # keep going even if one source is temporarily down
            print(f"  ! failed to fetch {name}: {exc}")
            continue
        new_items = [it for it in items if it.url not in seen]
        print(f"  {len(items)} item(s) examined, {len(new_items)} new since last run")
        all_new_items.extend(new_items)

    deduped_items = dedupe_items(all_new_items)
    if len(deduped_items) < len(all_new_items):
        merged_count = len(all_new_items) - len(deduped_items)
        print(f"Merged {merged_count} near-identical cross-source duplicate(s).")

    before_semantic = len(deduped_items)
    deduped_items = dedupe_semantic(deduped_items)
    if len(deduped_items) < before_semantic:
        print(
            f"Merged {before_semantic - len(deduped_items)} more duplicate(s) — same event, "
            "independently written articles (LLM pass)."
        )

    print(f"\nClassifying {len(deduped_items)} candidate item(s) for startup relevance...")
    classifications, unclassified_items = classify_items(deduped_items)
    print(f"{len(classifications)} flagged as relevant.")
    if unclassified_items:
        print(
            f"  ! {len(unclassified_items)} item(s) got no classification verdict this run "
            "(API failure) — will be retried next run, not marked as seen."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    md_path = OUTPUT_DIR / f"digest_{today.isoformat()}.md"
    html_path = OUTPUT_DIR / f"digest_{today.isoformat()}.html"
    md_path.write_text(render_markdown(classifications, since, today), encoding="utf-8")
    html_path.write_text(render_html(classifications, since, today), encoding="utf-8")

    print(f"\nDigest written to:\n  {md_path}\n  {html_path}")

    if not args.no_state:
        unclassified_urls = {it.url for it in unclassified_items}
        seen.update(it.url for it in all_new_items if it.url not in unclassified_urls)
        save_seen(STATE_PATH, seen)


if __name__ == "__main__":
    main()
