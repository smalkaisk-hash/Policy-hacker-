"""Tests for the 'every monitored source' coverage line in digest.py (2026-09-15) —
previously the digest only ever mentioned sources that happened to have a relevant hit
that run, so a source with zero matches looked indistinguishable from a source that
wasn't checked at all. Passing `all_sources` now adds a `Pārbaudītie avoti: ...` line
naming every monitored source with its relevant-item count, zero included.

Uses only the standard library, same as tests/test_reliability.py. Run with:

    python -m unittest discover tests
"""

import sys
import unittest
from datetime import date
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_digest.classify import Classification
from policy_digest.digest import render_html, render_markdown
from policy_digest.sources.base import Item

ALL_SOURCES = ["TAP portāls", "LIAA", "Altum"]


def _classification(source, title="T", category="funding"):
    item = Item(source=source, title=title, url=f"http://x/{source}/{title}", date="2026-09-10")
    return Classification(item=item, relevant=True, confidence=0.9, reason="r", category=category)


class SourcesCheckedLineTests(unittest.TestCase):
    def test_omitted_when_all_sources_not_passed(self):
        md = render_markdown([_classification("TAP portāls")], date(2026, 9, 1), date(2026, 9, 15))
        self.assertNotIn("Pārbaudītie avoti", md)

    def test_lists_every_source_including_zero_hits(self):
        md = render_markdown(
            [_classification("TAP portāls")], date(2026, 9, 1), date(2026, 9, 15), all_sources=ALL_SOURCES
        )
        self.assertIn("TAP portāls (1)", md)
        self.assertIn("LIAA (0)", md)
        self.assertIn("Altum (0)", md)

    def test_shown_even_when_no_items_are_relevant_at_all(self):
        md = render_markdown([], date(2026, 9, 1), date(2026, 9, 15), all_sources=ALL_SOURCES)
        self.assertIn("TAP portāls (0)", md)
        self.assertIn("LIAA (0)", md)
        self.assertIn("Altum (0)", md)

    def test_merged_cross_source_item_counts_toward_both_sources(self):
        # dedupe.py's _merge_cluster sets source to "A + B" for a merged item — the
        # coverage line must credit both original sources, not just the combined label.
        c = _classification("Ekonomikas ministrija + LIAA")
        md = render_markdown(
            [c], date(2026, 9, 1), date(2026, 9, 15),
            all_sources=["Ekonomikas ministrija", "LIAA", "Altum"],
        )
        self.assertIn("Ekonomikas ministrija (1)", md)
        self.assertIn("LIAA (1)", md)
        self.assertIn("Altum (0)", md)

    def test_html_render_also_includes_coverage_line(self):
        html = render_html(
            [_classification("TAP portāls")], date(2026, 9, 1), date(2026, 9, 15), all_sources=ALL_SOURCES
        )
        self.assertIn("Pārbaudītie avoti", html)
        self.assertIn("TAP portāls (1)", html)
        self.assertIn("Altum (0)", html)


if __name__ == "__main__":
    unittest.main()
