"""Tests for two additions to the digest pipeline (2026-09-15):

1. Deadline extraction — classify.py's LLM path gets a 'deadline' tool-use field,
   and the no-API-key keyword path gets a regex-based fallback extractor. Both feed
   Classification.deadline, an ISO date string or None.
2. Funding-vs-regulatory supergrouping in digest.py — items are split into two
   top-level sections ("Finansējuma iespējas" / "Regulējums un iniciatīvas") before
   the existing per-source grouping, and items with a soon-due deadline are sorted
   first within their source group and flagged as urgent.

Uses only the standard library (unittest + unittest.mock), same as
tests/test_reliability.py. Run with:

    python -m unittest discover tests
"""

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_digest import classify
from policy_digest.classify import Classification
from policy_digest.digest import render_html, render_markdown
from policy_digest.sources.base import Item


class DeadlineFallbackExtractionTests(unittest.TestCase):
    """No-API-key path: regex-based extraction of common Latvian deadline phrasings."""

    def test_extracts_long_form_genitive_date(self):
        text = "Pieteikumus var iesniegt līdz 2026. gada 30. septembrim plkst. 17:00."
        self.assertEqual(classify._extract_deadline_fallback(text), "2026-09-30")

    def test_extracts_numeric_dotted_date(self):
        text = "Pieteikšanās termiņš ir līdz 30.09.2026."
        self.assertEqual(classify._extract_deadline_fallback(text), "2026-09-30")

    def test_extracts_iso_date(self):
        text = "Komentārus var iesniegt līdz 2026-09-30."
        self.assertEqual(classify._extract_deadline_fallback(text), "2026-09-30")

    def test_returns_none_when_no_deadline_present(self):
        text = "Šis ir vienkāršs paziņojums bez jebkāda termiņa."
        self.assertIsNone(classify._extract_deadline_fallback(text))

    def test_returns_none_on_invalid_calendar_date(self):
        # "31. februārim" doesn't exist — must not raise, just report "no deadline found".
        text = "Pieteikumus var iesniegt līdz 2026. gada 31. februārim."
        self.assertIsNone(classify._extract_deadline_fallback(text))


class DeadlineLlmValidationTests(unittest.TestCase):
    """The LLM's 'deadline' field is untrusted input like every other tool-use field in
    classify.py — a malformed value must be dropped (with a warning), never crash the
    batch or propagate a garbage string into the digest."""

    def _item(self):
        return Item(source="A", title="Item 0", url="http://a/0", date="2026-09-10",
                    raw_text="Item 0\n\nSome long enough body text for a real item.")

    def _run(self, input_dict):
        client = MagicMock()
        block = MagicMock()
        block.type = "tool_use"
        block.input = input_dict
        message = MagicMock()
        message.content = [block]
        client.messages.create.return_value = message
        return classify._classify_batch_with_llm(client, [self._item()])

    def test_valid_iso_deadline_passes_through(self):
        out, _ = self._run({"results": [
            {"index": 0, "category": "funding", "reason": "ok", "relevant": True,
             "confidence": 0.9, "deadline": "2026-09-30"}
        ]})
        self.assertEqual(out[0].deadline, "2026-09-30")

    def test_missing_deadline_field_defaults_to_none(self):
        out, _ = self._run({"results": [
            {"index": 0, "category": "other", "reason": "ok", "relevant": True, "confidence": 0.9}
        ]})
        self.assertIsNone(out[0].deadline)

    def test_malformed_deadline_string_is_dropped_not_crashed(self):
        out, _ = self._run({"results": [
            {"index": 0, "category": "other", "reason": "ok", "relevant": True,
             "confidence": 0.9, "deadline": "30 September 2026"}
        ]})
        self.assertIsNone(out[0].deadline)

    def test_deadline_wrong_type_is_dropped_not_crashed(self):
        out, _ = self._run({"results": [
            {"index": 0, "category": "other", "reason": "ok", "relevant": True,
             "confidence": 0.9, "deadline": 20260930}
        ]})
        self.assertIsNone(out[0].deadline)

    def test_invalid_calendar_date_is_dropped_not_crashed(self):
        out, _ = self._run({"results": [
            {"index": 0, "category": "other", "reason": "ok", "relevant": True,
             "confidence": 0.9, "deadline": "2026-13-45"}
        ]})
        self.assertIsNone(out[0].deadline)


def _classification(source, category, deadline=None, item_date="2026-09-10", title="T"):
    item = Item(source=source, title=title, url=f"http://x/{source}/{title}", date=item_date)
    return Classification(item=item, relevant=True, confidence=0.9, reason="r",
                           category=category, deadline=deadline)


class SupergroupingTests(unittest.TestCase):
    """Items must be split into a 'funding' section and an 'everything else' section,
    each only rendered when it actually has items, with per-source grouping preserved
    inside each section."""

    def test_only_funding_items_renders_just_the_funding_section(self):
        md = render_markdown([_classification("LIAA", "funding")], date(2026, 9, 1), date(2026, 9, 15))
        self.assertIn("Finansējuma iespējas", md)
        self.assertNotIn("Regulējums un iniciatīvas", md)

    def test_only_regulatory_items_renders_just_the_regulatory_section(self):
        md = render_markdown([_classification("TAP portāls", "regulation")], date(2026, 9, 1), date(2026, 9, 15))
        self.assertIn("Regulējums un iniciatīvas", md)
        self.assertNotIn("Finansējuma iespējas", md)

    def test_mixed_categories_render_both_sections(self):
        items = [_classification("LIAA", "funding"), _classification("TAP portāls", "tax_labor")]
        md = render_markdown(items, date(2026, 9, 1), date(2026, 9, 15))
        self.assertIn("Finansējuma iespējas", md)
        self.assertIn("Regulējums un iniciatīvas", md)

    def test_html_render_also_splits_into_both_sections(self):
        items = [_classification("LIAA", "funding"), _classification("TAP portāls", "digitalization_innovation")]
        html = render_html(items, date(2026, 9, 1), date(2026, 9, 15))
        self.assertIn("Finansējuma iespējas", html)
        self.assertIn("Regulējums un iniciatīvas", html)

    def test_empty_classifications_does_not_crash_and_shows_no_sections(self):
        md = render_markdown([], date(2026, 9, 1), date(2026, 9, 15))
        self.assertNotIn("Finansējuma iespējas", md)
        html = render_html([], date(2026, 9, 1), date(2026, 9, 15))
        self.assertNotIn("Finansējuma iespējas", html)


class DeadlineUrgencyRenderingTests(unittest.TestCase):
    """A deadline within 14 days of the digest's run date must be flagged as urgent
    and sorted ahead of items without a deadline in the same source group; a distant
    or absent deadline must not be flagged."""

    def test_urgent_deadline_shown_in_markdown(self):
        run_date = date(2026, 9, 15)
        c = _classification("LIAA", "funding", deadline="2026-09-20")  # 5 days out
        md = render_markdown([c], date(2026, 9, 1), run_date)
        self.assertIn("drīzumā", md)
        self.assertIn("20.09.2026", md)

    def test_distant_deadline_not_flagged_urgent(self):
        run_date = date(2026, 9, 15)
        c = _classification("LIAA", "funding", deadline="2026-12-01")  # months out
        md = render_markdown([c], date(2026, 9, 1), run_date)
        self.assertNotIn("drīzumā", md)
        self.assertIn("01.12.2026", md)

    def test_past_deadline_not_flagged_urgent(self):
        run_date = date(2026, 9, 15)
        c = _classification("LIAA", "funding", deadline="2026-09-01")  # already past
        md = render_markdown([c], date(2026, 9, 1), run_date)
        self.assertNotIn("drīzumā", md)

    def test_no_deadline_renders_without_termins_label(self):
        c = _classification("LIAA", "funding", deadline=None)
        md = render_markdown([c], date(2026, 9, 1), date(2026, 9, 15))
        self.assertNotIn("Termiņš", md)

    def test_malformed_deadline_on_classification_does_not_crash_render(self):
        # Defense in depth: even if a bad value somehow reached this far, rendering
        # must degrade (no deadline shown) rather than raise.
        c = _classification("LIAA", "funding", deadline="not-a-date")
        md = render_markdown([c], date(2026, 9, 1), date(2026, 9, 15))
        self.assertNotIn("Termiņš", md)

    def test_item_with_deadline_sorted_before_item_without_in_same_source(self):
        run_date = date(2026, 9, 15)
        no_deadline = _classification("LIAA", "funding", item_date="2026-09-14", title="No deadline")
        with_deadline = _classification("LIAA", "funding", item_date="2026-09-01",
                                         deadline="2026-09-20", title="Has deadline")
        md = render_markdown([no_deadline, with_deadline], date(2026, 9, 1), run_date)
        self.assertLess(md.index("Has deadline"), md.index("No deadline"))

    def test_urgent_count_summary_line_in_html(self):
        run_date = date(2026, 9, 15)
        c = _classification("LIAA", "funding", deadline="2026-09-18")
        html = render_html([c], date(2026, 9, 1), run_date)
        self.assertIn("1 ierakstam termiņš", html)


if __name__ == "__main__":
    unittest.main()
