"""Tests for altum_news.py's WordPress REST API fetcher (2026-09-15) — replaced the old
scrape of the unpaginated news carousel (`/par-altum/aktualitates/`, ~12 most recent items,
no HTML pagination) with altum.lv's open `/wp-json/wp/v2/posts` endpoint, which supports
real pagination and date filtering and already embeds the full article body.

Uses only the standard library (unittest + unittest.mock), same as test_reliability.py.
Run with:

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

from policy_digest.sources import altum_news


def _post(title, link, published, content="<p>Saturs.</p>"):
    return {
        "title": {"rendered": title},
        "link": link,
        "date": published,
        "content": {"rendered": content},
    }


def _fake_get(pages, total_pages):
    """pages: dict[page_number] -> list[post dict]."""
    def fake_get(url, headers=None, params=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json.return_value = pages.get(params["page"], [])
        resp.headers = {"X-WP-TotalPages": str(total_pages)}
        return resp
    return fake_get


class AltumRestApiTests(unittest.TestCase):
    def test_parses_posts_into_items_with_full_body(self):
        pages = {1: [_post("Jaunums par jaunuzņēmumiem", "https://altum.lv/a/1",
                            "2026-09-10T12:00:00", "<p>Pilns <strong>saturs</strong> šeit.</p>")]}
        with patch("policy_digest.sources.altum_news.requests.get", _fake_get(pages, total_pages=1)):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertEqual(it.title, "Jaunums par jaunuzņēmumiem")
        self.assertEqual(it.date, "2026-09-10")
        self.assertIn("Pilns saturs šeit.", it.raw_text)
        self.assertEqual(it.source, "Altum")

    def test_follows_pagination_across_multiple_pages(self):
        pages = {
            1: [_post("Ziņa 1", "https://altum.lv/a/1", "2026-09-10T12:00:00")],
            2: [_post("Ziņa 2", "https://altum.lv/a/2", "2026-09-05T12:00:00")],
        }
        with patch("policy_digest.sources.altum_news.requests.get", _fake_get(pages, total_pages=2)):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual({it.title for it in items}, {"Ziņa 1", "Ziņa 2"})

    def test_stops_at_total_pages_header(self):
        pages = {1: [_post("Ziņa 1", "https://altum.lv/a/1", "2026-09-10T12:00:00")]}
        calls = []
        real_fake = _fake_get(pages, total_pages=1)

        def counting_get(url, headers=None, params=None, timeout=None):
            calls.append(params["page"])
            return real_fake(url, headers=headers, params=params, timeout=timeout)

        with patch("policy_digest.sources.altum_news.requests.get", counting_get):
            altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual(calls, [1])  # must not request page 2 when X-WP-TotalPages says 1

    def test_malformed_date_skips_only_that_post(self):
        pages = {1: [
            _post("Bad date", "https://altum.lv/a/1", "not-a-date"),
            _post("Good", "https://altum.lv/a/2", "2026-09-05T12:00:00"),
        ]}
        with patch("policy_digest.sources.altum_news.requests.get", _fake_get(pages, total_pages=1)):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual([it.title for it in items], ["Good"])

    def test_missing_date_field_does_not_crash(self):
        post = _post("No date field", "https://altum.lv/a/1", None)
        with patch("policy_digest.sources.altum_news.requests.get", _fake_get({1: [post]}, total_pages=1)):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual(items, [])

    def test_request_failure_does_not_crash(self):
        def failing_get(url, headers=None, params=None, timeout=None):
            import requests
            raise requests.RequestException("simulated network failure")

        with patch("policy_digest.sources.altum_news.requests.get", failing_get):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual(items, [])

    def test_html_content_is_stripped_to_plain_text(self):
        content = "<p>Teksts ar <a href='#'>saiti</a> un\n<strong>treknrakstu</strong>.</p>"
        post = _post("Title", "https://altum.lv/a/1", "2026-09-05T12:00:00", content)
        with patch("policy_digest.sources.altum_news.requests.get", _fake_get({1: [post]}, total_pages=1)):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertNotIn("<", items[0].raw_text)
        self.assertIn("Teksts ar saiti un treknrakstu", items[0].raw_text)

    def test_empty_page_stops_pagination(self):
        pages = {1: [_post("Ziņa 1", "https://altum.lv/a/1", "2026-09-10T12:00:00")], 2: []}
        with patch("policy_digest.sources.altum_news.requests.get", _fake_get(pages, total_pages=5)):
            items = altum_news.fetch_altum_news(since=date(2026, 9, 1))
        self.assertEqual(len(items), 1)


if __name__ == "__main__":
    unittest.main()
