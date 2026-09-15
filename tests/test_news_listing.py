"""Tests for news_listing.py's LIAA incubation-program exclusion (2026-09-16, "incubators
are not policy") — a source-level skip at fetch time, applied only to LIAA (this scraper
is shared with Ekonomikas ministrija, which doesn't run incubation programs).

Uses only the standard library (unittest + unittest.mock), same as test_reliability.py.
Run with:

    python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date

from policy_digest.sources import news_listing


def _listing_html(rows):
    rows_html = "".join(
        f"""<div class="views-row">
              <div class="title"><h2><a href="{url}">{title}</a></h2></div>
              <div class="date"><time datetime="{dt}">{dt}</time></div>
              <div class="text">Kopsavilkums.</div>
            </div>"""
        for title, url, dt in rows
    )
    return f'<div class="articles-wrapper">{rows_html}</div>'


def _fake_get(listing_html, article_body_html='<div class="text__text-content">Pilns saturs.</div>'):
    def fake_get(url, headers=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        # Listing pages carry "?page=" or are the bare listing path; anything else is an
        # individual article page.
        if "/lv/jaunumi" in url:
            resp.text = listing_html if "page=1" not in url else _listing_html([])
        else:
            resp.text = article_body_html
        return resp
    return fake_get


class LiaaIncubationExclusionTests(unittest.TestCase):
    def test_liaa_incubation_item_is_skipped(self):
        html = _listing_html([
            ("LIAA atver rudens uzņemšanu Biznesa inkubācijas programmā",
             "/lv/jaunums/incubation", "2026-09-09"),
        ])
        with patch("policy_digest.sources.news_listing.requests.get", _fake_get(html)):
            items = news_listing.fetch_news_listing(
                "https://www.liaa.gov.lv", "LIAA", since=date(2026, 9, 1),
            )
        self.assertEqual(items, [])

    def test_liaa_non_incubation_item_is_kept(self):
        html = _listing_html([
            ("LIAA izsludina jaunu programmu jaunuzņēmumiem",
             "/lv/jaunums/other", "2026-09-09"),
        ])
        with patch("policy_digest.sources.news_listing.requests.get", _fake_get(html)):
            items = news_listing.fetch_news_listing(
                "https://www.liaa.gov.lv", "LIAA", since=date(2026, 9, 1),
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "LIAA izsludina jaunu programmu jaunuzņēmumiem")

    def test_em_incubation_looking_title_is_not_skipped(self):
        # The exclusion is LIAA-only — this scraper is shared with Ekonomikas ministrija,
        # which doesn't run incubation programs, so the same keyword must not apply there.
        html = _listing_html([
            ("Ministrija atbalsta uzņēmumu inkubācijas iniciatīvas",
             "/lv/jaunums/em-item", "2026-09-09"),
        ])
        with patch("policy_digest.sources.news_listing.requests.get", _fake_get(html)):
            items = news_listing.fetch_news_listing(
                "https://www.em.gov.lv", "Ekonomikas ministrija", since=date(2026, 9, 1),
            )
        self.assertEqual(len(items), 1)

    def test_incubation_item_skipped_without_fetching_its_article_body(self):
        # The whole point is a source-level skip BEFORE the extra per-article HTTP
        # request — a fetched-but-then-discarded body would defeat that.
        html = _listing_html([
            ("Biznesa inkubācijas programmas rudens uzņemšana",
             "/lv/jaunums/incubation-2", "2026-09-09"),
        ])
        get = MagicMock(side_effect=_fake_get(html))
        with patch("policy_digest.sources.news_listing.requests.get", get):
            news_listing.fetch_news_listing(
                "https://www.liaa.gov.lv", "LIAA", since=date(2026, 9, 1),
            )
        called_urls = [c.args[0] for c in get.call_args_list]
        self.assertFalse(any("incubation-2" in u for u in called_urls))


if __name__ == "__main__":
    unittest.main()
