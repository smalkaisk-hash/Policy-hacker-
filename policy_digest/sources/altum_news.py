"""Altum (altum.lv) news — via its WordPress REST API.

The public listing page (`/par-altum/aktualitates/`) is a JS carousel showing only the
~12 most recent items, with no HTML pagination — capping any lookback window to about a
month (this was a documented known limitation). altum.lv runs WordPress, though, and its
default REST API is open with no auth: `/wp-json/wp/v2/posts` returns the exact same
articles (confirmed by matching titles/URLs against the carousel) with real pagination
(`page`/`per_page`, up to 100/page) and server-side date filtering (`after`) — so any
lookback window now actually works, not just the last ~12 items. Bonus: the full article
body comes back already embedded in `content.rendered`, so unlike every other source
module here, no second per-article HTTP request is needed at all.
"""

from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from .base import Item

BASE_URL = "https://www.altum.lv"
POSTS_API_URL = f"{BASE_URL}/wp-json/wp/v2/posts"
HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
SOURCE_NAME = "Altum"
PER_PAGE = 100  # WordPress REST API's own maximum
MAX_PAGES = 20  # safety cap; the loop normally stops once a page is fully older than `since`


def _strip_html(html_text: str | None) -> str:
    return BeautifulSoup(html_text or "", "html.parser").get_text(" ", strip=True)


def fetch_altum_news(since: date, seen: set[str] | None = None) -> list[Item]:
    items: list[Item] = []

    for page in range(1, MAX_PAGES + 1):
        try:
            resp = requests.get(
                POSTS_API_URL,
                headers=HEADERS,
                params={
                    "per_page": PER_PAGE,
                    "page": page,
                    "after": f"{since.isoformat()}T00:00:00",
                    "orderby": "date",
                    "order": "desc",
                },
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  ! Altum: page {page} request failed ({exc}) — stopping pagination early")
            break

        posts = resp.json()
        if not isinstance(posts, list) or not posts:
            break

        for post in posts:
            title = _strip_html((post.get("title") or {}).get("rendered", ""))
            published = post.get("date")  # site-local time, unambiguous enough for a date-only field
            try:
                article_date = datetime.fromisoformat(published).date()
            except (TypeError, ValueError):
                print(f"  ! Altum: could not parse post date {published!r} for {title[:60]!r} — skipped")
                continue

            url = post.get("link") or ""
            body = _strip_html((post.get("content") or {}).get("rendered", ""))
            items.append(
                Item(
                    source=SOURCE_NAME,
                    title=title,
                    url=url,
                    date=article_date.isoformat(),
                    summary=body[:200],
                    raw_text=f"{title}\n\n{body}",
                )
            )

        total_pages = resp.headers.get("X-WP-TotalPages")
        if total_pages and page >= int(total_pages):
            break
        if page == MAX_PAGES:
            print(
                f"  ! Altum: hit the {MAX_PAGES}-page safety cap while still within the "
                "requested window — some older items may be missing this run"
            )

    return items
