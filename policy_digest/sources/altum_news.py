"""Altum (altum.lv) news.

Altum runs a different CMS (WordPress) than EM/LIAA (Drupal), with its own
markup, so it gets its own small scraper rather than reusing news_listing.py.
The listing page (`/par-altum/aktualitates/`) isn't paginated in HTML — it's
a single page of the ~12 most recent items — which is enough for a weekly
digest but means a lookback window older than that won't find everything;
see README.
"""

import re
from datetime import date

import requests
from bs4 import BeautifulSoup

from .base import Item

BASE_URL = "https://www.altum.lv"
LISTING_URL = f"{BASE_URL}/par-altum/aktualitates/"
HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
SOURCE_NAME = "Altum"

# Month abbreviations are always English on this site regardless of server/
# process locale, so map them explicitly instead of relying on the
# locale-dependent %b strptime directive.
_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
_DATE_RE = re.compile(r"(\d{1,2})\.\s*([A-Za-z]{3}),\s*(\d{4})")


def _parse_date(text: str) -> date | None:
    # e.g. "02. Sep, 2026" -> 2026-09-02
    match = _DATE_RE.search(text.strip())
    if not match:
        return None
    day, month_abbr, year = match.groups()
    month = _MONTHS.get(month_abbr[:3].title())
    if month is None:
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def _fetch_article_body(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    node = soup.select_one(".newsContent__wrap")
    return node.get_text(" ", strip=True) if node else ""


def fetch_altum_news(since: date, seen: set[str] | None = None) -> list[Item]:
    seen = seen or set()
    resp = requests.get(LISTING_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items: list[Item] = []
    for block in soup.select(".mainNewsSwiper__content"):
        date_tag = block.select_one(".mainNewsSwiper__date")
        link_tag = block.select_one("a.mainNewsSwiper__link")
        text_tag = block.select_one(".mainNewsSwiperLink__text")
        if not date_tag or not link_tag:
            continue

        article_date = _parse_date(date_tag.get_text())
        if article_date is None:
            title_preview = (text_tag.get_text(strip=True) if text_tag else link_tag.get_text(strip=True))[:60]
            print(f"  ! Altum: could not parse article date {date_tag.get_text()!r} for {title_preview!r} — skipped")
            continue
        if article_date < since:
            continue

        title = text_tag.get_text(strip=True) if text_tag else link_tag.get_text(strip=True)
        href = link_tag["href"]
        url = href if href.startswith("http") else BASE_URL + href
        body = "" if url in seen else _fetch_article_body(url)

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

    return items
