"""Altum (altum.lv) news.

Altum runs a different CMS (WordPress) than EM/LIAA (Drupal), with its own
markup, so it gets its own small scraper rather than reusing news_listing.py.
The listing page (`/par-altum/aktualitates/`) isn't paginated in HTML — it's
a single page of the ~12 most recent items — which is enough for a weekly
digest but means a lookback window older than that won't find everything;
see README.
"""

from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from .base import Item

BASE_URL = "https://www.altum.lv"
LISTING_URL = f"{BASE_URL}/par-altum/aktualitates/"
HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
SOURCE_NAME = "Altum"


def _parse_date(text: str) -> date | None:
    # e.g. "02. Sep, 2026" -> 2026-09-02 (month abbreviations are in English on this site)
    try:
        return datetime.strptime(text.strip(), "%d. %b, %Y").date()
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


def fetch_altum_news(since: date) -> list[Item]:
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
        if article_date is None or article_date < since:
            continue

        title = text_tag.get_text(strip=True) if text_tag else link_tag.get_text(strip=True)
        url = link_tag["href"]
        body = _fetch_article_body(url)

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
