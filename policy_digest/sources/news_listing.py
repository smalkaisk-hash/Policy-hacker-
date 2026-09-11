"""Generic scraper for the shared government news-listing template used by
Ekonomikas ministrija (em.gov.lv) and LIAA (liaa.gov.lv) — both render the
same Drupal "articles-wrapper / views-row" markup with 0-indexed `?page=N`
pagination, so one implementation covers both sources.
"""

from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from .base import Item

HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
MAX_PAGES = 10  # safety cap; the loop normally stops early once past `since`


def _fetch_article_body(url: str) -> str:
    """Both em.gov.lv and liaa.gov.lv render the article body in the same
    Drupal CKEditor field, so one selector covers both sites."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    node = soup.select_one(".text__text-content")
    return node.get_text(" ", strip=True) if node else ""


def fetch_news_listing(
    base_url: str, source_name: str, since: date, listing_path: str = "/lv/jaunumi"
) -> list[Item]:
    items: list[Item] = []

    for page in range(MAX_PAGES):
        url = f"{base_url}{listing_path}" + (f"?page={page}" if page else "")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = soup.select("div.views-row")
        if not rows:
            break

        stop_pagination = False
        for row in rows:
            link = row.select_one(".title h2 a[href]")
            time_tag = row.select_one(".date time[datetime]")
            summary_tag = row.select_one(".text")

            if not link or not time_tag:
                continue

            try:
                article_date = datetime.fromisoformat(time_tag["datetime"]).date()
            except ValueError:
                continue

            if article_date < since:
                stop_pagination = True
                continue  # listing is newest-first; skip, keep checking rest of this page

            title = link.get_text(strip=True)
            href = link["href"]
            full_url = href if href.startswith("http") else base_url + href
            summary = summary_tag.get_text(strip=True) if summary_tag else ""
            body = _fetch_article_body(full_url)

            items.append(
                Item(
                    source=source_name,
                    title=title,
                    url=full_url,
                    date=article_date.isoformat(),
                    summary=summary,
                    raw_text=f"{title}\n\n{body or summary}",
                )
            )

        if stop_pagination:
            break

    return items
