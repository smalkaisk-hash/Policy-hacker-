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
MAX_PAGES = 40  # safety cap; the loop normally stops early once past `since` (see warning below)

# User decision 2026-09-16 ("incubators are not policy"): LIAA incubation-program content
# should never be scraped at all — stronger than "never classify as relevant"
# (classify._is_unfunded_training_round already rejects an unfunded incubation/mentorship
# round at classification time; this also excludes a program that states a real funding
# amount, e.g. "finanšu atbalstu līdz 70%"). This digest tracks funding/regulatory
# mechanisms, not institutions that happen to run incubation programs — a source-level
# exclusion, not a classification-time filter, so it applies before the article body is
# even fetched. Scoped to LIAA only (this scraper is shared with Ekonomikas ministrija,
# which doesn't run incubation programs). "inkub" (not "inkubat"/"inkubāc" separately):
# Latvian macron vowels are distinct characters ("inkubācija" has "ā", not "a"), so the
# shorter root common to both "inkubators" and "inkubācija" is needed to match either form.
LIAA_INCUBATION_TITLE_KEYWORD = "inkub"


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
    base_url: str,
    source_name: str,
    since: date,
    listing_path: str = "/lv/jaunumi",
    seen: set[str] | None = None,
) -> list[Item]:
    seen = seen or set()
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
                title_preview = link.get_text(strip=True)[:60]
                print(
                    f"  ! {source_name}: could not parse article date "
                    f"{time_tag['datetime']!r} for {title_preview!r} — skipped"
                )
                continue

            if article_date < since:
                stop_pagination = True
                continue  # listing is newest-first; skip, keep checking rest of this page

            title = link.get_text(strip=True)
            if source_name == "LIAA" and LIAA_INCUBATION_TITLE_KEYWORD in title.lower():
                print(f"  ! {source_name}: skipping incubation-program item {title[:60]!r} — never scraped, per product decision")
                continue

            href = link["href"]
            full_url = href if href.startswith("http") else base_url + href
            summary = summary_tag.get_text(strip=True) if summary_tag else ""
            # Already scraped this article's full body in a previous run — it won't have
            # changed, so don't re-fetch it.
            body = "" if full_url in seen else _fetch_article_body(full_url)

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
        if page == MAX_PAGES - 1:
            print(
                f"  ! {source_name}: hit the {MAX_PAGES}-page safety cap while still within "
                "the requested window — some older items may be missing this run"
            )

    return items
