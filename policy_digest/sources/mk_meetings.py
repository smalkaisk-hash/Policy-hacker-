"""Valsts sekretāru sanāksme + Ministru kabineta protokoli.

Both live on the same tapportals.mk.gov.lv "meetings" listing (server-rendered,
paginated), differing only by a `meeting_type` path segment. Each meeting row
optionally links to a protocol detail page listing the individual agenda
items actually decided/discussed — those items, not the generic meeting
title, are what's worth scanning for relevance.
"""

from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from .base import Item

BASE_URL = "https://tapportals.mk.gov.lv"
HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
MAX_PAGES = 10  # safety cap; the loop normally stops early once past `since`


def _parse_meeting_date(text: str) -> date | None:
    # e.g. "10.09.2026. 16:00" -> 2026-09-10
    day_part = text.strip().split(" ")[0]
    try:
        return datetime.strptime(day_part, "%d.%m.%Y.").date()
    except ValueError:
        return None


def _fetch_protocol_items(protocol_url: str, meeting_title: str) -> list[tuple[str, str, str]]:
    """Returns [(item_title, fragment_url, full_text), ...] for one protocol page.

    Each agenda item's full decision text sits in a `.structuralizer-tree` div that's a
    sibling (not a child) of the `.meeting-protocol-question--preview` block — already on
    this same page, so this costs no extra request.
    """
    resp = requests.get(protocol_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for block in soup.select(".meeting-protocol-question--preview"):
        name_span = block.select_one(".meeting-protocol-question__name span")
        if not name_span:
            continue
        title = name_span.get_text(strip=True)
        if not title:
            continue
        block_id = block.get("id", "")
        url = f"{protocol_url}#{block_id}" if block_id else protocol_url

        full_text = ""
        content_wrapper = block.find_next_sibling("div")
        if content_wrapper:
            tree = content_wrapper.select_one(".structuralizer-tree")
            if tree:
                full_text = tree.get_text(" ", strip=True)

        results.append((title, url, full_text))

    if not results:
        # Protocol page exists but had no parseable items (e.g. fully closed session) —
        # fall back to the meeting title itself so it isn't silently dropped.
        results.append((meeting_title, protocol_url, ""))

    return results


def fetch_mk_meetings(meeting_type: str, source_name: str, since: date) -> list[Item]:
    items: list[Item] = []
    listing_base = f"{BASE_URL}/meetings/{meeting_type}"

    for page in range(1, MAX_PAGES + 1):
        url = listing_base if page == 1 else f"{listing_base}?page={page}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = soup.select("div.flextable__row[data-url]")
        if not rows:
            break

        stop_pagination = False
        for row in rows:
            date_cell = row.select_one('[data-column-header-name="Datums"] .flextable__value')
            name_cell = row.select_one('[data-column-header-name="Nosaukums"] .flextable__value')
            protocol_link = row.select_one('[data-column-header-name="Protokols"] a[href]')

            meeting_date = _parse_meeting_date(date_cell.get_text()) if date_cell else None
            meeting_title = name_cell.get_text(strip=True) if name_cell else "(bez nosaukuma)"

            if meeting_date is None:
                continue
            if meeting_date < since:
                stop_pagination = True
                continue  # listing is newest-first; skip, keep checking rest of this page

            meeting_url = BASE_URL + row["data-url"]

            if protocol_link:
                protocol_url = BASE_URL + protocol_link["href"]
                for item_title, item_url, full_text in _fetch_protocol_items(protocol_url, meeting_title):
                    raw_text = f"{item_title}\n\n(No sēdes: {meeting_title})"
                    if full_text:
                        raw_text += f"\n\n{full_text}"
                    items.append(
                        Item(
                            source=source_name,
                            title=item_title,
                            url=item_url,
                            date=meeting_date.isoformat(),
                            summary=meeting_title,
                            raw_text=raw_text,
                        )
                    )
            else:
                items.append(
                    Item(
                        source=source_name,
                        title=meeting_title,
                        url=meeting_url,
                        date=meeting_date.isoformat(),
                        summary="Protokols vēl nav publicēts",
                        raw_text=meeting_title,
                    )
                )

        if stop_pagination:
            break

    return items
