"""Saeima committee sittings (Saeimas komisiju darba kārtības).

There's no unified feed on saeima.lv itself — each committee runs its own
mini-site. But saeima.lv's "Komisiju sēžu darba kārtības" link forwards to an
internal Domino system (titania.saeima.lv) whose public "webComisDK" view
returns *all* committees' sittings for a given day as inline JS calls, and
each sitting's own document exposes the full agenda text in a `#textBody`
element. No authentication required.

    .../saeimasnotikumi.nsf/webComisDK?OpenView&restrictToCategory=DD.MM.YYYY&count=1000
    .../saeimasnotikumi.nsf/0/{unid}?OpenDocument   (full agenda text)
"""

import re
from datetime import date, datetime, timedelta

import requests
from bs4 import BeautifulSoup

from .base import Item

BASE = "https://titania.saeima.lv/livs/saeimasnotikumi.nsf"
HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
SOURCE_NAME = "Saeimas komisiju darba kārtības"
MAX_MEETINGS = 300  # safety cap

DRAW_PE_RE = re.compile(
    r'draw_PE\(\{srt1:"[^"]*", time:"([^"]*)", title:"([^"]*)", next:"[^"]*", unid:"([^"]*)"\}\);'
)


def _daterange(since: date, today: date):
    d = since
    while d <= today:
        yield d
        d += timedelta(days=1)


def _fetch_full_text(unid: str) -> str:
    url = f"{BASE}/0/{unid}?OpenDocument"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    node = soup.select_one("#textBody")
    return node.get_text(" ", strip=True) if node else ""


def fetch_saeima_committees(
    since: date, today: date | None = None, seen: set[str] | None = None
) -> list[Item]:
    today = today or date.today()
    seen = seen or set()
    items: list[Item] = []
    seen_unids: set[str] = set()  # within this run only, to dedupe across day queries

    for day in _daterange(since, today):
        if len(items) >= MAX_MEETINGS:
            break

        day_str = day.strftime("%d.%m.%Y")
        url = f"{BASE}/webComisDK?OpenView&restrictToCategory={day_str}&count=1000"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        for _time, raw_title, unid in DRAW_PE_RE.findall(resp.text):
            if not unid or unid in seen_unids:
                continue
            seen_unids.add(unid)

            title = raw_title.strip()
            doc_url = f"{BASE}/0/{unid}?OpenDocument"
            # Already scraped this sitting's full agenda in a previous run — it's a past
            # meeting now, so its text won't change.
            full_text = "" if doc_url in seen else _fetch_full_text(unid)

            items.append(
                Item(
                    source=SOURCE_NAME,
                    title=title,
                    url=doc_url,
                    date=day.isoformat(),
                    summary=full_text[:200],
                    raw_text=f"{title}\n\n{full_text}" if full_text else title,
                )
            )

    return items
