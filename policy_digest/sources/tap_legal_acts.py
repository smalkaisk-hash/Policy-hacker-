"""TAP portāls (mandatory source) — draft legal acts.

Rather than scraping the tapportals.mk.gov.lv single-page app, this pulls the
official open dataset published daily on data.gov.lv (CKAN), which exposes
the same draft legal acts as clean JSON:API files, one per month, under
CC0. No authentication required.

Dataset: https://data.gov.lv/dati/lv/dataset/tap-publicetie-tiesibu-akti
"""

import html
import re
from datetime import date, datetime

import requests

from .base import Item

PACKAGE_SHOW_URL = "https://data.gov.lv/dati/api/3/action/package_show?id=tap-publicetie-tiesibu-akti"
SOURCE_NAME = "TAP portāls"
RESOURCE_DATE_RE = re.compile(r"legal_acts_(\d{4})-(\d{2})-\d{2}-\d{4}-\d{2}-\d{2}\.json")


def _months_between(start: date, end: date) -> set[tuple[int, int]]:
    months = set()
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.add((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _relevant_resource_urls(since: date, today: date) -> list[str]:
    resp = requests.get(PACKAGE_SHOW_URL, timeout=30)
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]
    needed_months = _months_between(since, today)

    urls = []
    for resource in resources:
        url = resource.get("url", "")
        match = RESOURCE_DATE_RE.search(url)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        if (year, month) in needed_months:
            urls.append(url)
    return urls


def _policy_area_names(included: list[dict]) -> dict[str, str]:
    return {
        entry["id"]: entry["attributes"]["name"]
        for entry in included
        if entry.get("type") == "policy_areas"
    }


def fetch_tap_legal_acts(since: date, today: date | None = None) -> list[Item]:
    today = today or date.today()
    items: list[Item] = []

    for resource_url in _relevant_resource_urls(since, today):
        resp = requests.get(resource_url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        area_names = _policy_area_names(payload.get("included", []))

        for entry in payload.get("data", []):
            attrs = entry.get("attributes", {})
            submitted_at = attrs.get("submitted_at")
            if not submitted_at:
                continue
            submitted_date = datetime.fromisoformat(submitted_at).date()
            if submitted_date < since:
                continue

            area_ids = [
                rel["id"]
                for rel in entry.get("relationships", {}).get("policy_areas", {}).get("data", [])
            ]
            areas = ", ".join(area_names.get(i, "") for i in area_ids if area_names.get(i))

            title = html.unescape(attrs.get("name", "")).strip()
            institution = attrs.get("responsible_institution_name", "")
            progress = attrs.get("progress_name", "")
            url = entry.get("links", {}).get("web", "")

            items.append(
                Item(
                    source=SOURCE_NAME,
                    title=title,
                    url=url,
                    date=submitted_date.isoformat(),
                    summary=f"{institution} · {progress}" + (f" · {areas}" if areas else ""),
                    raw_text=(
                        f"{title}\n\nAtbildīgā institūcija: {institution}\n"
                        f"Statuss: {progress}\nPolitikas jomas: {areas or '—'}"
                    ),
                )
            )

    return items
