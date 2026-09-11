"""TAP portāls (mandatory source) — draft legal acts.

Rather than scraping the tapportals.mk.gov.lv single-page app, this pulls the
official open dataset published daily on data.gov.lv (CKAN), which exposes
the same draft legal acts as clean JSON:API files, one per month, under
CC0. No authentication required.

Dataset: https://data.gov.lv/dati/lv/dataset/tap-publicetie-tiesibu-akti

Each entry also links to one or more "document_versions" (draft protocol
decision, annotation, etc.). Where a version is rendered by TAP's own
"structuralizer" preview (a public, no-auth HTML endpoint — file attachments
like .docx are skipped), we fetch it and pull the actual decision/annotation
text, not just the act's title.
"""

import html
import re
from datetime import date, datetime

import requests
from bs4 import BeautifulSoup

from .base import Item

PACKAGE_SHOW_URL = "https://data.gov.lv/dati/api/3/action/package_show?id=tap-publicetie-tiesibu-akti"
SOURCE_NAME = "TAP portāls"
RESOURCE_DATE_RE = re.compile(r"legal_acts_(\d{4})-(\d{2})-\d{2}-\d{4}-\d{2}-\d{2}\.json")
STRUCTURALIZER_URL_RE = re.compile(r"/structuralizer/data/nodes/[0-9a-f-]+/preview$")
HEADERS = {"User-Agent": "Mozilla/5.0 (policy-digest prototype; +startin.lv test task)"}
FULL_TEXT_CHAR_LIMIT = 4000


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


def _document_version_map(included: list[dict]) -> dict[str, dict]:
    return {
        entry["id"]: entry
        for entry in included
        if entry.get("type") == "legal_act_document_versions"
    }


def _fetch_full_text(entry: dict, doc_version_map: dict[str, dict]) -> str:
    version_ids = [
        rel["id"]
        for rel in entry.get("relationships", {}).get("document_versions", {}).get("data", [])
    ]
    for version_id in version_ids:
        version = doc_version_map.get(version_id)
        if not version:
            continue
        for doc_item in version.get("attributes", {}).get("items", []):
            preview_url = doc_item.get("url", "")
            if not STRUCTURALIZER_URL_RE.search(preview_url):
                continue  # a real file attachment (.docx etc.), not an inline preview — skip
            try:
                resp = requests.get(preview_url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
            except requests.RequestException:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            node = soup.select_one(".structuralizer-tree")
            if node:
                text = node.get_text(" ", strip=True)
                if text:
                    return text[:FULL_TEXT_CHAR_LIMIT]
    return ""


def fetch_tap_legal_acts(since: date, today: date | None = None) -> list[Item]:
    today = today or date.today()
    items: list[Item] = []

    for resource_url in _relevant_resource_urls(since, today):
        resp = requests.get(resource_url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        area_names = _policy_area_names(payload.get("included", []))
        doc_version_map = _document_version_map(payload.get("included", []))

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
            full_text = _fetch_full_text(entry, doc_version_map)

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
                        + (f"\n\nSaturs:\n{full_text}" if full_text else "")
                    ),
                )
            )

    return items
