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
    # These two functions run once per resource file, BEFORE the per-entry loop below even
    # starts — unlike a bad field on one act (skip that act, keep the rest), a KeyError here
    # from one malformed "included" entry used to lose every act in the whole file. Only
    # entries with a usable "id" are kept; a missing/null "name" degrades to "" rather than
    # dropping the entry (its id, needed to resolve other acts' policy_areas, is still good).
    result = {}
    for entry in included:
        if entry.get("type") != "policy_areas":
            continue
        entry_id = entry.get("id")
        if not entry_id:
            continue
        result[entry_id] = (entry.get("attributes") or {}).get("name") or ""
    return result


def _document_version_map(included: list[dict]) -> dict[str, dict]:
    result = {}
    for entry in included:
        if entry.get("type") != "legal_act_document_versions":
            continue
        entry_id = entry.get("id")
        if not entry_id:
            continue
        result[entry_id] = entry
    return result


def _fetch_full_text(entry: dict, doc_version_map: dict[str, dict]) -> str:
    version_ids = [
        rel["id"]
        for rel in entry.get("relationships", {}).get("document_versions", {}).get("data", [])
        if "id" in rel  # be tolerant of an unexpectedly-shaped relationship entry
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


def fetch_tap_legal_acts(
    since: date, today: date | None = None, seen: set[str] | None = None
) -> list[Item]:
    today = today or date.today()
    seen = seen or set()
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
            try:
                submitted_date = datetime.fromisoformat(submitted_at).date()
            except ValueError:
                # One malformed record must not cost us every other (valid) record in
                # this month's dataset — skip just this entry, not the whole source.
                name = attrs.get("name", "")[:60]
                print(f"  ! TAP portāls: could not parse submitted_at {submitted_at!r} for {name!r} — skipped")
                continue
            if submitted_date < since:
                continue

            area_ids = [
                rel["id"]
                for rel in entry.get("relationships", {}).get("policy_areas", {}).get("data", [])
                if "id" in rel  # be tolerant of an unexpectedly-shaped relationship entry
            ]
            areas = ", ".join(area_names.get(i, "") for i in area_ids if area_names.get(i))

            # `or ""`, not just `.get(key, "")`: a `.get` default only applies when the key
            # is MISSING — a value explicitly present as JSON null still comes back as None
            # and crashes html.unescape()/.strip() below (same bug class already hit and
            # fixed in classify.py's "results" and this function's "links" handling above).
            title = html.unescape(attrs.get("name") or "").strip()
            institution = attrs.get("responsible_institution_name") or ""
            progress = attrs.get("progress_name") or ""
            url = (entry.get("links") or {}).get("web", "")  # "links": null is valid JSON:API
            # Status/progress can still change after we've first seen an act, so we always
            # refetch that (cheap — it's already in this monthly JSON dump); the act's own
            # text doesn't change once published, so skip re-fetching that expensively.
            full_text = "" if url in seen else _fetch_full_text(entry, doc_version_map)

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
