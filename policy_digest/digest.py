"""Renders classified items into a Markdown + HTML digest. Output is Latvian throughout —
this is a digest for a Latvian team about Latvian government sources.
"""

import base64
import html as html_lib
from collections import defaultdict
from datetime import date
from pathlib import Path

from .classify import Classification

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logobig.png"
# Markdown files live one level down (output/, sample_digest/), both siblings of assets/.
LOGO_MARKDOWN_PATH = "../assets/logobig.png"

TITLE = "Politikas monitorings"  # "Policy monitoring" — matches the task's own naming

DEFINITION = (
    "Jaunuzņēmumiem atbilstošs ir ieraksts, kas pēc būtības ietekmē jaunuzņēmumu darbību, "
    "finansējumu vai izaugsmes vidi — piemēram, jauns atbalsts vai finansējuma iespēja, "
    "izmaiņas nodokļu vai darba tiesību regulējumā, vai iniciatīva inovāciju un digitalizācijas "
    "jomā. Katru ierakstu izvērtē mākslīgais intelekts, izlasot tā pilno saturu, nevis meklējot "
    "atsevišķus atslēgvārdus."
)

# (Latvian label, accent hex) — one accent color used consistently, not a rainbow per
# category; categories are told apart by label text, not by hue.
CATEGORY_META = {
    "funding": ("Finansējums", "#c0173f"),
    "regulation": ("Regulējums", "#c0173f"),
    "tax_labor": ("Nodokļi un darbs", "#c0173f"),
    "digitalization_innovation": ("Digitalizācija un inovācijas", "#c0173f"),
    "other": ("Cits", "#71717a"),
}

# Two top-level sections, reflecting the two different things a reader does with an
# item: "funding" is something to apply for (deadline-driven), everything else is
# something to monitor/react to (a regulatory or initiative change). Source-level
# grouping still happens inside each section — this is one level above it, not a
# replacement for it.
SUPERGROUPS = [
    ("Finansējuma iespējas", lambda c: c.category == "funding"),
    ("Regulējums un iniciatīvas", lambda c: c.category != "funding"),
]

# A deadline inside this many days of the digest's run date is called out as urgent —
# close enough that it needs acting on before the next weekly run, not just noting.
URGENT_DEADLINE_DAYS = 14


def _category_label(category: str) -> str:
    return CATEGORY_META.get(category, CATEGORY_META["other"])[0]


def _parse_deadline(deadline: str | None) -> date | None:
    """classify.py already validates the ISO-date shape before it reaches a
    Classification, but this is the last stage of the pipeline (see _oneline's own
    comment on the same principle) — tolerate a bad value here too rather than crash
    the whole render over one item's deadline field."""
    if not deadline:
        return None
    try:
        return date.fromisoformat(deadline)
    except ValueError:
        return None


def _is_urgent(c: Classification, run_date: date) -> bool:
    d = _parse_deadline(c.deadline)
    if d is None:
        return False
    return 0 <= (d - run_date).days <= URGENT_DEADLINE_DAYS


def _oneline(s: str) -> str:
    """Collapses any whitespace run — including embedded newlines — to a single space.
    Scraped titles can contain a literal line break (BeautifulSoup's get_text(strip=True)
    only trims the outer edges of each text node, not an internal "\\n" from a line break
    in the source HTML), and an LLM's "one-line" reason isn't a hard guarantee either.
    Both get embedded in single-line Markdown constructs (a list item, a link) where an
    embedded newline corrupts the structure — splitting the item, and often the link,
    across two lines instead of just looking odd.

    Also tolerates None: this is the very last stage of the whole pipeline, after fetch,
    dedupe, and classify have all already succeeded for potentially dozens of items — a
    single bad field here (upstream defenses should already prevent it, but this is the
    last line of defense) must not crash the write and discard all of that finished work.
    """
    return " ".join((s or "").split())


def _group_by_source(classifications: list[Classification]) -> dict[str, list[Classification]]:
    grouped: dict[str, list[Classification]] = defaultdict(list)
    for c in classifications:
        grouped[c.item.source].append(c)
    for source_items in grouped.values():
        source_items.sort(key=lambda c: c.item.date, reverse=True)
        # Second, stable sort: items with a deadline float to the top, soonest first;
        # items without one keep the date-descending order the first sort just gave
        # them. Two items both without a deadline are therefore unaffected by this.
        source_items.sort(key=lambda c: (_parse_deadline(c.deadline) is None, _parse_deadline(c.deadline) or date.max))
    return grouped


def _deadline_markdown(c: Classification, run_date: date) -> str:
    d = _parse_deadline(c.deadline)
    if d is None:
        return ""
    formatted = d.strftime("%d.%m.%Y")
    if _is_urgent(c, run_date):
        return f" · **Termiņš: {formatted} — drīzumā!**"
    return f" · Termiņš: {formatted}"


def _sources_checked_text(all_sources: list[str], classifications: list[Classification]) -> str:
    """'{source} ({count})' for every monitored source, in the given order — including
    ones with zero relevant items this period, so the digest shows what was actually
    checked, not just where something happened to be found. A merged cross-source item
    (source field "A + B", see dedupe.py's _merge_cluster) counts toward every source it
    credits, not just the first."""
    counts = {name: 0 for name in all_sources}
    for c in classifications:
        for part in c.item.source.split(" + "):
            if part in counts:
                counts[part] += 1
    return ", ".join(f"{name} ({counts[name]})" for name in all_sources)


def _deadline_html(c: Classification, run_date: date, esc) -> str:
    d = _parse_deadline(c.deadline)
    if d is None:
        return ""
    formatted = d.strftime("%d.%m.%Y")
    urgent = _is_urgent(c, run_date)
    css_class = "deadline urgent" if urgent else "deadline"
    suffix = " — drīzumā" if urgent else ""
    return f" · <span class='{css_class}'>Termiņš: {esc(formatted)}{esc(suffix)}</span>"


def render_markdown(
    classifications: list[Classification],
    since: date,
    run_date: date,
    all_sources: list[str] | None = None,
) -> str:
    lines = [
        f"![startin.lv]({LOGO_MARKDOWN_PATH})",
        "",
        f"# {TITLE}",
        f"#### {since.isoformat()} – {run_date.isoformat()}",
        "",
        f"> {DEFINITION}",
        "",
    ]

    if not classifications:
        lines.append("Šajā periodā nav atrasts neviens jaunuzņēmumiem atbilstošs ieraksts.")
        if all_sources:
            lines.append("")
            lines.append(f"*Pārbaudītie avoti: {_sources_checked_text(all_sources, classifications)}*")
        return "\n".join(lines)

    n_sources = len({c.item.source for c in classifications})
    lines.append(f"**{len(classifications)} atbilstoši ieraksti no {n_sources} avotiem.**")
    if all_sources:
        lines.append(f"*Pārbaudītie avoti: {_sources_checked_text(all_sources, classifications)}*")
    urgent_count = sum(1 for c in classifications if _is_urgent(c, run_date))
    if urgent_count:
        lines.append(f"*{urgent_count} ar termiņu tuvāko {URGENT_DEADLINE_DAYS} dienu laikā.*")
    lines.append("")

    for supergroup_label, in_supergroup in SUPERGROUPS:
        supergroup_items = [c for c in classifications if in_supergroup(c)]
        if not supergroup_items:
            continue
        lines.append(f"## {supergroup_label} ({len(supergroup_items)})")
        lines.append("")
        grouped = _group_by_source(supergroup_items)
        for source, items in sorted(grouped.items()):
            lines.append(f"### {source} ({len(items)})")
            lines.append("")
            for c in items:
                label = _category_label(c.category)
                title = _oneline(c.item.title)
                reason = _oneline(c.reason)
                deadline_part = _deadline_markdown(c, run_date)
                lines.append(f"- **[{title}]({c.item.url})** — {c.item.date}{deadline_part} · `{label}`")
                lines.append(f"  > {reason}")
                if c.verification_url:
                    lines.append(f"  > Pārbaudīts pret oriģinālo tekstu: [{c.verification_url}]({c.verification_url})")
            lines.append("")

    return "\n".join(lines)


def _logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html(
    classifications: list[Classification],
    since: date,
    run_date: date,
    all_sources: list[str] | None = None,
) -> str:
    def esc(s: str) -> str:
        return html_lib.escape(s)

    logo_uri = _logo_data_uri()
    logo_html = f"<img src='{logo_uri}' alt='startin.lv' class='logo'>" if logo_uri else ""

    body_parts = [
        "<header class='masthead'>",
        logo_html,
        f"<h1>{TITLE}</h1>",
        f"<p class='date-range'>{since.isoformat()} – {run_date.isoformat()}</p>",
        "</header>",
        f"<div class='scope'><p>{esc(DEFINITION)}</p></div>",
    ]

    if not classifications:
        body_parts.append("<p class='empty'>Šajā periodā nav atrasts neviens jaunuzņēmumiem atbilstošs ieraksts.</p>")
        if all_sources:
            body_parts.append(
                f"<p class='sources-checked'>Pārbaudītie avoti: "
                f"{esc(_sources_checked_text(all_sources, classifications))}</p>"
            )
    else:
        n_items = len(classifications)
        n_sources = len({c.item.source for c in classifications})
        item_word = "ieraksts" if n_items == 1 else "ieraksti"
        source_word = "avota" if n_sources == 1 else "avotiem"
        body_parts.append(
            f"<p class='lede'><strong>{n_items}</strong> atbilstoši {item_word} "
            f"no <strong>{n_sources}</strong> {source_word}</p>"
        )
        if all_sources:
            body_parts.append(
                f"<p class='sources-checked'>Pārbaudītie avoti: "
                f"{esc(_sources_checked_text(all_sources, classifications))}</p>"
            )
        urgent_count = sum(1 for c in classifications if _is_urgent(c, run_date))
        if urgent_count:
            urgent_word = "ierakstam" if urgent_count == 1 else "ierakstiem"
            body_parts.append(
                f"<p class='urgent-note'>{urgent_count} {urgent_word} termiņš tuvāko "
                f"{URGENT_DEADLINE_DAYS} dienu laikā</p>"
            )
        for supergroup_label, in_supergroup in SUPERGROUPS:
            supergroup_items = [c for c in classifications if in_supergroup(c)]
            if not supergroup_items:
                continue
            body_parts.append(
                f"<h2 class='supergroup'>{esc(supergroup_label)} "
                f"<span class='count'>{len(supergroup_items)}</span></h2>"
            )
            grouped = _group_by_source(supergroup_items)
            for source, items in sorted(grouped.items()):
                body_parts.append(f"<h3>{esc(source)} <span class='count'>{len(items)}</span></h3><ul>")
                for c in items:
                    label = _category_label(c.category)
                    deadline_html = _deadline_html(c, run_date, esc)
                    verified_html = (
                        f"<p class='verified'>Pārbaudīts pret oriģinālo tekstu: "
                        f"<a href='{esc(c.verification_url)}' target='_blank' rel='noopener'>{esc(c.verification_url)}</a></p>"
                        if c.verification_url else ""
                    )
                    body_parts.append(
                        "<li>"
                        "<div class='item-head'>"
                        f"<a href='{esc(c.item.url)}' target='_blank' rel='noopener'>{esc(_oneline(c.item.title))}</a>"
                        f"<span class='tag'>{esc(label)}</span>"
                        "</div>"
                        f"<div class='item-meta'>{c.item.date}{deadline_html}</div>"
                        f"<p class='reason'>{esc(_oneline(c.reason))}</p>"
                        f"{verified_html}"
                        "</li>"
                    )
                body_parts.append("</ul>")

    body = "\n".join(body_parts)
    return f"""<!DOCTYPE html>
<html lang="lv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE} — {run_date.isoformat()}</title>
<style>
  * {{ box-sizing: border-box; }}
  html {{ background: #eef0f3; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #eef0f3;
    color: #1c1d21;
    margin: 0;
    padding: clamp(1.5rem, 5vw, 4rem) 0;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{
    max-width: 900px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 0 1px 2px rgba(20, 20, 30, 0.04), 0 12px 32px rgba(20, 20, 30, 0.07);
    padding: clamp(1.75rem, 6vw, 3.25rem);
  }}
  .masthead {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 0.4rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #edeef1;
  }}
  .logo {{ height: 64px; width: auto; margin-bottom: 0.6rem; }}
  h1 {{
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.015em;
    margin: 0;
    line-height: 1.2;
  }}
  .date-range {{
    align-self: flex-end;
    font-size: 0.85rem;
    color: #8b8d97;
    margin: 0.2rem 0 0;
    font-weight: 500;
  }}
  .scope {{
    background: #f7f5f6;
    border-radius: 14px;
    padding: 1.1rem 1.35rem;
    margin-bottom: 1.75rem;
  }}
  .scope p {{
    margin: 0;
    font-size: 0.88rem;
    color: #5c5e68;
    line-height: 1.65;
  }}
  .lede {{
    font-size: 1rem;
    font-weight: 600;
    color: #1c1d21;
    margin: 0 0 0.25rem;
  }}
  .empty {{ color: #5c5e68; font-size: 0.95rem; }}
  .sources-checked {{
    font-size: 0.8rem;
    color: #9a9ca5;
    margin: 0.2rem 0 0;
  }}
  .urgent-note {{
    font-size: 0.85rem;
    font-weight: 700;
    color: #c0173f;
    margin: 0.2rem 0 0;
  }}
  h2.supergroup {{
    font-size: 1.05rem;
    font-weight: 800;
    color: #1c1d21;
    margin: 2.5rem 0 0.6rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #c0173f;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
  }}
  h2.supergroup:first-of-type {{ margin-top: 1.75rem; }}
  h3 {{
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #c0173f;
    margin: 1.5rem 0 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}
  h2.supergroup + h3 {{ margin-top: 1rem; }}
  .count {{
    color: #a9abb5;
    font-weight: 700;
    letter-spacing: normal;
    text-transform: none;
  }}
  ul {{ list-style: none; padding: 0; margin: 0; }}
  li {{
    padding: 1rem 0;
    border-bottom: 1px solid #f0f0f2;
  }}
  li:last-child {{ border-bottom: none; padding-bottom: 0.25rem; }}
  li:first-of-type {{ padding-top: 0; }}
  .item-head {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
  }}
  .item-head a {{
    font-size: 0.98rem;
    font-weight: 650;
    color: #1c1d21;
    text-decoration: none;
    line-height: 1.4;
  }}
  .item-head a:hover {{ color: #c0173f; }}
  .tag {{
    flex-shrink: 0;
    margin-top: 0.15rem;
    font-size: 0.66rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #c0173f;
    background: #fdedf1;
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    white-space: nowrap;
  }}
  .item-meta {{ font-size: 0.78rem; color: #9a9ca5; margin-top: 0.25rem; }}
  .deadline {{ font-weight: 700; color: #5c5e68; }}
  .deadline.urgent {{ color: #c0173f; }}
  .reason {{
    color: #4a4b54;
    margin: 0.4rem 0 0;
    font-size: 0.9rem;
    line-height: 1.55;
  }}
  .verified {{
    color: #1a7a4c;
    margin: 0.3rem 0 0;
    font-size: 0.78rem;
  }}
  .verified a {{ color: inherit; }}
  @media (max-width: 480px) {{
    .item-head {{ flex-direction: column; gap: 0.35rem; }}
  }}
</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>
"""
