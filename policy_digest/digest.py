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
    '"Startapiem atbilstošs" = finansējuma un atbalsta programmas (granti, ES fondi, '
    "akseleratoru/inkubatoru programmas, LIAA/Altum iniciatīvas), tiesiskas vai regulatīvas "
    "izmaiņas, kas skar startapus, MVU vai tehnoloģiju uzņēmumus (komerctiesības, nodokļu "
    "režīms, darba tiesības, digitālo pakalpojumu/MI regulējums, publiskie iepirkumi), vai "
    "likumprojekti/iniciatīvas par inovācijām, digitalizāciju un uzņēmējdarbību."
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


def _category_label(category: str) -> str:
    return CATEGORY_META.get(category, CATEGORY_META["other"])[0]


def _group_by_source(classifications: list[Classification]) -> dict[str, list[Classification]]:
    grouped: dict[str, list[Classification]] = defaultdict(list)
    for c in classifications:
        grouped[c.item.source].append(c)
    for source_items in grouped.values():
        source_items.sort(key=lambda c: c.item.date, reverse=True)
    return grouped


def render_markdown(classifications: list[Classification], since: date, run_date: date) -> str:
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
        lines.append("Šajā periodā nav atrasts neviens startapiem atbilstošs ieraksts.")
        return "\n".join(lines)

    grouped = _group_by_source(classifications)
    lines.append(f"**{len(classifications)} atbilstoši ieraksti no {len(grouped)} avotiem.**")
    lines.append("")

    for source, items in sorted(grouped.items()):
        lines.append(f"## {source} ({len(items)})")
        lines.append("")
        for c in items:
            label = _category_label(c.category)
            lines.append(f"- **[{c.item.title}]({c.item.url})** — {c.item.date} · `{label}`")
            lines.append(f"  > {c.reason}")
        lines.append("")

    return "\n".join(lines)


def _logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_html(classifications: list[Classification], since: date, run_date: date) -> str:
    def esc(s: str) -> str:
        return html_lib.escape(s)

    logo_uri = _logo_data_uri()
    logo_html = f"<img src='{logo_uri}' alt='startin.lv' class='logo'>" if logo_uri else ""

    body_parts = [
        "<header class='masthead'>",
        logo_html,
        "<div>",
        f"<h1>{TITLE}</h1>",
        f"<p class='date-range'>{since.isoformat()} – {run_date.isoformat()}</p>",
        "</div>",
        "</header>",
        f"<div class='scope'><p>{esc(DEFINITION)}</p></div>",
    ]

    if not classifications:
        body_parts.append("<p class='empty'>Šajā periodā nav atrasts neviens startapiem atbilstošs ieraksts.</p>")
    else:
        grouped = _group_by_source(classifications)
        n_items, n_sources = len(classifications), len(grouped)
        item_word = "ieraksts" if n_items == 1 else "ieraksti"
        source_word = "avota" if n_sources == 1 else "avotiem"
        body_parts.append(
            f"<p class='lede'><strong>{n_items}</strong> atbilstoši {item_word} "
            f"no <strong>{n_sources}</strong> {source_word}</p>"
        )
        for source, items in sorted(grouped.items()):
            body_parts.append(f"<h2>{esc(source)} <span class='count'>{len(items)}</span></h2><ul>")
            for c in items:
                label = _category_label(c.category)
                body_parts.append(
                    "<li>"
                    "<div class='item-head'>"
                    f"<a href='{esc(c.item.url)}' target='_blank' rel='noopener'>{esc(c.item.title)}</a>"
                    f"<span class='tag'>{esc(label)}</span>"
                    "</div>"
                    f"<div class='item-meta'>{c.item.date}</div>"
                    f"<p class='reason'>{esc(c.reason)}</p>"
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
    max-width: 700px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 0 1px 2px rgba(20, 20, 30, 0.04), 0 12px 32px rgba(20, 20, 30, 0.07);
    padding: clamp(1.75rem, 6vw, 3.25rem);
  }}
  .masthead {{
    display: flex;
    align-items: center;
    gap: 1.1rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #edeef1;
  }}
  .logo {{ height: 34px; width: auto; flex-shrink: 0; }}
  h1 {{
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.015em;
    margin: 0;
    line-height: 1.2;
  }}
  .date-range {{
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
  h2 {{
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #c0173f;
    margin: 2.25rem 0 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }}
  h2:first-of-type {{ margin-top: 1.75rem; }}
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
  .reason {{
    color: #4a4b54;
    margin: 0.4rem 0 0;
    font-size: 0.9rem;
    line-height: 1.55;
  }}
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
