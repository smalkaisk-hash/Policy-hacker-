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
    "funding": ("Finansējums", "#a3123c"),
    "regulation": ("Regulējums", "#a3123c"),
    "tax_labor": ("Nodokļi un darbs", "#a3123c"),
    "digitalization_innovation": ("Digitalizācija un inovācijas", "#a3123c"),
    "other": ("Cits", "#6b6b70"),
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
        f"# Politikas apkopojums — {run_date.isoformat()}",
        "",
        f"> Periods: {since.isoformat()} — {run_date.isoformat()}.",
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
        "<div class='rule'></div>",
        f"<time>{run_date.isoformat()}</time>",
        "</header>",
        f"<h1>Politikas apkopojums</h1>",
        f"<p class='dek'>Periods: {since.isoformat()} — {run_date.isoformat()}. {esc(DEFINITION)}</p>",
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
            f"no <strong>{n_sources}</strong> {source_word}.</p>"
        )
        for source, items in sorted(grouped.items()):
            body_parts.append(f"<h2>{esc(source)} <span class='count'>{len(items)}</span></h2><ul>")
            for c in items:
                label = _category_label(c.category)
                body_parts.append(
                    "<li>"
                    "<div class='item-head'>"
                    f"<a href='{esc(c.item.url)}' target='_blank' rel='noopener'>{esc(c.item.title)}</a>"
                    "</div>"
                    f"<div class='item-meta'>{c.item.date} &nbsp;&middot;&nbsp; "
                    f"<span class='tag'>{esc(label)}</span></div>"
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
<title>Politikas apkopojums — {run_date.isoformat()}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    background: #f2f1ee;
    color: #201f1d;
    margin: 0;
    padding: 3rem 1.25rem 5rem;
  }}
  .page {{
    max-width: 700px;
    margin: 0 auto;
    background: #fffdfb;
    padding: 0 0 1rem;
  }}
  .masthead {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
  }}
  .logo {{ height: 30px; width: auto; filter: grayscale(1); opacity: 0.85; }}
  .rule {{ flex: 1; height: 1px; background: #201f1d; }}
  time {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: #6b6b70;
    text-transform: uppercase;
  }}
  h1 {{
    font-size: 2.3rem;
    font-weight: 400;
    margin: 0 0 0.6rem;
    letter-spacing: -0.01em;
  }}
  .dek {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 0.92rem;
    color: #4a4a4d;
    line-height: 1.6;
    border-top: 1px solid #ddd9d2;
    border-bottom: 1px solid #ddd9d2;
    padding: 0.9rem 0;
    margin: 0 0 1.6rem;
  }}
  .lede {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.05rem;
    margin: 0 0 2rem;
  }}
  .empty {{ font-family: "Helvetica Neue", Arial, sans-serif; color: #4a4a4d; }}
  h2 {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #a3123c;
    margin: 2.6rem 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #201f1d;
  }}
  .count {{ color: #a19f98; font-weight: 400; }}
  ul {{ list-style: none; padding: 0; margin: 0; }}
  li {{
    padding: 1.1rem 0;
    border-bottom: 1px solid #e6e2da;
  }}
  li:first-child {{ padding-top: 0; }}
  .item-head a {{
    font-family: Georgia, serif;
    font-size: 1.12rem;
    font-weight: 700;
    color: #201f1d;
    text-decoration: none;
    line-height: 1.35;
  }}
  .item-head a:hover {{ color: #a3123c; text-decoration: underline; }}
  .item-meta {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 0.76rem;
    color: #8a8a8f;
    margin-top: 0.3rem;
  }}
  .tag {{
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700;
    color: #a3123c;
  }}
  .reason {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: #3a3a3d;
    margin: 0.5rem 0 0;
    font-size: 0.94rem;
    line-height: 1.55;
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
