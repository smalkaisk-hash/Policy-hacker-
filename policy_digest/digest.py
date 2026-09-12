"""Renders classified items into a Markdown + HTML digest."""

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
    '"Startup-relevant" = funding/support programs (grants, EU funds, accelerator/incubator '
    "programs, LIAA/Altum initiatives), legal or regulatory changes affecting startups/SMEs/tech "
    "companies (company law, tax treatment, employee stock options, labor law, digital/AI "
    "regulation, public procurement), or draft legislation/initiatives on innovation, "
    "digitalization, and entrepreneurship."
)

# label, text color, background color — used by the HTML category badges;
# the label alone is reused in the Markdown render.
CATEGORY_META = {
    "funding": ("Funding", "#047857", "#d1fae5"),
    "regulation": ("Regulation", "#1d4ed8", "#dbeafe"),
    "tax_labor": ("Tax & Labor", "#b45309", "#fef3c7"),
    "digitalization_innovation": ("Digital & Innovation", "#7e22ce", "#f3e8ff"),
    "other": ("Other", "#475569", "#e2e8f0"),
}
ACCENT_BORDER = {cat: meta[1] for cat, meta in CATEGORY_META.items()}


def _category_meta(category: str) -> tuple[str, str, str]:
    return CATEGORY_META.get(category, CATEGORY_META["other"])


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
        f"# Policy Digest — {run_date.isoformat()}",
        "",
        f"> Window: {since.isoformat()} to {run_date.isoformat()}.",
        f"> {DEFINITION}",
        "",
    ]

    if not classifications:
        lines.append("No startup-relevant items found in this window.")
        return "\n".join(lines)

    grouped = _group_by_source(classifications)
    lines.append(f"**{len(classifications)} relevant item(s) across {len(grouped)} source(s).**")
    lines.append("")

    for source, items in sorted(grouped.items()):
        lines.append(f"## {source} ({len(items)})")
        lines.append("")
        for c in items:
            label, _, _ = _category_meta(c.category)
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
        "<div class='masthead-text'>",
        "<span class='eyebrow'>Policy Monitoring Digest</span>",
        f"<h1>{run_date.isoformat()}</h1>",
        "</div>",
        "</header>",
        "<div class='meta-box'>",
        f"<p><strong>Window</strong> {since.isoformat()} &rarr; {run_date.isoformat()}</p>",
        f"<p>{esc(DEFINITION)}</p>",
        "</div>",
    ]

    if not classifications:
        body_parts.append("<p class='empty'>No startup-relevant items found in this window.</p>")
    else:
        grouped = _group_by_source(classifications)
        body_parts.append(
            "<div class='stats'>"
            f"<div class='stat'><span class='stat-num'>{len(classifications)}</span>"
            "<span class='stat-label'>relevant items</span></div>"
            f"<div class='stat'><span class='stat-num'>{len(grouped)}</span>"
            "<span class='stat-label'>sources</span></div>"
            "</div>"
        )
        for source, items in sorted(grouped.items()):
            body_parts.append(
                f"<section class='source'><h2>{esc(source)} "
                f"<span class='count'>{len(items)}</span></h2><ul>"
            )
            for c in items:
                label, color, bg = _category_meta(c.category)
                body_parts.append(
                    f"<li class='item' style='border-left-color:{color}'>"
                    "<div class='item-head'>"
                    f"<a href='{esc(c.item.url)}' target='_blank' rel='noopener'>{esc(c.item.title)}</a>"
                    f"<span class='badge' style='color:{color};background:{bg}'>{esc(label)}</span>"
                    "</div>"
                    f"<div class='date'>{c.item.date}</div>"
                    f"<p class='reason'>{esc(c.reason)}</p>"
                    "</li>"
                )
            body_parts.append("</ul></section>")

    body = "\n".join(body_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy Digest — {run_date.isoformat()}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: #f4f4f6;
    color: #17181c;
    margin: 0;
    padding: 2.5rem 1rem;
  }}
  .page {{
    max-width: 840px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(15, 15, 20, 0.06), 0 8px 24px rgba(15, 15, 20, 0.06);
    padding: 2.75rem 3rem 3rem;
  }}
  .masthead {{
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding-bottom: 1.5rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #ececef;
  }}
  .logo {{ height: 40px; width: auto; flex-shrink: 0; }}
  .masthead-text {{ display: flex; flex-direction: column; }}
  .eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    font-weight: 700;
    color: #c2255c;
  }}
  h1 {{ margin: 0.1rem 0 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.01em; }}
  .meta-box {{
    background: #faf7f8;
    border: 1px solid #f0e4e8;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #55565c;
    line-height: 1.5;
  }}
  .meta-box p {{ margin: 0.25rem 0; }}
  .meta-box strong {{ color: #17181c; }}
  .stats {{ display: flex; gap: 2rem; margin: 1.75rem 0 0.5rem; }}
  .stat {{ display: flex; flex-direction: column; }}
  .stat-num {{ font-size: 2rem; font-weight: 800; color: #c2255c; line-height: 1; }}
  .stat-label {{ font-size: 0.8rem; color: #77787f; margin-top: 0.15rem; }}
  .empty {{ color: #55565c; margin-top: 1.5rem; }}
  h2 {{
    font-size: 1.05rem;
    font-weight: 700;
    margin: 2.5rem 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .count {{
    background: #17181c;
    color: #fff;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.1rem 0.55rem;
  }}
  ul {{ list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.7rem; }}
  .item {{
    background: #fbfbfc;
    border: 1px solid #ececef;
    border-left: 4px solid #ccc;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
  }}
  .item:hover {{ box-shadow: 0 4px 14px rgba(15, 15, 20, 0.08); transform: translateY(-1px); }}
  .item-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap; }}
  .item a {{ font-weight: 650; color: #17181c; text-decoration: none; font-size: 0.98rem; }}
  .item a:hover {{ color: #c2255c; text-decoration: underline; }}
  .badge {{
    display: inline-block;
    border-radius: 999px;
    padding: 0.12rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 700;
    white-space: nowrap;
  }}
  .date {{ color: #9a9ba1; font-size: 0.78rem; margin-top: 0.15rem; }}
  .reason {{ color: #45464c; margin: 0.45rem 0 0; font-size: 0.9rem; line-height: 1.5; }}
</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>
"""
