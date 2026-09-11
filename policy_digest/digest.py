"""Renders classified items into a Markdown + HTML digest."""

import html as html_lib
from collections import defaultdict
from datetime import date

from .classify import Classification

DEFINITION = (
    '"Startup-relevant" = funding/support programs (grants, EU funds, accelerator/incubator '
    "programs, LIAA/Altum initiatives), legal or regulatory changes affecting startups/SMEs/tech "
    "companies (company law, tax treatment, employee stock options, labor law, digital/AI "
    "regulation, public procurement), or draft legislation/initiatives on innovation, "
    "digitalization, and entrepreneurship."
)


def _group_by_source(classifications: list[Classification]) -> dict[str, list[Classification]]:
    grouped: dict[str, list[Classification]] = defaultdict(list)
    for c in classifications:
        grouped[c.item.source].append(c)
    for source_items in grouped.values():
        source_items.sort(key=lambda c: c.item.date, reverse=True)
    return grouped


def render_markdown(classifications: list[Classification], since: date, run_date: date) -> str:
    lines = [
        f"# Policy Digest — {run_date.isoformat()}",
        "",
        f"_Window: {since.isoformat()} to {run_date.isoformat()}. {DEFINITION}_",
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
            lines.append(f"- **[{c.item.title}]({c.item.url})** — {c.item.date} · _{c.category}_")
            lines.append(f"  {c.reason}")
        lines.append("")

    return "\n".join(lines)


def render_html(classifications: list[Classification], since: date, run_date: date) -> str:
    def esc(s: str) -> str:
        return html_lib.escape(s)

    body_parts = [
        f"<h1>Policy Digest — {run_date.isoformat()}</h1>",
        f"<p class='meta'>Window: {since.isoformat()} to {run_date.isoformat()}.<br>{esc(DEFINITION)}</p>",
    ]

    if not classifications:
        body_parts.append("<p>No startup-relevant items found in this window.</p>")
    else:
        grouped = _group_by_source(classifications)
        body_parts.append(
            f"<p class='summary'><strong>{len(classifications)} relevant item(s)"
            f" across {len(grouped)} source(s).</strong></p>"
        )
        for source, items in sorted(grouped.items()):
            body_parts.append(f"<h2>{esc(source)} ({len(items)})</h2><ul>")
            for c in items:
                body_parts.append(
                    "<li>"
                    f"<a href='{esc(c.item.url)}' target='_blank' rel='noopener'>{esc(c.item.title)}</a>"
                    f" <span class='date'>{c.item.date}</span>"
                    f" <span class='category'>{esc(c.category)}</span>"
                    f"<div class='reason'>{esc(c.reason)}</div>"
                    "</li>"
                )
            body_parts.append("</ul>")

    body = "\n".join(body_parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Policy Digest — {run_date.isoformat()}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 780px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .meta {{ color: #555; font-size: 0.9rem; }}
  .summary {{ margin-top: 1rem; }}
  h2 {{ border-bottom: 2px solid #eee; padding-bottom: 0.25rem; margin-top: 2rem; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: 0.75rem 0; border-bottom: 1px solid #eee; }}
  a {{ font-weight: 600; color: #0b5fff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .date {{ color: #888; font-size: 0.85rem; margin-left: 0.5rem; }}
  .category {{ display: inline-block; background: #eef2ff; color: #3346a6; border-radius: 999px;
               padding: 0.1rem 0.6rem; font-size: 0.75rem; margin-left: 0.5rem; }}
  .reason {{ color: #444; margin-top: 0.25rem; font-size: 0.9rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
