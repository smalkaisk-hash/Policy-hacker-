"""Shared data model for everything the fetchers pull in."""

from dataclasses import dataclass


@dataclass
class Item:
    source: str  # e.g. "TAP portāls", "Ekonomikas ministrija"
    title: str
    url: str
    date: str  # ISO date string, YYYY-MM-DD (best effort if source is imprecise)
    summary: str = ""  # short excerpt/description, if the source provides one
    raw_text: str = ""  # fuller text used for classification when available

    @property
    def id(self) -> str:
        return self.url
