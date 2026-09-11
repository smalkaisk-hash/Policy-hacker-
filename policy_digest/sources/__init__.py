from .altum_news import fetch_altum_news
from .base import Item
from .mk_meetings import fetch_mk_meetings
from .news_listing import fetch_news_listing
from .saeima_committees import fetch_saeima_committees
from .tap_legal_acts import fetch_tap_legal_acts

__all__ = [
    "Item",
    "fetch_tap_legal_acts",
    "fetch_mk_meetings",
    "fetch_news_listing",
    "fetch_altum_news",
    "fetch_saeima_committees",
]
