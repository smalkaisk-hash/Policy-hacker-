"""Tests for verify.py's deep-verification pass — see the module docstring and CLAUDE.md
for the cybercrime-convention false positive that motivated it. Uses only unittest.mock,
no real API calls (see evals/classification_evals.py for a real-API smoke test).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_digest import verify
from policy_digest.classify import Classification
from policy_digest.sources.base import Item


def _legislative_item(url="http://saeima/1"):
    return Item(
        source="Saeimas komisiju darba kārtības",
        title="Budžeta un finanšu (nodokļu) komisijas sēde",
        url=url, date="2026-09-08",
        raw_text="Budžeta un finanšu (nodokļu) komisijas sēde\n\nLikumprojekts par Mikrouzņēmumu nodokļa likuma grozījumiem.",
    )


def _news_item(url="http://liaa/1"):
    return Item(
        source="LIAA", title="LIAA izsludina programmu", url=url, date="2026-09-08",
        raw_text="LIAA izsludina programmu\n\nPilns raksta teksts ar finansējuma detaļām.",
    )


def _relevant_classification(item):
    return Classification(
        item=item, relevant=True, confidence=0.9, category="tax_labor",
        reason="Likumprojekts skar mikrouzņēmumu nodokli jaunuzņēmumiem.",
    )


class _FakeToolUseBlock:
    def __init__(self, name, input_):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def _mock_client(content_blocks):
    client = MagicMock()
    message = MagicMock()
    message.content = content_blocks
    client.messages.create.return_value = message
    return client


class VerifyLegislativeItemsTests(unittest.TestCase):
    def test_no_api_key_leaves_classifications_untouched(self):
        c = _relevant_classification(_legislative_item())
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            out = verify.verify_legislative_items([c])
        self.assertEqual(out, [c])

    def test_non_legislative_source_is_never_sent_to_the_model(self):
        c = _relevant_classification(_news_item())
        client = _mock_client([])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        client.messages.create.assert_not_called()
        self.assertEqual(out, [c])

    def test_already_not_relevant_item_is_untouched(self):
        item = _legislative_item()
        c = Classification(item=item, relevant=False, confidence=0.9, category="other", reason="Nesaistīts.")
        client = _mock_client([])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        client.messages.create.assert_not_called()
        self.assertEqual(out, [c])

    def test_confirmed_relevant_item_keeps_relevant_and_gets_evidence(self):
        c = _relevant_classification(_legislative_item())
        block = _FakeToolUseBlock("record_verification", {
            "found_primary_source": True,
            "source_url": "https://likumi.lv/ta/id/12345",
            "evidence_quote": "Likums attiecas uz mikrouzņēmumiem, kuru gada apgrozījums nepārsniedz 40 000 eiro.",
            "verified_relevant": True,
            "reason": "Primārais teksts apstiprina mikrouzņēmumu/jaunuzņēmumu tvērumu.",
        })
        client = _mock_client([block])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].relevant)
        self.assertEqual(out[0].verification_url, "https://likumi.lv/ta/id/12345")
        self.assertIn("apgrozījums", out[0].verification_quote)
        self.assertNotIn("Atcelts", out[0].reason)

    def test_unconfirmed_item_is_overridden_to_not_relevant(self):
        # The real production case this module exists for: the primary text doesn't
        # confirm the provisional claim.
        c = _relevant_classification(_legislative_item())
        block = _FakeToolUseBlock("record_verification", {
            "found_primary_source": True,
            "source_url": "https://titania.saeima.lv/livs/saeimasnotikumi.nsf/0/801DFEB953442F6EC2258E67004B7810?OpenDocument",
            "evidence_quote": "Teksts ir tikai darba kārtības ieraksts bez satura par jaunuzņēmumiem.",
            "verified_relevant": False,
            "reason": "Primārais teksts neapstiprina jaunuzņēmumu/MVU tvērumu.",
        })
        client = _mock_client([block])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].relevant)
        self.assertIn("Atcelts", out[0].reason)
        self.assertIsNone(out[0].verification_url)

    def test_no_primary_source_found_is_overridden_to_not_relevant(self):
        c = _relevant_classification(_legislative_item())
        block = _FakeToolUseBlock("record_verification", {
            "found_primary_source": False,
            "evidence_quote": "Meklēšana neatrada pieejamu primāro tekstu.",
            "verified_relevant": False,
            "reason": "Nevarēja atrast primāro avotu.",
        })
        client = _mock_client([block])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        self.assertFalse(out[0].relevant)
        self.assertIn("Atcelts", out[0].reason)

    def test_api_failure_fails_closed_not_open(self):
        c = _relevant_classification(_legislative_item())
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("simulated outage")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].relevant)
        self.assertIn("Atcelts", out[0].reason)

    def test_no_usable_tool_result_fails_closed(self):
        c = _relevant_classification(_legislative_item())
        client = _mock_client([_FakeTextBlock("some stray text with no tool call")])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items([c])
        self.assertFalse(out[0].relevant)
        self.assertIn("Atcelts", out[0].reason)

    def test_mixed_batch_only_legislative_relevant_items_call_the_model(self):
        legislative_relevant = _relevant_classification(_legislative_item("http://saeima/1"))
        legislative_not_relevant = Classification(
            item=_legislative_item("http://saeima/2"), relevant=False, confidence=0.9,
            category="other", reason="Nesaistīts.",
        )
        news_relevant = _relevant_classification(_news_item("http://liaa/1"))

        block = _FakeToolUseBlock("record_verification", {
            "found_primary_source": True, "source_url": "https://likumi.lv/x",
            "evidence_quote": "...", "verified_relevant": True, "reason": "OK.",
        })
        client = _mock_client([block])
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}):
            with patch("anthropic.Anthropic", lambda api_key: client):
                out = verify.verify_legislative_items(
                    [legislative_relevant, legislative_not_relevant, news_relevant]
                )
        self.assertEqual(client.messages.create.call_count, 1)
        self.assertEqual(len(out), 3)
        # Order and identity of untouched items preserved.
        self.assertIs(out[1], legislative_not_relevant)
        self.assertIs(out[2], news_relevant)
        self.assertTrue(out[0].relevant)


if __name__ == "__main__":
    unittest.main()
