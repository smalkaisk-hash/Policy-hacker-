"""Tests for TAP portāls' .docx attachment parsing (2026-09-15) — the one gap called
out in CLAUDE.md/README as a known limitation: an act whose only document version is a
plain file attachment (not TAP's own "structuralizer" HTML preview) used to be skipped
entirely, so the classifier only ever saw the act's title. .docx is by far the most
common attachment type here, so it's now parsed directly via python-docx.

Unlike tests/test_reliability.py, this file needs python-docx (already a hard
dependency of the feature itself, see requirements.txt) to build real .docx fixtures
in memory. Run with:

    python -m unittest discover tests
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_digest.sources import tap_legal_acts as tap


def _docx_bytes(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


class ExtractDocxTextTests(unittest.TestCase):
    def test_extracts_paragraph_text_in_order(self):
        content = _docx_bytes(["Pirmā rindkopa.", "Otrā rindkopa ar detaļām."])
        text = tap._extract_docx_text(content)
        self.assertEqual(text, "Pirmā rindkopa.\nOtrā rindkopa ar detaļām.")

    def test_skips_empty_paragraphs(self):
        content = _docx_bytes(["Saturs.", "", "   ", "Vēl saturs."])
        text = tap._extract_docx_text(content)
        self.assertEqual(text, "Saturs.\nVēl saturs.")

    def test_malformed_content_returns_empty_string_not_crash(self):
        # Not a real .docx at all (e.g. an HTML error page served at the download URL
        # instead of the actual file) — must degrade gracefully like every other
        # untrusted-input path in this codebase, never raise.
        self.assertEqual(tap._extract_docx_text(b"not a docx file, just garbage bytes"), "")

    def test_empty_bytes_does_not_crash(self):
        self.assertEqual(tap._extract_docx_text(b""), "")


def _entry_with_versions(version_ids):
    return {"relationships": {"document_versions": {"data": [{"id": v} for v in version_ids]}}}


class FetchFullTextDocxFallbackTests(unittest.TestCase):
    """_fetch_full_text must prefer a structuralizer preview when one exists, fall back
    to parsing a .docx attachment when it doesn't, and still skip other attachment
    types (.pdf, .xlsx, ...) it can't parse — never crash on any of these paths."""

    def _fake_get(self, docx_content=b"", struct_html="", should_fail=False):
        def fake_get(url, headers=None, timeout=None):
            resp = MagicMock()
            if should_fail:
                import requests
                raise requests.RequestException("simulated network failure")
            resp.raise_for_status = lambda: None
            if "structuralizer" in url:
                resp.text = struct_html
            else:
                resp.content = docx_content
            return resp
        return fake_get

    def test_falls_back_to_docx_when_no_structuralizer_preview(self):
        entry = _entry_with_versions(["v1"])
        doc_version_map = {
            "v1": {"attributes": {"items": [
                {"url": "https://tapportals.mk.gov.lv/attachments/.../download",
                 "file_name": "LMzin_260526.docx"},
            ]}}
        }
        content = _docx_bytes(["Informatīvais ziņojums par jaunuzņēmumu atbalstu."])
        with patch("policy_digest.sources.tap_legal_acts.requests.get", self._fake_get(docx_content=content)):
            text = tap._fetch_full_text(entry, doc_version_map)
        self.assertIn("jaunuzņēmumu atbalstu", text)

    def test_prefers_structuralizer_preview_over_docx_when_both_present(self):
        entry = _entry_with_versions(["v1"])
        doc_version_map = {
            "v1": {"attributes": {"items": [
                {"url": "https://tapportals.mk.gov.lv/structuralizer/data/nodes/abc-123/preview",
                 "file_name": ""},
                {"url": "https://tapportals.mk.gov.lv/attachments/.../download",
                 "file_name": "other.docx"},
            ]}}
        }
        html = '<div class="structuralizer-tree">Struktūrā renderēts saturs.</div>'
        with patch("policy_digest.sources.tap_legal_acts.requests.get", self._fake_get(struct_html=html)):
            text = tap._fetch_full_text(entry, doc_version_map)
        self.assertEqual(text, "Struktūrā renderēts saturs.")

    def test_skips_unparseable_attachment_types(self):
        entry = _entry_with_versions(["v1"])
        doc_version_map = {
            "v1": {"attributes": {"items": [
                {"url": "https://tapportals.mk.gov.lv/attachments/.../download",
                 "file_name": "annex.pdf"},
            ]}}
        }
        with patch("policy_digest.sources.tap_legal_acts.requests.get", self._fake_get()):
            text = tap._fetch_full_text(entry, doc_version_map)
        self.assertEqual(text, "")

    def test_download_failure_does_not_crash(self):
        entry = _entry_with_versions(["v1"])
        doc_version_map = {
            "v1": {"attributes": {"items": [
                {"url": "https://tapportals.mk.gov.lv/attachments/.../download",
                 "file_name": "x.docx"},
            ]}}
        }
        with patch("policy_digest.sources.tap_legal_acts.requests.get", self._fake_get(should_fail=True)):
            text = tap._fetch_full_text(entry, doc_version_map)
        self.assertEqual(text, "")

    def test_missing_url_does_not_crash(self):
        entry = _entry_with_versions(["v1"])
        doc_version_map = {
            "v1": {"attributes": {"items": [{"file_name": "x.docx"}]}}  # no "url" key at all
        }
        text = tap._fetch_full_text(entry, doc_version_map)
        self.assertEqual(text, "")

    def test_unknown_version_id_does_not_crash(self):
        entry = _entry_with_versions(["missing-version"])
        text = tap._fetch_full_text(entry, {})
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
