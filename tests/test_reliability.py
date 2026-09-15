"""Regression tests for reliability bugs found by adversarial testing of the pipeline
(2026-09-14). Each test corresponds to a failure mode that used to crash the whole run
or silently and permanently lose items — see CLAUDE.md's "Gotchas hit while building
this" section and the commit that introduced this file for the full writeup.

Uses only the standard library (unittest + unittest.mock) so it needs no extra
dependency beyond what's already in requirements.txt. Run with:

    python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

# Several of the modules under test print warnings containing Latvian text (e.g. source
# names like "TAP portāls") when they hit a malformed record — see CLAUDE.md: "Windows
# console can't print Latvian text by default (cp1252)". run_digest.py's entry point
# already reconfigures stdout for this; this test suite is effectively its own entry
# point (it calls those modules directly, without going through run_digest.py), so it
# needs the same fix or those warnings crash the test run on Windows instead of just
# being printed.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy_digest import classify, dedupe, state
from policy_digest.digest import render_markdown
from policy_digest.classify import Classification
from policy_digest.sources import tap_legal_acts as tap
from policy_digest.sources.base import Item
from policy_digest.sources.mk_meetings import _parse_meeting_date


def _items():
    return [
        Item(source="A", title="Test 1", url="http://a/1", date="2026-09-10",
             raw_text="Test 1\n\nSome body text about jaunuzņēmumi funding."),
        Item(source="B", title="Test 2", url="http://b/2", date="2026-09-10",
             raw_text="Test 2\n\nAnother body about something else entirely."),
    ]


# At least classify.MIN_SUBSTANTIVE_BODY_LEN (150) chars — these malformed-response/
# retry/confidence tests are about response-parsing and confidence filtering, not about
# the separate discussion-only-event backstop (see DiscussionOnlyEventGateTests below),
# so a body missing that threshold here would make relevant=True results fail for the
# wrong reason.
_SUBSTANTIVE_BODY = (
    "Item 0\n\nThis is a sufficiently long article body describing a real government "
    "decision to fund jaunuzņēmumi in enough detail that it clears the no-substantive-"
    "body threshold used elsewhere in classify.py, so these tests exercise only what "
    "they intend to."
)


class FakeAnthropicRaises:
    """Simulates a transient API failure (rate limit, timeout, overload, ...)."""

    def __init__(self, api_key=None):
        self.messages = MagicMock()
        self.messages.create.side_effect = RuntimeError("simulated API outage")


class FakeAnthropicWithFixedResults:
    """Returns a fixed tool-use response for whatever batch it's asked to classify —
    used to test classify_items()'s confidence-based filtering without hitting the real
    API. Assumes a single batch (len(items) <= classify.BATCH_SIZE)."""

    def __init__(self, results_by_index):
        self._results_by_index = results_by_index
        self.messages = MagicMock()
        self.messages.create.side_effect = self._create

    def _create(self, **kwargs):
        block = MagicMock()
        block.type = "tool_use"
        block.input = {"results": self._results_by_index}
        message = MagicMock()
        message.content = [block]
        message.stop_reason = "tool_use"
        return message


def _confidence_test_items():
    return [
        Item(source="A", title="Confident item", url="http://a/1", date="2026-09-10",
             raw_text="Confident item\n\n" + _SUBSTANTIVE_BODY),
        Item(source="B", title="Shaky item", url="http://b/2", date="2026-09-10",
             raw_text="Shaky item\n\n" + _SUBSTANTIVE_BODY),
    ]


class ConfidenceFilterTests(unittest.TestCase):
    """Caught in production 2026-09-15: tightening SYSTEM_PROMPT reduced but didn't
    eliminate the hedge-word/category-truism failure — the model sometimes still says
    relevant=True with weak, hedge-qualified reasoning (confirmed: confidence 0.60-0.75
    on those cases vs. 0.92-0.98 on solid ones). Per product direction, those items must
    not reach the digest at all — held back entirely, not shown with a caveat — so
    classify_items() enforces MIN_CONFIDENCE_TO_INCLUDE as a hard filter, not just a
    display hint."""

    def _run(self, results_by_index):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("anthropic.Anthropic", lambda api_key: FakeAnthropicWithFixedResults(results_by_index)):
                return classify.classify_items(_confidence_test_items())

    def test_high_confidence_relevant_item_is_included(self):
        out, _ = self._run([
            {"index": 0, "relevant": True, "confidence": 0.95, "category": "funding", "reason": "Solid."},
            {"index": 1, "relevant": False, "confidence": 0.95, "category": "other", "reason": "No."},
        ])
        self.assertEqual([c.item.url for c in out], ["http://a/1"])

    def test_low_confidence_relevant_item_is_held_back(self):
        out, unclassified = self._run([
            {"index": 0, "relevant": True, "confidence": 0.6, "category": "funding", "reason": "Reaching."},
            {"index": 1, "relevant": False, "confidence": 0.95, "category": "other", "reason": "No."},
        ])
        self.assertEqual(out, [])
        # Held back, not "unclassified" — it WAS classified, just not confident enough
        # to show; must not be retried next run as if it never got a verdict.
        self.assertEqual(unclassified, [])

    def test_confidence_exactly_at_threshold_is_included(self):
        out, _ = self._run([
            {"index": 0, "relevant": True, "confidence": classify.MIN_CONFIDENCE_TO_INCLUDE,
             "category": "funding", "reason": "Right at the line."},
            {"index": 1, "relevant": False, "confidence": 0.95, "category": "other", "reason": "No."},
        ])
        self.assertEqual([c.item.url for c in out], ["http://a/1"])

    def test_low_confidence_not_relevant_item_stays_excluded_for_its_own_reason(self):
        # A NOT-relevant item's confidence doesn't matter — it's excluded either way, and
        # must not somehow end up included by a confidence-filter bug.
        out, _ = self._run([
            {"index": 0, "relevant": False, "confidence": 0.4, "category": "other", "reason": "Not it."},
            {"index": 1, "relevant": False, "confidence": 0.95, "category": "other", "reason": "No."},
        ])
        self.assertEqual(out, [])

    def test_keyword_fallback_path_is_unaffected_by_confidence_filter(self):
        # No API key -> keyword fallback path, whose confidence=0.5 is a flat
        # placeholder, not a calibrated score. MIN_CONFIDENCE_TO_INCLUDE (0.75) must only
        # gate the real LLM path — applying it here would silently disable the entire
        # no-key fallback mode (every match would be < 0.75 and vanish).
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            out, unclassified = classify.classify_items(_items())
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].item.title, "Test 1")
        self.assertEqual(out[0].confidence, 0.5)
        self.assertEqual(unclassified, [])


class ClassifyApiFailureTests(unittest.TestCase):
    """A transient Anthropic API failure must degrade gracefully, not crash the run
    or silently mark the affected items as already-processed (they'd be lost forever —
    see run_digest.py's seen-tracking, which relies on classify_items reporting which
    items it failed to get a verdict for)."""

    def test_classify_items_does_not_raise_on_api_failure(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("anthropic.Anthropic", FakeAnthropicRaises):
                classifications, unclassified = classify.classify_items(_items())
        self.assertEqual(classifications, [])
        self.assertEqual({it.url for it in unclassified}, {"http://a/1", "http://b/2"})

    def test_dedupe_semantic_does_not_raise_on_api_failure(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            with patch("anthropic.Anthropic", FakeAnthropicRaises):
                result = dedupe.dedupe_semantic(_items())
        # Safe fallback: items pass through unmerged rather than the whole digest failing.
        self.assertEqual(len(result), 2)


class StateFileTests(unittest.TestCase):
    """state.json is a best-effort dedupe cache. Corruption in it (e.g. from a run
    killed mid-write) must never crash every future run, and writes must be atomic so
    a kill mid-write can't corrupt it in the first place."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / "state.json"

    def test_corrupt_state_file_does_not_crash_load(self):
        self.path.write_text('["http://a/1", "http://a/2"', encoding="utf-8")  # truncated
        self.assertEqual(state.load_seen(self.path), set())

    def test_save_then_load_round_trip(self):
        urls = {"http://x/1", "http://x/2"}
        state.save_seen(self.path, urls)
        self.assertEqual(state.load_seen(self.path), urls)

    def test_save_is_atomic_no_leftover_tmp_file(self):
        state.save_seen(self.path, {"http://x/1"})
        leftovers = list(self.path.parent.glob(f"{self.path.name}*.tmp"))
        self.assertEqual(leftovers, [])

    def test_failed_write_leaves_original_file_untouched(self):
        state.save_seen(self.path, {"http://original/1"})
        with patch("json.dump", side_effect=KeyboardInterrupt("simulated interrupt")):
            with self.assertRaises(KeyboardInterrupt):
                state.save_seen(self.path, {"http://should-not-be-saved/1"})
        self.assertEqual(state.load_seen(self.path), {"http://original/1"})
        leftovers = list(self.path.parent.glob(f"{self.path.name}*.tmp"))
        self.assertEqual(leftovers, [])


class MkMeetingsDateParsingTests(unittest.TestCase):
    """Real government sites often separate date/time with a non-breaking space
    (U+00A0), not a plain space — a plain .split(" ") silently fails on that and drops
    the row with no trace."""

    def test_parses_plain_space_separated_date(self):
        self.assertEqual(_parse_meeting_date("10.09.2026. 16:00"), date(2026, 9, 10))

    def test_parses_nbsp_separated_date(self):
        self.assertEqual(_parse_meeting_date("10.09.2026. 16:00"), date(2026, 9, 10))

    def test_returns_none_not_crash_on_garbage(self):
        self.assertIsNone(_parse_meeting_date(""))
        self.assertIsNone(_parse_meeting_date("not a date"))


def _mock_batch_response(input_dict, block_type="tool_use", stop_reason="end_turn"):
    block = MagicMock()
    block.type = block_type
    block.input = input_dict
    message = MagicMock()
    message.content = [block]
    message.stop_reason = stop_reason
    return message


class ClassifyMalformedResponseTests(unittest.TestCase):
    """The LLM's tool-use response is untrusted input — its shape isn't guaranteed to
    match the JSON schema exactly (see CLAUDE.md: a real batch once came back with every
    "index" as a numeric string). A single malformed batch response must not crash the
    whole run, the way an outright API failure must not either."""

    def _run(self, input_dict, block_type="tool_use", stop_reason="end_turn"):
        client = MagicMock()
        client.messages.create.return_value = _mock_batch_response(input_dict, block_type, stop_reason)
        batch = [Item(source="A", title="Item 0", url="http://a/0", date="2026-09-10",
                       raw_text=_SUBSTANTIVE_BODY)]
        return classify._classify_batch_with_llm(client, batch)

    def test_results_explicitly_null_does_not_crash(self):
        # .get("results", []) only substitutes the default for a MISSING key — a response
        # with "results": null still passes a bare None through, which used to crash
        # enumerate() below it.
        out, unclassified = self._run({"results": None})
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_missing_results_key_does_not_crash(self):
        out, unclassified = self._run({})
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_non_dict_result_entries_do_not_crash(self):
        out, unclassified = self._run({"results": ["oops", 42, None]})
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_no_tool_use_block_does_not_crash(self):
        out, unclassified = self._run({}, block_type="text")
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_null_reason_does_not_crash_and_defaults_to_empty(self):
        # "reason": null is present-but-null. r.get("reason", "") only substitutes the
        # default for a MISSING key, so a bare None used to flow all the way to
        # render_markdown()/render_html() and crash there (str.split() on None) — the very
        # last step of the pipeline, after fetch/dedupe/classify had all already succeeded.
        out, _ = self._run({"results": [
            {"index": 0, "relevant": True, "confidence": 0.9, "reason": None, "category": "funding"}
        ]})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].reason, "")

    def test_all_fields_null_does_not_crash(self):
        out, _ = self._run({"results": [
            {"index": 0, "relevant": None, "confidence": None, "reason": None, "category": None}
        ]})
        self.assertEqual(len(out), 1)
        c = out[0]
        self.assertEqual((c.relevant, c.confidence, c.reason, c.category), (False, 0.5, "", "other"))
        # Must render without crashing too (defense in depth at the render layer itself).
        render_markdown(out, date(2026, 9, 7), date(2026, 9, 14))

    def test_results_as_double_encoded_json_string_is_recovered(self):
        # Seen in practice on a real 10-item batch of long Saeima committee agenda items:
        # the model wrapped its own valid JSON array in quotes, returning "results" as a
        # string instead of a native array. Without recovery, enumerate() iterates the
        # string character-by-character — thousands of spurious "not an object" warnings
        # and the whole batch wrongly treated as unclassified even though the model's
        # answer was actually fine.
        payload = json.dumps([
            {"index": 0, "relevant": True, "confidence": 0.82, "reason": "ok", "category": "other"}
        ])
        out, unclassified = self._run({"results": payload})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].relevant, True)
        self.assertEqual(unclassified, [])

    def test_results_as_unparseable_string_does_not_crash(self):
        out, unclassified = self._run({"results": "not valid json at all {["})
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_results_as_json_string_decoding_to_non_list_does_not_crash(self):
        # Valid JSON, but not an array (e.g. a bare object) — must not be treated as a
        # one-item-per-character iterable either.
        out, unclassified = self._run({"results": json.dumps({"not": "a list"})})
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_max_tokens_truncated_batch_does_not_crash(self):
        # A batch whose output was cut off mid-JSON by the max_tokens limit — the
        # incomplete "results" string won't parse as JSON at all (unlike the
        # double-encoded-but-complete case above). Must degrade to "unclassified",
        # not crash, regardless of whether the truncation is correctly diagnosed.
        incomplete = '[{"index": 0, "relevant": true, "confidence": 0.9, "reason": "cut off mid-s'
        out, unclassified = self._run({"results": incomplete}, stop_reason="max_tokens")
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])

    def test_unusable_results_batch_is_recovered_on_automatic_retry(self):
        # The actual intended behavior, not just graceful failure: an unusable first
        # response (results double-encoded as an unparseable string — the real failure
        # mode hit on production data) followed by a good second response must recover
        # the whole batch, not just avoid crashing on the first attempt.
        client = MagicMock()
        client.messages.create.side_effect = [
            _mock_batch_response({"results": "not valid json {["}),
            _mock_batch_response({"results": [
                {"index": 0, "relevant": True, "confidence": 0.9, "reason": "ok", "category": "funding"}
            ]}),
        ]
        batch = [Item(source="A", title="Item 0", url="http://a/0", date="2026-09-10",
                       raw_text=_SUBSTANTIVE_BODY)]
        out, unclassified = classify._classify_batch_with_llm(client, batch)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].relevant, True)
        self.assertEqual(unclassified, [])
        self.assertEqual(client.messages.create.call_count, 2)

    def test_retry_happens_at_most_once(self):
        # Two bad responses in a row must not trigger a third call — the retry is bounded
        # to exactly one extra attempt, never an unbounded loop.
        client = MagicMock()
        client.messages.create.return_value = _mock_batch_response({"results": "still not json {["})
        batch = [Item(source="A", title="Item 0", url="http://a/0", date="2026-09-10",
                       raw_text=_SUBSTANTIVE_BODY)]
        out, unclassified = classify._classify_batch_with_llm(client, batch)
        self.assertEqual(out, [])
        self.assertEqual([it.url for it in unclassified], ["http://a/0"])
        self.assertEqual(client.messages.create.call_count, 2)


class DiscussionOnlyEventGateTests(unittest.TestCase):
    """Caught in production 2026-09-16: the real item "Valainis ar piecu ES valstu
    kolēģiem PĀRRUNĀ Digitālā omnibusa ietekmi uz uzņēmējdarbību" genuinely mentions
    startups/SMEs but is still just a ministerial discussion with no policy outcome —
    SYSTEM_PROMPT's "Events are not policy" section already says this should be rejected,
    but the model marked it relevant anyway on real data. classify._is_discussion_only_event
    is the deterministic backstop: a title verb naming a discussion/meeting rather than a
    decision."""

    def _run_with_item(self, item, input_dict):
        client = MagicMock()
        client.messages.create.return_value = _mock_batch_response(input_dict)
        return classify._classify_batch_with_llm(client, [item])

    def test_real_production_case_digital_omnibus_discussion_is_overridden(self):
        item = Item(
            source="Ekonomikas ministrija",
            title="Valainis ar piecu ES valstu kolēģiem pārrunā Digitālā omnibusa ietekmi uz uzņēmējdarbību",
            url="http://em/1", date="2026-09-03",
            raw_text=(
                "Valainis ar piecu ES valstu kolēģiem pārrunā Digitālā omnibusa ietekmi uz "
                "uzņēmējdarbību\n\nEkonomikas ministrs Valainis videokonferencē ar piecu ES "
                "valstu kolēģiem pārrunāja, kā Digitālais omnibuss ietekmē mazos un vidējos "
                "uzņēmumus un jaunuzņēmumus. Nekādi konkrēti grozījumi vai termiņi vēl nav "
                "izziņoti."
            ),
        )
        out, _ = self._run_with_item(item, {"results": [
            {"index": 0, "relevant": True, "confidence": 0.85, "category": "regulation",
             "reason": "Teksts skaidri min jaunuzņēmumus un digitālā regulējuma ietekmi."}
        ]})
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].relevant)
        self.assertIn("Atcelts", out[0].reason)

    def test_decision_verb_title_is_not_overridden(self):
        # Contrast case: an actual decision/action verb ("izsludina") must not trip this
        # gate just because it also names a general topic.
        item = Item(
            source="LIAA", title="LIAA izsludina jaunu programmu jaunuzņēmumiem",
            url="http://liaa/1", date="2026-09-03",
            raw_text=(
                "LIAA izsludina jaunu programmu jaunuzņēmumiem\n\nLIAA izsludinājusi jaunu "
                "atbalsta programmu jaunuzņēmumiem ar finansējumu līdz 50 000 eiro agrīnās "
                "stadijas uzņēmumiem."
            ),
        )
        out, _ = self._run_with_item(item, {"results": [
            {"index": 0, "relevant": True, "confidence": 0.9, "category": "funding",
             "reason": "Jauna programma ar konkrētu finansējuma summu jaunuzņēmumiem."}
        ]})
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].relevant)
        self.assertNotIn("Atcelts", out[0].reason)

    def test_discussion_verb_on_already_not_relevant_item_is_unaffected(self):
        item = Item(
            source="Ekonomikas ministrija", title="Ministrs pārrunā ceļu infrastruktūras jautājumus",
            url="http://em/2", date="2026-09-03",
            raw_text="Ministrs pārrunā ceļu infrastruktūras jautājumus\n\nSarunas par ceļu remontu.",
        )
        out, _ = self._run_with_item(item, {"results": [
            {"index": 0, "relevant": False, "confidence": 0.9, "category": "other",
             "reason": "Nesaistīts ar jaunuzņēmumiem."}
        ]})
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].relevant)
        self.assertNotIn("Atcelts", out[0].reason)  # nothing to override, verdict was already correct


class RenderMarkdownWhitespaceTests(unittest.TestCase):
    """A scraped title can contain an embedded literal newline (BeautifulSoup's
    get_text(strip=True) only trims the OUTER whitespace of each text node, not an
    internal "\\n" from a line break in the source HTML). Embedding that raw into a
    single-line Markdown list item corrupts the list/link structure."""

    def test_render_tolerates_none_reason_directly(self):
        # Defense-in-depth check on _oneline() itself, independent of classify.py's own
        # fix above — the render layer must survive a None slipping through regardless.
        item = Item(source="A", title="T", url="http://a/1", date="2026-09-10")
        c = Classification(item=item, relevant=True, confidence=0.9, reason=None, category="other")
        render_markdown([c], date(2026, 9, 7), date(2026, 9, 14))  # must not raise

    def test_embedded_newline_in_title_does_not_break_markdown_structure(self):
        item = Item(source="TAP portāls", title="Noteikumi par atbalstu\njaunuzņēmumiem",
                    url="http://a/1", date="2026-09-10")
        c = Classification(item=item, relevant=True, confidence=0.9, reason="ok", category="funding")
        md = render_markdown([c], date(2026, 9, 7), date(2026, 9, 14))
        # The full markdown link for this item must appear intact on a single line.
        self.assertIn("[Noteikumi par atbalstu jaunuzņēmumiem](http://a/1)", md)


_TAP_PACKAGE_SHOW_PAYLOAD = {"result": {"resources": [
    {"url": "https://data.gov.lv/dati/dataset/legal_acts_2026-09-01-0000-00-00.json"},
]}}


def _tap_fake_get(resource_payload):
    def fake_get(url, *a, **kw):
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json.return_value = _TAP_PACKAGE_SHOW_PAYLOAD if "package_show" in url else resource_payload
        return resp
    return fake_get


class TapLegalActsMalformedEntryTests(unittest.TestCase):
    """TAP portāls is CLAUDE.md's "mandatory source". Its fetcher processes nested JSON:API
    data one entry at a time — a single malformed entry anywhere in a month's dataset must
    not cost every other (valid) entry in that same dataset, the same way a bad row in an
    HTML listing elsewhere in this codebase doesn't take down the rest of that page."""

    def _fetch(self, entries):
        payload = {"included": [], "data": entries}
        with patch("policy_digest.sources.tap_legal_acts.requests.get", side_effect=_tap_fake_get(payload)):
            return tap.fetch_tap_legal_acts(since=date(2026, 9, 1), today=date(2026, 9, 14))

    def test_malformed_submitted_at_skips_only_that_entry(self):
        items = self._fetch([
            {"attributes": {"name": "Good 1", "submitted_at": "2026-09-05T10:00:00"},
             "relationships": {}, "links": {"web": "http://tap/1"}},
            {"attributes": {"name": "Bad (malformed date)", "submitted_at": "2026-13-45"},
             "relationships": {}, "links": {"web": "http://tap/2"}},
            {"attributes": {"name": "Good 2", "submitted_at": "2026-09-08T10:00:00"},
             "relationships": {}, "links": {"web": "http://tap/3"}},
        ])
        self.assertEqual({it.title for it in items}, {"Good 1", "Good 2"})

    def test_policy_area_relationship_missing_id_does_not_crash(self):
        items = self._fetch([
            {"attributes": {"name": "Good 1", "submitted_at": "2026-09-05T10:00:00"},
             "relationships": {"policy_areas": {"data": [{"type": "policy_areas"}]}},  # no "id"
             "links": {"web": "http://tap/1"}},
        ])
        self.assertEqual([it.title for it in items], ["Good 1"])

    def test_null_links_does_not_crash(self):
        items = self._fetch([
            {"attributes": {"name": "Good 1", "submitted_at": "2026-09-05T10:00:00"},
             "relationships": {}, "links": None},
        ])
        self.assertEqual([it.title for it in items], ["Good 1"])

    def test_document_version_relationship_missing_id_does_not_crash(self):
        items = self._fetch([
            {"attributes": {"name": "Good 1", "submitted_at": "2026-09-05T10:00:00"},
             "relationships": {"document_versions": {"data": [{"type": "x"}]}},  # no "id"
             "links": {"web": "http://tap/1"}},
        ])
        self.assertEqual([it.title for it in items], ["Good 1"])

    def test_null_name_does_not_crash(self):
        # "name": null is present-but-null — a plain .get("name", "") default only covers
        # a MISSING key, not an explicit null, and used to crash html.unescape()/.strip().
        items = self._fetch([
            {"attributes": {"name": None, "submitted_at": "2026-09-05T10:00:00"},
             "relationships": {}, "links": {"web": "http://tap/1"}},
            {"attributes": {"name": "Good 2", "submitted_at": "2026-09-06T10:00:00"},
             "relationships": {}, "links": {"web": "http://tap/2"}},
        ])
        self.assertEqual({it.title for it in items}, {"", "Good 2"})


class TapIncludedLookupTests(unittest.TestCase):
    """_policy_area_names/_document_version_map run once per resource file, BEFORE the
    per-entry loop even starts — a KeyError from one malformed "included" entry here used
    to lose every act in the whole file, not just one."""

    def test_policy_area_missing_id_is_skipped_not_fatal(self):
        self.assertEqual(tap._policy_area_names([{"type": "policy_areas", "attributes": {"name": "X"}}]), {})

    def test_policy_area_missing_attributes_does_not_crash(self):
        self.assertEqual(tap._policy_area_names([{"type": "policy_areas", "id": "pa1"}]), {"pa1": ""})

    def test_policy_area_missing_name_does_not_crash(self):
        self.assertEqual(tap._policy_area_names([{"type": "policy_areas", "id": "pa1", "attributes": {}}]), {"pa1": ""})

    def test_document_version_missing_id_is_skipped_not_fatal(self):
        self.assertEqual(tap._document_version_map([{"type": "legal_act_document_versions", "attributes": {}}]), {})


if __name__ == "__main__":
    unittest.main()
