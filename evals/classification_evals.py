#!/usr/bin/env python
"""Regression eval for classify.py's SYSTEM_PROMPT.

Deliberately NOT part of `python -m unittest discover tests`: it calls the real Anthropic
API (costs money, needs ANTHROPIC_API_KEY, and an LLM judgment call isn't guaranteed
bit-for-bit deterministic run to run), so it isn't meant to run on every commit. Run it by
hand whenever SYSTEM_PROMPT changes, to check the edit didn't reintroduce a known mistake or
break a case that used to pass:

    python evals/classification_evals.py

Each fixture pairs a real or realistic government/policy item with the verdict a human
reviewer would give. The first one is the actual case that motivated this file: an ALTUM
financing program open to any Latvian company (no size/stage restriction) got wrongly marked
relevant on 2026-09-14 just because ALTUM is a funder the prompt name-checks as an example —
see the SYSTEM_PROMPT edit in the same commit as this file, and CLAUDE.md.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from policy_digest import classify
from policy_digest.sources.base import Item

FIXTURES = [
    {
        "name": "altum_general_export_loan_specific_company",
        "expect_relevant": False,
        "note": (
            "Caught in production 2026-09-14: ALTUM financing open to any Latvian exporter, "
            "no size/stage restriction, wrongly marked relevant just because ALTUM was the "
            "funder — the digest even invented an 'SME instrument' framing the source text "
            "never stated."
        ),
        "source": "Altum",
        "title": (
            'ALTUM piešķir 842 tūkstošus eiro Latvijas uzņēmumam "FORTES" enerģētikas '
            "infrastruktūras atjaunošanai Ukrainā"
        ),
        "body": (
            'ALTUM piešķīrusi finansējumu 842 300 eiro apmērā Latvijas uzņēmumam "FORTES", '
            "kas ražo un piegādā koģenerācijas stacijas klientiem Ukrainā. Uzņēmums "
            "izmantojis ALTUM piedāvāto atbalsta programmu uzņēmumiem, kas iesaistās "
            "Ukrainas infrastruktūras rekonstrukcijā. ALTUM šobrīd piedāvā aizdevumus "
            "investīcijām un apgrozāmajiem līdzekļiem līdz 1 000 000 eiro apmērā, ja "
            "Latvijas uzņēmums projekta īstenošanas rezultātā plāno eksportēt preces vai "
            "pakalpojumus uz Ukrainu un spēj pamatot, kā šis eksports veicinās Ukrainas "
            "ekonomiskās darbības atjaunošanu. Aizdevumam piemērojamas samazinātas "
            "nodrošinājuma prasības. Programma pieejama jebkuram Latvijas uzņēmumam "
            "neatkarīgi no tā lieluma, nozares vai darbības ilguma."
        ),
    },
    {
        "name": "altum_program_actually_startup_scoped",
        "expect_relevant": True,
        "note": (
            "Contrast case for the fixture above: same funder, but eligibility is "
            "explicitly capped to early-stage/small companies, so this one should still "
            "pass — the fix must not make the prompt reject every Altum/LIAA mention."
        ),
        "source": "Altum",
        "title": "ALTUM izsludina jaunu riska kapitāla programmu jaunuzņēmumiem",
        "body": (
            "ALTUM izsludinājusi jaunu riska kapitāla programmu jaunuzņēmumiem, kas "
            "dibināti pēdējo trīs gadu laikā un kuru gada apgrozījums nepārsniedz 200 000 "
            "eiro. Programmas ietvaros pieejams finansējums līdz 100 000 eiro ar "
            "atvieglotiem nosacījumiem agrīnās stadijas uzņēmumiem, kas izstrādā jaunus "
            "produktus vai tehnoloģijas."
        ),
    },
    {
        "name": "unemployment_program_generic_entrepreneurship_line",
        "expect_relevant": False,
        "note": (
            "Existing SYSTEM_PROMPT guardrail: a general jobseeker/requalification program "
            "with a generic 'this also helps entrepreneurship' line tacked on."
        ),
        "source": "LM",
        "title": "NVA izsludina jaunu bezdarbnieku pārkvalifikācijas programmu",
        "body": (
            "Nodarbinātības valsts aģentūra (NVA) izsludina jaunu bezdarbnieku "
            "pārkvalifikācijas programmu, kas paredz apmācības IT, būvniecības un "
            "pakalpojumu nozarēs ilgstošiem bezdarbniekiem un ekonomiski neaktīvām "
            "personām. NVA norāda, ka programma kopumā veicinās uzņēmējdarbības vidi un "
            "jaunuzņēmumu ekosistēmu Latvijā."
        ),
    },
    {
        "name": "universal_per_employee_levy",
        "expect_relevant": False,
        "note": (
            "Existing SYSTEM_PROMPT guardrail: a duty/fee applying identically to every "
            "company regardless of size, not startup/SME-specific."
        ),
        "source": "MK",
        "title": "Grozījumi noteikumos par valsts nodevu par darba aizsardzības uzraudzību",
        "body": (
            "Ministru kabinets apstiprinājis grozījumus noteikumos, ar kuriem tiek "
            "palielināta valsts nodeva par darba aizsardzības uzraudzību — 15 eiro par "
            "katru nodarbināto personu gadā. Nodeva attiecas uz visiem darba devējiem "
            "Latvijā neatkarīgi no uzņēmuma lieluma vai nozares."
        ),
    },
    {
        "name": "startup_stock_option_tax_change",
        "expect_relevant": True,
        "note": "Should pass: a concrete regulatory change scoped specifically to startups.",
        "source": "Saeima",
        "title": "Saeima pieņem grozījumus par kapitāla daļu opciju nodokļu režīmu jaunuzņēmumos",
        "body": (
            "Saeima otrajā lasījumā atbalstījusi grozījumus likumā 'Par iedzīvotāju "
            "ienākuma nodokli', kas paredz atvieglotu nodokļu režīmu kapitāla daļu "
            "opcijām, ko jaunuzņēmumi piešķir saviem darbiniekiem. Grozījumi attiecas "
            "tikai uz uzņēmumiem, kas atbilst likumā 'Par jaunuzņēmumu darbības atbalstu' "
            "noteiktajiem kritērijiem."
        ),
    },
]


def run() -> int:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        print(
            "ANTHROPIC_API_KEY not set — this eval needs the real classifier, not the "
            "no-key keyword fallback. Set it in .env and retry."
        )
        return 1

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    items = [
        Item(
            source=f["source"],
            title=f["title"],
            url=f"eval://{f['name']}",
            date="2026-01-01",
            raw_text=f"{f['title']}\n\n{f['body']}",
        )
        for f in FIXTURES
    ]

    classifications, unclassified = classify._classify_batch_with_llm(client, items)
    by_url = {c.item.url: c for c in classifications}
    unclassified_urls = {it.url for it in unclassified}

    failures = 0
    for fixture, item in zip(FIXTURES, items):
        if item.url in unclassified_urls:
            print(f"FAIL  {fixture['name']}: got no classification result at all")
            failures += 1
            continue
        c = by_url[item.url]
        ok = c.relevant == fixture["expect_relevant"]
        print(
            f"{'ok' if ok else 'FAIL':4}  {fixture['name']}: expected "
            f"relevant={fixture['expect_relevant']}, got relevant={c.relevant} "
            f"(confidence {c.confidence:.2f})"
        )
        print(f"      reason: {c.reason}")
        if not ok:
            failures += 1
            print(f"      note: {fixture['note']}")

    print()
    if failures:
        print(f"{failures}/{len(FIXTURES)} fixture(s) failed.")
        return 1
    print(f"All {len(FIXTURES)} fixture(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
