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
    # --- Fixtures below caught in production 2026-09-15, from a full real-data digest run
    # that the user manually fact-checked item by item against primary sources. 9 of 19
    # digest items (47%) turned out not to hold up under strict scrutiny — see the commit
    # this file changed in alongside for the full writeup and CLAUDE.md.
    {
        "name": "trade_mission_event_startups_merely_invited",
        "expect_relevant": False,
        "note": (
            "An event/delegation announcement, not a policy or funding mechanism — "
            "startups are one invited category among general Latvian exporters, with no "
            "eligibility criterion or mechanism of its own."
        ),
        "source": "Ekonomikas ministrija",
        "title": "Latvijas uzņēmēji Valsts prezidenta vizītes laikā dosies tirdzniecības misijā uz ASV",
        "body": (
            "Valsts prezidenta vizītes laikā ASV Latvijas uzņēmēju delegācija dosies "
            "tirdzniecības misijā, lai stiprinātu divpusējo ekonomisko sadarbību. "
            "Misijā piedalīsies dažādu nozaru uzņēmumi, tostarp jaunuzņēmumi, kuriem "
            "organizētas tikšanās ar potenciālajiem partneriem un investoriem ASV tirgū. "
            "Delegācijas sastāvs un pilna programma tiks izziņota vēlāk."
        ),
    },
    {
        "name": "venture_capital_program_named_but_no_body_text",
        "expect_relevant": False,
        "note": (
            "Correctly rejected even with the VC/SME eligibility-cap carve-out restored "
            "2026-09-16: this item has NO body text at all (title only), so there's "
            "nothing to confirm an actual eligibility cap from — exactly the fabrication "
            "risk NO_BODY_MARKER exists to prevent (this is the real MK protocol item "
            "that motivated the 2026-09-15 audit fix in the first place). Contrast with "
            "venture_capital_for_startups_word_present_stays_relevant and "
            "altum_program_actually_startup_scoped below, which have real body text "
            "stating the eligibility cap and correctly pass."
        ),
        "source": "Ministru kabineta protokoli",
        "title": (
            'Noteikumu projekts "Grozījumi Ministru kabineta 2023. gada 15. augusta '
            'noteikumos Nr. 463 "Programmas \'Iespējkapitāla ieguldījumi\' īstenošanas '
            'noteikumi""'
        ),
        "body": "",  # genuinely no body text available — matches the real production case
    },
    {
        "name": "no_body_mk_protocol_item_new_products_not_new_companies",
        "expect_relevant": False,
        "note": (
            "Caught in production: this exact MK protocol item (Nr. 501) has no real "
            "body text and was wrongly marked relevant with an invented 'clearly "
            "includes startup support' claim. The title says 'jaunu produktu' (new "
            "PRODUCTS), not jaunuzņēmumu (new companies/startups) — an easy but wrong "
            "keyword association for the model to make."
        ),
        "source": "Ministru kabineta protokoli",
        "title": (
            'Noteikumu projekts "Grozījumi Ministru kabineta 2024. gada 23. jūlija '
            "noteikumos Nr. 501 \"Eiropas Savienības kohēzijas politikas programmas "
            "2021.–2027. gadam 1.2.1. specifiskā atbalsta mērķa 'Pētniecības un inovāciju "
            "kapacitātes stiprināšana un progresīvu tehnoloģiju ieviešana uzņēmumiem' "
            "1.2.1.1. pasākuma 'Atbalsts jaunu produktu attīstībai un "
            'internacionalizācijai\' pirmās kārtas īstenošanas noteikumi""'
        ),
        "body": "",
    },
    {
        "name": "youth_school_entrepreneurship_program",
        "expect_relevant": False,
        "note": (
            "Teaches schoolchildren to run a mock 'skolēnu mācību uzņēmums' — not "
            "startups by any definition, regardless of how much the program's own "
            "materials use the word 'uzņēmējdarbība'."
        ),
        "source": "LIAA",
        "title": "Sākusies pieteikšanās “Jauno uzņēmēju skolā 2026” piecās Latvijas pilsētās",
        "body": (
            "LIAA rīko reģionālas darbnīcas skolēniem un skolotājiem par skolēnu mācību "
            "uzņēmumu (SMU) izveidi un attīstību piecās Latvijas pilsētās. Programmas "
            "ietvaros skolēni mācās izveidot un vadīt savu mācību uzņēmumu, iepazīstot "
            "uzņēmējdarbības pamatprincipus. Projekts tiek īstenots ERAF programmas "
            "'Atbalsts MVU inovatīvas uzņēmējdarbības attīstībai' ietvaros."
        ),
    },
    {
        "name": "hedge_word_meeting_discussion_no_policy_yet",
        "expect_relevant": False,
        "note": (
            "Verified against the source: 'purely a meeting/discussion summary' about an "
            "EU regulation's general effects, describing current problems, not a new "
            "policy. The 'īpaši MVU un jaunuzņēmumiem' framing in a digest reason for "
            "this kind of item is exactly the hedge-word failure the prompt now names "
            "with a worked example."
        ),
        "source": "Ekonomikas ministrija",
        "title": "Valainis ar piecu ES valstu kolēģiem pārrunā Digitālā omnibusa ietekmi uz uzņēmējdarbību",
        "body": (
            "Ekonomikas ministrs Valainis videokonferencē ar piecu ES valstu kolēģiem "
            "pārrunāja, kā Eiropas Savienības esošais Digitālais omnibuss ietekmē "
            "uzņēmējdarbības vidi un konkurētspēju. Dalībnieki norādīja, ka mazie un "
            "vidējie uzņēmumi, un jaunuzņēmumi saskaras ar administratīvo slogu saistībā "
            "ar datu izmantošanas prasībām. Ministri vienojās turpināt ekspertu līmeņa "
            "koordināciju un aicināja Eiropas Komisiju uz skaidrākiem noteikumiem. Nekādi "
            "konkrēti grozījumi vai ieviešanas termiņi vēl nav izziņoti."
        ),
    },
    {
        "name": "hedge_word_national_position_no_substantive_detail",
        "expect_relevant": False,
        "note": (
            "Verified against the source: the protocol only records that this national "
            "position was approved, with no substantive detail about its content. The "
            "digest reason used the hedge word 'potenciāli' ('potentially affects "
            "startups') — exactly what the prompt says to reject."
        ),
        "source": "Ministru kabineta protokoli",
        "title": (
            'Latvijas Republikas nacionālā pozīcija Nr. 1 "Par priekšlikumu Padomes '
            'lēmumam par Nolīguma par elektronisko komerciju noslēgšanu"'
        ),
        "body": "",
    },
    {
        "name": "hedge_word_media_law_broad_sector",
        "expect_relevant": False,
        "note": (
            "Verified against the source: a general Electronic Mass Media Law amendment "
            "(pre-first-reading), a 'broad sectoral policy discussion' with invited "
            "broadcasting/film industry groups, no SME/startup provision at all."
        ),
        "source": "Saeimas komisiju darba kārtības",
        "title": "Cilvēktiesību un sabiedrisko lietu komisijas sēde",
        "body": (
            "Darba kārtībā: likumprojekts 'Grozījumi Elektronisko plašsaziņas līdzekļu "
            "likumā' (Nr. 1454/Lp14) un likumprojekts 'Grozījumi Filmu likumā' "
            "(Nr. 1455/Lp14), abi pirms pirmā lasījuma. Uz sēdi aicināti Kultūras "
            "ministrijas, Nacionālās elektronisko plašsaziņas līdzekļu padomes, "
            "Nacionālā kino centra un raidorganizāciju asociāciju pārstāvji."
        ),
    },
    {
        "name": "hedge_word_cybercrime_convention_affects_all_businesses",
        "expect_relevant": False,
        "note": (
            "Verified against the source: a Cybercrime Convention ratification package "
            "(criminal procedure / electronic evidence disclosure) — 'standard digital "
            "crime prevention measures affecting all businesses, not startups "
            "specifically'. No startup/SME-specific provision."
        ),
        "source": "Saeimas komisiju darba kārtības",
        "title": "Tautsaimniecības, agrārās, vides un reģionālās politikas komisijas sēde",
        "body": (
            "Darba kārtībā izskatāma piecu likumprojektu pakete, ar ko tiek ieviests "
            "Konvencijas par kibernoziegumiem Otrais papildu protokols: grozījumi "
            "Elektronisko sakaru likumā, Informācijas sabiedrības pakalpojumu likumā un "
            "Kriminālprocesa likumā. Pakete paredz pastiprinātu starptautisko sadarbību "
            "un elektronisko pierādījumu izpaušanas kārtību."
        ),
    },
    {
        "name": "funding_discovery_tool_without_literal_word",
        "expect_relevant": True,
        "note": (
            "Reinstated 2026-09-16: the funding-discovery-tool carve-out is back in "
            "SYSTEM_PROMPT — a tool that helps founders (including newly-founded "
            "companies, 'jaundibinātiem uzņēmumiem') find the right government support "
            "is relevant even open to companies of any size, since its function is "
            "funding discovery, not the financing itself."
        ),
        "source": "LIAA",
        "title": "LIAA piedāvā rīku, kas palīdz uzņēmējiem atrast piemērotāko atbalstu",
        "body": (
            "LIAA laidusi klajā jaunu digitālu rīku 'Latvijas biznesa atbalsta vednis', "
            "kas palīdz uzņēmējiem atrast viņu vajadzībām piemērotāko valsts atbalstu. "
            "Tas ir paredzēts plašam uzņēmēju lokam – gan tiem, kuri tikai apsver biznesa "
            "uzsākšanu, gan jaundibinātiem uzņēmumiem, kā arī pieredzējušiem "
            "komersantiem, kas vēlas paplašināties ārvalstu tirgos vai attīstīt jaunus "
            "produktus un pakalpojumus."
        ),
    },
    # --- Fixtures below caught 2026-09-16: user manually reviewed a real digest run and
    # flagged that several "Finansējuma iespējas" items were events (contests, mentor
    # calls, course cohorts), not funding programs or policy changes — see SYSTEM_PROMPT's
    # "Events are not policy" section and CLAUDE.md.
    {
        "name": "contest_prize_fund_announcement",
        "expect_relevant": False,
        "note": (
            "'Events are not policy': a business-idea contest's prize fund is not a "
            "funding program with ongoing eligibility — it's a one-off competition."
        ),
        "source": "LIAA",
        "title": "Biznesa ideju konkursa “Ideju Kauss” balvu fonds šogad 40 000 eiro",
        "body": (
            "Konkurss 'Ideju kauss' ir īpaši paredzēts biznesa ideju atbalstam agrīnā "
            "attīstības posmā ar mentoru atbalstu, ekspertu palīdzību un reālām "
            "investīcijām jaunuzņēmumiem. Šogad konkursa balvu fonds ir 40 000 eiro, un "
            "pieteikšanās ilgst līdz 13. septembrim."
        ),
    },
    {
        "name": "contest_registration_count_update",
        "expect_relevant": False,
        "note": (
            "'Events are not policy': an update on how many ideas were submitted to a "
            "contest is a step further still from being a funding mechanism or "
            "regulatory change."
        ),
        "source": "LIAA",
        "title": "Biznesa ideju konkursam \"Ideju kauss\" pieteiktas gandrīz 400 idejas",
        "body": (
            "'Ideju kauss' ir konkurss agrīnā biznesa idejas attīstības posmā, ko rīko "
            "LIAA un kurā dalībniekiem tiek piešķirtas naudas balvas (kopā 40 000 eiro) "
            "ideju attīstībai. Konkursam līdz šim pieteiktas gandrīz 400 idejas."
        ),
    },
    {
        "name": "mentor_recruitment_call",
        "expect_relevant": False,
        "note": (
            "'Events are not policy': recruiting experienced entrepreneurs to volunteer "
            "as mentors is staffing an advisory pool, not funding or regulating "
            "startups themselves."
        ),
        "source": "LIAA",
        "title": (
            "LIAA aicina pieredzējušus uzņēmējus un nozaru profesionāļus kļūt par "
            "mentoriem jauno uzņēmumu izaugsmei"
        ),
        "body": (
            "LIAA aicina pieredzējušus uzņēmējus un profesionāļus kļūt par mentoriem "
            "LIAA biznesa pirmsinkubācijas un inkubācijas programmu dalībniekiem, kas ir "
            "jaunuzņēmumi, sniedzot praktisku atbalstu jaunuzņēmumu izaugsmei."
        ),
    },
    {
        "name": "training_course_cohort_launch",
        "expect_relevant": False,
        "note": (
            "'Events are not policy': a mentorship/training program's next phase "
            "starting is a course with a start date, not a funding mechanism, even "
            "though it's ERAF-funded."
        ),
        "source": "LIAA",
        "title": "Kad satiekas pieredze, zinātkāre un vēlme augt - tiek uzsākts “MENTORS BIZNESA IZAUGSMEI” 3. posms!",
        "body": (
            "'Mentors biznesa izaugsmei' ir LIAA programma biznesa pirmsinkubācijas un "
            "inkubācijas dalībniekiem — jaunuzņēmumiem un MVU inovatīvās "
            "uzņēmējdarbības attīstībai, finansēta ar ERAF. Sācies programmas 3. posms, "
            "kurā pieredzējuši mentori dalīsies zināšanās ar dalībniekiem."
        ),
    },
    {
        "name": "training_course_enrollment_milestone",
        "expect_relevant": False,
        "note": (
            "'Events are not policy': reporting that a training program's application "
            "slots filled up in 28 minutes is a course enrollment milestone, not a "
            "funding program or regulatory change."
        ),
        "source": "LIAA",
        "title": "Pieteikumu limits LIAA Mini MBA programmai sasniegts 28 minūtēs",
        "body": (
            "LIAA īstenotā Mini MBA programma 'Inovāciju vadība' ir paredzēta Latvijas "
            "mazo un vidējo uzņēmumu vadītājiem un inovāciju procesu vadītājiem. "
            "Pieteikumu limits šī gada programmai tika sasniegts 28 minūšu laikā pēc "
            "pieteikšanās atvēršanas."
        ),
    },
    {
        "name": "incubation_program_sme_scoped_no_literal_word",
        "expect_relevant": True,
        "note": (
            "Reinstated 2026-09-16: real direct funding (up to 70%) with a formal "
            "deadline, scoped to young SMEs ('mikro, mazajiem un vidējiem uzņēmumiem, "
            "kuri nav vecāki par pieciem gadiem') — an explicit age/size eligibility cap "
            "counts as startup/SME-scoped even without the literal word jaunuzņēmums."
        ),
        "source": "LIAA",
        "title": "No prototipa līdz eksporta tirgum: LIAA atver rudens uzņemšanu Biznesa inkubācijas programmā",
        "body": (
            "LIAA aicina jaunos uzņēmumus pieteikties Biznesa inkubācijas programmai ar "
            "finanšu atbalstu līdz 70% no attiecināmajām izmaksām, ekspertu "
            "konsultācijām un mentoru atbalstu. Programma paredzēta mikro, mazajiem un "
            "vidējiem uzņēmumiem, kuri nav vecāki par pieciem gadiem. Pieteikšanās "
            "rudens uzņemšanai ir atvērta līdz noteiktam termiņam."
        ),
    },
    # --- Fixtures below caught 2026-09-16: user reviewed the digest again and flagged
    # that several items were "business in general", not startup-specific — a program
    # open to companies of any size/age was still passing because the model's reasoning
    # treated "oriented toward companies with X" as if it were an eligibility cap. See
    # SYSTEM_PROMPT's "Be very specific for startups" section and CLAUDE.md.
    {
        "name": "general_business_open_competition_no_size_cap",
        "expect_relevant": False,
        "note": (
            "An open project competition for 'high-readiness' industrial/dual-use R&D "
            "is orientation toward mature projects, not a startup/SME eligibility cap — "
            "any company can apply regardless of size or age."
        ),
        "source": "Ekonomikas ministrija",
        "title": "Valdība novirza papildu finansējumu divējāda lietojuma tehnoloģiju attīstībai",
        "body": (
            "Ekonomikas ministrija papildina divējāda lietojuma tehnoloģiju attīstības "
            "atbalsta programmu ar 106 035 eiro. Programma ir atklāts projektu konkurss, "
            "kurā var pieteikties jebkurš Latvijā reģistrēts uzņēmums, kas īsteno "
            "rūpnieciskus pētījumus vai izstrādā jaunus produktus divējāda lietojuma "
            "tehnoloģiju jomā. Konkursam nav noteikts uzņēmuma lieluma, apgrozījuma vai "
            "dibināšanas gada ierobežojums."
        ),
    },
    {
        "name": "regional_business_development_no_size_cap",
        "expect_relevant": False,
        "note": (
            "Regional/territorial economic development funding based on 'local business "
            "needs' is general business support, not a startup/SME-specific mechanism."
        ),
        "source": "Ministru kabineta protokoli",
        "title": (
            'Noteikumu projekts "Grozījumi Ministru kabineta 2015. gada 13. oktobra '
            "noteikumos Nr. 593 \"Darbības programmas 'Izaugsme un nodarbinātība' 3.3.1. "
            "specifiskā atbalsta mērķa 'Palielināt privāto investīciju apjomu reģionos, "
            "veicot ieguldījumus uzņēmējdarbības attīstībai atbilstoši pašvaldību "
            "attīstības programmās noteiktajai teritoriju ekonomiskajai specializācijai "
            "un balstoties uz vietējo uzņēmēju vajadzībām' īstenošanas noteikumi\"\""
        ),
        "body": (
            "Noteikumi īsteno ES darbības programmas mērķi palielināt privāto "
            "investīciju apjomu reģionos, atbalstot uzņēmējdarbības attīstību "
            "pašvaldībās atbilstoši vietējo uzņēmēju vajadzībām. Atbalsts pieejams "
            "jebkuram reģionā darbojošamies uzņēmumam neatkarīgi no tā lieluma vai "
            "dibināšanas gada."
        ),
    },
    {
        "name": "portal_modernization_for_all_clients",
        "expect_relevant": False,
        "note": (
            "A loan-application portal modernization explicitly serving 'all clients' "
            "is not a startup-specific product, even though startups are among its users."
        ),
        "source": "Altum",
        "title": "Jauns ALTUM klientu portāls jauniem aizdevumu pieteikumiem",
        "body": (
            "ALTUM laidusi klajā modernizētu klientu portālu aizdevumu pieteikumu "
            "iesniegšanai un apstrādei. Jaunais portāls ir pieejams visiem ALTUM "
            "klientiem neatkarīgi no uzņēmuma lieluma vai nozares, un tā mērķis ir "
            "vienkāršot pieteikumu iesniegšanas procesu kopumā."
        ),
    },
    {
        "name": "specific_company_investment_no_startup_label",
        "expect_relevant": False,
        "note": (
            "An international company opening an R&D center with agency support is not "
            "relevant unless the item's own text says the company is a startup/SME — "
            "'international company' plus a vague 'strengthens the ecosystem' closing "
            "line is the ecosystem-truism failure, not a real connection."
        ),
        "source": "LIAA",
        "title": "“NestAI” paplašina darbību Latvijā, lai attīstītu aizsardzības mākslīgo intelektu NATO austrumu flangā",
        "body": (
            "Starptautisks uzņēmums 'NestAI' atver pētniecības un attīstības centru "
            "Latvijā ar ieguldījumiem 10 miljonu eiro apmērā un LIAA atbalstu. Uzņēmuma "
            "pārstāvji norāda, ka jaunais centrs stiprinās Latvijas aizsardzības "
            "tehnoloģiju ekosistēmu un radīs jaunas augsti kvalificētas darba vietas. "
            "Uzņēmuma lielums, dibināšanas gads vai statuss netiek minēts."
        ),
    },
    {
        "name": "hedge_word_credit_institutions_law",
        "expect_relevant": False,
        "note": (
            "A credit institutions / investment broker regulation amendment justified "
            "only as something that 'could affect investment capital availability for "
            "startups' — the hedge-word failure, no startup-specific provision stated."
        ),
        "source": "Ministru kabineta protokoli",
        "title": 'Likumprojekts "Grozījumi Kredītiestāžu un ieguldījumu brokeru sabiedrību darbības atjaunošanas un noregulējuma likumā"',
        "body": (
            "Likumprojekts groza kārtību, kādā tiek atjaunota kredītiestāžu un "
            "ieguldījumu brokeru sabiedrību darbība maksātnespējas vai citu krīzes "
            "situāciju gadījumā, ieviešot ES direktīvas prasības. Grozījumi attiecas uz "
            "finanšu sektora uzraudzības iestādēm un regulētajām finanšu institūcijām "
            "kopumā, nevis uz konkrētiem jaunuzņēmumu finansējuma nosacījumiem."
        ),
    },
    # --- Fixtures below added 2026-09-16, originally for a hard "literal word required,
    # no SME/VC/incubator carve-out" gate that was itself reversed the same day (see
    # CLAUDE.md and memory/project_policy_hacker_startup_word_gate.md) — kept because
    # they're still valid regression cases under the reinstated eligibility-cap prompt.
    {
        "name": "money_to_non_startup_company_no_word_at_all",
        "expect_relevant": False,
        "note": (
            "ALTUM financing a solar park project, open to 'jebkuram Latvijas "
            "energoražotājam neatkarīgi no uzņēmuma lieluma vai darbības ilguma' — no "
            "size/stage eligibility cap at all, so this stays NOT relevant even without "
            "the literal-word gate: 'atjaunojamās enerģijas uzņēmums' (renewable energy "
            "company) is general business, not a startup/SME-scoped program."
        ),
        "source": "Altum",
        "title": 'ALTUM piešķir aizdevumu SIA "Saules Parks" saules elektrostacijas būvniecībai',
        "body": (
            "ALTUM piešķīrusi aizdevumu 1,2 miljonu eiro apmērā SIA 'Saules Parks' jaunas "
            "saules elektrostacijas būvniecībai Zemgalē. Uzņēmums ir viens no "
            "lielākajiem atjaunojamās enerģijas ražotājiem reģionā, un projekts "
            "palielinās tā jaudu par 40%. Aizdevums pieejams jebkuram Latvijas "
            "energoražotājam neatkarīgi no uzņēmuma lieluma vai darbības ilguma."
        ),
    },
    {
        "name": "venture_capital_for_startups_word_present_stays_relevant",
        "expect_relevant": True,
        "note": (
            "Venture capital explicitly scoped to early-stage jaunuzņēmumi — passes "
            "cleanly under the eligibility-cap wording regardless of the literal word "
            "also being present."
        ),
        "source": "Altum",
        "title": "ALTUM izsludina jaunu riska kapitāla fondu jaunuzņēmumiem",
        "body": (
            "ALTUM izsludinājusi jaunu riska kapitāla fondu, kas veiks ieguldījumus "
            "Latvijas jaunuzņēmumos to agrīnajā attīstības stadijā. Fonds paredzēts "
            "jaunuzņēmumiem, kas meklē finansējumu produkta attīstībai un tirgus "
            "paplašināšanai."
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
