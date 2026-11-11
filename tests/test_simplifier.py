"""Offline tests for the Simplifier."""
import json
import unittest

from faithfulmed.agents import simplifier
from faithfulmed.agents.simplifier import (
    SimplifiedOutput,
    SimplifierError,
    build_grounding,
    format_atoms,
    simplify,
)

ATOMS = [
    {
        "fact_id": "F001",
        "text": "The patient reports a mild headache.",
        "type": "symptom",
        "assertion": "present",
        "source_quote": "The patient reports a mild headache.",
    },
    {
        "fact_id": "F002",
        "text": "He denies nausea.",
        "type": "symptom",
        "assertion": "absent",
        "source_quote": "He denies nausea.",
    },
    {
        "fact_id": "F003",
        "text": "His father has questionable celiac disease.",
        "type": "diagnosis",
        "assertion": "family",
        "source_quote": "His father has questionable celiac disease.",
    },
]

HEADACHE_PASSAGE = {
    "title": "Headache",
    "url": "https://medlineplus.gov/headache.html",
    "section": None,
    "text": "A headache is pain in your head.",
    "distance": 0.28,
}

GROUNDING = {
    "F001": [HEADACHE_PASSAGE],
    "F002": [],
    "F003": [],
}

HEADACHE = "Your head hurts a little."
NO_NAUSEA = "You do not have an upset stomach."
FATHER = "Your father may have a problem digesting gluten."


def rewrite(fact_id, text, ungrounded=()):
    return {"fact_id": fact_id, "text": text, "ungrounded_terms": list(ungrounded)}


def reply(rewrites, sources_used=()):
    return json.dumps({"fact_rewrites": list(rewrites), "sources_used": list(sources_used)})


def good_reply():
    return reply(
        [
            rewrite("F001", HEADACHE),
            rewrite("F002", NO_NAUSEA),
            rewrite("F003", FATHER, ["celiac disease"]),
        ],
        sources_used=["Headache"],
    )


def scripted(*replies):
    """A sender that returns the scripted replies in order and fails loudly when empty."""
    remaining = list(replies)

    def send(prompt, model):
        if not remaining:
            raise AssertionError("the test script ran out of replies")
        return remaining.pop(0)

    return send


class AcceptedDraftTest(unittest.TestCase):
    def test_good_reply_is_assembled_in_fact_order(self):
        output = simplify(ATOMS, GROUNDING, send=scripted(good_reply()))

        self.assertIsInstance(output, SimplifiedOutput)
        self.assertTrue(output.simplified_text.startswith(HEADACHE))
        self.assertIn(NO_NAUSEA, output.simplified_text)
        self.assertIn(FATHER, output.simplified_text)
        self.assertEqual(output.atoms_covered, [1, 2, 3])
        self.assertEqual(output.fact_ids_covered, ["F001", "F002", "F003"])

    def test_used_passage_text_comes_from_the_store(self):
        output = simplify(ATOMS, GROUNDING, send=scripted(good_reply()))

        self.assertEqual(len(output.source_metadata), 1)
        self.assertEqual(output.source_metadata[0].title, "Headache")
        self.assertEqual(output.source_metadata[0].text, HEADACHE_PASSAGE["text"])
        self.assertEqual(output.source_metadata[0].url, HEADACHE_PASSAGE["url"])

    def test_ungrounded_term_is_reported_not_rejected(self):
        output = simplify(ATOMS, GROUNDING, send=scripted(good_reply()))

        self.assertTrue(any("celiac disease" in warning for warning in output.review_warnings))

    def test_clean_draft_has_no_warnings(self):
        output = simplify(
            ATOMS[:1], {"F001": [HEADACHE_PASSAGE]},
            send=scripted(reply([rewrite("F001", HEADACHE)], sources_used=["Headache"])),
        )

        self.assertEqual(output.review_warnings, [])


class RepairTest(unittest.TestCase):
    def test_missing_fact_is_repaired(self):
        first = reply([rewrite("F001", HEADACHE), rewrite("F002", NO_NAUSEA)], sources_used=["Headache"])
        second = reply([rewrite("F003", FATHER, ["celiac disease"])])

        output = simplify(ATOMS, GROUNDING, send=scripted(first, second))

        self.assertIn(FATHER, output.simplified_text)
        self.assertEqual(output.fact_ids_covered, ["F001", "F002", "F003"])

    def test_invented_quantity_triggers_repair(self):
        first = reply([rewrite("F001", "Your head has hurt for 3 days.")], sources_used=["Headache"])
        second = reply([rewrite("F001", HEADACHE)], sources_used=["Headache"])

        output = simplify(ATOMS[:1], {"F001": [HEADACHE_PASSAGE]}, send=scripted(first, second))

        self.assertEqual(output.simplified_text, HEADACHE)
        self.assertEqual(output.review_warnings, [])

    def test_unrepaired_quantity_falls_back_to_source_wording(self):
        bad = reply([rewrite("F001", "Your head has hurt for 3 days.")], sources_used=["Headache"])

        output = simplify(ATOMS[:1], {"F001": [HEADACHE_PASSAGE]}, send=scripted(bad, bad))

        self.assertEqual(output.simplified_text, ATOMS[0]["text"])
        self.assertTrue(any("quantities" in warning for warning in output.review_warnings))


class MeaningCheckTest(unittest.TestCase):
    def assert_repair_requested(self, atom, bad_text, expected_word):
        first = reply([rewrite(atom["fact_id"], bad_text)])
        second = reply([rewrite(atom["fact_id"], bad_text)])

        output = simplify([atom], {}, send=scripted(first, second))

        self.assertEqual(output.simplified_text, atom["text"])
        self.assertTrue(
            any(expected_word in warning for warning in output.review_warnings),
            output.review_warnings,
        )

    def test_absent_fact_rewritten_as_present_is_repaired(self):
        self.assert_repair_requested(ATOMS[1], "You have an upset stomach.", "absence")

    def test_possible_or_family_fact_losing_its_owner_is_repaired(self):
        self.assert_repair_requested(ATOMS[2], "You have celiac disease.", "owner")

    def test_family_fact_kept_uncertain_passes(self):
        output = simplify([ATOMS[2]], {}, send=scripted(reply([rewrite("F003", FATHER, ["celiac disease"])])))

        self.assertEqual(output.simplified_text, FATHER)

    def test_present_fact_gaining_a_negation_is_repaired(self):
        self.assert_repair_requested(ATOMS[0], "Your head does not hurt.", "negation")


class CitationTest(unittest.TestCase):
    def test_unknown_citation_is_dropped_with_a_warning(self):
        output = simplify(
            ATOMS[:1], {"F001": [HEADACHE_PASSAGE]},
            send=scripted(reply([rewrite("F001", HEADACHE)], sources_used=["Migraine"])),
        )

        self.assertEqual(output.source_metadata, [])
        self.assertTrue(any("not supplied" in warning for warning in output.review_warnings))

    def test_title_is_not_cited_twice(self):
        output = simplify(
            ATOMS[:1], {"F001": [HEADACHE_PASSAGE]},
            send=scripted(reply([rewrite("F001", HEADACHE)], sources_used=["Headache", "Headache"])),
        )

        self.assertEqual(len(output.source_metadata), 1)


class ContractTest(unittest.TestCase):
    def test_no_atoms_is_named(self):
        with self.assertRaises(SimplifierError) as caught:
            simplify([], {}, send=scripted())

        self.assertIn("no atoms", str(caught.exception))

    def test_unparseable_reply_is_named(self):
        with self.assertRaises(SimplifierError) as caught:
            simplify(ATOMS[:1], {"F001": []}, send=scripted("not json"))

        self.assertIn("not a valid simplification", str(caught.exception))

    def test_format_atoms_numbers_and_tags_every_fact(self):
        rendered = format_atoms(ATOMS)

        self.assertIn("1. F001 [symptom/present]", rendered)
        self.assertIn("2. F002 [symptom/absent]", rendered)

    def test_retrieval_failure_fails_the_run_instead_of_ungrounding_it(self):
        def broken_retrieval(query, limit=3):
            raise RuntimeError("store unavailable")

        original = simplifier.retrieve_passages
        simplifier.retrieve_passages = broken_retrieval
        try:
            with self.assertRaises(SimplifierError) as caught:
                build_grounding(ATOMS[:1])
        finally:
            simplifier.retrieve_passages = original

        self.assertIn("retrieval failed", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
