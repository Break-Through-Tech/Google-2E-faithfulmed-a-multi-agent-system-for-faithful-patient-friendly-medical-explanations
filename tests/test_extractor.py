"""Offline tests for the Extractor."""
import json
import unittest

from faithfulmed.agents.extractor import (
    Atom,
    ExtractionRecord,
    ExtractionValidationError,
    ExtractorError,
    extract_atoms,
    missing_numbers,
)

NOTE = (
    "HISTORY OF PRESENT ILLNESS:, The patient reports a mild headache. "
    "He denies nausea. His father has questionable celiac disease. "
    "Blood pressure was 120/80. Oxygen saturation was 98% on O2. "
    "ADMITTING DIAGNOSES:,1. Fever."
)

HEADACHE = {
    "text": "The patient reports a mild headache.",
    "type": "symptom",
    "assertion": "present",
    "source_quote": "The patient reports a mild headache.",
}
NO_NAUSEA = {
    "text": "He denies nausea.",
    "type": "symptom",
    "assertion": "absent",
    "source_quote": "He denies nausea.",
}
FATHER = {
    "text": "His father has questionable celiac disease.",
    "type": "diagnosis",
    "assertion": "family",
    "source_quote": "His father has questionable celiac disease.",
}
PARAPHRASED = {
    "text": "The patient reports a severe headache.",
    "type": "symptom",
    "assertion": "present",
    "source_quote": "The patient reports a severe headache.",
}
FURNITURE = {
    "text": "ADMITTING DIAGNOSES:,1. Fever.",
    "type": "diagnosis",
    "assertion": "present",
    "source_quote": "ADMITTING DIAGNOSES:,1. Fever.",
}


def reply(*atoms):
    """One scripted model reply, in the JSON shape the schema expects."""
    return json.dumps({"atoms": list(atoms)})


def scripted(*replies):
    """A sender that returns the scripted replies in order and fails loudly when empty."""
    remaining = list(replies)

    def send(prompt, model):
        if not remaining:
            raise AssertionError("the test script ran out of replies")
        return remaining.pop(0)

    return send


def checks_in(error):
    return {record["check"] for record in error.diagnostics}


class AcceptedExtractionTest(unittest.TestCase):
    def test_valid_reply_gets_identity_and_spans(self):
        record = extract_atoms(NOTE, send=scripted(reply(HEADACHE, NO_NAUSEA)))

        self.assertIsInstance(record, ExtractionRecord)
        self.assertEqual([atom.fact_id for atom in record.atoms], ["F001", "F002"])
        for atom in record.atoms:
            self.assertEqual(NOTE[atom.char_start:atom.char_end], atom.source_quote)

    def test_repair_that_fixes_the_problem_is_accepted(self):
        record = extract_atoms(NOTE, send=scripted(reply(PARAPHRASED), reply(HEADACHE)))

        self.assertEqual(len(record.atoms), 1)
        self.assertEqual(record.atoms[0].source_quote, HEADACHE["source_quote"])

    def test_blank_note_is_named_and_rejected(self):
        with self.assertRaises(ExtractorError) as caught:
            extract_atoms("   ", send=scripted())

        self.assertIn("blank", str(caught.exception))


class RejectedExtractionTest(unittest.TestCase):
    def assert_check_fires(self, expected_check, *atoms):
        with self.assertRaises(ExtractionValidationError) as caught:
            extract_atoms(NOTE, send=scripted(reply(*atoms), reply(*atoms)))

        self.assertIn(expected_check, checks_in(caught.exception))
        return caught.exception

    def test_paraphrased_quote_is_rejected(self):
        self.assert_check_fires("quote", PARAPHRASED)

    def test_text_must_come_from_the_note(self):
        atom = dict(HEADACHE, text="The patient has a bad head pain")
        self.assert_check_fires("text", atom)

    def test_section_header_furniture_is_rejected(self):
        self.assert_check_fires("furniture", FURNITURE)

    def test_nested_quote_is_dropped(self):
        nested = {
            "text": "He denies nausea. His father has questionable celiac disease.",
            "type": "symptom",
            "assertion": "absent",
            "source_quote": "He denies nausea. His father has questionable celiac disease.",
        }
        record = extract_atoms(NOTE, send=scripted(reply(NO_NAUSEA, nested), reply(NO_NAUSEA, nested)))

        self.assertEqual([atom.fact_id for atom in record.atoms], ["F001"])
        self.assertTrue(any(warning.startswith("F002 dropped") for warning in record.review_warnings))

    def test_duplicate_span_is_rejected(self):
        with self.assertRaises(ExtractionValidationError) as caught:
            extract_atoms(NOTE, send=scripted(reply(HEADACHE, HEADACHE), reply(HEADACHE, HEADACHE)))

        self.assertIn("overlap", checks_in(caught.exception))

    def test_absent_fact_without_a_cue_is_rejected(self):
        atom = {"text": "nausea", "type": "symptom", "assertion": "absent", "source_quote": "nausea"}
        self.assert_check_fires("assertion cue", atom)

    def test_family_fact_must_name_the_relative(self):
        atom = {
            "text": "questionable celiac disease",
            "type": "diagnosis",
            "assertion": "family",
            "source_quote": "questionable celiac disease",
        }
        self.assert_check_fires("family owner", atom)

    def test_patient_fact_must_not_name_a_relative(self):
        atom = dict(FATHER, assertion="present")
        self.assert_check_fires("family owner", atom)

    def test_repair_that_drops_a_fact_is_recorded(self):
        record = extract_atoms(NOTE, send=scripted(reply(HEADACHE, PARAPHRASED), reply(HEADACHE)))

        self.assertEqual(len(record.atoms), 1)
        self.assertTrue(any("down from 2" in warning for warning in record.review_warnings))

    def test_one_bad_atom_does_not_discard_the_note(self):
        record = extract_atoms(NOTE, send=scripted(reply(HEADACHE, FURNITURE), reply(HEADACHE, FURNITURE)))

        self.assertEqual([atom.fact_id for atom in record.atoms], ["F001"])
        self.assertTrue(any(warning.startswith("F002 dropped") for warning in record.review_warnings))

    def test_family_fact_is_accepted_when_it_names_the_relative(self):
        record = extract_atoms(NOTE, send=scripted(reply(FATHER)))

        self.assertEqual(record.atoms[0].assertion, "family")


class DiagnosticsTest(unittest.TestCase):
    def test_message_carries_no_patient_text(self):
        with self.assertRaises(ExtractionValidationError) as caught:
            extract_atoms(NOTE, send=scripted(reply(PARAPHRASED), reply(PARAPHRASED)))

        message = str(caught.exception)
        self.assertNotIn("headache", message.lower())
        self.assertNotIn("celiac", message.lower())
        self.assertIn("quote", message)

    def test_each_record_names_the_fact_and_the_check(self):
        with self.assertRaises(ExtractionValidationError) as caught:
            extract_atoms(NOTE, send=scripted(reply(PARAPHRASED), reply(PARAPHRASED)))

        record = caught.exception.diagnostics[0]
        self.assertEqual(record["fact"], 1)
        self.assertEqual(record["check"], "quote")
        self.assertTrue(record["message"])

    def test_attempts_are_reported(self):
        with self.assertRaises(ExtractionValidationError) as caught:
            extract_atoms(NOTE, send=scripted(reply(PARAPHRASED), reply(PARAPHRASED)))

        self.assertEqual(caught.exception.attempts, 2)


class NumberCoverageTest(unittest.TestCase):
    def test_lettered_digits_are_not_measurements(self):
        missing = missing_numbers(NOTE, [])

        self.assertIn("120", missing)
        self.assertIn("98", missing)
        self.assertNotIn("2", missing)

    def test_quoted_numbers_are_covered(self):
        atom = {
            "text": "Blood pressure was 120/80.",
            "type": "lab_value",
            "assertion": "present",
            "source_quote": "Blood pressure was 120/80.",
        }
        missing = missing_numbers(NOTE, [Atom.model_validate(atom)])

        self.assertNotIn("120", missing)


if __name__ == "__main__":
    unittest.main()
