import tempfile
import unittest
from pathlib import Path

from faithfulmed.agents.extractor import (
    Atom,
    Extraction,
    attach_topics,
    check_quotes,
    find_topic,
    load_aliases,
    load_extraction,
    missing_numbers,
    save_extraction,
)


class ExtractorHelperTests(unittest.TestCase):
    def make_extraction(self, quote="No history of asthma.", text="No history of asthma."):
        return Extraction(atoms=[
            Atom(
                text=text,
                type="diagnosis",
                assertion="absent",
                source_quote=quote,
            )
        ])

    def test_load_aliases_returns_string_mapping(self):
        aliases = load_aliases()
        self.assertTrue(aliases)
        self.assertTrue(all(isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()))

    def test_find_topic_prefers_longest_phrase(self):
        aliases = {
            "heart": "Heart",
            "heart failure": "Heart Failure",
            "congestive heart failure": "Congestive Heart Failure",
        }
        self.assertEqual(
            find_topic("Possible congestive heart failure.", aliases),
            "Congestive Heart Failure",
        )

    def test_find_topic_returns_none_without_match(self):
        self.assertIsNone(find_topic("No matching phrase here", {}))

    def test_attach_topics_preserves_atom_fields(self):
        extraction = self.make_extraction(text="History of asthma")
        linked = attach_topics(extraction, {"asthma": "Asthma"})
        self.assertEqual(linked[0].topic, "Asthma")
        self.assertEqual(linked[0].assertion, "absent")
        self.assertEqual(linked[0].source_quote, "No history of asthma.")

    def test_check_quotes_requires_verbatim_match(self):
        source = "No history of asthma."
        extraction = self.make_extraction(quote="no history of asthma.")
        self.assertEqual(check_quotes(source, extraction), extraction.atoms)

    def test_missing_numbers_reports_unrepresented_values(self):
        extraction = self.make_extraction(quote="A1C was 7.2%.", text="A1C was 7.2%.")
        self.assertEqual(missing_numbers("A1C was 7.2%; glucose was 105.", extraction), ["105"])

    def test_save_and_load_round_trip(self):
        extraction = self.make_extraction()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = save_extraction("sample-1", extraction.atoms, Path(temp_dir) / "nested")
            payload = load_extraction(output_path)

        self.assertEqual(payload["doc_id"], "sample-1")
        self.assertEqual(payload["atoms"][0], extraction.atoms[0].model_dump(mode="json"))

    def test_save_rejects_path_traversal_doc_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                save_extraction("../outside", [], Path(temp_dir))


if __name__ == "__main__":
    unittest.main()