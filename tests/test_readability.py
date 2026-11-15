import unittest
from unittest.mock import patch

from faithfulmed.agents import readability
from faithfulmed.agents.readability import (
    MAX_ATTEMPTS,
    ReadabilityError,
    improve_readability,
    measure_readability,
)


def reply(text):
    """A model reply in the JSON shape ReadabilityReply expects."""
    return '{"revised_text": "' + text + '"}'


def sequence_sender(replies):
    """A fake send() that returns the given replies in order and records each call."""
    remaining = list(replies)
    calls = []

    def send(prompt, model):
        calls.append(prompt)
        if not remaining:
            return reply("unused")
        return remaining.pop(0)

    send.calls = calls
    return send


def fixed_grades(grades):
    """A measure_readability replacement that hands out the given grades in order."""
    remaining = list(grades)

    def measure(text):
        grade = remaining.pop(0) if remaining else grades[-1]
        return {"grade": grade, "smog": 9.0}

    return measure


class MeasureTests(unittest.TestCase):
    def test_blank_text_raises(self):
        with self.assertRaisesRegex(ReadabilityError, "empty"):
            measure_readability("   ")

    def test_reports_grade_and_smog(self):
        scores = measure_readability("The patient has a headache. Rest at home. Drink water.")
        self.assertIn("grade", scores)
        self.assertIn("smog", scores)


class ImproveTests(unittest.TestCase):
    def test_text_that_already_passes_uses_no_model_call(self):
        send = sequence_sender([])
        result = improve_readability("note", [{"text": "x"}], "Short simple text.", send=send)
        self.assertEqual(result.revised_text, "Short simple text.")
        self.assertTrue(result.meets_target)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(send.calls, [])

    def test_grade_exactly_at_target_passes(self):
        with patch.object(readability, "measure_readability", fixed_grades([8.0])):
            result = improve_readability("note", [{"text": "x"}], "Original.")
        self.assertTrue(result.meets_target)
        self.assertEqual(result.attempts, 0)

    def test_grade_just_above_target_is_not_rounded_down(self):
        send = sequence_sender([])
        with patch.object(readability, "measure_readability", fixed_grades([8.04])):
            result = improve_readability("note", [{"text": "x"}], "Original.", send=send)
        self.assertFalse(result.meets_target)

    def test_accepted_candidate_is_selected(self):
        send = sequence_sender([reply("You have a headache.")])
        with patch.object(readability, "measure_readability", fixed_grades([12.0, 6.0])):
            result = improve_readability("note", [{"text": "x"}], "Original.", send=send)
        self.assertEqual(result.revised_text, "You have a headache.")
        self.assertTrue(result.meets_target)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(send.calls), 1)

    def test_two_failed_candidates_keep_the_incoming_text(self):
        send = sequence_sender([reply("Hard read."), reply("Still hard.")])
        with patch.object(readability, "measure_readability", fixed_grades([12.0, 9.5, 9.0])):
            result = improve_readability("note", [{"text": "x"}], "Original.", send=send)
        self.assertEqual(result.revised_text, "Original.")
        self.assertFalse(result.meets_target)
        self.assertEqual(result.attempts, MAX_ATTEMPTS)
        self.assertTrue(result.review_warnings)

    def test_candidate_that_drops_a_number_is_rejected(self):
        send = sequence_sender([reply("Take it twice daily."), reply("Take it twice daily.")])
        with patch.object(readability, "measure_readability", fixed_grades([12.0])):
            result = improve_readability("note", [{"text": "x"}], "Take 5 mg twice daily.", send=send)
        self.assertEqual(result.revised_text, "Take 5 mg twice daily.")
        self.assertFalse(result.meets_target)
        self.assertTrue(any("dropped" in warning for warning in result.review_warnings))

    def test_blank_text_raises(self):
        with self.assertRaisesRegex(ReadabilityError, "empty"):
            improve_readability("note", [{"text": "x"}], "   ")

    def test_no_facts_raises(self):
        with self.assertRaisesRegex(ReadabilityError, "no facts"):
            improve_readability("note", [], "Some text.")

    def test_model_failure_raises_named_error(self):
        def failing_send(prompt, model):
            raise RuntimeError("boom")

        with patch.object(readability, "measure_readability", fixed_grades([12.0])):
            with self.assertRaises(ReadabilityError):
                improve_readability("note", [{"text": "x"}], "Original.", send=failing_send)


if __name__ == "__main__":
    unittest.main()
