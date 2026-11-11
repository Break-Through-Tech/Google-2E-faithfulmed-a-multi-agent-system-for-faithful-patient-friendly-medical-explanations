import unittest
from unittest.mock import patch

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from streamlit.testing.v1 import AppTest

from faithfulmed.pipeline_ui import PipelineResult, run_pipeline


class FixtureAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        outputs = [
            ("Extractor", "atoms", {"atoms": [{"text": "No nausea."}]}),
            ("Simplifier", "draft", "Initial draft"),
            ("Verifier", "verdict", "First review"),
            ("Refiner", "draft", "Revised draft"),
            ("Verifier", "verdict", "Second review"),
            ("Refiner", "draft", "Second revision"),
            ("Readability", "final", "Final explanation"),
        ]
        for author, key, value in outputs:
            yield Event(author=author, actions=EventActions(state_delta={key: value}))


class ErrorAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        yield Event(author="Extractor", actions=EventActions(state_delta={"atoms": []}))
        yield Event(author="Verifier", error_code="TEST_ERROR", error_message="failed")


class MissingFinalAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        yield Event(author="Extractor", actions=EventActions(state_delta={"atoms": []}))


class EmptyFinalAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        yield Event(author="Readability", actions=EventActions(state_delta={"final": "  "}))


class PipelineRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_adk_runner_records_outputs_and_loop_history(self):
        result = PipelineResult()
        updates = []
        await run_pipeline(
            "A sample note",
            result,
            lambda current: updates.append(sum(map(len, current.outputs.values()))),
            agent=FixtureAgent(name="FixturePipeline"),
        )
        self.assertEqual(updates, list(range(1, 8)))
        self.assertEqual(result.outputs["Verifier"], ["First review", "Second review"])
        self.assertEqual(result.outputs["Refiner"], ["Revised draft", "Second revision"])
        self.assertEqual(result.outputs["Readability"], ["Final explanation"])

    async def test_adk_error_keeps_partial_outputs_and_raises(self):
        result = PipelineResult()
        with self.assertRaisesRegex(RuntimeError, "TEST_ERROR"):
            await run_pipeline("A sample note", result, lambda _: None, ErrorAgent(name="Error"))
        self.assertEqual(result.outputs["Extractor"], [[]])

    async def test_missing_final_output_is_failure(self):
        result = PipelineResult()
        with self.assertRaisesRegex(RuntimeError, "without a final"):
            await run_pipeline("A sample note", result, lambda _: None, MissingFinalAgent(name="Missing"))

    async def test_empty_final_output_is_failure(self):
        result = PipelineResult()
        with self.assertRaisesRegex(RuntimeError, "without a final"):
            await run_pipeline("A sample note", result, lambda _: None, EmptyFinalAgent(name="Empty"))

    async def test_empty_note_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Enter a medical note"):
            await run_pipeline("  ", PipelineResult(), lambda _: None)

    def test_only_expected_author_and_key_are_recorded(self):
        result = PipelineResult()
        self.assertFalse(result.record("Other", {"draft": "ignored"}))
        self.assertFalse(result.record("Verifier", {"draft": "ignored"}))
        self.assertEqual(result.outputs, {})


class StreamlitAppTests(unittest.TestCase):
    def test_initial_screen_and_empty_input(self):
        app = AppTest.from_file("streamlit_app.py").run()
        self.assertFalse(app.exception)
        self.assertIn("fictional", app.warning[0].value.lower())
        app.text_area[0].set_value("  ")
        app.button[0].click().run()
        self.assertIn("Enter a medical note", app.error[0].value)
        self.assertFalse(app.success)

    def test_success_and_failure_show_actual_outputs(self):
        async def successful_run(note, result, on_update):
            result.record("Extractor", {"atoms": {"atoms": [{"source_quote": "No nausea."}]}})
            result.record("Readability", {"final": "Final explanation"})
            on_update(result)

        with patch("faithfulmed.pipeline_ui.run_pipeline", successful_run):
            app = AppTest.from_file("streamlit_app.py").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
            self.assertIn("Final explanation", [item.value for item in app.text])
            self.assertTrue(app.json)

        async def failed_run(note, result, on_update):
            result.record("Extractor", {"atoms": []})
            on_update(result)
            raise RuntimeError("model unavailable")

        with patch("faithfulmed.pipeline_ui.run_pipeline", failed_run):
            app.button[0].click().run()
            self.assertIn("model unavailable", app.error[0].value)
            self.assertFalse(app.success)
            self.assertNotIn("Final explanation", [item.value for item in app.text])


if __name__ == "__main__":
    unittest.main()
