"""Interactive, local visualizer for the FaithfulMed ADK scaffold."""

import asyncio
import json

import streamlit as st

from faithfulmed.pipeline_ui import PipelineResult, STAGE_LABELS, STAGE_OUTPUTS, run_pipeline
from faithfulmed.readability import score_readability


SAMPLE_NOTE = (
    "The patient reports a mild headache. The patient denies nausea. "
    "No known allergies."
)
STAGE_DESCRIPTIONS = {
    "Extractor": "Extracts source-backed clinical facts.",
    "Simplifier": "Drafts a plain-language explanation.",
    "readability": "Rewrites the draft for readability.",
    "Verifier": "Checks the draft against the extracted facts.",
    "Refiner": "Revises the draft using verifier feedback.",
}


def display_value(value: object, *, formatted_text: bool = False) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        if formatted_text:
            st.markdown(value)
            return
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            st.text(value)
            return
    if isinstance(value, (dict, list)):
        st.json(value)
    else:
        st.text(str(value))


def display_progress(result: PipelineResult) -> None:
    columns = st.columns(len(STAGE_OUTPUTS))
    for column, stage in zip(columns, STAGE_OUTPUTS):
        count = len(result.outputs.get(stage, []))
        with column:
            st.markdown(f"**{STAGE_LABELS.get(stage, stage)}**")
            st.caption(f"Output received ({count})" if count else "Waiting for output")
    st.caption("Verifier and Refiner run in a loop of up to two iterations.")


def display_verifier_report(result: PipelineResult) -> None:
    reports = result.outputs.get("Verifier", [])
    if not reports:
        return
    st.subheader("Verifier results")
    st.caption(
        "Latest verifier report; these are model judgments, not proof of clinical accuracy. "
        "The refiner runs only when the verifier requests revision."
    )
    if result.current_text is None or result.current_text != result.last_verified_text:
        st.warning(
            "The explanation changed after this report. These results do not assess the final text."
        )
    else:
        st.caption("This report assessed the current explanation.")
    report = reports[-1]
    if not isinstance(report, dict):
        st.warning("The verifier report is not a structured object; scores are unavailable.")
        return

    fields = (
        ("faithfulness_score", "Faithfulness", float),
        ("addition_count", "Additions", int),
        ("omission_count", "Omissions", int),
        ("total_facts", "Total facts", int),
        ("hallucination", "Major hallucination flagged", bool),
        ("needs_revision", "Revision needed", bool),
    )
    for columns, row in ((st.columns(3), fields[:3]), (st.columns(3), fields[3:])):
        for column, (key, label, expected) in zip(columns, row):
            value = report.get(key)
            if expected is float:
                claims = report.get("claims")
                valid = (
                    type(value) in (float, int)
                    and 0 <= value <= 1
                    and isinstance(claims, list)
                    and bool(claims)
                )
            else:
                valid = type(value) is expected and (expected is not int or value >= 0)
            if not valid:
                rendered = "Unavailable"
            elif expected is float:
                rendered = f"{value:.1%}"
            elif expected is bool:
                rendered = "Yes" if value else "No"
            else:
                rendered = str(value)
            column.metric(label, rendered)
    if report.get("claims") == []:
        st.caption("Faithfulness is unavailable because the verifier assessed no claims.")
    coverage = report.get("false_coverage")
    if isinstance(coverage, list) and all(type(item) is int and item > 0 for item in coverage):
        st.write(
            "False coverage (fact positions): "
            + (", ".join(str(item) for item in coverage) if coverage else "None reported")
        )
    else:
        st.write("False coverage: Unavailable")
    with st.expander("How verifier fields are calculated"):
        st.write(
            "Faithfulness is the fraction of claims marked supported or background. "
            "Additions count claims marked background or unsupported. Omissions count facts "
            "judged missing. Total facts is the number of extractor atoms. A major hallucination "
            "is flagged for a major unsupported or distorted claim. Revision is needed for a "
            "major hallucination or major omission. False coverage lists one-based fact positions "
            "the Simplifier marked covered but the Verifier judged missing."
        )


def display_results(result: PipelineResult) -> None:
    final = result.current_text
    if final is not None:
        st.subheader("Patient-friendly explanation")
        with st.container(border=True):
            display_value(final, formatted_text=True)
        with st.expander("View original output"):
            st.text(final)

        st.subheader("Readability scores")
        try:
            scores = score_readability(final)
        except ValueError as exc:
            st.warning(f"Readability scores unavailable: {exc}")
        else:
            grade, ease, words = st.columns(3)
            grade.metric("Flesch-Kincaid grade", f"{scores.grade:.2f}")
            ease.metric("Flesch reading ease", f"{scores.reading_ease:.2f}")
            words.metric("Words", scores.word_count)
            if scores.grade <= 8:
                st.success("The estimated grade meets the project's grade 8 or below target.")
            else:
                st.warning("The estimated grade is above the project's grade 8 target.")
            if scores.word_count < 30:
                st.caption("Short passages can produce less reliable readability estimates.")
            st.caption(
                "Grade estimates US school reading level; for reading ease, higher is generally easier. "
                "These are automated estimates of the final output, including headings. "
                "The target comparison uses the unrounded grade. "
                "They do not assess medical jargon, accuracy, or whether a patient understands the text."
            )

    display_verifier_report(result)
    st.subheader("Explore agent outputs")
    for stage, description in STAGE_DESCRIPTIONS.items():
        outputs = result.outputs.get(stage, [])
        with st.expander(f"{STAGE_LABELS.get(stage, stage)} · {len(outputs)} output(s)"):
            st.caption(description)
            if not outputs:
                st.write("No output received.")
            for index, value in enumerate(outputs, start=1):
                if len(outputs) > 1:
                    st.markdown(f"**Output {index}**")
                display_value(value, formatted_text=stage in {"Refiner", "readability"})


st.set_page_config(page_title="FaithfulMed", page_icon="🩺", layout="wide")
st.title("FaithfulMed")
st.write("Explore how a clinical note moves through the multi-agent pipeline.")
st.warning(
    "Prototype only: not medical advice. Use fictional or approved de-identified "
    "notes; text is sent to the configured model provider."
)
st.info(
    "Outputs are experimental. A final refinement at the iteration limit may not "
    "be checked again by the verifier."
)

with st.form("pipeline_form"):
    note = st.text_area("Medical note", value=SAMPLE_NOTE, height=180)
    submitted = st.form_submit_button("Run pipeline", type="primary")

progress = st.empty()
message = st.empty()
results = st.empty()

if submitted:
    result = PipelineResult()
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_error"] = None

    def update(current: PipelineResult) -> None:
        with progress.container():
            display_progress(current)
        with results.container():
            display_results(current)

    update(result)
    if not note.strip():
        st.session_state["pipeline_error"] = "Enter a medical note before running the pipeline."
    else:
        try:
            with message.container():
                with st.spinner("Running the pipeline. This can take a few minutes."):
                    asyncio.run(run_pipeline(note, result, update))
        except Exception as exc:
            st.session_state["pipeline_error"] = f"{type(exc).__name__}: {exc}"

if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    with progress.container():
        display_progress(result)
    message.empty()
    if st.session_state.get("pipeline_error"):
        st.error(f"Pipeline failed: {st.session_state['pipeline_error']}")
        st.caption("Any outputs below are partial and should not be treated as a completed run.")
    else:
        st.success("The pipeline finished.")
    with results.container():
        display_results(result)
