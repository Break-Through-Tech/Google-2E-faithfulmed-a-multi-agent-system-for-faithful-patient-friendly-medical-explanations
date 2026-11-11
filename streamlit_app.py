"""Interactive, local visualizer for the FaithfulMed ADK scaffold."""

import asyncio
import json

import streamlit as st

from faithfulmed.pipeline_ui import PipelineResult, STAGE_OUTPUTS, run_pipeline


SAMPLE_NOTE = (
    "The patient reports a mild headache. The patient denies nausea. "
    "No known allergies."
)
STAGE_DESCRIPTIONS = {
    "Extractor": "Extracts source-backed clinical facts.",
    "Simplifier": "Drafts a plain-language explanation.",
    "Verifier": "Checks the draft against the extracted facts.",
    "Refiner": "Revises the draft using verifier feedback.",
    "Readability": "Produces a final readability pass.",
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
            st.markdown(f"**{stage}**")
            st.caption(f"Output received ({count})" if count else "Waiting for output")
    st.caption("Verifier and Refiner run in a loop of up to two iterations.")


def display_results(result: PipelineResult) -> None:
    final = result.outputs.get("Readability", [])
    if final:
        st.subheader("Patient-friendly explanation")
        with st.container(border=True):
            display_value(final[-1], formatted_text=True)
        with st.expander("View original output"):
            st.text(final[-1])

    st.subheader("Explore agent outputs")
    for stage, description in STAGE_DESCRIPTIONS.items():
        outputs = result.outputs.get(stage, [])
        with st.expander(f"{stage} · {len(outputs)} output(s)"):
            st.caption(description)
            if not outputs:
                st.write("No output received.")
            for index, value in enumerate(outputs, start=1):
                if len(outputs) > 1:
                    st.markdown(f"**Output {index}**")
                display_value(value, formatted_text=stage in {"Simplifier", "Refiner", "Readability"})


st.set_page_config(page_title="FaithfulMed", page_icon="🩺", layout="wide")
st.title("FaithfulMed")
st.write("Explore how a clinical note moves through the multi-agent pipeline.")
st.warning(
    "Prototype only: not medical advice. Use fictional or approved de-identified "
    "notes; text is sent to the configured model provider."
)
st.info(
    "This interface uses the current ADK scaffold. Outputs are experimental, "
    "and the final readability pass is not re-verified by this pipeline."
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
