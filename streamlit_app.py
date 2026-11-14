"""Interactive, local visualizer for the FaithfulMed ADK scaffold."""

import asyncio
import json
from pathlib import Path
import streamlit as st
from textstat import textstat

from faithfulmed.icons import CHECK, STAGE_ICONS, WARNING
from faithfulmed.pipeline_ui import PipelineResult, STAGE_LABELS, STAGE_OUTPUTS, run_pipeline
from faithfulmed.readability_scores import score_readability


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
    # Pydantic results expose model_dump. getattr keeps the call type-safe.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
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


def refiner_was_skipped(result: PipelineResult) -> bool:
    reports = result.outputs.get("Verifier", [])
    return (
        result.phase == "completed" and not result.outputs.get("Refiner")
        and bool(reports) and isinstance(reports[-1], dict)
        and reports[-1].get("needs_revision") is False
    )


JOURNEY_STATE_WORDS = {
    "is-done": "done",
    "is-live": "running",
    "is-failed": "failed",
    "is-skipped": "skipped",
    "is-empty": "no output",
    "is-waiting": "waiting",
}


def journey_states(result: PipelineResult) -> list[str]:
    """One CSS state per stage, in pipeline order."""
    states = []
    for stage in STAGE_OUTPUTS:
        if result.active_stage == stage and result.phase == "failed":
            states.append("is-failed")
        elif result.active_stage == stage and result.phase == "running":
            states.append("is-live")
        elif result.outputs.get(stage):
            states.append("is-done")
        elif stage == "Refiner" and refiner_was_skipped(result):
            states.append("is-skipped")
        elif result.phase == "completed":
            states.append("is-empty")
        else:
            states.append("is-waiting")
    return states


def journey_headline(result: PipelineResult) -> str:
    """The label on the left of the meter: what is happening now."""
    if result.phase == "failed":
        if result.active_stage:
            label = STAGE_LABELS.get(result.active_stage, result.active_stage)
            return f"Stopped at {label}"
        return "Stopped"
    if result.phase == "completed":
        return "Complete"
    if result.phase == "running":
        if result.active_stage:
            # A stage that has been announced but has no output yet still reads live,
            # so the bar pulses from the first moment of a run rather than stalling.
            return STAGE_LABELS.get(result.active_stage, result.active_stage)
        return "Preparing"
    return "Not started"


def journey_icon(result: PipelineResult) -> str:
    """The icon beside the headline: the stage it is on, or its outcome."""
    if result.phase == "failed":
        return STAGE_ICONS.get(result.active_stage or "", WARNING)
    if result.phase == "completed":
        return CHECK
    if result.phase == "running" and result.active_stage:
        return STAGE_ICONS.get(result.active_stage, "")
    return ""


def journey_tone(result: PipelineResult) -> str:
    """The colour the headline takes: accent while working, red when stopped."""
    if result.phase == "failed":
        return "is-failed"
    if result.phase in {"running", "completed"}:
        return "is-live"
    return ""


def build_meter_html(result: PipelineResult) -> str:
    """The journey as one measure: what is happening, how far, and how many settled.

    Returned as a single block on purpose. display_progress is rewritten into an
    st.empty placeholder on every stage event, and a block count that changes
    between writes is what crashed the page before.
    """
    states = journey_states(result)
    settled = sum(1 for state in states if state in {"is-done", "is-skipped"})
    total = len(states)
    if result.phase == "running" and not any(state == "is-live" for state in states):
        states[0] = "is-live"

    segments = "".join(
        f'<span class="fm-meter-seg {state}"></span>' for state in states
    )
    # The headline only shows a stage for as long as that stage runs, and the
    # readability stage is over in seconds. These labels sit in the same grid as
    # the segments, so a finished run can still be read stage by stage.
    labels = "".join(
        f'<span class="fm-meter-label">{STAGE_LABELS.get(stage, stage)}</span>'
        for stage in STAGE_OUTPUTS
    )
    spoken = ", ".join(
        f"{STAGE_LABELS.get(stage, stage)} {JOURNEY_STATE_WORDS[state]}"
        for stage, state in zip(STAGE_OUTPUTS, states)
    )
    stage_class = f"fm-meter-stage {journey_tone(result)}".strip()
    return (
        '<div class="fm-meter">'
        '<div class="fm-meter-head">'
        f'<span class="{stage_class}">{journey_icon(result)}'
        f'<span>{journey_headline(result)}</span></span>'
        f'<span class="fm-meter-count">{settled} of {total}</span>'
        "</div>"
        f'<div class="fm-meter-bar" role="img" aria-label="Pipeline stages: {spoken}">'
        f"{segments}</div>"
        f'<div class="fm-meter-labels" aria-hidden="true">{labels}</div>'
        "</div>"
    )


def display_progress(result: PipelineResult) -> None:
    st.subheader("Pipeline journey")
    st.markdown(build_meter_html(result), unsafe_allow_html=True)


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
                # isinstance narrows the value; bool is excluded because it is an int.
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
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

        st.subheader("Readability scores")
        try:
            scores = score_readability(final)
        except ValueError as exc:
            st.warning(f"Readability scores unavailable: {exc}")
        else:
            # Reading ease and word count are shown for context only.
            # The grade target is decided by scores.grade.
            reading_ease = textstat.flesch_reading_ease(final)
            word_count = textstat.lexicon_count(final, removepunct=True)
            grade, ease, words = st.columns(3)
            grade.metric("Flesch-Kincaid grade", f"{scores.grade:.2f}")
            ease.metric("Flesch reading ease", f"{reading_ease:.2f}")
            words.metric("Words", word_count)
            if scores.grade <= 8:
                st.success("The estimated grade meets the project's grade 8 or below target.")
            else:
                st.warning("The estimated grade is above the project's grade 8 target.")
            if word_count < 30:
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
                if stage == "Refiner" and refiner_was_skipped(result):
                    st.write("Skipped: the Verifier did not request a revision.")
                else:
                    st.write("No output received.")
            for index, value in enumerate(outputs, start=1):
                if len(outputs) > 1:
                    st.markdown(f"**Output {index}**")
                display_value(value, formatted_text=stage in {"Refiner", "readability"})


st.set_page_config(page_title="FaithfulMed", page_icon="🩺", layout="wide")
styles = Path(__file__).resolve().parent / "faithfulmed" / "app.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
with st.container(key="page_header"):
    st.title("FaithfulMed")
    st.write("Explore how a clinical note moves through the multi-agent pipeline.")
st.warning(
    "This is the prototype. Do not use for medical advice. "
    "Use fictional or de-identified notes."
)

with st.form("pipeline_form"):
    note = st.text_area("Medical note", value=SAMPLE_NOTE, height=180)
    submitted = st.form_submit_button("Run pipeline", type="primary")

progress = st.empty()

if "pipeline_result" not in st.session_state and not submitted:
    with progress.container():
        display_progress(PipelineResult(phase="idle"))

if submitted:
    result = PipelineResult()
    st.session_state["pipeline_result"] = result
    st.session_state["pipeline_error"] = None

    # The results panel is deliberately NOT written here. display_results builds a
    # different number of blocks depending on what exists yet (explanation,
    # verifier report, per-stage expanders), and rewriting a placeholder with a
    # tree that changes shape sends deltas the client cannot place: the page dies
    # with "Bad 'setIn' index" and goes blank. It is rendered once, after the run.
    def update(current: PipelineResult) -> None:
        with progress.container():
            display_progress(current)

    update(result)
    if not note.strip():
        st.session_state["pipeline_error"] = "Enter a medical note before running the pipeline."
        result.phase = "failed"
    else:
        try:
            # No spinner: the journey meter pulses on the live stage, and a second
            # loading signal beside it read as two different things happening.
            asyncio.run(run_pipeline(note, result, update))
            result.phase = "completed"
            result.active_stage = None
        except Exception as exc:
            result.phase = "failed"
            st.session_state["pipeline_error"] = f"{type(exc).__name__}: {exc}"

if "pipeline_result" in st.session_state:
    result = st.session_state["pipeline_result"]
    with progress.container():
        display_progress(result)
    if st.session_state.get("pipeline_error"):
        st.error(f"Pipeline failed: {st.session_state['pipeline_error']}")
        st.caption("Any outputs below are partial and should not be treated as a completed run.")
    display_results(result)
