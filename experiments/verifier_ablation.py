
import os
import getpass
import asyncio
import time
import uuid
import random
from pathlib import Path
from typing import AsyncGenerator
from asyncio import to_thread

import pandas as pd
import textstat

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import SequentialAgent, LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types


# --------------------------------------------------
# API configuration
# --------------------------------------------------

os.environ["GOOGLE_API_KEY"] = getpass.getpass(
    "Enter your Gemini API key: "
)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"


# Import existing team code (no modifications)
from faithfulmed.agents.refiner import refine

from faithfulmed.adk_pipeline import (
    faithfulmed,
    ExtractAtomsStep,
    SimplifyStep,
    readability_agent,
    MODEL,
)


# --------------------------------------------------
# Experiment output
# --------------------------------------------------

RESULTS_FILE = Path(
    "experiments/verifier_ablation_results.csv"
)


# --------------------------------------------------
# Refiner wrapper for Verifier-only ablation
# --------------------------------------------------

class RefinerWithoutVerifier(BaseAgent):
    """
    Runs the existing Refiner without Verifier feedback.

    An empty verdict means there are no flagged
    problems, so the Refiner normally leaves
    the patient explanation unchanged.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:

        state = ctx.session.state

        empty_verdict = {
            "claims": [],
            "omissions": [],
        }

        result = await to_thread(
            refine,
            state.get("source_note", ""),
            state["text"],
            empty_verdict,
            state.get("atoms") or [],
        )

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={
                    "text": result.refined_text,
                    "refiner_changes": [
                        c.model_dump()
                        for c in result.changes
                    ],
                    "refiner_unresolved": result.unresolved,
                }
            ),
        )


# --------------------------------------------------
# Experiment 1: Remove Verifier AND Refiner
# --------------------------------------------------

readability_ablation = LlmAgent(
    name="ReadabilityAblation",
    model=MODEL,
    description=readability_agent.description,
    instruction=readability_agent.instruction,
    output_key="text",
)

faithfulmed_no_verification_loop = SequentialAgent(
    name="FaithfulMedNoVerificationLoop",
    sub_agents=[
        ExtractAtomsStep(name="ExtractorAblation"),
        SimplifyStep(name="SimplifierAblation"),
        readability_ablation,
    ],
)


# --------------------------------------------------
# Experiment 2: Remove ONLY the Verifier
# --------------------------------------------------

faithfulmed_no_verifier = SequentialAgent(
    name="FaithfulMedNoVerifier",
    sub_agents=[
        ExtractAtomsStep(
            name="ExtractorWithoutVerifier"
        ),
        SimplifyStep(
            name="SimplifierWithoutVerifier"
        ),
        LlmAgent(
            name="ReadabilityWithoutVerifier",
            model=MODEL,
            description=readability_agent.description,
            instruction=readability_agent.instruction,
            output_key="text",
        ),
        RefinerWithoutVerifier(
            name="RefinerWithoutVerifier"
        ),
    ],
)


# --------------------------------------------------
# Run one pipeline
# --------------------------------------------------

async def run_pipeline(agent, medical_text):
    session_service = InMemorySessionService()

    app_name = "faithfulmed_ablation"
    user_id = "researcher"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service,
    )

    message = types.Content(
        role="user",
        parts=[
            types.Part(text=medical_text)
        ],
    )

    start = time.perf_counter()

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.error_code:
            raise RuntimeError(
                event.error_message
                or event.error_code
            )

    latency = time.perf_counter() - start

    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    return {
        "output": session.state.get("text", ""),
        "latency_seconds": latency,
    }


# --------------------------------------------------
# Retry temporary Gemini API errors
# --------------------------------------------------

async def run_with_retries(
    agent,
    medical_text,
    max_attempts=3,
):
    for attempt in range(1, max_attempts + 1):

        try:
            return await run_pipeline(
                agent,
                medical_text,
            )

        except Exception as error:
            error_text = str(error).lower()

            temporary_error = any(
                phrase in error_text
                for phrase in [
                    "503",
                    "unavailable",
                    "high demand",
                    "429",
                    "resource_exhausted",
                    "rate limit",
                ]
            )

            # Do not retry unrelated programming errors.
            if not temporary_error:
                raise

            if attempt == max_attempts:
                raise

            wait_seconds = min(
                60,
                5 * (2 ** (attempt - 1))
                + random.uniform(0, 2),
            )

            print(
                f"Temporary Gemini error. "
                f"Attempt {attempt}/{max_attempts}. "
                f"Retrying in {wait_seconds:.1f} seconds...",
                flush=True,
            )

            await asyncio.sleep(wait_seconds)


# --------------------------------------------------
# Save results safely
# --------------------------------------------------

def save_results(results):
    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    # Write to a temporary file first.
    temporary_file = RESULTS_FILE.with_suffix(
        ".csv.tmp"
    )

    df.to_csv(
        temporary_file,
        index=False,
    )

    temporary_file.replace(RESULTS_FILE)


# --------------------------------------------------
# Load existing results to resume experiment
# --------------------------------------------------

def load_results():
    if not RESULTS_FILE.exists():
        return []

    try:
        df = pd.read_csv(RESULTS_FILE)

        if not {
            "case_id",
            "pipeline",
            "output",
        }.issubset(df.columns):
            raise ValueError(
                "Existing CSV has unexpected columns."
            )

        return df.to_dict("records")

    except pd.errors.EmptyDataError:
        return []


# --------------------------------------------------
# Evaluate one output
# --------------------------------------------------

def evaluate_output(case_id, name, source, result):
    output = result["output"] or ""

    refusal_phrases = [
        "i cannot",
        "i can't",
        "i'm unable",
        "i am unable",
    ]

    return {
        "case_id": case_id,
        "pipeline": name,
        "source_note": source,
        "output": output,
        "latency_seconds": round(
            result["latency_seconds"], 2
        ),
        "reading_grade": (
            round(
                textstat.flesch_kincaid_grade(output),
                2,
            )
            if output.strip()
            else None
        ),
        "output_word_count": len(output.split()),
        "refusal": any(
            phrase in output.lower()
            for phrase in refusal_phrases
        ),
    }


# --------------------------------------------------
# Compare all three configurations
# --------------------------------------------------

async def main():

    medical_notes = [
        {
            "id": "case_1",
            "text": (
                "The patient has type 2 diabetes. "
                "Blood glucose was 180 mg/dL. "
                "Metformin 500 mg was prescribed twice daily."
            ),
        },
        {
            "id": "case_2",
            "text": (
                "The patient has hypertension. "
                "Blood pressure was 150/95 mmHg. "
                "Lisinopril 10 mg was prescribed once daily."
            ),
        },
        {
            "id": "case_3",
            "text": (
                "The patient has iron deficiency anemia. "
                "Hemoglobin was 9.5 g/dL. "
                "Ferrous sulfate 325 mg was prescribed once daily."
            ),
        },
        {
            "id": "case_4",
            "text": (
                "The patient has asthma. "
                "An albuterol inhaler was prescribed: "
                "two puffs every four to six hours "
                "as needed for wheezing."
            ),
        },
        {
            "id": "case_5",
            "text": (
                "The patient has hypothyroidism. "
                "TSH was 8.2 mIU/L. "
                "Levothyroxine 50 mcg was prescribed once daily."
            ),
        },
    ]

    pipelines = [
        ("full", faithfulmed),
        ("no_verifier", faithfulmed_no_verifier),
        (
            "no_verification_loop",
            faithfulmed_no_verification_loop,
        ),
    ]

    # Resume from previous successful runs.
    results = load_results()

    completed = {
        (
            str(row["case_id"]),
            str(row["pipeline"]),
        )
        for row in results
        if pd.notna(row.get("output"))
        and str(row.get("output", "")).strip()
    }

    print(
        f"\nLoaded {len(completed)} "
        "previously completed runs."
    )

    failures = []

    for case in medical_notes:

        print(
            f"\n========== {case['id']} =========="
        )

        for name, agent in pipelines:

            run_key = (
                case["id"],
                name,
            )

            if run_key in completed:
                print(
                    f"Skipping completed: "
                    f"{case['id']} / {name}"
                )
                continue

            print(
                f"Running {name}...",
                flush=True,
            )

            try:
                result = await run_with_retries(
                    agent,
                    case["text"],
                )

                row = evaluate_output(
                    case["id"],
                    name,
                    case["text"],
                    result,
                )

                if not row["output"].strip():
                    raise ValueError(
                        "Pipeline returned an empty output."
                    )

                # Replace any previous incomplete
                # record for this case and pipeline.
                results = [
                    existing
                    for existing in results
                    if (
                        str(existing["case_id"]),
                        str(existing["pipeline"]),
                    ) != run_key
                ]

                results.append(row)

                # Save immediately after success.
                save_results(results)

                completed.add(run_key)

                print(
                    "Reading grade:",
                    row["reading_grade"],
                )
                print(
                    "Latency:",
                    row["latency_seconds"],
                )
                print(
                    "Word count:",
                    row["output_word_count"],
                )
                print("Result saved.")

            except Exception as error:
                import traceback
                traceback.print_exception(type(error), error, error.__traceback__)

                failures.append(
                    {
                        "case_id": case["id"],
                        "pipeline": name,
                        "error": str(error),
                    }
                )

                print(
                    f"FAILED: {case['id']} / {name}"
                )
                print(
                    f"Reason: {error}"
                )
                print(
                    "Continuing to the next run..."
                )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print(
        "\n========== SUMMARY =========="
    )

    if not results:
        print(
            "No successful runs yet. "
            "Try again when Gemini is available."
        )
        return

    df = pd.DataFrame(results)

    summary = df.groupby("pipeline")[
        [
            "latency_seconds",
            "reading_grade",
            "output_word_count",
        ]
    ].mean().round(2)

    print(summary)

    print(
        "\n========== COMPLETION =========="
    )

    expected_runs = (
        len(medical_notes) * len(pipelines)
    )

    valid_keys = {
        (case["id"], name)
        for case in medical_notes
        for name, _ in pipelines
    }

    successful_runs = len(
        completed.intersection(valid_keys)
    )

    print(
        f"Completed: {successful_runs}"
        f"/{expected_runs}"
    )

    print(
        f"Missing: "
        f"{expected_runs - successful_runs}"
    )

    if failures:
        print(
            "\n========== FAILED RUNS =========="
        )

        for failure in failures:
            print(
                f"{failure['case_id']} / "
                f"{failure['pipeline']}: "
                f"{failure['error']}"
            )

        print(
            "\nRun the experiment again later "
            "to retry unfinished cases."
        )

    print(
        f"\nResults saved to: {RESULTS_FILE}"
    )

# --------------------------------------------------
# Offline ablation metrics
# Uses existing CSV; does not call Gemini
# --------------------------------------------------

import re


# Source facts to check for each synthetic case.
# These patterns are specific to our five test notes.
EXPECTED_FACTS = {
    "case_1": {
        "diagnosis": r"\btype 2 diabetes\b",
        "glucose_value": r"\b180\s*mg/dl\b",
        "medication": r"\bmetformin\b",
        "dose": r"\b500\s*mg\b",
        "frequency": r"\b(twice|two times|2 times)\b",
    },
    "case_2": {
        "diagnosis": r"\b(hypertension|high blood pressure)\b",
        "blood_pressure": r"\b150\s*/\s*95\b",
        "medication": r"\blisinopril\b",
        "dose": r"\b10\s*mg\b",
        "frequency": r"\b(once|one time|1 time)\b",
    },
    "case_3": {
        "diagnosis": r"\b(iron deficiency anemia|low iron)\b",
        "hemoglobin": r"\b9\.5\s*g/dl\b",
        "medication": r"\bferrous sulfate\b",
        "dose": r"\b325\s*mg\b",
        "frequency": r"\b(once|one time|1 time)\b",
    },
    "case_4": {
        "diagnosis": r"\basthma\b",
        "medication": r"\balbuterol\b",
        "dose": r"\b(two|2)\s*puffs?\b",
        "interval": r"\b(four|4)\s*to\s*(six|6)\s*hours?\b",
        "as_needed": r"\b(as needed|when needed)\b",
    },
    "case_5": {
        "diagnosis": r"\b(hypothyroidism|underactive thyroid)\b",
        "tsh_value": r"\b8\.2\s*miu/l\b",
        "medication": r"\blevothyroxine\b",
        "dose": r"\b50\s*mcg\b",
        "frequency": r"\b(once|one time|1 time)\b",
    },
}


# Fixed jargon vocabulary for this experiment.
# Expand only if you document the change.
MEDICAL_TERMS = {
    "diabetes",
    "glucose",
    "metformin",
    "hypertension",
    "lisinopril",
    "anemia",
    "hemoglobin",
    "ferrous",
    "sulfate",
    "asthma",
    "albuterol",
    "hypothyroidism",
    "tsh",
    "levothyroxine",
}


def calculate_extra_metrics(row):
    case_id = str(row["case_id"])
    output = str(row["output"])
    output_lower = output.lower()

    # Faithfulness proxy: source facts retained.
    facts = EXPECTED_FACTS.get(case_id, {})

    if facts:
        matched = sum(
            bool(re.search(pattern, output_lower))
            for pattern in facts.values()
        )

        faithfulness = round(
            100 * matched / len(facts), 2
        )
    else:
        faithfulness = None

    # Jargon density: matched medical tokens / all tokens.
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        output_lower,
    )

    jargon_count = sum(
        word in MEDICAL_TERMS
        for word in words
    )

    jargon_density = (
        round(100 * jargon_count / len(words), 2)
        if words else None
    )

    return pd.Series({
        "fact_retention_percent": faithfulness,
        "medical_jargon_density_percent": jargon_density,
        "medical_jargon_count": jargon_count,
    })


def score_saved_results():
    if not RESULTS_FILE.exists():
        print("No saved results found.")
        return

    df = pd.read_csv(RESULTS_FILE)

    # Score only successful, nonempty outputs.
    df = df.dropna(subset=["output"]).copy()

    df = df[
        df["output"].astype(str).str.strip() != ""
    ].copy()

    if df.empty:
        print("No completed outputs to score.")
        return

    extra = df.apply(
        calculate_extra_metrics,
        axis=1,
    )

    for column in extra.columns:
        df[column] = extra[column]

    # Do not invent a hallucination rate.
    # Unsupported claims require an explicit
    # claim-by-claim review.
    df["hallucination_rate_percent"] = pd.NA

    output_file = Path(
        "experiments/verifier_ablation_scored.csv"
    )

    df.to_csv(output_file, index=False)

    print("\n========== ADDITIONAL METRICS ==========")

    print(
        df.groupby("pipeline")[
            [
                "fact_retention_percent",
                "medical_jargon_density_percent",
                "medical_jargon_count",
            ]
        ].mean().round(2)
    )

    print(
        "\nHallucination rate: pending claim review."
    )

    print(
        f"\nScored results saved to: {output_file}"
    )


if __name__ == "__main__":
    import sys

    if "--score-only" in sys.argv:
        score_saved_results()
    else:
        asyncio.run(main())


