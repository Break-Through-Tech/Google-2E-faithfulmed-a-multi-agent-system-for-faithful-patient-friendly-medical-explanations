"""
simplifier_ablation.py

Tests the necessity of the Simplifier agent and the RAG grounding tool.
Conditions:
1. full_pipeline: Extractor -> Simplifier (with RAG) -> Readability -> Verifier -> Refiner
2. no_simplifier: Extractor -> Readability -> Verifier -> Refiner (Simplifier is completely skipped)
3. simplifier_no_rag: Simplifier runs, but RAG grounding is removed (empty context).

Run with: python -m experiments.simplifier_ablation
"""

import os
import getpass
import asyncio
import time
import uuid
import random
import ssl
from pathlib import Path
from typing import AsyncGenerator
from asyncio import to_thread

import pandas as pd
import textstat
import nltk

# Bypass macOS Python SSL certificate verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('cmudict', quiet=True)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import SequentialAgent, LlmAgent, BaseAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

# Import pipeline components
from faithfulmed.adk_pipeline import (
    ExtractAtomsStep,
    SimplifyStep,
    VerifyStep,
    RefineStep,
    readability,
    MODEL
)
from faithfulmed.agents.simplifier import simplify

# --------------------------------------------------
# API configuration
# --------------------------------------------------

if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

RESULTS_FILE = Path("experiments/simplifier_ablation_results.csv")

# --------------------------------------------------
# Ablation Steps
# --------------------------------------------------

class NoSimplifierStep(BaseAgent):
    """Bypasses the Simplifier entirely. Concatenates the raw extracted facts."""
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        atoms = ctx.session.state.get("atoms", [])
        text = " ".join([a.get("text", "") for a in atoms])
        
        # Provide a fully-formed fake draft so the downstream Verifier doesn't crash
        fake_draft = {
            "simplified_text": text,
            "source_metadata": [],
            "atoms_covered": [a.get("position", i+1) for i, a in enumerate(atoms)],
            "fact_ids_covered": [a.get("fact_id", f"F{i+1:03d}") for i, a in enumerate(atoms)],
            "review_warnings": []
        }
        
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={"draft": fake_draft, "text": text}),
        )


class NoRagSimplifyStep(BaseAgent):
    """Runs the Simplifier, but passes an empty grounding dict to simulate no RAG."""
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        atoms = ctx.session.state.get("atoms")
        if not atoms:
            raise ValueError("Nothing to simplify. Run the Extractor first.")

        # INTENTIONAL ABLATION: Bypass build_grounding
        grounding = {}  
        draft = await to_thread(simplify, atoms, grounding)
        
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={
                "grounding": grounding,
                "draft": draft.model_dump(),
                "text": draft.simplified_text,
            }),
        )

# --------------------------------------------------
# Pipeline Builders (Avoids cloning conflicts)
# --------------------------------------------------

def build_readability(prefix: str):
    return LlmAgent(
        name=f"{prefix}Readability",
        model=MODEL,
        description=readability.description,
        instruction=readability.instruction,
        output_key="text"
    )

def build_refine_loop(prefix: str):
    return LoopAgent(
        name=f"{prefix}RefineLoop",
        sub_agents=[VerifyStep(name=f"{prefix}Verifier"), RefineStep(name=f"{prefix}Refiner")],
        max_iterations=2
    )

pipeline_full = SequentialAgent(
    name="FullPipeline",
    sub_agents=[
        ExtractAtomsStep(name="FullExtractor"),
        SimplifyStep(name="FullSimplifier"),
        build_readability("Full"),
        build_refine_loop("Full")
    ]
)

pipeline_no_simplifier = SequentialAgent(
    name="NoSimplifierPipeline",
    sub_agents=[
        ExtractAtomsStep(name="NoSimpExtractor"),
        NoSimplifierStep(name="NoSimpSimplifier"),
        build_readability("NoSimp"),
        build_refine_loop("NoSimp")
    ]
)

pipeline_no_rag = SequentialAgent(
    name="NoRagPipeline",
    sub_agents=[
        ExtractAtomsStep(name="NoRagExtractor"),
        NoRagSimplifyStep(name="NoRagSimplifier"),
        build_readability("NoRag"),
        build_refine_loop("NoRag")
    ]
)

# --------------------------------------------------
# Execution Logic
# --------------------------------------------------

async def run_pipeline(agent, medical_text):
    session_service = InMemorySessionService()
    app_name = "simplifier_ablation"
    user_id = "researcher"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"source_note": medical_text}
    )

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text="Start")])

    start = time.perf_counter()
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        if event.error_code:
            raise RuntimeError(event.error_message or event.error_code)

    latency = time.perf_counter() - start
    session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)

    return {
        "output": session.state.get("text", ""),
        "verdict": session.state.get("verdict", {}),
        "latency_seconds": latency,
    }

async def run_with_retries(agent, medical_text, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        try:
            return await run_pipeline(agent, medical_text)
        except Exception as error:
            error_text = str(error).lower()
            temporary_error = any(
                phrase in error_text for phrase in 
                ["503", "unavailable", "high demand", "429", "resource_exhausted", "rate limit"]
            )
            
            if not temporary_error or attempt == max_attempts:
                raise

            wait_seconds = min(70, 15 * (2 ** (attempt - 1)) + random.uniform(0, 3))
            print(f"Temporary API error. Attempt {attempt}/{max_attempts}. Retrying in {wait_seconds:.1f}s...")
            await asyncio.sleep(wait_seconds)

def save_results(results):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    temp_file = RESULTS_FILE.with_suffix(".csv.tmp")
    df.to_csv(temp_file, index=False)
    temp_file.replace(RESULTS_FILE)

def load_results():
    if not RESULTS_FILE.exists():
        return []
    try:
        return pd.read_csv(RESULTS_FILE).to_dict("records")
    except pd.errors.EmptyDataError:
        return []

def evaluate_output(case_id, name, source, result):
    output = result["output"] or ""
    verdict = result["verdict"]
    
    return {
        "case_id": case_id,
        "pipeline": name,
        "source_note": source,
        "output": output,
        "latency_seconds": round(result["latency_seconds"], 2),
        "reading_grade": round(textstat.flesch_kincaid_grade(output), 2) if output.strip() else None,
        "smog_score": round(textstat.smog_index(output), 2) if output.strip() else None,
        "faithfulness_score": verdict.get("faithfulness_score", None),
        "omission_count": verdict.get("omission_count", None),
        "hallucination": verdict.get("hallucination", None)
    }

# --------------------------------------------------
# Main Loop
# --------------------------------------------------

async def main():
    medical_notes = [
        {
            "id": "case_1_diabetes",
            "text": "Patient is a 45yo male with a 5-year history of type 2 diabetes. HbA1c is 8.2%. He was started on metformin 500mg BID. No complaints of hypoglycemia."
        },
        {
            "id": "case_2_cardiac",
            "text": "The patient presents with acute shortness of breath and bilateral lower extremity edema. EKG shows atrial fibrillation. BNP is elevated at 1200. Started on furosemide."
        },
        {
            "id": "case_3_infection",
            "text": "Patient with a history of COPD presents with increased sputum production and fever. Chest X-ray reveals a right lower lobe infiltrate. Prescribed azithromycin."
        }
    ]

    pipelines = [
        ("full_pipeline", pipeline_full),
        ("no_simplifier", pipeline_no_simplifier),
        ("simplifier_no_rag", pipeline_no_rag)
    ]

    results = load_results()
    completed = { (str(row["case_id"]), str(row["pipeline"])) for row in results if pd.notna(row.get("output")) }
    
    print(f"\nLoaded {len(completed)} previously completed runs.")
    failures = []

    for case in medical_notes:
        print(f"\n========== {case['id']} ==========")
        
        for name, agent in pipelines:
            run_key = (case["id"], name)
            
            if run_key in completed:
                print(f"Skipping completed: {case['id']} / {name}")
                continue
                
            print(f"Running {name}...", flush=True)
            
            try:
                result = await run_with_retries(agent, case["text"])
                
                if not result["output"].strip():
                    raise ValueError("Pipeline returned an empty output.")

                row = evaluate_output(case["id"], name, case["text"], result)

                # Remove any incomplete runs for this key, append new, and save
                results = [r for r in results if (str(r["case_id"]), str(r["pipeline"])) != run_key]
                results.append(row)
                save_results(results)
                completed.add(run_key)

                print(f"Latency: {row['latency_seconds']} | FKGL: {row['reading_grade']} | Hallucination: {row['hallucination']}")
                
            except Exception as error:
                failures.append({"case_id": case["id"], "pipeline": name, "error": str(error)})
                print(f"FAILED: {case['id']} / {name}\nReason: {error}")

    print("\n========== SUMMARY ==========")
    if results:
        df = pd.DataFrame(results)
        summary = df.groupby("pipeline")[
            ["latency_seconds", "reading_grade", "smog_score", "faithfulness_score", "omission_count"]
        ].mean().round(2)
        print(summary)

    if failures:
        print("\n========== FAILED RUNS ==========")
        for failure in failures:
            print(f"{failure['case_id']} / {failure['pipeline']}: {failure['error']}")

if __name__ == "__main__":
    asyncio.run(main())