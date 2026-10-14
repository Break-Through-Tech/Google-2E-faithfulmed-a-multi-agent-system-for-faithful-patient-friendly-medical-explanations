"""
FaithfulMed — minimal Google ADK pipeline skeleton.

Shows the *shape* the SME review asked for: a deterministic DAG (SequentialAgent), not open-ended
recursion, with a single BOUNDED loop (LoopAgent, max_iterations=2) for Refiner <-> Verifier.

This is a scaffold to react to, not a finished pipeline. Each fellow fleshes out their own agent's
instruction, output schema, and tools.

Setup:
    Create the Python 3.12 environment and configure .env as described in
    Getting-Started-for-Fellows.md, then run: adk web faithfulmed
Docs: https://google.github.io/adk-docs

Note: ADK's exact class/kwarg names evolve — check the docs for your installed version. The intent
(SequentialAgent for the DAG, LoopAgent with a hard iteration cap for the refine step) is the point.
"""
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent
from faithfulmed.agents.extractor import Extraction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL = "gemini-3.8-flash"  # start on the free tier; swap per-agent for the leaderboard

# 1) Extractor — pull clinical "atoms" (facts + types) from the source document.
extractor = LlmAgent(
    name="Extractor",
    model=MODEL,
    instruction=(
        "Extract every clinically meaningful fact from the user's source note as one atom per fact. "
        "Return only an object matching the required schema. Use symptom for reported symptoms and "
        "diagnosis only for a diagnosed condition. Preserve negation, uncertainty, history, and "
        "family context in assertion. Copy text and source_quote as the shortest COMPLETE clause "
        "from the note that identifies the subject and establishes the fact. For example, quote "
        "'The patient reports a mild headache,' not only 'mild headache'; quote 'The clinician "
        "diagnosed migraine,' not only 'migraine'; and quote 'The patient denies nausea,' not only "
        "'nausea.' Preserve these exact source words. Do not add implied words, infer, or normalize "
        "clinical facts."
    ),
    output_schema=Extraction,
    output_key="atoms",
)

# 2) Simplifier — rewrite into plain language, grounded by RAG (wire retrieval into instruction/tools).
simplifier = LlmAgent(
    name="Simplifier",
    model=MODEL,
    instruction=(
        "Rewrite the SOURCE for a patient at an 8th-grade reading level using only the extracted "
        "{atoms}. Use plain-language definitions retrieved from the glossary. Add nothing new."
    ),
    output_key="draft",
)

# 3) Verifier — LLM-as-judge: is every claim in {draft} supported by {atoms}? Flag omissions/additions.
verifier = LlmAgent(
    name="Verifier",
    model=MODEL,
    instruction=(
        "Compare {draft} against {atoms}. Return JSON: {faithful: bool, unsupported_claims: [...], "
        "omissions: [...], reading_level_ok: bool}. Be strict: an unsupported clinical claim is a failure."
    ),
    output_key="verdict",
)

# 4) Refiner — fix only what the Verifier flagged. No free rewriting.
refiner = LlmAgent(
    name="Refiner",
    model=MODEL,
    instruction=(
        "Given {draft} and {verdict}, produce a corrected draft that removes unsupported claims and "
        "restores omissions, changing nothing else. If {verdict.faithful} is true, return {draft} unchanged."
    ),
    output_key="draft",  # overwrites the draft so the loop re-verifies the corrected version
)

# Bounded refine loop: Verifier -> Refiner, at most twice. This is the ONLY loop in the system.
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[verifier, refiner],
    max_iterations=2,
)

# 5) Readability — final pass to hit the target grade level and tone.
readability = LlmAgent(
    name="Readability",
    model=MODEL,
    instruction=(
        "Polish the final draft to a Flesch-Kincaid grade level of 8 or below while preserving every "
        "fact. Do not introduce new clinical content."
    ),
    output_key="final",
)

# The deterministic DAG: Extractor -> Simplifier -> (Verifier <-> Refiner, bounded) -> Readability.
faithfulmed = SequentialAgent(
    name="FaithfulMed",
    sub_agents=[extractor, simplifier, refine_loop, readability],
)

if __name__ == "__main__":
    print("FaithfulMed ADK pipeline assembled:")
    print("  Extractor -> Simplifier -> [Verifier <-> Refiner x<=2] -> Readability")
    print("Run it with an ADK Runner + session (see the ADK quickstart) once GEMINI_API_KEY is set.")
