"""
FaithfulMed ADK pipeline

Follows the agreed flowchart: a deterministic DAG (SequentialAgent), not open-ended recursion.

    Extractor -> Simplifier -> Readability -> [Verifier <-> Refiner] -> final text

Readability runs before the Verifier, so faithfulness is judged on the patient-facing text a person
would actually read, not on the draft as first written. Verifier <-> Refiner is a bounded loop
(LoopAgent, max_iterations=2).

The flowchart also draws a dotted "bounded loop?" arrow between Simplifier and Readability. That is
still an open decision, so it is not built here.

State flow: source_note -> atoms -> draft (structured record) + text (working prose) -> verdict ->
text. `text` is what the reader ends up with; `draft` keeps the Simplifier's citations and coverage.

This is a scaffold: Readability, the Verifier, and the Refiner are still single-prompt agents.
"""
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext, LlmAgent, LoopAgent, SequentialAgent
from google.adk.events import Event

from faithfulmed.agents.extractor import extract_atoms
from faithfulmed.agents.simplifier import build_grounding, simplify

MODEL = "gemini-3.5-flash-lite"


def _note_from_message(ctx: InvocationContext) -> str:
    content = ctx.user_content
    if content is None or not content.parts:
        return ""
    return "".join(part.text for part in content.parts if part.text)


class ExtractAtomsStep(BaseAgent):
    """ The step reads the note, writes its atoms to session state, and emits
    one event, which is what the Simplifier reads as {atoms}.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        note = ctx.session.state.get("source_note") or _note_from_message(ctx)
        if not note.strip():
            raise ValueError(
                "Nothing to extract. Send the note as the message, or set source_note in state."
            )

        extraction = extract_atoms(note)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            state={
                "source_note": note,
                "atoms": [atom.model_dump() for atom in extraction.atoms],
            },
        )


class SimplifyStep(BaseAgent):
    """
    The facts are already in state. Grounding is fetched once in Python and stored beside 
    the draft, so the Verifier can check the draft's citations against the passages the model 
    was actually given.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        atoms = ctx.session.state.get("atoms")
        if not atoms:
            raise ValueError("Nothing to simplify. Run the Extractor first.")

        grounding = build_grounding(atoms)
        draft = simplify(atoms, grounding)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            state={
                "grounding": grounding,
                "draft": draft.model_dump(),
                "text": draft.simplified_text,
            },
        )


# 1) Extractor - pull clinical "atoms" (facts + types) from the source document.
extractor = ExtractAtomsStep(name="Extractor")

# 2) Simplifier - rewrite the facts into plain language, grounded in MedlinePlus passages.
simplifier = SimplifyStep(name="Simplifier")

# 3) Readability - grade-level pass before faithfulness is judged, per the agreed flowchart.
readability = LlmAgent(
    name="Readability",
    model=MODEL,
    instruction=(
        "Rewrite {text} to reach a Flesch-Kincaid grade level of 8 or below: shorter sentences, everyday "
        "words, active voice. Keep every fact, number, unit, negation, and uncertainty exactly as written. "
        "Add no clinical content and drop none. Return only the rewritten text."
    ),
    output_key="text",
)

# 4) Verifier - LLM-as-judge over the readability-improved text.
verifier = LlmAgent(
    name="Verifier",
    model=MODEL,
    instruction=(
        "Compare {text} against {atoms}. Return JSON: {faithful: bool, unsupported_claims: [...], "
        "omissions: [...], invented_numbers: [...]}. Be strict: an unsupported clinical claim is a failure."
    ),
    output_key="verdict",
)

# 5) Refiner - fix only what the Verifier flagged. No free rewriting.
refiner = LlmAgent(
    name="Refiner",
    model=MODEL,
    instruction=(
        "Given {text} and {verdict}, remove every claim the verdict lists as unsupported and restore "
        "every omission it lists, changing nothing else. Keep the reading level at or below grade 8. "
        "If the verdict reports no problems, return {text} unchanged."
    ),
    output_key="text",  # overwrites the working text so the loop re-verifies the corrected version
)

# Bounded refine loop: Verifier -> Refiner, at most twice.
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[verifier, refiner],
    max_iterations=2,
)

# The agreed DAG: Extractor -> Simplifier -> Readability -> (Verifier <-> Refiner, bounded).
faithfulmed = SequentialAgent(
    name="FaithfulMed",
    sub_agents=[extractor, simplifier, readability, refine_loop],
)

if __name__ == "__main__":
    print("FaithfulMed ADK pipeline assembled:")
    print("  Extractor -> Simplifier -> Readability -> [Verifier <-> Refiner x<=2]")
    print("Final patient text is state['text']; state['draft'] keeps the Simplifier's record.")
    print("Run it with an ADK Runner + session (see the ADK quickstart) once GEMINI_API_KEY is set.")
