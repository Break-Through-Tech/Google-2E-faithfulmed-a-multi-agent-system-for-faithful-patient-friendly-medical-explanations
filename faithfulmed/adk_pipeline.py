"""
FaithfulMed ADK pipeline

Follows the agreed flowchart: a deterministic DAG (SequentialAgent), not open-ended recursion.

    Extractor -> Simplifier -> Readability -> [Verifier <-> Refiner] -> final text

Readability runs before the Verifier, so faithfulness is judged on the patient-facing text a person
would actually read, not on the draft as first written. Verifier <-> Refiner is a bounded loop
(LoopAgent, max_iterations=2).

The bounded loop is not built yet between simplifier agent and readability agent. (NOTE: not needed)

State flow: source_note -> atoms -> draft (structured record) + text (working prose) -> verdict ->
text. `text` is what the reader ends up with; `draft` keeps the Simplifier's citations and coverage.

"""
from asyncio import to_thread
from typing import AsyncGenerator, List, Dict
from pydantic import BaseModel, Field
import textstat

from google.genai import types
from google.adk.agents import BaseAgent, InvocationContext, LlmAgent, LoopAgent, SequentialAgent
from google.adk.events import Event, EventActions

from faithfulmed.agents.extractor import extract_atoms
from faithfulmed.agents.simplifier import build_grounding, simplify
from faithfulmed.agents.verifier import verify

# Session state is a shared dictionary: each step reads earlier results and publishes
# its own results. Agent definitions at the bottom set the order, not these class names.
MODEL = "gemini-3.5-flash-lite"



# output for Readability agent -> pydantic model for structured output -> Flesch-Kincaid score and SMOG score (before and after rewriting), simplified text
# class ReadabilityEvaluation(BaseModel):
#    pre_flesch_kincaid_grade: float = Field(description="Calculated Flesch-Kincaid grade level score before revision")
#    post_flesch_kincaid_grade: float = Field(description="Calculated Flesch-Kincaid grade level score after revision")
#    pre_smog: float = Field(description="SMOG readability score before revision")
#    post_smog: float = Field(description="SMOG readability score after revision")
#    revised_text: str = Field(description="Text rewritten at or below target grade level")

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

        # The model call waits for a network reply. to_thread does that waiting on
        # another thread so it does not block ADK's asynchronous event loop.
        extraction = await to_thread(extract_atoms, note)
        # yield publishes an event. state_delta contains only the state entries
        # this step updates, so the next agent can read the accepted atoms.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={
                "source_note": note,
                "atoms": [atom.model_dump() for atom in extraction.atoms],
            }),
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

        # Fetch reference passages before writing. The Simplifier receives these
        # same passages, and the state keeps them available for later inspection.
        grounding = await to_thread(build_grounding, atoms)
        draft = await to_thread(simplify, atoms, grounding)
        # Keep two versions: draft is the original structured record, while text
        # is the working prose that Readability and Refiner are allowed to replace.
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={
                "grounding": grounding,
                "draft": draft.model_dump(),
                "text": draft.simplified_text,
            }),
        )

class VerifyStep(BaseAgent):
    """
    Checks the current patient text (after Readability or the Refiner) against the note, the atoms,
    and the passages the Simplifier used. Writes the report as {verdict} for the Refiner, and ends
    the loop early when nothing needs revising.
    """
 
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        atoms = state.get("atoms")
        if not atoms:
            raise ValueError("Nothing to verify against. Run the Extractor first.")
        if not state.get("draft"):
            raise ValueError("Nothing to verify. Run the Simplifier first.")
 
        # Build a temporary dictionary with current prose and original draft metadata.
        # ** copies the draft fields, then simplified_text replaces that one field.
        # This does not change state["draft"] or certify its citations for later edits.
        draft = {**state["draft"], "simplified_text": state.get("text") or state["draft"]["simplified_text"]}
        report = await to_thread(verify, state.get("source_note", ""), atoms, draft)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(
                state_delta={"verdict": report.model_dump()},
                escalate=not report.needs_revision,  # stop the loop, skipping the Refiner
            ),
        )

# 1) Extractor - pull clinical "atoms" (facts + types) from the source document.
extractor = ExtractAtomsStep(name="Extractor")

# 2) Simplifier - rewrite the facts into plain language, grounded in MedlinePlus passages.
simplifier = SimplifyStep(name="Simplifier")

# 3) Readability - grade-level pass before faithfulness is judged, per the agreed flowchart.
readability_agent = LlmAgent(
    name="readability",
    model=MODEL,
    description="Rewrites clinical text to make it readable for patients at or below an 8th-grade reading level",
    instruction="""
        You are an expert at evaluating the readability of clinical text and rewriting the text to be more readable.
        Evaluate the draft's readability using the Flesch-Kincaid grade level and SMOG metrics.
        Then, if the reading level required for the text is above the 8th-grade level, rewrite the text
        to make it readable at the 8th-grade level or below while maintaining the text's original meaning.
    
        Follow these guidelines:
        1) Do not insert new information, change the meaning of existing information, or remove significant pieces of information
        from the input text.
        2) You can replace complex medical terms with plain language equivalents or keep the medical term followed by
        a plain language definition.
        3) Output only the simplified text.

        Draft: {draft}
    """,
    output_key="text"
)

# 4) Verifier - LLM-as-judge over the readability-improved text.
verifier = VerifyStep(name="Verifier")

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

# Each iteration runs Verifier, then Refiner unless Verifier stops the loop early.
# The cap prevents endless repairs. At the cap, the last Refiner output may leave
# without another Verifier check. Loop completion alone does not mean approval.
# Nothing in this loop measures the final reading grade after a Refiner rewrite.
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[verifier, refiner],
    max_iterations=2,
)

faithfulmed = SequentialAgent(
    name="FaithfulMed",
    sub_agents=[extractor, simplifier, readability, refine_loop],
)

if __name__ == "__main__":
    print("FaithfulMed ADK pipeline assembled:")
    print("  Extractor -> Simplifier -> Readability -> [Verifier <-> Refiner x<=2]")
    print("Final patient text is state['text']; state['draft'] keeps the Simplifier's record.")
    print("Run it with an ADK Runner + session (see the ADK quickstart) once GEMINI_API_KEY is set.")
