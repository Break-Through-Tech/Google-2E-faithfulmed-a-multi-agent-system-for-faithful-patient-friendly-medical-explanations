"""Verifier agent: source + extracted facts + Simplifier draft -> faithfulness report.

Scores the draft with the advisor's four metrics (faithfulness, omission, addition,
hallucination) and tells the Refiner whether the draft needs revision.
"""
import json
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from faithfulmed.agents.simplifier import SimplifiedOutput, format_atoms

MODEL = "gemini-3.5-flash-lite"


# A claim is one statement in the explanation. An omission is an extracted fact
# that the explanation leaves out or states incorrectly. The model judges these,
# then Python calculates the report numbers from those judgments.
Severity = Literal["minor", "major"]


class ClaimCheck(BaseModel):
    claim: str = Field(description="One specific statement from the patient explanation")
    label: Literal["supported", "background", "unsupported", "distorted"] = Field(
        description="supported: the source states it; background: not in the source but backed by the "
                    "reference material; unsupported: backed by neither; distorted: changes a fact from the source")
    evidence: str = Field(description="Exact quote from the source or reference passage, or '' if none")
    severity: Severity = Field(description="The clinical importance of claim being incorrect.")


class Omission(BaseModel):
    fact_number: int = Field(description="Number of the fact in FACTS that the explanation leaves out or states incorrectly")
    fact: str = Field(description="The text of that fact")
    severity: Severity = Field(description="The clinical importance of omission being made.")


class VerifierJudgment(BaseModel):
    """What the model returns."""

    claims: List[ClaimCheck] = Field(description="Every claim in the explanation, including supported ones")
    omissions: List[Omission] = Field(description="Every fact the explanation leaves out or states incorrectly")


class VerifierReport(VerifierJudgment):
    """What we hand back: the judgment plus metrics computed in code."""

    faithfulness_score: float = Field(description="Fraction of claims that are supported or background")
    omission_count: int = Field(description="Facts missing")
    total_facts: int = Field(description="Number of Extractor facts")
    addition_count: int = Field(description="Claims labelled background or unsupported")
    hallucination: bool = Field(description="At least one major unsupported or distorted claim.")
    needs_revision: bool = Field(description="Hallucination, or any major omission. Sent to refiner agent.")
    false_coverage: List[int] = Field(description="Fact numbers the Simplifier reported as covered but are missing")


INSTRUCTION = """
You are a medical verification agent.

Your task is to compare a patient-friendly medical
explanation against the original medical source.

Check for:
1. Incorrect or unsupported medical claims.
2. Contradictions with the original source.
3. Missing important medical information.
4. Incorrect medication names, doses, or instructions.
5. Incorrect numbers, test results, or follow-up dates.
6. Changes in medical risk, severity, or uncertainty.

Break the patient-friendly explanation into individual
factual medical claims.

Check every claim separately against the original source.

Include all claims in your evaluation, even those that
are correct. Label each claim as supported, background,
unsupported, or distorted.

A claim that changes a fact from the source (a number,
unit, timing, or a hedge turned into certainty) is
distorted. General information not in the source is
background only if a REFERENCE MATERIAL passage backs it.

Each numbered fact is tagged with its type and assertion
(for example present, absent, possible, family). Go through
the FACTS one by one and list every fact the explanation
leaves out or states incorrectly as an omission, using its
number as fact_number.

Pay special attention to negation errors.

Compare statements containing words such as
"no", "not", "never", "without", and "negative"
against the original source.

If the source reports a symptom or finding,
but the explanation says it is absent,
mark the claim as distorted.

A family member's condition stated as the patient's,
or a possible condition stated as confirmed, is distorted.

Check each part of a sentence separately when
it contains multiple medical facts.

For every unsupported or distorted claim,
provide the relevant evidence from the original source.

If the source does not mention the claim, clearly state:
"Not mentioned in the original source."

Important rules:
- Use only the original medical source and the reference material to verify facts.
- Do not invent medical information.
- Do not treat suspected conditions as confirmed diagnoses.
- Prioritize errors that could affect patient safety.
- Check the entire explanation, not just a few claims.
- Do not overlook a contradiction just because most
  of the explanation is accurate.
- Return only the structured output that matches the schema.
"""


def as_draft(draft: Union[SimplifiedOutput, dict]) -> SimplifiedOutput:
    """Accept the Simplifier's output either as the model or as a dict from load_draft."""
    return draft if isinstance(draft, SimplifiedOutput) else SimplifiedOutput.model_validate(draft)


def format_references(draft: SimplifiedOutput) -> str:
    """Join the MedlinePlus passages the draft used into one reference section."""
    if not draft.source_metadata:
        return "No reference passages were used."
    return "\n\n---\n\n".join(f"{p.title}\n{p.text}" for p in draft.source_metadata)


def build_report(judgment: VerifierJudgment, atom_count: int,
                 atoms_covered: Optional[List[int]] = None) -> VerifierReport:
    """Computes the advisor's metrics in code from the model's judgment."""
    # The categories overlap on purpose: reference-backed background counts as
    # faithful, but also as an addition because it is not stated in the patient note.
    supported = [c for c in judgment.claims if c.label in ("supported", "background")]
    additions = [c for c in judgment.claims if c.label in ("background", "unsupported")]
    # Only major unsupported or distorted claims set the hallucination flag.
    # A major omission can also require revision even when no claim was invented.
    hallucinated = [c for c in judgment.claims if c.label in ("unsupported", "distorted") and c.severity == "major"]
    serious_omissions = [o for o in judgment.omissions if o.severity == "major"]

    n = len(judgment.claims)
    hallucination = len(hallucinated) > 0
    # These numbers are one-based fact positions, not IDs such as F007.
    # Set intersection below finds facts claimed as covered but judged missing.
    omitted = {o.fact_number for o in judgment.omissions}

    # Scores summarize the model's labels. They are not human-verified accuracy.
    # With no claims the formula returns 1.0, but verify rejects an empty judgment.
    return VerifierReport(
        **judgment.model_dump(),
        faithfulness_score=len(supported) / n if n else 1.0,
        omission_count=len(judgment.omissions),
        total_facts=atom_count,
        addition_count=len(additions),
        hallucination=hallucination,
        needs_revision=hallucination or len(serious_omissions) > 0,
        false_coverage=sorted(omitted & set(atoms_covered or [])),
    )


def verify(
    source: str,
    atoms: List[dict],
    draft: Union[SimplifiedOutput, dict],
    model: str = MODEL,
) -> VerifierReport:
    """Check the Simplifier's draft against the source, the numbered facts and its references."""
    if not atoms:
        raise ValueError("No facts to verify against. Run the Extractor first, or pass the atoms in.")
    draft = as_draft(draft)

    from google import genai
    from google.genai import types

    prompt = (
        f"SOURCE:\n{source}\n\n"
        f"FACTS:\n{format_atoms(atoms)}\n\n"
        f"REFERENCE MATERIAL:\n{format_references(draft)}\n\n"
        f"EXPLANATION:\n{draft.simplified_text}"
    )

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            response_mime_type="application/json",
            response_schema=VerifierJudgment,
            temperature=0,
        ),
    )

    if response.parsed is None:
        raise ValueError("Verifier did not return a structured judgment.")
    judgment = response.parsed

    # A correctly shaped JSON reply can still refer to facts that do not exist.
    # Reject those references before calculating metrics or asking for a repair.
    issues = []
    if not judgment.claims:
        issues.append("no claims listed")
    if check_fact_numbers(judgment, len(atoms)):
        issues.append("omissions reference unknown fact numbers")
    if issues:
        raise ValueError("Verifier validation failed: " + ", ".join(issues))

    return build_report(judgment, len(atoms), draft.atoms_covered)


# Is the judgment usable?
def check_fact_numbers(judgment: VerifierJudgment, atom_count: int) -> List[int]:
    """Return omission fact numbers that are not in the fact list."""
    return [o.fact_number for o in judgment.omissions if o.fact_number < 1 or o.fact_number > atom_count]


def print_report(report: VerifierReport) -> None:
    print("Verifier Report")
    print(f"Faithfulness score: {report.faithfulness_score:.2f}")
    print(f"Omissions: {report.omission_count} of {report.total_facts} | Additions: {report.addition_count}")
    print(f"Hallucination: {report.hallucination} | Needs revision: {report.needs_revision}")
    if report.false_coverage:
        print(f"Simplifier reported as covered but missing: {report.false_coverage}")
    for c in report.claims:
        if c.label != "supported":
            print(f"  [{c.label}, {c.severity}] {c.claim}")
    for o in report.omissions:
        print(f"  [omitted fact {o.fact_number}, {o.severity}] {o.fact}")


# Hand off the file
def save_report(doc_id: str, report: VerifierReport, out_dir: Path) -> Path:
    """Write one document's report to <out_dir>/<doc_id>.verdict.json and return the path."""
    if not doc_id or Path(doc_id).name != doc_id or doc_id in {".", ".."}:
        raise ValueError("doc_id must be a non-empty filename stem")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{doc_id}.verdict.json"
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_report(path: Path) -> dict:
    """Read a saved report back into a dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Report file must contain a JSON object")
    return payload


# Optional ADK integration, separate from the active VerifyStep in adk_pipeline.py.
# Callers of this builder must supply the state keys below. In particular, it uses
# "source", while the active pipeline uses "source_note" and a separate working "text".
SOURCE_KEY = "source"                  # set when the session is created
ATOMS_KEY = "atoms"                    # Extractor output_key
DRAFT_KEY = "draft"                    # explanation and references for this optional builder
OUTPUT_KEY = "verdict"                 # Refiner reads {verdict}
REPORT_KEY = "verifier_report"


def build_adk_agents():
    """Return (verifier_agent, verifier_stop_check) for a LoopAgent with the Refiner:
    LoopAgent(sub_agents=[verifier_agent, verifier_stop_check, refiner], max_iterations=2)
    """
    from google.adk.agents import BaseAgent, LlmAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event, EventActions
    from google.genai import types

    pipeline_instruction = INSTRUCTION + f"""
SOURCE:
{{{SOURCE_KEY}}}

FACTS:
{{{ATOMS_KEY}}}

EXPLANATION AND REFERENCE MATERIAL (if this is JSON, check simplified_text against source_metadata):
{{{DRAFT_KEY}}}
"""

    def save_adk_report(callback_context):
        """Runs after the agent: turns the raw judgment in state into the full report."""
        # A callback runs after the model reply. It saves the Python-computed report
        # separately from the raw judgment, then leaves a flag for the stop-check agent.
        state = callback_context.state
        judgment = VerifierJudgment.model_validate(state[OUTPUT_KEY])
        atoms = state.get(ATOMS_KEY)
        draft = state.get(DRAFT_KEY)
        atoms_covered = draft.get("atoms_covered", []) if isinstance(draft, dict) else []
        report = build_report(judgment, len(atoms) if isinstance(atoms, list) else 0, atoms_covered)
        state[REPORT_KEY] = report.model_dump()
        state["needs_revision"] = report.needs_revision   # the stop check reads this
        return None

    class StopIfNoRevision(BaseAgent):
        """Ends the loop when the Verifier found nothing to fix."""
        async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
            # Only an explicit False means no revision is needed. An absent flag
            # does not count as success. Escalation tells the surrounding loop to stop.
            done = ctx.session.state.get("needs_revision") is False
            yield Event(author=self.name, actions=EventActions(escalate=done))

    verifier_agent = LlmAgent(
        name="Verifier",
        model=MODEL,
        description="Checks patient-friendly medical explanations for accuracy and completeness.",
        instruction=pipeline_instruction,
        output_schema=VerifierJudgment,
        output_key=OUTPUT_KEY,
        generate_content_config=types.GenerateContentConfig(temperature=0),
        after_agent_callback=save_adk_report,
    )
    return verifier_agent, StopIfNoRevision(name="VerifierStopCheck")
