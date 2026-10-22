from typing import AsyncGenerator, Literal

from pydantic import BaseModel, Field
from google.genai import Client, types
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

MODEL = "gemini-3.5-flash-lite"

# Shared-state keys used in the ADK pipeline.
SOURCE_KEY = "source"                  
FACTS_KEY = "atoms"                    
SIMPLIFIED_KEY = "draft"               
PASSAGES_KEY = "retrieved_passages"    
OUTPUT_KEY = "verdict"               
REPORT_KEY = "verifier_report"         


VERIFIER_INSTRUCTION = """
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
background only if a RETRIEVED BACKGROUND passage backs it.

If KEY FACTS are given, go through them one by one and
list every key fact the explanation leaves out or states
incorrectly as an omission, using its fact_id.

Pay special attention to negation errors.

Compare statements containing words such as
"no", "not", "never", "without", and "negative"
against the original source.

If the source reports a symptom or finding,
but the explanation says it is absent,
mark the claim as distorted.

Check each part of a sentence separately when
it contains multiple medical facts.

For every unsupported or distorted claim,
provide the relevant evidence from the original source.

If the source does not mention the claim, clearly state:
"Not mentioned in the original source."

Important rules:
- Use only the original medical source and the retrieved background to verify facts.
- Do not invent medical information.
- Do not treat suspected conditions as confirmed diagnoses.
- Prioritize errors that could affect patient safety.
- Check the entire explanation, not just a few claims.
- Do not overlook a contradiction just because most
  of the explanation is accurate.
"""

# Output schema
Severity = Literal["minor", "major"]

class ClaimCheck(BaseModel):
    claim: str = Field(description="One specific statement from the patient explanation")
    label: Literal["supported", "background", "unsupported", "distorted"] = Field(
        description="supported: the source states it; background: not in the source but backed by the "
                    "retrieved background; unsupported: backed by neither; distorted: changes a fact from the source")
    evidence: str = Field(description="Exact quote from the source or background passage, or '' if none")
    severity: Severity = Field(description="The clinical importance of claim being incorrect.")

class Omission(BaseModel):
    fact_id: str = Field(description="fact_id from KEY FACTS, or '' if no key facts were given")
    fact: str = Field(description="An important fact from the source that the explanation leaves out")
    severity: Severity = Field(description="The clinical importance of omission being made.")

class VerifierJudgment(BaseModel):
    claims: list[ClaimCheck]
    omissions: list[Omission]

class VerifierReport(VerifierJudgment):
    faithfulness_score: float = Field(description="Fraction of claims that are supported or background")
    omission_count: int = Field(description="Key facts missing")
    total_key_facts: int = Field(description="Number of Extractor key facts (0 if none given)")
    addition_count: int = Field(description="Claims labelled background or unsupported")
    hallucination: bool = Field(description="At least one major unsupported or distorted claim.")
    needs_revision: bool = Field(description="Hallucination, or any major omission. Sent to refiner agent.")
    false_coverage: list[str] = Field(description="fact_ids the Simplifier claimed to cover but are missing")


def build_report(judgment: VerifierJudgment, facts: list[dict] | None = None,
                 facts_covered: list[str] | None = None) -> VerifierReport:
    """Computes the advisor's metrics in code from the model's judgment."""
    supported = [c for c in judgment.claims if c.label in ("supported", "background")]
    additions = [c for c in judgment.claims if c.label in ("background", "unsupported")]
    hallucinated = [c for c in judgment.claims if c.label in ("unsupported", "distorted") and c.severity == "major"]
    serious_omissions = [o for o in judgment.omissions if o.severity == "major"]

    n = len(judgment.claims)
    hallucination = len(hallucinated) > 0
    omitted_ids = {o.fact_id for o in judgment.omissions}

    return VerifierReport(
        **judgment.model_dump(),
        faithfulness_score=len(supported) / n if n else 1.0,
        omission_count=len(judgment.omissions),
        total_key_facts=len(facts or []),
        addition_count=len(additions),
        hallucination=hallucination,
        needs_revision=hallucination or len(serious_omissions) > 0,
        false_coverage=sorted(omitted_ids & set(facts_covered or [])),
    )


# testing: call gemini directly
_client = None

def build_message(source: str, explanation: str, facts: list[dict] | None = None,
                  passages: list[str] | None = None) -> str:
    """Puts the inputs into labelled sections. Each fact needs a 'text' key; 'fact_id' is used if present."""
    message = f"SOURCE:\n{source}\n\n"
    if facts:
        message += "KEY FACTS:\n" + "\n".join(
            f"{f.get('fact_id', f'F{i}')}: {f['text']}" for i, f in enumerate(facts, 1)) + "\n\n"
    if passages:
        message += "RETRIEVED BACKGROUND:\n" + "\n\n".join(passages) + "\n\n"
    message += f"EXPLANATION:\n{explanation}"
    return message


def create_verifier_chat():
    global _client
    if _client is None:
        _client = Client()   # reads GOOGLE_API_KEY from the environment
    config = types.GenerateContentConfig(
        system_instruction=VERIFIER_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=VerifierJudgment,
        temperature=0,
    )
    return _client.chats.create(model=MODEL, config=config)


def run_verifier_agent(source: str, explanation: str, facts: list[dict] | None = None,
                       passages: list[str] | None = None, facts_covered: list[str] | None = None,
                       verbose: bool = True) -> VerifierReport:
    """Checks one explanation and returns the full Verifier report."""
    # new chat for every check, so the judge never sees earlier drafts
    chat = create_verifier_chat()
    response = chat.send_message(build_message(source, explanation, facts, passages))

    judgment = VerifierJudgment.model_validate_json(response.text)
    report = build_report(judgment, facts, facts_covered)

    if verbose:
        print_report(report)
    return report


def print_report(report: VerifierReport) -> None:
    print("Verifier Report")
    print(f"Faithfulness score: {report.faithfulness_score:.2f}")
    if report.total_key_facts:
        print(f"Omissions: {report.omission_count} of {report.total_key_facts} | Additions: {report.addition_count}")
    else:
        print(f"Omissions: {report.omission_count} | Additions: {report.addition_count}")
    print(f"Hallucination: {report.hallucination} | Needs revision: {report.needs_revision}")
    if report.false_coverage:
        print(f"Simplifier claimed but missing: {report.false_coverage}")
    for c in report.claims:
        if c.label != "supported":
            print(f"  [{c.label}, {c.severity}] {c.claim}")
    for o in report.omissions:
        print(f"  [omitted {o.fact_id}, {o.severity}] {o.fact}")


# ADK pipeline: LlmAgent that reads inputs from shared state
# {key?} means optional: the run continues if that key is not in state yet.
PIPELINE_INSTRUCTION = VERIFIER_INSTRUCTION + f"""
SOURCE:
{{{SOURCE_KEY}}}

KEY FACTS:
{{{FACTS_KEY}?}}

RETRIEVED BACKGROUND:
{{{PASSAGES_KEY}?}}

EXPLANATION (if this is JSON, check its simplified_text field):
{{{SIMPLIFIED_KEY}}}
"""


def _save_report(callback_context):
    """Runs after the agent: turns the raw judgment in state into the full report."""
    state = callback_context.state
    judgment = VerifierJudgment.model_validate(state[OUTPUT_KEY])
    simplified = state.get(SIMPLIFIED_KEY) or {}
    facts_covered = simplified.get("facts_covered", []) if isinstance(simplified, dict) else []
    facts = state.get(FACTS_KEY) if isinstance(state.get(FACTS_KEY), list) else None

    report = build_report(judgment, facts, facts_covered)
    state[REPORT_KEY] = report.model_dump()
    state["needs_revision"] = report.needs_revision   # the LoopAgent stop check reads this
    return None


verifier_agent = LlmAgent(
    name="Verifier",
    model=MODEL,
    description="Checks patient-friendly medical explanations for accuracy and completeness.",
    instruction=PIPELINE_INSTRUCTION,
    output_schema=VerifierJudgment,
    output_key=OUTPUT_KEY,
    generate_content_config=types.GenerateContentConfig(temperature=0),
    after_agent_callback=_save_report,
)


# Loop stop check: put between the Verifier and the Refiner in the LoopAgent
class StopIfNoRevision(BaseAgent):
    """Ends the loop when the Verifier found nothing to fix."""
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        done = ctx.session.state.get("needs_revision") is False
        yield Event(author=self.name, actions=EventActions(escalate=done))


verifier_stop_check = StopIfNoRevision(name="VerifierStopCheck")
