"""Verifier agent: source + facts + current patient text -> faithfulness report.

The text judged is whatever the pipeline has selected at this point: the Readability
rewrite when it accepted one, otherwise the Simplifier's draft, or a Refiner revision
on later loop passes. Only the reference passages come from the Simplifier, because
Readability and Refiner never retrieve material of their own.

Scores that text with the four metrics (faithfulness, omission, addition, hallucination)
and tells the Refiner whether it needs revision.
"""
import json
from pathlib import Path
from typing import AsyncGenerator, Callable, List, Literal, Optional, Union

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
    stated_incorrectly: bool = Field(
        default=False,
        description="True when the explanation does mention this fact but gets it wrong, "
                    "False when the fact is absent from the explanation altogether"
    )


class VerifierJudgment(BaseModel):
    """What the model returns."""

    claims: List[ClaimCheck] = Field(description="Every claim in the explanation, including supported ones")
    omissions: List[Omission] = Field(description="Every fact the explanation leaves out or states incorrectly")


class VerifierReport(VerifierJudgment):
    """What we hand back: the judgment plus metrics computed in code."""

    faithfulness_score: float = Field(description="Fraction of claims labelled supported or background")
    omission_count: int = Field(
        description="Facts absent from the explanation. A fact the explanation states "
                    "incorrectly is listed in omissions but not counted here."
    )
    total_facts: int = Field(description="Number of Extractor facts")
    addition_count: int = Field(
        description="Claims the source does not state: those labelled background or unsupported"
    )
    hallucination: bool = Field(
        description="At least one major unsupported or distorted claim. Severity is the model's own judgement."
    )
    needs_revision: bool = Field(
        description="Any unsupported or distorted claim, or any omission, minor or major. Sent to refiner agent."
    )
    false_coverage: List[int] = Field(
        description="Fact numbers the Simplifier reported as covered that the judgement finds "
                    "missing or stated incorrectly"
    )


MAX_ATTEMPTS = 2


class ContradictionCheck(BaseModel):
    """The answer to one narrow question, asked in its own call."""

    contradicted: bool = Field(
        description="True only when SOURCE states something the explanation cannot both be true with"
    )
    explanation_phrase: str = Field(
        description="Verbatim words from EXPLANATION that conflict, or '' when contradicted is false"
    )
    source_quote: str = Field(
        description="Verbatim words from SOURCE that conflict with it, or '' when contradicted is false"
    )


INSTRUCTION = """
You are a medical verification agent. You decide whether a patient-facing
explanation is faithful to the source it was written from.

You get four blocks: SOURCE (the note or abstract), FACTS (numbered facts taken
from the source, often empty), REFERENCE MATERIAL (background articles, often
absent), and EXPLANATION (the text to judge).

STEP 1. Split the explanation into claims.
- A claim is one thing that can be true or false on its own: one symptom, one
  number, one dose, one timing, one cause, one recommendation.
- A sentence that makes several statements becomes several claims. Do not merge
  statements to keep the count down.
- Keep one comparison inside one claim. "As effective as", "no worse than" and
  "more than" are a single claim and may not be split across two, because neither
  half can be checked against the source on its own.
- A claim carries the number, the direction and the negation the explanation gave
  it. "The risk fell" and "the risk fell by 30%" are different claims, and
  dropping a "not" changes what has to be verified.
- Write each claim with the words of the explanation.

STEP 2. Test the claim for contradiction before you label it.
A claim is contradicted when SOURCE says something about the SAME comparison that
the claim cannot both be true with. Check these in order:
- a different number, dose, threshold, date or timeframe for that comparison;
- the opposite direction: more or fewer, higher or lower, increased or decreased,
  associated or not associated;
- no difference, no association, no benefit, or "not statistically significant"
  for that comparison. A null result is a finding about the comparison, not
  silence. If the claim says the thing works, helps or is linked, and SOURCE
  reports no effect, the claim is contradicted;
- the same outcome restated as its opposite: present described as absent, a risk
  described as safe, a possibility described as certain.
If one of these holds, label the claim distorted and quote SOURCE's conflicting
words.

Guard the other way. SOURCE discussing a related but different population, dose,
endpoint or timeframe, and saying nothing about this comparison, is silence. Label
it unsupported, not distorted. A different group is not a contradiction of the
claim.

STEP 3. Label every claim that STEP 2 did not contradict.
- supported: SOURCE states the claim, or a strictly weaker form of it.
- unsupported: SOURCE says nothing about this comparison, so it can neither be
  confirmed nor contradicted.
- background: only when a REFERENCE MATERIAL passage states the claim. With no
  reference passages, nothing is background.

STEP 4. Give evidence.
- evidence is a short verbatim quote from SOURCE or from a REFERENCE MATERIAL
  passage. Never quote the EXPLANATION back as its own evidence, and never cite a
  FACTS line as evidence.
- For a distorted claim, quote the words in SOURCE that conflict with it.
- For an unsupported claim, write "Not mentioned in the original source."

STEP 5. List omissions.
- If no numbered facts were supplied, return an empty omissions list.
- Otherwise go through FACTS in order and list every fact the explanation leaves
  out or states incorrectly, with its number as fact_number.
- Set stated_incorrectly to true when the explanation does mention the fact but gets
  it wrong: a changed number, a dropped hedge, a flipped negation, a family history
  stated as the patient's. Set it to false when the fact is absent altogether. A fact
  that is genuinely missing is not the same finding as a fact that was changed.

Rules that decide the rest:
- Judge the idea, not the wording. Stating the same thing in different words is
  still support: "high blood pressure" supports "hypertension", and "did not
  reduce the risk" supports "had no effect on the risk". Sharing only a topic is
  not support, but different wording is not a reason to call a claim unsupported.
- A claim can be true in general and still be unsupported by this SOURCE. Label
  it unsupported.
- Never label a claim supported on the strength of the FACTS block. FACTS records
  what was extracted, not what SOURCE proves.
- Return only the structured output that matches the schema.
"""


CONTRADICTION_INSTRUCTION = """
You answer one question about a SOURCE and an EXPLANATION written from it: does the
source contradict the explanation?

Answer contradicted = true only when the source makes a statement that the
explanation cannot both be true with:
- a different number, dose, threshold, date or timeframe for the same comparison;
- the opposite direction: more or fewer, higher or lower, increased or decreased,
  associated or not associated;
- no difference, no association, no benefit, or "not statistically significant"
  where the explanation says the thing works, helps, or is linked;
- the same outcome restated as its opposite: present described as absent, a risk
  described as safe, a possibility described as certain.

Answer contradicted = false when the source discusses a related but different
population, dose, endpoint or timeframe, and when the source says nothing about the
explanation at all. Silence is not contradiction. A neighbouring topic is not
contradiction. Being unproven is not contradiction.

When contradicted is true, copy two short verbatim quotes: the words in the
explanation that conflict, and the words in the source that conflict with them.
Both must be copied from the text, never paraphrased. A conflict you cannot quote
is not one you can report. When contradicted is false, leave both quotes empty.

Return only the structured output that matches the schema.
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
    """Computes the four metrics in code from the model's judgment.

    The labels are the model's, the arithmetic is not: each score is a count over the
    labels, so it cannot drift from the definitions in the scoring rubric.
    """
    # The categories overlap on purpose: reference-backed background counts as
    # faithful, but also as an addition because it is not stated in the patient note.
    supported = [c for c in judgment.claims if c.label in ("supported", "background")]
    additions = [c for c in judgment.claims if c.label in ("background", "unsupported")]
    # Only major unsupported or distorted claims set the hallucination flag.
    # An omission of any severity can also require revision even when no claim was invented.
    hallucinated = [c for c in judgment.claims if c.label in ("unsupported", "distorted") and c.severity == "major"]
    # A fact the explanation gets wrong is a restated fact, not a missing one, so it is kept
    # out of omission_count: the rubric scores a changed fact as neither an addition nor an
    # omission. The entry itself is still reported so the Refiner can act on it.
    missing = [o for o in judgment.omissions if not o.stated_incorrectly]
    # Any invented claim, not only a major one. Severity is the model's own call, and an
    # invented sentence left in a patient's text is not a severity question.
    invented = [c for c in judgment.claims if c.label == "unsupported"]
    # A distorted claim changes a fact from the source, so it always needs revisiting.
    # Severity only decides whether it also counts as a hallucination. Before this,
    # a minor distortion was reported and then left in the text, because a minor
    # negation flip neither set the hallucination flag nor blocked the loop.
    distorted = [c for c in judgment.claims if c.label == "distorted"]

    n = len(judgment.claims)
    # A judgement with no claims has no faithfulness to report. Returning 1.0 here made an
    # empty reply look like a perfect score to every caller that does not pass through
    # verify(), which rejects the empty case earlier with a repair attempt.
    if not n:
        raise ValueError("Verifier judgement has no claims, so faithfulness is undefined")

    hallucination = len(hallucinated) > 0
    # These numbers are one-based fact positions, not IDs such as F007.
    # This set is the lookup for false coverage: a fact the Simplifier claimed to cover is
    # wrong whether it went missing or was restated incorrectly.
    omitted = {o.fact_number for o in judgment.omissions}
    # Only genuinely absent facts are counted, and each fact only once.
    missing_numbers = {o.fact_number for o in missing}

    # Scores summarize the model's labels. They are not human-verified accuracy.
    return VerifierReport(
        **judgment.model_dump(),
        faithfulness_score=len(supported) / n,
        omission_count=len(missing_numbers),
        total_facts=atom_count,
        addition_count=len(additions),
        hallucination=hallucination,
        # The flag above stays severity-gated so it matches the rubric's "clinically
        # meaningful" wording. The loop is deliberately stricter than the metric: any
        # distortion, any invented claim and any omission send the text back, because
        # severity is the model's own judgement and a defect that survives is a defect
        # the patient reads.
        needs_revision=(
            hallucination
            or bool(distorted)
            or bool(invented)
            or len(judgment.omissions) > 0
        ),
        false_coverage=sorted(omitted & set(atoms_covered or [])),
    )


def format_facts(atoms: List[dict]) -> str:
    """The numbered facts, or a plain statement that this judgement has none."""
    if not atoms:
        return "No numbered facts were supplied. Judge the explanation against SOURCE only."
    return format_atoms(atoms)


def build_prompt(source: str, atoms: List[dict], draft: SimplifiedOutput) -> str:
    """The one prompt shape: source, numbered facts, reference passages, then the text."""
    return (
        f"SOURCE:\n{source}\n\n"
        f"FACTS:\n{format_facts(atoms)}\n\n"
        f"REFERENCE MATERIAL:\n{format_references(draft)}\n\n"
        f"EXPLANATION:\n{draft.simplified_text}"
    )


def repair_prompt(prompt: str, previous: str, issues: List[str]) -> str:
    """Ask again, naming what was wrong with the reply we could not use."""
    listed = "\n".join(f"- {message}" for message in issues)
    return (
        prompt
        + "\n\nYOUR PREVIOUS REPLY:\n" + (previous or "(empty)")
        + "\n\nWHAT WAS WRONG:\n" + listed
        + "\n\nReturn a corrected judgement that follows the schema."
    )


def live_send(prompt: str, model: str) -> str:
    """Send one prompt to Gemini and return the raw reply text."""
    from google import genai
    from google.genai import types

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
    return response.text or ""


def build_contradiction_prompt(source: str, draft: SimplifiedOutput) -> str:
    """The narrow question, given the same two texts the judgement was given."""
    return f"SOURCE:\n{source}\n\nEXPLANATION:\n{draft.simplified_text}"


def live_contradiction_send(prompt: str, model: str) -> str:
    """Send the contradiction question to Gemini and return the raw reply text."""
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=CONTRADICTION_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ContradictionCheck,
            temperature=0,
        ),
    )
    return response.text or ""


def parse_contradiction(reply_text: str) -> Optional[ContradictionCheck]:
    """Validate a contradiction reply, returning None when it does not parse."""
    try:
        return ContradictionCheck.model_validate_json(reply_text)
    except Exception:
        return None


def find_contradiction(
    source: str,
    draft: SimplifiedOutput,
    model: str = MODEL,
    send: Optional[Callable[[str, str], str]] = None,
) -> Optional[ContradictionCheck]:
    """Ask the narrow question once and return the finding, or None when there is none.

    An unparseable reply, a plain "no", and a "yes" that cannot quote both sides all
    count as no finding. The quotes are the point of the pass: they make a reported
    distortion cheap to check by hand and impossible to invent quietly.
    """
    checker = send or live_contradiction_send
    finding = parse_contradiction(checker(build_contradiction_prompt(source, draft), model))
    if finding is None or not finding.contradicted:
        return None
    if not finding.source_quote.strip() or not finding.explanation_phrase.strip():
        return None
    return finding


def parse_reply(reply_text: str) -> Optional[VerifierJudgment]:
    """Validate a reply against the schema, returning None when it does not parse."""
    try:
        return VerifierJudgment.model_validate_json(reply_text)
    except Exception:
        return None


def judgment_problems(judgment: Optional[VerifierJudgment], atom_count: int) -> List[str]:
    """The reasons this judgement cannot be used, as sentences a repair can act on."""
    if judgment is None:
        return ["the reply did not match the judgement schema"]
    issues = []
    if not judgment.claims:
        issues.append("no claims listed")
    unknown = check_fact_numbers(judgment, atom_count)
    if unknown:
        numbers = ", ".join(str(number) for number in unknown)
        issues.append(f"omissions reference unknown fact numbers: {numbers}")
    return issues


def verify(
    source: str,
    atoms: List[dict],
    draft: Union[SimplifiedOutput, dict],
    model: str = MODEL,
    send: Optional[Callable[[str, str], str]] = None,
    send_contradiction: Optional[Callable[[str, str], str]] = None,
) -> VerifierReport:
    """Check the current patient text against the source, the facts and the reference passages.

    An empty `atoms` list is allowed and means the claim labels are the whole
    judgement, which is how the MedAESQA calibration scores a sentence against an
    abstract. The pipeline's own step still refuses to run without facts.

    `send` can be swapped for a fake one so tests run without an API key. A fake `send`
    does not cover the contradiction pass; pass `send_contradiction` for that.
    """
    atoms = list(atoms or [])
    draft = as_draft(draft)
    prompt = build_prompt(source, atoms, draft)
    sender = send or live_send

    # One repair attempt, the same shape the Extractor and the Simplifier use.
    # Before this, a single unusable reply ended a run that had already produced text.
    reply = sender(prompt, model)
    judgment = parse_reply(reply)
    issues = judgment_problems(judgment, len(atoms))
    if issues:
        reply = sender(repair_prompt(prompt, reply, issues), model)
        judgment = parse_reply(reply)
        issues = judgment_problems(judgment, len(atoms))
    if issues:
        raise ValueError("Verifier validation failed: " + ", ".join(issues))

    assert judgment is not None  # judgment_problems rejects None

    # Second pass, and the reason it exists: the five-label judgement found 10 of 27
    # contradictions on the MedAESQA dev pairs, and seven prompt rewrites left that recall
    # at 0.37. One call offered five labels takes the cheapest one. This asks the same
    # model one narrower question in its own call, where the schema has room to say no.
    # A missed contradiction is the expensive error, because "the source is silent" reads
    # as harmless and leaves the text unchanged.
    checker = send_contradiction
    if checker is None and send is None:
        checker = live_contradiction_send
    # An injected `send` speaks one prompt shape, so it is never reused for this question.
    # Requiring the caller to name a contradiction sender keeps an offline test off the
    # network and keeps the provider explicit rather than silently falling back to Gemini.
    if checker is not None and not any(c.label == "distorted" for c in judgment.claims):
        found = find_contradiction(source, draft, model=model, send=checker)
        if found is not None:
            judgment.claims.append(
                ClaimCheck(
                    claim=found.explanation_phrase,
                    label="distorted",
                    evidence=found.source_quote,
                    # A contradiction that can be quoted is a fidelity failure, not a nit.
                    severity="major",
                )
            )

    return build_report(judgment, len(atoms), draft.atoms_covered)


# Is the judgment usable?
def check_fact_numbers(judgment: VerifierJudgment, atom_count: int) -> List[int]:
    """Return omission fact numbers that are not in the fact list."""
    return [o.fact_number for o in judgment.omissions if o.fact_number < 1 or o.fact_number > atom_count]


def print_report(report: VerifierReport) -> None:
    print("Verifier Report")
    print(f"Faithfulness score: {report.faithfulness_score:.2f}")
    print(f"Omissions (missing): {report.omission_count} of {report.total_facts} | Additions: {report.addition_count}")
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
