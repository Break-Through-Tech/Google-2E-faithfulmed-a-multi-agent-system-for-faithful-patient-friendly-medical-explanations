"""Readability agent: patient text -> the same text at or below the reading target.

Every score is measured in Python with textstat, never reported by the model. A model
asked to grade its own writing will name a grade it did not reach.

The policy is conservative. A rewrite is used only when it passes the local checks and
reaches TARGET_GRADE. Otherwise the incoming text is kept unchanged and the reason is
listed in review_warnings. Keeping the incoming text does not make it faithful: the
Verifier still compares the selected text against the source note and the facts.
"""
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from faithfulmed.agents.extractor import NEGATION_CUE_PATTERN
from faithfulmed.agents.simplifier import format_atoms, numbers_in
from faithfulmed.readability_scores import score_readability

MODEL = "gemini-3.5-flash-lite"
TARGET_GRADE = 8.0
MAX_ATTEMPTS = 2


class ReadabilityError(Exception):
    """A failure this agent can name, instead of a raw SDK or validation traceback."""


class ReadabilityReply(BaseModel):
    """What the model returns. Every score is added by Python, not by the model."""

    revised_text: str = Field(description="The whole text rewritten at or below the target grade.")


class ReadabilityResult(BaseModel):
    """What improve_readability() returns to the pipeline."""

    revised_text: str = Field(description="Selected text, which may be the unchanged original.")
    before_grade: float = Field(description="Flesch-Kincaid grade of the incoming text.")
    after_grade: float = Field(description="Flesch-Kincaid grade of the selected text.")
    before_smog: float = Field(description="SMOG index of the incoming text.")
    after_smog: float = Field(description="SMOG index of the selected text.")
    meets_target: bool = Field(description="True when after_grade is at or below TARGET_GRADE.")
    attempts: int = Field(description="Rewrite calls made, from 0 to MAX_ATTEMPTS.")
    changed: bool = Field(description="True when the selected text differs from the incoming text.")
    review_warnings: List[str] = Field(default_factory=list)


INSTRUCTION = """
You rewrite a patient-friendly medical explanation so an ordinary adult can read it.

Keep every fact. These must survive the rewrite unchanged:
- every number, unit, dose, date and instruction, written exactly as it is now
- every denial: "no", "not", "never", "without", "negative", "free of"
- uncertainty: "possible", "may", "suspected" must not become certainty
- timing: a past condition must not read as a current one
- ownership: a relative's condition must not become the patient's condition

How to lower the reading grade:
- shorten sentences and split crowded ideas, one idea per sentence
- use active voice, and "you" for the patient's own facts
- replace hard words with familiar words where the meaning stays the same
- when a medical term has to stay, explain it in the same sentence in a few plain words
- never invent a definition for a term you cannot explain plainly

Add no new diagnosis, advice, reassurance, cause, or unsupported definition.
Do not drop facts to make the text shorter.
Return only the revised text, and nothing else.
"""


def measure_readability(text: str) -> dict:
    """Return the reading scores of one text.

    Blank text raises ReadabilityError, so a missing explanation is never reported as
    a perfect score of zero.
    """
    if not text or not text.strip():
        raise ReadabilityError("Nothing to measure: the text is empty.")

    try:
        scores = score_readability(text)
    except ValueError as error:
        raise ReadabilityError("Nothing to measure: the text has no words.") from error

    return {"grade": scores.grade, "smog": scores.smog}


def format_references(source_metadata) -> str:
    """Return the draft's own reference passages, or an empty string when there are none.

    These explain terms only. This agent never retrieves new material.
    """
    passages = []
    for passage in source_metadata or []:
        if hasattr(passage, "model_dump"):
            passage = passage.model_dump()
        title = passage.get("title") or "Reference"
        text = passage.get("text") or ""
        if text:
            passages.append(f"{title}\n{text}")
    return "\n\n---\n\n".join(passages)


def build_prompt(
    source: str,
    atoms,
    text: str,
    grade: float,
    feedback: Optional[str] = None,
) -> str:
    """Assemble the labelled prompt: source note, facts, current text, and grade.

    `feedback` names what went wrong on the previous attempt. Without it this is the
    first attempt.
    """
    sections = [
        f"SOURCE NOTE:\n{source}",
        f"FACTS:\n{format_atoms(atoms)}",
        f"CURRENT TEXT:\n{text}",
        f"MEASURED READING LEVEL: Flesch-Kincaid grade {grade:.2f}",
        f"TARGET: grade {TARGET_GRADE:.1f} or below",
    ]
    if feedback:
        sections.append(f"WHAT WAS WRONG WITH THE PREVIOUS REWRITE:\n{feedback}")
    return "\n\n".join(sections)


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
            response_schema=ReadabilityReply,
            temperature=0,
        ),
    )
    return response.text or ""


def parse_reply(reply_text: str) -> Optional[ReadabilityReply]:
    """Return a ReadabilityReply, or None when the reply cannot be parsed."""
    try:
        return ReadabilityReply.model_validate_json(reply_text)
    except Exception:
        return None


def check_candidate(original: str, candidate: str) -> List[str]:
    """Return what is wrong with one rewrite. An empty list means it may be selected.

    Only quantities are blocked here. Matching numbers does not prove matching meaning:
    "5 mg twice daily" and "5 mg once daily" both pass. The Verifier makes the fuller
    comparison with the source note and the facts.
    """
    if not candidate.strip():
        return ["the rewrite is empty"]

    problems = []

    added = numbers_in(candidate) - numbers_in(original)
    if added:
        problems.append(f"numbers that are not in the incoming text: {', '.join(sorted(added))}")

    dropped = numbers_in(original) - numbers_in(candidate)
    if dropped:
        problems.append(f"numbers dropped from the incoming text: {', '.join(sorted(dropped))}")

    return problems


def lost_denial(original: str, candidate: str) -> bool:
    """True when the incoming text denies something and the rewrite denies nothing.

    This is a review warning, not a rejection. One cue word cannot be trusted to carry
    the meaning of a whole paragraph.
    """
    original_denies = bool(NEGATION_CUE_PATTERN.search(original))
    candidate_denies = bool(NEGATION_CUE_PATTERN.search(candidate))
    return original_denies and not candidate_denies


def improve_readability(
    source: str,
    atoms,
    text: str,
    source_metadata=None,
    send: Optional[Callable[[str, str], str]] = None,
    model: str = MODEL,
) -> ReadabilityResult:
    """Return the text at or below TARGET_GRADE, or the incoming text with warnings.

    `send` can be replaced by a fake function so tests run without an API key.
    """
    if not atoms:
        raise ReadabilityError("Nothing to protect: no facts were supplied.")
    if not text or not text.strip():
        raise ReadabilityError("Nothing to rewrite: the text is empty.")

    original = text
    before = measure_readability(original)

    # Already easy enough: no model call, no cost, and no reason to change the wording.
    if before["grade"] <= TARGET_GRADE:
        return ReadabilityResult(
            revised_text=original,
            before_grade=before["grade"],
            after_grade=before["grade"],
            before_smog=before["smog"],
            after_smog=before["smog"],
            meets_target=True,
            attempts=0,
            changed=False,
            review_warnings=[],
        )

    sender = send or live_send
    references = format_references(source_metadata)
    warnings: List[str] = []
    feedback: Optional[str] = None
    attempts = 0
    best_grade: Optional[float] = None

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        prompt = build_prompt(source, atoms, original, before["grade"], feedback)
        if references:
            prompt = f"{prompt}\n\nREFERENCE MATERIAL (explains terms only):\n{references}"

        # An outage says nothing about the text, so it ends the run as a named error
        # rather than being recorded as a readability failure.
        try:
            reply_text = sender(prompt, model)
        except Exception as error:
            raise ReadabilityError("The readability model call failed.") from error

        reply = parse_reply(reply_text)
        if reply is None or not reply.revised_text.strip():
            feedback = "The previous reply was empty or not valid JSON. Return the revised text only."
            warnings.append(f"attempt {attempts} returned no usable text")
            continue

        problems = check_candidate(original, reply.revised_text)
        if problems:
            feedback = "\n".join(f"- {problem}" for problem in problems)
            warnings.append(f"attempt {attempts} was rejected: {'; '.join(problems)}")
            continue

        candidate = measure_readability(reply.revised_text)
        best_grade = candidate["grade"]

        if candidate["grade"] <= TARGET_GRADE:
            revised = reply.revised_text.strip()
            if lost_denial(original, revised):
                warnings.append("the rewrite no longer denies anything; review it before use")
            return ReadabilityResult(
                revised_text=revised,
                before_grade=before["grade"],
                after_grade=candidate["grade"],
                before_smog=before["smog"],
                after_smog=candidate["smog"],
                meets_target=True,
                attempts=attempts,
                changed=revised != original,
                review_warnings=warnings,
            )

        feedback = (
            f"The rewrite is still grade {candidate['grade']:.2f}. "
            "Shorten long sentences without dropping facts."
        )

    # No candidate qualified, so keep the incoming text and say why.
    if best_grade is None:
        warnings.append(
            f"kept the incoming text: no usable rewrite in {attempts} attempt(s), "
            f"grade {before['grade']:.2f} is above {TARGET_GRADE:.1f}"
        )
    else:
        warnings.append(
            f"kept the incoming text: the closest rewrite was grade {best_grade:.2f}, "
            f"above {TARGET_GRADE:.1f}"
        )

    return ReadabilityResult(
        revised_text=original,
        before_grade=before["grade"],
        after_grade=before["grade"],
        before_smog=before["smog"],
        after_smog=before["smog"],
        meets_target=False,
        attempts=attempts,
        changed=False,
        review_warnings=warnings,
    )
