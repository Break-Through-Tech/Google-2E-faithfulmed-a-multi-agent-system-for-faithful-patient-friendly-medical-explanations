"""Refiner agent: patient text + Verifier report -> text with the flagged problems fixed."""
import json
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from faithfulmed.agents.simplifier import format_atoms

MODEL = "gemini-3.5-flash-lite"

# Claims with these labels get fixed. Background claims are fine, so they are left alone.
FLAGGED_LABELS = ("unsupported", "distorted")

class RefinerError(Exception):
    """A failure the Refiner can name."""

class RefinedOutput(BaseModel):
    refined_text: str = Field(description="The full patient text with only the listed problems fixed.")
    changes: List[str] = Field(default_factory=list, description="One short line per change made.")


INSTRUCTION = """
You fix a patient-friendly medical explanation that a verifier has checked.

Fix only the problems listed:
- unsupported claim: remove it.
- distorted claim: rewrite it to match the SOURCE.
- omitted fact: add one short sentence that states it.

Rules:
- Change nothing else. Keep every other sentence word for word.
- Use only the SOURCE and FACTS. Do not add new facts, numbers, or advice.
- Keep negations, uncertainty, timing, and who the fact is about.
- Write at an 8th-grade reading level or below.

Return the full corrected text and one short line per change.
"""

def find_problems(verdict: dict, atoms: List[dict]) -> List[str]:
    """List the claims and omissions the Refiner should fix."""
    problems = []
    for c in verdict.get("claims", []):
        if c["label"] in FLAGGED_LABELS:
            problems.append(f"[{c['label']}] {c['claim']} (evidence: {c.get('evidence') or 'none'})")
    for o in verdict.get("omissions", []):
        # fact_number is 1-based, the same numbering format_atoms prints
        n = o["fact_number"]
        if 1 <= n <= len(atoms):
            problems.append(f"[omitted] fact {n}: {atoms[n - 1]['text']}")
    return problems

def build_prompt(source: str, atoms: List[dict], text: str, problems: List[str]) -> str:
    return (
        f"SOURCE:\n{source}\n\n"
        f"FACTS:\n{format_atoms(atoms)}\n\n"
        f"TEXT:\n{text}\n\n"
        "PROBLEMS:\n" + "\n".join(f"- {p}" for p in problems)
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
            response_schema=RefinedOutput,
            temperature=0,
        ),
    )
    return response.text or ""


def refine(
    source: str,
    text: str,
    verdict,
    atoms: List[dict],
    send: Optional[Callable[[str, str], str]] = None,
    model: str = MODEL,
) -> RefinedOutput:
    """Fix the problems in the verdict. Returns the text unchanged if there are none.

    `send` can be swapped for a fake one so tests run without an API key.
    """
    if hasattr(verdict, "model_dump"):
        verdict = verdict.model_dump()

    problems = find_problems(verdict, atoms)
    if not problems:
        return RefinedOutput(refined_text=text)

    sender = send or live_send
    reply = sender(build_prompt(source, atoms, text, problems), model)
    try:
        result = RefinedOutput.model_validate_json(reply)
    except Exception:
        raise RefinerError("The model reply was not valid JSON for RefinedOutput.")

    # TODO: check the new text in code (numbers, negations, reading grade) like the Simplifier does.
    # TODO: if a check fails, ask the model once to repair it.
    if not result.refined_text.strip():
        raise RefinerError("The model returned empty text.")
    return result


# Hand off the file
def save_refinement(doc_id: str, result: RefinedOutput, out_dir: Path) -> Path:
    """Write one document's result to <out_dir>/<doc_id>.refined.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc_id}.refined.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_refinement(path: Path) -> dict:
    """Read a saved result back into a dict."""
    return json.loads(path.read_text(encoding="utf-8"))
