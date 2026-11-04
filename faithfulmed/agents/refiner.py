"""Refiner agent: patient text + Verifier report -> text with only the flagged passages fixed."""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from faithfulmed.agents.simplifier import FactRewrite, assess_fact_rewrite, format_atoms, normalize_atoms, numbers_in

MODEL = "gemini-3.5-flash-lite"

# Claims with these labels get fixed. Background claims are fine, so they are left alone.
FLAGGED_LABELS = ("unsupported", "distorted")


class RefinerError(Exception):
    """A failure the Refiner can name."""


class Change(BaseModel):
    problem_id: str = Field(description="The id of the problem this fixes, copied from PROBLEMS, e.g. C3 or F2.")
    old_text: str = Field(description="Exact words copied from TEXT that this change replaces. '' to add a new sentence at the end.")
    new_text: str = Field(description="The new wording. '' to remove old_text.")


class RefinerReply(BaseModel):
    """What the model returns."""
    changes: List[Change]


class RefinedOutput(BaseModel):
    """What refine() returns."""
    refined_text: str
    changes: List[Change] = Field(default_factory=list, description="The changes that passed the checks and were applied.")
    unresolved: List[str] = Field(default_factory=list, description="Problem ids left as they were.")
    warnings: List[str] = Field(default_factory=list)


INSTRUCTION = """
You fix a patient-friendly medical explanation that a verifier has checked.

Fix only the problems listed. Each problem has an id.
- unsupported claim: remove it (new_text = ""), or keep only the supported part of its sentence.
- distorted claim: replace it with wording that matches the SOURCE. Use the evidence.
- omitted fact: add one short sentence that states it (old_text = ""), or replace the wrong
  wording if the text states the fact incorrectly.

Rules:
- old_text must be copied exactly from TEXT and appear only once in it.
- Only use the SOURCE and FACTS. Do not add new facts, numbers, or advice.
- Keep negations, uncertainty, timing, and who the fact is about.
- Use plain words, like the text around it.
- Return at least one change per problem, and nothing for anything else.
"""


def find_problems(verdict, atoms: List[dict]):
    """Return the problems to fix, keyed by id (C<n> = claim n, F<n> = fact n), plus warnings."""
    if hasattr(verdict, "model_dump"):
        verdict = verdict.model_dump()
    atoms = normalize_atoms(atoms)

    problems, warnings = {}, []
    for n, c in enumerate(verdict.get("claims", []), start=1):
        if c.get("label") in FLAGGED_LABELS:
            problems[f"C{n}"] = {"kind": "claim", **c}
    for o in verdict.get("omissions", []):
        # fact_number is 1-based, the same numbering format_atoms prints
        n = o.get("fact_number")
        if isinstance(n, int) and 1 <= n <= len(atoms):
            problems[f"F{n}"] = {"kind": "omission", "atom": atoms[n - 1]}
        else:
            warnings.append(f"omission points to unknown fact number {n}; skipped.")
    return problems, warnings


def format_problems(problems: Dict[str, dict]) -> str:
    lines = []
    for pid, p in problems.items():
        if p["kind"] == "claim":
            lines.append(f"{pid} [{p['label']}] {p['claim']} (evidence: {p.get('evidence') or 'none'})")
        else:
            lines.append(f"{pid} [omitted] fact {p['atom']['position']}: {p['atom']['text']}")
    return "\n".join(lines)


def build_prompt(source: str, atoms: List[dict], text: str, problems: Dict[str, dict]) -> str:
    return (
        f"SOURCE:\n{source}\n\n"
        f"FACTS:\n{format_atoms(atoms)}\n\n"
        f"TEXT:\n{text}\n\n"
        f"PROBLEMS:\n{format_problems(problems)}"
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
            response_schema=RefinerReply,
            temperature=0,
        ),
    )
    return response.text or ""


def parse_reply(reply: str) -> RefinerReply:
    try:
        return RefinerReply.model_validate_json(reply)
    except Exception as error:
        raise RefinerError("The model reply was not valid JSON for RefinerReply.") from error


def check_changes(text: str, source: str, changes: List[Change], problems: Dict[str, dict]) -> Dict[str, List[str]]:
    """Return what is wrong, per problem id. An empty dict means every change passed."""
    issues: Dict[str, List[str]] = {}

    def flag(pid, message):
        issues.setdefault(pid, []).append(message)

    for pid in problems:
        if not any(c.problem_id == pid for c in changes):
            flag(pid, "no change was returned")

    spans = []
    for c in changes:
        p = problems.get(c.problem_id)
        if p is None:
            continue  # unknown ids are ignored

        # Where does it go?
        if c.old_text:
            count = text.count(c.old_text)
            if count != 1:
                flag(c.problem_id, f"old_text appears {count} times in TEXT (needs exactly 1): {c.old_text!r}")
            else:
                start = text.find(c.old_text)
                spans.append((start, start + len(c.old_text), c.problem_id))
        elif p["kind"] == "claim":
            flag(c.problem_id, "a claim fix needs old_text")

        # Does it fix the flagged words?
        if p["kind"] == "claim" and p["claim"] in text and p["claim"] not in c.old_text and c.old_text not in p["claim"]:
            flag(c.problem_id, "old_text does not cover the flagged claim")

        # Is the new wording faithful?
        if p["kind"] == "claim":
            invented = numbers_in(c.new_text) - numbers_in(source)
            if invented:
                flag(c.problem_id, f"numbers not in the source: {', '.join(sorted(invented))}")
        else:
            if not c.new_text.strip():
                flag(c.problem_id, "an omitted fact can't be fixed by removing text")
            else:
                # Same meaning checks the Simplifier uses: numbers, negation, uncertainty, timing, family
                atom = p["atom"]
                for message in assess_fact_rewrite(atom, FactRewrite(fact_id=atom["fact_id"], text=c.new_text)):
                    flag(c.problem_id, message)

    # Two changes can't touch the same words
    for i, (s1, e1, id1) in enumerate(spans):
        for s2, e2, id2 in spans[i + 1:]:
            if s1 < e2 and s2 < e1:
                flag(id1, f"overlaps the change for {id2}")
                flag(id2, f"overlaps the change for {id1}")
    return issues


def repair_prompt(source, atoms, text, problems, changes, issues) -> str:
    """Ask again, only for the problems whose changes failed."""
    failing = {pid: problems[pid] for pid in issues}
    previous = [c.model_dump_json() for c in changes if c.problem_id in issues]
    listed = [f"- {pid}: {'; '.join(msgs)}" for pid, msgs in issues.items()]
    return (
        build_prompt(source, atoms, text, failing)
        + "\n\nYOUR PREVIOUS CHANGES:\n" + ("\n".join(previous) or "(none)")
        + "\n\nWHAT WAS WRONG:\n" + "\n".join(listed)
        + "\n\nReturn fixed changes for these problems only."
    )


def apply_changes(text: str, changes: List[Change]) -> str:
    """Apply changes that passed the checks. Replacements first (from the end), then additions."""
    replacements = sorted((c for c in changes if c.old_text), key=lambda c: text.find(c.old_text), reverse=True)
    for c in replacements:
        start, end = text.find(c.old_text), text.find(c.old_text) + len(c.old_text)
        # A removal that empties its line takes the line with it
        line_start, line_end = text.rfind("\n", 0, start) + 1, text.find("\n", end)
        line_end = len(text) if line_end == -1 else line_end
        if not c.new_text and text[line_start:start].strip(" -*•") == "" and not text[end:line_end].strip():
            start, end = line_start, min(line_end + 1, len(text))
        text = text[:start] + c.new_text + text[end:]
    for c in changes:
        if not c.old_text:
            text = text.rstrip() + "\n" + c.new_text.strip()

    # Clean up what removals leave behind: double spaces and empty bullet lines
    lines = [" ".join(line.split()) if line.strip() else "" for line in text.split("\n")]
    lines = [line for line in lines if line not in ("-", "*", "•")]
    return "\n".join(lines).strip()


def refine(
    source: str,
    text: str,
    verdict,
    atoms: List[dict],
    send: Optional[Callable[[str, str], str]] = None,
    model: str = MODEL,
) -> RefinedOutput:
    """Fix only the flagged passages. Returns the text unchanged if nothing is flagged.

    `send` can be swapped for a fake one so tests run without an API key.
    """
    atoms = normalize_atoms(atoms)
    problems, warnings = find_problems(verdict, atoms)
    if not problems:
        return RefinedOutput(refined_text=text, warnings=warnings)

    sender = send or live_send
    changes = parse_reply(sender(build_prompt(source, atoms, text, problems), model)).changes
    issues = check_changes(text, source, changes, problems)

    # One repair attempt for the problems that failed
    if issues:
        repair = parse_reply(sender(repair_prompt(source, atoms, text, problems, changes, issues), model)).changes
        answered = {c.problem_id for c in repair if c.problem_id in issues}
        changes = [c for c in changes if c.problem_id not in answered] + [c for c in repair if c.problem_id in answered]
        issues = check_changes(text, source, changes, problems)

    # Still failing: leave those passages as they were, so the next Verifier pass sees them again
    for pid, msgs in issues.items():
        warnings.append(f"{pid} left unchanged: {'; '.join(msgs)}")
    kept = [c for c in changes if c.problem_id in problems and c.problem_id not in issues]

    return RefinedOutput(
        refined_text=apply_changes(text, kept),
        changes=kept,
        unresolved=list(issues),
        warnings=warnings,
    )


# Hand off the file
def save_refinement(doc_id: str, result: RefinedOutput, out_dir: Path) -> Path:
    """Write one document's result to <out_dir>/<doc_id>.refined.json."""
    if not doc_id or Path(doc_id).name != doc_id or doc_id in {".", ".."}:
        raise RefinerError("doc_id must be a non-empty filename stem")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc_id}.refined.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_refinement(path: Path) -> dict:
    """Read a saved result back into a dict."""
    return json.loads(path.read_text(encoding="utf-8"))
