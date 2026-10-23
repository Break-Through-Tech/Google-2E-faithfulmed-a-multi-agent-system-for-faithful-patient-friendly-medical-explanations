"""Extractor agent: clinical text -> typed atoms backed by verbatim quotes."""
import json
import re
from pathlib import Path
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, Field

MODEL = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 2


class ExtractorError(Exception):
    """A failure the Extractor can name, instead of a raw SDK or validation traceback."""


class ExtractionValidationError(ExtractorError):
    """The model replied, but the reply failed the source checks.

    `diagnostics` holds one record per problem: the fact position, the check that fired,
    and the quote it fired on. `str(error)` reports counts and check names only, because
    the message may be logged or shown in a UI and must not carry patient text.
    """

    def __init__(self, diagnostics: List[dict], attempts: int):
        self.diagnostics = diagnostics
        self.attempts = attempts
        checks = sorted({record["check"] for record in diagnostics})
        super().__init__(
            f"Extraction failed source validation after {attempts} attempt(s). "
            f"{len(diagnostics)} issue(s) from {len(checks)} check(s): {', '.join(checks)}."
        )


# An atom is one medical fact. Its assertion tells us whether
# the fact is present, denied, uncertain, historical, or about a family member.
# Pydantic checks the shape of the reply. Later functions check it against the note.
class Atom(BaseModel):
    text: str = Field(
        description=(
            "One clinically meaningful fact copied as a complete source clause, including wording "
            "that establishes who has the fact and its status."
        )
    )
    type: Literal["diagnosis", "symptom", "medication", "lab_value", "procedure", "instruction"] = Field(
        description=
            "The category of the extracted fact: diagnosis for a diagnosed disease or condition; " \
            "symptom for a symptom reported or denied by the patient; " \
            "medication for a drug or its use; lab_value for a test result or clinical measurement;" \
            "procedure for a clinical procedure or intervention; instruction for advice or directions given to the patient."
        )
    assertion: Literal["present", "absent", "possible", "past", "family"] = Field(
        description=
            "The assertion status of the fact: present for an affirmed current fact; " \
            "absent for an explicitly denied or absent fact; possible for an uncertain or suspected fact;" \
            "past for a historical fact that is no longer current; family for a fact about a family member's medical history."
        )
    source_quote: str = Field(
        description=(
            "An exact, contiguous source clause that supports both the fact and its assertion. "
            "Include the subject and assertion wording, not only the clinical term."
        )
    )

class Extraction(BaseModel):
    atoms: List[Atom]


class StoredAtom(Atom):
    """An accepted atom, with identity and position filled in by Python.

    The model is not asked for these: an offset it computed could not be trusted, and a
    position it invented would not survive a repair attempt.
    """

    # Python adds these fields after checking the reply. The model supplies words,
    # but Python finds where those words occur and assigns an ID inside this result.
    fact_id: str = Field(description="Stable handle inside this extraction, such as F001.")
    char_start: int = Field(description="Inclusive offset of source_quote in the note.")
    char_end: int = Field(description="Exclusive offset of source_quote in the note.")


class ExtractionRecord(BaseModel):
    """What extract_atoms returns and what the session state carries."""

    atoms: List[StoredAtom]
    review_warnings: List[str] = Field(
        default_factory=list,
        description="Facts dropped after repair, and any coverage lost, for a human to review.",
    )


INSTRUCTION = """
ROLE
You extract clinical facts from one note. You do not interpret, summarize, diagnose, or advise.

WHAT AN ATOM IS
One atom is one fact, quoted from the note, with two labels attached.
- text and source_quote must be the SAME string: one complete clause, copied character for
  character. If you cannot copy it exactly, do not create the atom.
- One fact per atom. Keep a measurement with its value and unit, and a medicine with its
  documented dose.
- Do not split one fact into several atoms, and do not quote the same words twice.
- Items that follow one shared lead-in are ONE fact. "Negative for heart disease, lung
  disease, or bleeding problems" is one atom, and so is "Possibilities include A, B and
  C". Do not create one atom per item and repeat the lead-in each time.

QUOTE RULES
1. The quote must appear in the note exactly: same spelling, capitalization, punctuation.
   Do not expand abbreviations or correct anything.
2. Do not paraphrase, tidy, or join two places in the note.
3. Never include a list number, or a section label that introduces a list, in the fact. From
   "ADMITTING DIAGNOSES:,1. Fever.,2. Otitis media." quote "Fever." and
   "Otitis media.", not the header and not the numbers. A label on its own, such as
   "HEENT:", may stay when the note gives the clause no other subject.
4. Never quote a span that contains another atom's quote.

ASSERTION: choose the value, then quote the words that prove it
- present: the note affirms it. Quote the finding itself.
- absent: the note denies it. The quote MUST contain the denying words: "no", "not",
  "denies", "without", "negative for", "free of", "no history of". Quote
  "denies nausea", never only "nausea". A symptom that improved or resolved is not
  absent: use past when it is no longer current.
- possible: the note is unsure. The quote MUST contain the hedging words: "possible",
  "questionable", "suspected", "may", "cannot be excluded".
- past: historical, stopped, improved, or resolved. The quote MUST contain the time
  wording: "history of", "status post", "previously", "discontinued", "resolved",
  "improved", "no longer", or the year.
- family: about a relative. The quote MUST name the relative, such as "mother",
  "father", "sister", "brother", "son", "daughter", or say "family history". Quote
  "Her father has questionable celiac disease", never only "questionable celiac
  disease".

TYPE
diagnosis: a disease, condition, or clinical finding.
symptom: a symptom reported or denied, including improvement or resolution.
medication: a drug, its use, its dose, or its discontinuation.
lab_value: a test, measurement, or reading, with its value and unit.
procedure: a procedure or intervention performed or planned.
instruction: advice or directions given to the patient.

WHAT TO EXTRACT
- Every clinically meaningful fact within those types, including negatives, uncertain
  findings, historical items, stopped medicines, family history, and improvements.
- Keep numbers and units exactly as written.
- Never infer a fact from medical knowledge, and never treat silence as absence.
- Never turn a possible condition into a confirmed one, or a relative's condition into
  the patient's.
- Do not classify a medicine as current unless the note establishes it.

WHAT TO SKIP
- Note furniture: section headers, list numbers, page markers, signature and dictation
  lines.
- Any fact you cannot quote exactly.

OUTPUT
Return only the structured extraction, one atom per fact, no commentary.
"""

# Checks. Each returns issue records, and each is wired into extract_atoms.
# These notes are comma-joined, so "sentence" is not a unit that exists here: a check
# is written over a quote, a clause of the note, or a pair of spans.

# A list number is furniture, and so is a section label that introduces one. A label on its
# own is not, because in these comma-joined notes the label is sometimes the only subject the
# clause has. The (?!\d) stops a decimal such as "0.25%" from reading as list number "0.".
LIST_NUMBER_PATTERN = re.compile(r"^\s*(?:[A-Z][A-Z0-9 /&\-]{2,}:\s*,?\s*\d+\.|\d+\.(?!\d))")
RELATIVE_PATTERN = re.compile(
    r"\b(?:mother|father|mom|dad|sister|brother|daughter|son|wife|husband|partner|"
    r"grandmother|grandfather|aunt|uncle|cousin|sibling)\b",
    flags=re.IGNORECASE,
)
FAMILY_HISTORY_PATTERN = re.compile(r"\bfamily\b", flags=re.IGNORECASE)
NEGATION_CUE_PATTERN = re.compile(
    r"\b(?:no|not|never|none|den(?:y|ies|ied)|without|negative|absent|free of|"
    r"ruled out|absence of)\b|n't\b",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?"


def issue(position: int, check: str, quote: str, message: str) -> dict:
    """One problem, addressed to a fact position and to the check that found it."""
    return {"fact": position, "check": check, "quote": quote, "message": message}


def assign_spans(source_text: str, atoms: List[Atom]) -> List[tuple[int, int]]:
    """Find each quote's offsets, claiming a distinct occurrence where one exists.

    Two atoms quoting the same words must land on different spans, otherwise overlap
    detection cannot tell a duplicate from a coincidence.
    """
    # A span is a pair of character positions: include start, stop just before end.
    # (-1, -1) means the quote was not found. Each exact occurrence can be claimed once.
    claimed = set()
    spans: List[tuple[int, int]] = []
    for atom in atoms:
        quote = atom.source_quote
        start = source_text.find(quote) if quote else -1
        first = (start, start + len(quote)) if start != -1 else (-1, -1)
        chosen = None
        while start != -1:
            # The same words may appear more than once in a note. Look for an unused
            # occurrence before treating the atom as a duplicate of another atom.
            candidate = (start, start + len(quote))
            if candidate not in claimed:
                chosen = candidate
                break
            start = source_text.find(quote, start + 1)
        # No free occurrence left means the quote was reused, so reuse the first span and
        # let check_overlap report it instead of silently accepting the duplicate.
        chosen = chosen or first
        spans.append(chosen)
        if chosen != (-1, -1):
            claimed.add(chosen)
    return spans


def check_quotes(source_text: str, atoms: List[Atom]) -> List[dict]:
    """Every quote is non-empty and appears verbatim in the note."""
    issues = []
    for position, atom in enumerate(atoms, start=1):
        if not atom.source_quote.strip():
            issues.append(issue(position, "quote", atom.source_quote, "source_quote is empty."))
        elif atom.source_quote not in source_text:
            issues.append(issue(position, "quote", atom.source_quote,
                                "source_quote is not a verbatim substring of the note."))
    return issues


def check_texts(source_text: str, atoms: List[Atom]) -> List[dict]:
    """Every text is non-empty and copied from the note."""
    issues = []
    for position, atom in enumerate(atoms, start=1):
        if not atom.text.strip():
            issues.append(issue(position, "text", atom.source_quote, "text is empty."))
        elif atom.text not in source_text:
            issues.append(issue(position, "text", atom.source_quote,
                                "text is not a verbatim substring of the note."))
    return issues


def check_furniture(atoms: List[Atom]) -> List[dict]:
    """No atom starts with a list number or a label that introduces a list."""
    issues = []
    for position, atom in enumerate(atoms, start=1):
        for field in ("text", "source_quote"):
            value = getattr(atom, field)
            match = LIST_NUMBER_PATTERN.match(value)
            if match:
                issues.append(issue(position, "furniture", value,
                                    f"{field} starts with a list number: {match.group(0).strip()!r}."))
    return issues


def check_overlap(atoms: List[Atom], spans: List[tuple[int, int]]) -> List[dict]:
    """No two atoms quote the same span, and no quote sits inside another."""
    # Compare each quote with the others. If this quote contains another quote,
    # report an issue on this atom. Equal spans also count as overlap.
    issues = []
    for position, (start, end) in enumerate(spans, start=1):
        if start < 0:
            continue
        for other_position, (other_start, other_end) in enumerate(spans, start=1):
            if other_position == position or other_start < 0:
                continue
            if start <= other_start and other_end <= end:
                if (start, end) == (other_start, other_end):
                    message = f"quote is identical to the quote of fact {other_position}."
                elif (end - start) > (other_end - other_start):
                    message = f"quote contains the quote of fact {other_position}."
                else:
                    message = f"quote sits inside the quote of fact {other_position}."
                issues.append(issue(position, "overlap", atoms[position - 1].source_quote, message))
                break
    return issues


def check_assertion_cues(atoms: List[Atom]) -> List[dict]:
    """An absent atom's quote carries the wording that makes it absent."""
    issues = []
    for position, atom in enumerate(atoms, start=1):
        if atom.assertion == "absent" and not NEGATION_CUE_PATTERN.search(atom.source_quote):
            issues.append(issue(position, "assertion cue", atom.source_quote,
                                "absent fact has no negation cue in its quote."))
    return issues


def check_family_owner(atoms: List[Atom]) -> List[dict]:
    """A family atom names its relative, and a patient atom names no relative."""
    issues = []
    for position, atom in enumerate(atoms, start=1):
        names_relative = bool(RELATIVE_PATTERN.search(atom.source_quote))
        names_family = bool(FAMILY_HISTORY_PATTERN.search(atom.source_quote))
        if atom.assertion == "family" and not (names_relative or names_family):
            issues.append(issue(position, "family owner", atom.source_quote,
                                "family fact names no relative and no family history."))
        if atom.assertion != "family" and names_relative:
            issues.append(issue(position, "family owner", atom.source_quote,
                                "quote names a relative but the assertion is not family."))
    return issues


def missing_numbers(source_text: str, atoms: List[Atom]) -> List[str]:
    """Numbers in the note that no atom mentions. A review signal, never a gate."""
    quoted = " ".join(atom.text + " " + atom.source_quote for atom in atoms)
    covered = set(re.findall(NUMBER_PATTERN, quoted))
    return sorted(set(re.findall(NUMBER_PATTERN, source_text)) - covered)


def collect_issues(source_text: str, atoms: List[Atom], spans: List[tuple[int, int]]) -> List[dict]:
    """Run every blocking check and return the combined record."""
    # Collect all kinds of issues here, including ones that can be fixed or dropped.
    # extract_atoms decides which remaining issues must stop the entire note.
    issues: List[dict] = []
    issues.extend(check_quotes(source_text, atoms))
    issues.extend(check_texts(source_text, atoms))
    issues.extend(check_furniture(atoms))
    issues.extend(check_overlap(atoms, spans))
    issues.extend(check_assertion_cues(atoms))
    issues.extend(check_family_owner(atoms))
    return issues


# Checks a repair can fix, or that can be dropped one atom at a time. A quote that is not
# verbatim, or a document that will not parse, makes the whole reply untrustworthy instead.
FIXABLE_CHECKS = frozenset({"furniture", "overlap", "assertion cue", "family owner"})


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
            response_schema=Extraction,
            temperature=0,
        ),
    )
    return response.text or ""


def parse_reply(reply_text: str) -> Optional[Extraction]:
    """Validate a reply against the schema, returning None when it does not parse."""
    try:
        return Extraction.model_validate_json(reply_text)
    except Exception:
        return None


def build_stored_atoms(atoms: List[Atom], spans: List[tuple[int, int]]) -> List[StoredAtom]:
    """Add the identity and the offsets the model is not asked for."""
    stored = []
    for position, (atom, (start, end)) in enumerate(zip(atoms, spans), start=1):
        stored.append(
            StoredAtom(**atom.model_dump(), fact_id=f"F{position:03d}", char_start=start, char_end=end)
        )
    return stored


def correction_prompt(source_text: str, extraction: Optional[Extraction], diagnostics: List[dict]) -> str:
    """Ask for a corrected extraction, naming each problem without restating the note."""
    problems = "\n".join(
        f"- fact {record['fact']}: [{record['check']}] {record['message']}"
        for record in diagnostics
    )
    previous = extraction.model_dump_json() if extraction else "No valid extraction."
    return (
        f"SOURCE:\n{source_text}\n\nPREVIOUS EXTRACTION:\n{previous}\n\n"
        f"PROBLEMS:\n{problems}\n\n"
        "Return a corrected, complete extraction that keeps every fact from the previous "
        "attempt. Every text and source_quote must be a nonempty exact excerpt from SOURCE. "
        "Quote the fact itself, not the section header or list number in front of it. "
        "Preserve the subject, uncertainty, and timing. Do not discard facts to make "
        "validation pass."
    )


def coverage_warnings(first_count: Optional[int], final_count: int) -> List[str]:
    """Name a repair that came back with fewer facts than the first reply."""
    if first_count is None or final_count >= first_count:
        return []
    lost = first_count - final_count
    return [
        f"the final reply has {final_count} atoms, down from {first_count}; "
        f"{lost} fact(s) are not in the extraction."
    ]


def extract_atoms(
    source_text: str,
    model: str = MODEL,
    send: Optional[Callable[[str, str], str]] = None,
) -> ExtractionRecord:
    """Extract atoms from one note, or raise ExtractionValidationError with the record.

    A reply is repaired once. After that, a check that can be attributed to single atoms
    drops those atoms and names them in `review_warnings`, so one bad atom cannot discard
    a whole note. A reply that cannot be trusted at all still fails the run.

    `send` is injected so the tests can run with no API key and no network. Every
    sentinel here is a check in this module, never a judgement the model makes about
    its own output.
    """
    if not source_text.strip():
        raise ExtractorError("Source note must not be blank.")

    # Tests can supply a scripted sender. Normal use calls the live model instead.
    sender = send or live_send
    prompt = f"SOURCE:\n{source_text}"
    first_count = None
    attempts_made = 0
    extraction = None
    atoms: List[Atom] = []
    spans: List[tuple[int, int]] = []
    issues: List[dict] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # There are at most two replies: the first extraction and one correction.
        # Recheck the whole reply because a correction can introduce new problems.
        attempts_made = attempt
        extraction = parse_reply(sender(prompt, model))
        atoms = extraction.atoms if extraction else []

        if extraction is None:
            issues = [issue(0, "reply", "", "the reply is not a valid extraction document.")]
        elif not atoms:
            issues = [issue(0, "empty", "", "the reply contained no atoms.")]
        else:
            spans = assign_spans(source_text, atoms)
            issues = collect_issues(source_text, atoms, spans)
            if first_count is None:
                first_count = len(atoms)
            if not issues:
                return ExtractionRecord(
                    atoms=build_stored_atoms(atoms, spans),
                    review_warnings=coverage_warnings(first_count, len(atoms)),
                )

        if attempt < MAX_ATTEMPTS:
            prompt = correction_prompt(source_text, extraction, issues)

    # A made-up quote or non-copied fact text prevents us from trusting the reply.
    # Other named issues can be handled one atom at a time after correction fails.
    blocking = [item for item in issues if item["check"] not in FIXABLE_CHECKS]
    if blocking or not atoms:
        raise ExtractionValidationError(diagnostics=issues, attempts=attempts_made)

    # Assign IDs before removing bad atoms. Surviving IDs can therefore have gaps,
    # which keeps each surviving atom tied to its position in this checked reply.
    stored = build_stored_atoms(atoms, spans)
    dropped_positions = {item["fact"] for item in issues}
    kept = [atom for position, atom in enumerate(stored, start=1) if position not in dropped_positions]

    # Dropping facts is a loss of coverage, so name every drop for human review.
    # These checks verify selected rules, not whether every clinical fact was found.
    review_warnings = []
    for item in issues:
        fact_id = f"F{item['fact']:03d}"
        review_warnings.append(
            f"{fact_id} dropped after repair: {item['check']} - {item['message']}"
        )
    review_warnings.extend(coverage_warnings(first_count, len(kept)))

    if not kept:
        raise ExtractionValidationError(
            diagnostics=issues + [issue(0, "empty", "", "every atom was dropped after repair.")],
            attempts=attempts_made,
        )

    return ExtractionRecord(atoms=kept, review_warnings=review_warnings)


# Hand off the file
def save_extraction(doc_id: str, record: ExtractionRecord, out_dir: Path) -> Path:
    """Write one document's atoms to <out_dir>/<doc_id>.json and return the path."""
    if not doc_id or Path(doc_id).name != doc_id or doc_id in {".", ".."}:
        raise ExtractorError("doc_id must be a non-empty filename stem")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{doc_id}.json"
    payload = {
        "doc_id": doc_id,
        "atoms": [atom.model_dump(mode="json") for atom in record.atoms],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_extraction(path: Path) -> dict:
    """Read a saved extraction back into a dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExtractorError("Extraction file must contain a JSON object")
    return payload