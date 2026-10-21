"""Simplifier agent: extracted facts plus MedlinePlus grounding -> patient-friendly draft."""
import json
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from faithfulmed.retrieval_utils import format_passages, retrieve_passages

MODEL = "gemini-3.5-flash-lite"
TARGET_GRADE = 8


# Define what the model returns, and what we hand back
class FactRewrite(BaseModel):
    fact_number: int
    text: str


class SimplifierReply(BaseModel):
    """What the model returns: the prose, plus which supplied blocks it used."""

    fact_rewrites: List[FactRewrite] = Field(
        description="Exactly one plain-language rewrite per numbered fact, preserving its owner and status."
    )
    sources_used: List[str] = Field(
        description="The title of every MedlinePlus reference block the draft actually used."
    )
    glossary_terms_used: List[str] = Field(
        description="Every term that was explained from a MedlinePlus reference block."
    )
    ungrounded_terms: List[str] = Field(
        description="Every term that had no reference block and was explained from the facts alone."
    )


class SourcePassage(BaseModel):
    title: str
    url: Optional[str] = None
    section: Optional[str] = None
    text: str


class SimplifiedOutput(BaseModel):
    simplified_text: str
    fact_rewrites: List[FactRewrite] = Field(default_factory=list)
    sources_used: List[str]
    source_metadata: List[SourcePassage] = Field(
        description="The full text of every supplied passage the draft used, filled in by the caller."
    )
    glossary_terms_used: List[str]
    ungrounded_terms: List[str]
    atoms_covered: List[int]


INSTRUCTION = """
ROLE
You are a patient-education writer. You turn a numbered list of clinical facts about one
patient into a short explanation that the patient can read and understand.
You are not a clinician: you do not diagnose, advise, prescribe, or reassure.

THE ONLY SOURCE OF TRUTH
The numbered facts are the only source of truth about this patient.
Never add a fact, diagnosis, symptom, medicine, dose, date, number, unit, or cause that the
facts do not state. Never correct, soften, or upgrade what the facts say.
Never drop a fact because it is negative, uncertain, historical, or about a family member.

READING LEVEL
Write at an 8th-grade reading level or below.
- Keep sentences short: one idea per sentence, about 15 words or fewer.
- Use everyday words. Explain a medical term the first time it appears, in the same sentence.
- Use active voice, and address the patient as "you" only when the fact is about the patient.
- Keep paragraphs short: two to four sentences each.
- Do not use tables or long headings.

REFERENCE MATERIAL
Reference blocks are MedlinePlus definitions supplied for this draft. They explain words.
They never describe this patient.
- Use a definition only to explain a term that the facts already contain.
- Do not copy clinical detail from a definition into this patient's situation.
- Do not use a definition to fill a gap that the facts leave.
- Do not import numerical thresholds, frequencies, durations, or advice from references.
    This includes numbers written as words, such as "three bowel movements a week".
- If a term has no reference block, explain it in the plainest words the facts support,
  invent no definition, and list that term in ungrounded_terms.
- List every term you explained from a block in glossary_terms_used, and list the title of each
  block you actually used in sources_used, copied exactly.

PRESERVE MEANING
- Keep negations: a denied symptom stays denied.
- Keep uncertainty: "possible" or "suspected" never becomes confirmed.
- Keep timing: a past fact or a stopped medicine stays in the past.
- Keep the owner: a family member's condition never becomes the patient's condition.
- Read each supporting quote for the subject and uncertainty. A father's possible
    condition must remain the father's possible condition, never "you have" it.
- Keep numbers and units exactly as written. Do not convert, round, or restate them loosely.

COVERAGE AND OUTPUT
Return exactly one fact_rewrites entry for each numbered fact. Copy its number into
fact_number and put its patient-friendly explanation in text. Do not merge or omit entries.
Each entry must stand alone: name the relative instead of depending on an earlier sentence.
Keep improvement, resolution, and timing explicit, not just the fact that a diet started.
Explain terms in that entry only when a supplied reference supports the explanation.
Do not list a reference or glossary term as used unless you actually explained a term from it.
Return only the structured output that matches the schema: no markdown, no commentary.
"""


def distinct_texts(atoms: List[dict]) -> List[str]:
    """Return each fact's text once, keeping the order of the fact list."""
    texts = []
    for atom in atoms:
        text = atom["text"].strip()
        if text not in texts:
            texts.append(text)
    return texts


def build_grounding(atoms: List[dict]) -> dict:
    """Look up MedlinePlus once per distinct fact and keep the passages, not just their text."""
    grounding = {}
    for text in distinct_texts(atoms):
        passages = retrieve_passages(text)
        if passages:
            grounding[text] = passages
    return grounding


def format_atoms(atoms: List[dict]) -> str:
    """Number the facts so the draft can report which ones it covered."""
    lines = []
    for position, atom in enumerate(atoms, start=1):
        lines.append(f"{position}. [{atom['type']}, {atom['assertion']}] {atom['text']}")
        if atom.get("source_quote"):
            lines.append(f"   Supporting quote: {atom['source_quote']}")
    return "\n".join(lines)


def format_grounding(grounding: dict) -> str:
    """Join the MedlinePlus blocks into one reference section, each labelled with its term."""
    if not grounding:
        return "No reference blocks were found for these facts."
    blocks = []
    for text, passages in grounding.items():
        blocks.append(f"Term: {text}\n{format_passages(passages)}")
    return "\n\n---\n\n".join(blocks)


def collect_sources(grounding: dict, sources_used: List[str]) -> List[SourcePassage]:
    """Return the exact passages behind the titles the draft used, once each."""
    wanted = {title.strip().lower() for title in sources_used}
    sources = []
    seen = set()
    for passages in grounding.values():
        for passage in passages:
            if passage["title"].strip().lower() not in wanted:
                continue
            key = (passage["title"], passage["text"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourcePassage(
                    title=passage["title"],
                    url=passage["url"],
                    section=passage["section"],
                    text=passage["text"],
                )
            )
    return sources


def rewrite_context_issues(atom: dict, text: str) -> List[str]:
    """Flag lost explicit context in one fact's rewrite, without claiming semantic verification."""
    source = atom.get("source_quote") or atom["text"]
    issues = []
    if atom["assertion"] == "absent":
        if not re.search(r"\b(?:no|not|none|never|without|denies|resolved|gone|free of)\b", text, flags=re.IGNORECASE):
            issues.append("lost negation")
    if atom["assertion"] == "family":
        relatives = re.findall(
            r"\b(?:mother|father|sister|brother|daughter|son|wife|husband|grandmother|grandfather)\b",
            source, flags=re.IGNORECASE,
        )
        if not relatives:
            relatives = ["family"]
        if any(not re.search(r"\b" + relative + r"\b", text, flags=re.IGNORECASE) for relative in relatives):
            issues.append("lost family ownership")
        if re.search(r"\byou (?:have|had|may have|might have)\b", text, flags=re.IGNORECASE):
            issues.append("family fact attributed to patient")
    uncertainty = r"\b(?:possible|possibly|questionable|suspected|uncertain|may|might)\b"
    if atom["assertion"] == "possible" or re.search(uncertainty, source, flags=re.IGNORECASE):
        if not re.search(uncertainty, text, flags=re.IGNORECASE):
            issues.append("lost uncertainty")
    if re.search(r"\bsymptoms?\b", source, flags=re.IGNORECASE):
        if re.search(r"\b(?:improved|resolved|none of these symptoms)\b", source, flags=re.IGNORECASE):
            if not re.search(r"\b(?:improved|better|resolved|gone|no longer|none|stopped)\b", text, flags=re.IGNORECASE):
                issues.append("lost symptom improvement or resolution")
    if numeric_values(text) - numeric_values(atom["text"]):
        issues.append("unsupported quantities in fact rewrite")
    return issues


def simplify(
    atoms: List[dict],
    grounding: Optional[dict] = None,
    model: str = MODEL,
) -> SimplifiedOutput:
    """Write the patient-friendly draft for a numbered fact list, grounded in MedlinePlus."""
    if not atoms:
        raise ValueError("No facts to simplify. Run the Extractor first, or pass the atoms in.")

    from google import genai
    from google.genai import types

    if grounding is None:
        grounding = build_grounding(atoms)

    prompt = (
        f"FACTS:\n{format_atoms(atoms)}\n\n"
        f"REFERENCE MATERIAL:\n{format_grounding(grounding)}"
    )

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SimplifierReply,
            temperature=0,
        ),
    )

    if response.parsed is None:
        raise ValueError("Simplifier did not return a structured draft.")

    reply = response.parsed
    rewrites = sorted(reply.fact_rewrites, key=lambda rewrite: rewrite.fact_number)
    if [rewrite.fact_number for rewrite in rewrites] != list(range(1, len(atoms) + 1)):
        raise ValueError("Simplifier validation failed: invalid reported coverage")
    for rewrite, atom in zip(rewrites, atoms):
        issues = rewrite_context_issues(atom, rewrite.text)
        if not rewrite.text.strip():
            issues.append("blank fact rewrite")
        if issues:
            raise ValueError(
                f"Simplifier validation failed: fact {rewrite.fact_number}: " + ", ".join(issues)
            )
    paragraphs = []
    for start in range(0, len(rewrites), 3):
        paragraphs.append(" ".join(rewrite.text.strip() for rewrite in rewrites[start:start + 3]))
    draft = SimplifiedOutput(
        simplified_text="\n\n".join(paragraphs),
        fact_rewrites=rewrites,
        sources_used=reply.sources_used,
        source_metadata=collect_sources(grounding, reply.sources_used),
        glossary_terms_used=reply.glossary_terms_used,
        ungrounded_terms=reply.ungrounded_terms,
        atoms_covered=[rewrite.fact_number for rewrite in rewrites],
    )
    issues = []
    if not draft.simplified_text.strip():
        issues.append("blank text")
    if check_coverage(draft, len(atoms)) or missing_atoms(draft, len(atoms)):
        issues.append("invalid reported coverage")
    if check_citations(draft, grounding):
        issues.append("unknown citation titles")
    if invented_numbers(draft, atoms):
        issues.append("unsupported quantities")
    if issues:
        raise ValueError("Simplifier validation failed: " + ", ".join(issues))
    return draft


# Are the citations and the coverage real?
def check_coverage(draft: SimplifiedOutput, atom_count: int) -> List[int]:
    """Return reported fact numbers that are not in the fact list."""
    return [
        position for position in draft.atoms_covered
        if position < 1 or position > atom_count
    ]


def missing_atoms(draft: SimplifiedOutput, atom_count: int) -> List[int]:
    """Return the numbers of facts that the draft did not report as covered."""
    return [
        position for position in range(1, atom_count + 1)
        if position not in draft.atoms_covered
    ]


def check_citations(draft: SimplifiedOutput, grounding: dict) -> List[str]:
    """Return cited titles that appear in none of the supplied passages."""
    supplied = {
        passage["title"].strip().lower()
        for passages in grounding.values()
        for passage in passages
    }
    return [
        title for title in draft.sources_used
        if title.strip().lower() not in supplied
    ]


NUMBER_PATTERN = r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?"


def numeric_values(text: str) -> set[str]:
    """Recognize digits and English whole numbers from zero through ninety-nine.

    This is a quantity warning, not proof that a number belongs to the right fact.
    Larger written numbers, fractions, and ordinals still need review.
    """
    names = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    values = {name: str(position) for position, name in enumerate(names)}
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    for position, name in enumerate(tens, start=2):
        values[name] = str(position * 10)
        for unit in range(1, 10):
            values[f"{name} {names[unit]}"] = str(position * 10 + unit)
    alternatives = sorted(values, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(name).replace(r"\ ", r"[ -]") for name in alternatives) + r")\b"
    found = set(re.findall(NUMBER_PATTERN, text))
    for match in re.finditer(pattern, text.lower()):
        found.add(values[match.group().replace("-", " ")])
    return found


def invented_numbers(draft: SimplifiedOutput, atoms: List[dict]) -> List[str]:
    """Return numbers that appear in the draft and in none of the facts."""
    fact_numbers = numeric_values(" ".join(atom["text"] for atom in atoms))
    return sorted(numeric_values(draft.simplified_text) - fact_numbers)


# Hand off the file
def save_draft(doc_id: str, draft: SimplifiedOutput, out_dir: Path) -> Path:
    """Write one document's draft to <out_dir>/<doc_id>.draft.json and return the path."""
    if not doc_id or Path(doc_id).name != doc_id or doc_id in {".", ".."}:
        raise ValueError("doc_id must be a non-empty filename stem")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{doc_id}.draft.json"
    output_path.write_text(
        json.dumps(draft.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_draft(path: Path) -> dict:
    """Read a saved draft back into a dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Draft file must contain a JSON object")
    return payload
