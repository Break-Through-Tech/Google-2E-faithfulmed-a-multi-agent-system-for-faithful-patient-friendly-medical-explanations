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
class SimplifierReply(BaseModel):
    """What the model returns: the prose, plus which supplied blocks it used."""

    simplified_text: str = Field(
        description="The patient explanation, in short paragraphs at an 8th-grade reading level."
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
    atoms_covered: List[int] = Field(
        description="The number of every fact the draft states, using the numbers from the fact list."
    )


class SourcePassage(BaseModel):
    title: str
    url: Optional[str] = None
    section: Optional[str] = None
    text: str


class SimplifiedOutput(BaseModel):
    simplified_text: str
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
- If a term has no reference block, explain it in the plainest words the facts support,
  invent no definition, and list that term in ungrounded_terms.
- List every term you explained from a block in glossary_terms_used, and list the title of each
  block you actually used in sources_used, copied exactly.

PRESERVE MEANING
- Keep negations: a denied symptom stays denied.
- Keep uncertainty: "possible" or "suspected" never becomes confirmed.
- Keep timing: a past fact or a stopped medicine stays in the past.
- Keep the owner: a family member's condition never becomes the patient's condition.
- Keep numbers and units exactly as written. Do not convert, round, or restate them loosely.

COVERAGE AND OUTPUT
State every fact from the list in the draft, and report the number of each fact you stated in
atoms_covered. Those numbers come from the fact list, not from the fact text.
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
        raise ValueError(f"Gemini didn't return a parsed result. Raw reply: {response.text}")

    reply = response.parsed
    return SimplifiedOutput(
        simplified_text=reply.simplified_text,
        sources_used=reply.sources_used,
        source_metadata=collect_sources(grounding, reply.sources_used),
        glossary_terms_used=reply.glossary_terms_used,
        ungrounded_terms=reply.ungrounded_terms,
        atoms_covered=reply.atoms_covered,
    )


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


NUMBER_PATTERN = r"\d+(?:[.,]\d+)?"


def invented_numbers(draft: SimplifiedOutput, atoms: List[dict]) -> List[str]:
    """Return numbers that appear in the draft and in none of the facts."""
    fact_numbers = set(re.findall(NUMBER_PATTERN, " ".join(atom["text"] for atom in atoms)))
    draft_numbers = re.findall(NUMBER_PATTERN, draft.simplified_text)
    return sorted({number for number in draft_numbers if number not in fact_numbers})


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
