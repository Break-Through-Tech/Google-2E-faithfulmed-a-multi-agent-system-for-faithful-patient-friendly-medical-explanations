"""Extractor agent: clinical text -> typed atoms backed by verbatim quotes."""
import json
import re
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

MODEL = "gemini-3.5-flash-lite"
ALIAS_PATH = Path(__file__).resolve().parent.parent / "medlineplus_aliases.json"


# Talking to Gemini
def ask(prompt: str) -> str:
    """Send a prompt to Gemini and return the reply text."""
    from google import genai

    client = genai.Client()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


# Define what an atom is
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

INSTRUCTION = """
ROLE & OVERVIEW
You are a clinical information extraction agent. Extract every clinically meaningful fact from the input, with one fact per atom.
Each atom should have a type and a corresponding source_quote that is copied EXACTLY from the source.
Never ever infer, add, or correct anything that isn't written in the source.
Keep any and all abbreviations as written - do not expand upon or rename them.
Text uses only words from the source. An absence or possibility of a disease is not a diagnosis.
Your task is extraction, not diagnosis, interpretation, summarization, or medical advice.

CORE RULES
1. Extract each distinct clinically meaningful fact as a separate atom.
2. Use only information supported by the source text. Never invent, infer,
   assume, or add clinical facts that are not explicitly stated.
3. Preserve the source's meaning, including negation, uncertainty, timing,
   medication status, and whether a fact concerns the patient or a family member.
4. Keep abbreviations as written. Do not expand, rename, correct, or silently
   reinterpret abbreviations.
5. Keep each atom's text concise and faithful to the source. Do not add
   information merely to make the sentence sound more natural.
6. Every atom must have a source_quote copied exactly from one contiguous
   portion of the source text. Do not paraphrase the quote, combine separate
   excerpts, or insert words that are not present in the source.
    Include the complete clause that establishes the fact and assertion. For
    "The patient reports a mild headache," use that complete phrase, not only
    "mild headache." For "The clinician diagnosed migraine," include
    "diagnosed," not only the condition name.
    For an absent fact, quote the denial too: for "The patient denies nausea,"
    quote that phrase, not just "nausea."
7. If the source does not provide enough information to determine a fact,
   do not fill in the missing information from medical knowledge.
8. Do not omit a clinically meaningful fact merely because it is negative,
   uncertain, historical, or related to family history.

FIELD: text
Copy the shortest complete source clause that identifies the fact and makes
its subject and assertion clear. Do not return a bare clinical term when the
surrounding words identify who has it or how it is asserted.
Use the same complete excerpt as source_quote. Do not add words not present.
Do not turn a negative statement into a positive one.
Do not turn a possible condition into a confirmed condition.
Do not add a patient, diagnosis, medication, unit, date, or other detail
unless it is supported by the source.
Include relevant numbers and units exactly as documented.

FIELD: type
Choose exactly one of the permitted types:
- diagnosis: a disease, condition, or clinical finding.
- medication: a named medication or drug, including information about its
  use or discontinuation when stated.
- lab_value: a test result, measurement, or clinical reading, including
  the available test name, value, and unit.
- procedure: a clinical procedure or intervention explicitly described
  in the source.
- instruction: advice, directions, or instructions given to the patient.

Choose the most appropriate type based on the source. Do not invent a type
of fact that the source does not support. Do not classify a medication
as current merely because its name appears in the source.

FIELD: assertion
Choose exactly one permitted value:
- present: the source affirms the fact as current or applicable.
- absent: the source explicitly denies the fact or states that it is absent.
- possible: the source expresses uncertainty, suspicion, or possibility.
- past: the source explicitly describes a historical fact that is no longer
  current.
- family: the fact concerns a family member's medical history, not the
  patient's own medical history.

Apply these rules:
- Explicit negation takes precedence over an otherwise affirmative mention
  of a condition. For example, "No history of diabetes" must not be marked
  as present.
- Explicit uncertainty must be preserved. For example, "Possible pneumonia"
  must not be treated as a confirmed diagnosis.
- Use past for a fact explicitly described as historical and no longer
  current. Do not assume that a condition has resolved just because it
  occurred in the past.
- Use family when the fact concerns a family member, rather than the patient.
- If a fact concerns family history, use family even if the condition
  occurred in the relative's past. Do not attribute the relative's condition
  to the patient.
- If a medication is explicitly stated to have been stopped, discontinued,
  or ceased, do not represent it as a current medication. Use past when
  the medication's use is historical and no longer current.
- Do not assume a medication is current or discontinued if the source
  does not establish its status.
- When the source does not establish a status, do not invent certainty.
  Use the closest supported value only when justified by the source.

FIELD: source_quote
Copy the shortest complete clause that adequately supports the atom.
The quote must appear verbatim in the source, including its original
abbreviations, spelling, capitalization, and punctuation.
The quote must identify the subject and support the assertion as well as the
fact. Include wording such as "reports" for a reported symptom, "diagnosed"
for a diagnosis, and "denies" or "no" for an absent fact. Never quote only
the clinical term when surrounding words establish its context.
Do not use a paraphrase as a quote.
Do not combine words from different parts of the source into one quote.
If a fact cannot be supported by a suitable source excerpt, do not create
an atom for that fact.

GRANULARITY
Extract one distinct fact per atom.
Keep a test or measurement together with its documented value and unit
when they belong to the same result.
For example, a blood pressure reading of "120/80" is one measurement,
not two separate atoms for 120 and 80.
If the source documents multiple distinct measurements, extract each
measurement separately.
Keep a medication's name and its documented dose or schedule together
when they describe the same medication instruction.
Do not split a single fact into multiple atoms unless the source contains
distinct clinically meaningful facts.

NEGATION, UNCERTAINTY, AND HISTORY
Preserve clinically important wording such as "no", "denies", "possible",
"suspected", "history of", "discontinued", and "family history".
Do not infer that a patient has a condition simply because the condition
is mentioned.
Do not infer that a condition is absent merely because it is not mentioned.
Do not convert a family member's condition into the patient's condition.
Do not convert a suspected diagnosis into a confirmed diagnosis.
Do not infer that a past condition has resolved unless the source says so.

COMPLETENESS
Extract all clinically meaningful facts that fall within the permitted
types. Preserve all documented numerical values, relevant units, and
important contextual details.
Do not add unrelated information or duplicate the same fact without reason.

OUTPUT
Return only the structured extraction matching the supplied schema.
Do not include explanations, commentary, markdown, or additional fields.
"""
# It should tell the model to:
#   - extract every clinically meaningful fact, one fact per atom
#   - give each atom a type and a source_quote copied EXACTLY from the source
#   - never infer, add, or correct anything not written in the source
#   - keep abbreviations as written (don't expand or rename them)

def extract_atoms(source_text: str, model: str = MODEL) -> Extraction:
    from google import genai
    from google.genai import types

    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents= f"SOURCE:\n{source_text}",
        config=types.GenerateContentConfig(
            system_instruction= INSTRUCTION,
            response_mime_type="application/json",
            response_schema= Extraction,
            temperature=0,
        ),
    )
    if resp.parsed is None:
        raise ValueError(f"Gemini didn't return a parsed result. Raw reply: {resp.text}")
    return resp.parsed


# Are the atoms real?
def check_quotes(source_text: str, extraction: Extraction) -> List[Atom]:
    """Return atoms whose source quote is not an exact substring of the source."""
    return [atom for atom in extraction.atoms if atom.source_quote not in source_text]


def check_atom_texts(source_text: str, extraction: Extraction) -> List[Atom]:
    """Return atoms whose text is not copied verbatim from the source."""
    return [atom for atom in extraction.atoms if atom.text not in source_text]


NEGATION_CUE_PATTERN = re.compile(
    r"\b(?:no|not|never|none|den(?:y|ies|ied)|without|negative|absent|free of)\b",
    flags=re.IGNORECASE,
)


def check_assertion_cues(extraction: Extraction) -> List[Atom]:
    """Return absent atoms whose quotes omit an explicit negation cue."""
    return [
        atom for atom in extraction.atoms
        if atom.assertion == "absent" and not NEGATION_CUE_PATTERN.search(atom.source_quote)
    ]


SYMPTOM_CUE_PATTERN = re.compile(
    r"\b(?:report(?:s|ed)?|complain(?:s|ed)?|experienc(?:e|es|ed)|"
    r"endorse(?:s|d)?|present(?:s|ed) with|has|have)\b",
    flags=re.IGNORECASE,
)
DIAGNOSIS_CUE_PATTERN = re.compile(
    r"\b(?:diagnos(?:e|ed|is)|impression|assessment|confirmed|consistent with)\b",
    flags=re.IGNORECASE,
)


def check_quote_context(extraction: Extraction) -> List[Atom]:
    """Return present symptom/diagnosis atoms whose quotes omit context cues."""
    invalid_atoms = []
    for atom in extraction.atoms:
        if atom.assertion != "present":
            continue
        if atom.type == "symptom" and not SYMPTOM_CUE_PATTERN.search(atom.source_quote):
            invalid_atoms.append(atom)
        elif atom.type == "diagnosis" and not DIAGNOSIS_CUE_PATTERN.search(atom.source_quote):
            invalid_atoms.append(atom)
    return invalid_atoms


# were facts dropped?
NUMBER_PATTERN = r"\d+(?:\.\d+)?"   # matches 10 and 8.2


def missing_numbers(source_text: str, extraction: Extraction) -> List[str]:
    """Returns numbers in the source that no atom mentions."""
    covered = set(re.findall(NUMBER_PATTERN, " ".join(atom.text + " " + atom.source_quote for atom in extraction.atoms)))
    return sorted(set(re.findall(NUMBER_PATTERN, source_text)) - covered)


# Attaching relevant aliases or abbreviations
class LinkedAtom(Atom):
    topic: Optional[str] = None


def load_aliases() -> dict:
    """Load and validate the lowercase alias-to-topic mapping."""
    aliases = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
    if not isinstance(aliases, dict) or any(
        not isinstance(alias, str) or not isinstance(topic, str)
        for alias, topic in aliases.items()
    ):
        raise ValueError(f"Alias file must contain a JSON object of strings: {ALIAS_PATH}")
    return aliases


def find_topic(text: str, aliases: dict) -> Optional[str]:
    """Return the MedlinePlus topic for some text, or None."""
    words = re.findall(r"[a-z0-9\-]+", text.lower())
    for phrase_length in range(4, 0, -1):
        for start in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[start:start + phrase_length])
            if phrase in aliases:
                return aliases[phrase]
    return None


def attach_topics(extraction: Extraction, aliases: dict) -> List[LinkedAtom]:
    """Return copies of the extracted atoms with matching MedlinePlus topics."""
    return [
        LinkedAtom(**atom.model_dump(), topic=find_topic(atom.text, aliases))
        for atom in extraction.atoms
    ]

# Hand off the file
def save_extraction(doc_id: str, atoms: List[Atom], out_dir: Path) -> Path:
    """Write one document's atoms to <out_dir>/<doc_id>.json and return the path."""
    if not doc_id or Path(doc_id).name != doc_id or doc_id in {".", ".."}:
        raise ValueError("doc_id must be a non-empty filename stem")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{doc_id}.json"
    payload = {
        "doc_id": doc_id,
        "atoms": [atom.model_dump(mode="json") for atom in atoms],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_extraction(path: Path) -> dict:
    """Read a saved extraction back into a dict."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Extraction file must contain a JSON object")
    return payload