"""Extractor agent: clinical text -> typed atoms backed by verbatim quotes."""
import json
import re
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Started with 3.8 but kept getting 503s and eventually a 429 limit... 3.6 hasn't been much better on the 503 end...
MODEL = "gemini-3.6-flash"
ALIAS_PATH = Path(__file__).resolve().parent.parent / "medlineplus_aliases.json"


# ---- Step 1: talking to Gemini -------------------------------------------
def ask(prompt: str) -> str:
    """Send a prompt to Gemini and return the reply text."""
    client = genai.Client()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


# ---- Step 2: what an atom is ---------------------------------------------
class Atom(BaseModel):
    text: str = Field(
        description="A concise statement of one clinically meaningful fact explicitly supported by the source.")
    type: Literal["diagnosis", "medication", "lab_value", "procedure", "instruction"] = Field(
        description=
            "The category of the extracted fact: diagnosis for a disease, condition, or clinical finding;" \
            "medication for a drug or its use; lab_value for a test result or clinical measurement;" \
            "procedure for a clinical procedure or intervention; instruction for advice or directions given to the patient."
        )
    assertion: Literal["present", "absent", "possible", "past", "family"] = Field(
        description=
            "The assertion status of the fact: present for an affirmed current fact; " \
            "absent for an explicitly denied or absent fact; possible for an uncertain or suspected fact;" \
            "past for a historical fact that is no longer current; family for a fact about a family member's medical history."
        )
    source_quote: str = Field(description="The exact words copied from the source text that supports the extracted fact.")


class Extraction(BaseModel):
    atoms: List[Atom]


# ---- Step 3: the instruction ---------------------------------------------
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
7. If the source does not provide enough information to determine a fact,
   do not fill in the missing information from medical knowledge.
8. Do not omit a clinically meaningful fact merely because it is negative,
   uncertain, historical, or related to family history.

FIELD: text
Write one concise statement expressing the extracted fact.
Preserve the original meaning and wording as closely as possible.
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
Copy the shortest contiguous excerpt that adequately supports the atom.
The quote must appear verbatim in the source, including its original
abbreviations, spelling, capitalization, and punctuation.
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


# ---- Step 4: structured extraction ---------------------------------------
def extract_atoms(source_text: str, model: str = MODEL) -> Extraction:
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


# ---- Step 5: are the atoms real? -----------------------------------------
def _norm(s: str) -> str:
    """Makes two strings comparable by collapsing whitespace and lowercasing everything."""
    return " ".join(s.split().lower())


def check_quotes(source_text: str, extraction: Extraction) -> List[Atom]:
    """Return the atoms whose quote is NOT found in the source."""
    source = _norm(source_text)
    return [atom for atom in extraction.atoms if _norm(atom.source_quote) not in source]

# ---- Step 6: were facts dropped? -----------------------------------------
NUMBER_PATTERN = r"\d+(?:\.\d+)?"   # matches 10 and 8.2


def missing_numbers(source_text: str, extraction: Extraction) -> List[str]:
    """Returns numbers in the source that no atom mentions."""
    covered = set(re.findall(NUMBER_PATTERN, " ".join(atom.text + " " + atom.source_quote for atom in extraction.atoms)))
    return sorted(set(re.findall(NUMBER_PATTERN, source_text)) - covered)


# ---- Step 7: Attaching relevant aliases or abbreviations ------------------------------------------
class LinkedAtom(Atom):
    topic: Optional[str] = None


def load_aliases() -> dict:
    # TODO: open ALIAS_PATH and return the parsed JSON.
    raise NotImplementedError


def find_topic(text: str, aliases: dict) -> Optional[str]:
    """Return the MedlinePlus topic for some text, or None."""
    # TODO, in this order:
    #   1. words = lowercase the text, then pull out words with
    #      re.findall(r"[a-z0-9\-]+", ...)
    #   2. for n from 4 down to 1 (longest phrase first):
    #        for each start position i: phrase = " ".join(words[i:i+n])
    #          if phrase is a key in aliases: return aliases[phrase]
    #   3. nothing matched -> return None
    raise NotImplementedError


def attach_topics(extraction: Extraction, aliases: dict) -> List[LinkedAtom]:
    # TODO: for each atom, build a LinkedAtom with the same fields plus
    # topic=find_topic(atom.text, aliases).
    raise NotImplementedError

# ---- Step 8: hand-off file ------------------------------------------------
def save_extraction(doc_id: str, atoms: List[Atom], out_dir: Path) -> Path:
    """Write one document's atoms to <out_dir>/<doc_id>.json and return the path."""
    # TODO:
    #   1. make sure out_dir exists (Path.mkdir with parents=True, exist_ok=True)
    #   2. build a dict: {"doc_id": doc_id, "atoms": [one dict per atom]}
    #   3. write it to out_dir / f"{doc_id}.json" as indented JSON
    raise NotImplementedError


def load_extraction(path: Path) -> dict:
    """Read a saved extraction back into a dict."""
    # TODO: open the path and return the parsed JSON.
    raise NotImplementedError