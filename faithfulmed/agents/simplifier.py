"""Simplifier agent: extracted atoms plus MedlinePlus grounding -> patient-friendly draft."""
import re
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from faithfulmed.agents.extractor import (
    NEGATION_CUE_PATTERN,
    NUMBER_PATTERN,
    RELATIVE_PATTERN,
)
from faithfulmed.retrieval_utils import format_passages, retrieve_passages

# Reading path: build_grounding finds reference passages, then simplify writes and
# checks a draft. The small helper functions below handle each part of that process.
# Grounding means reference text for explaining terms, not new facts about the patient.
MODEL = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 2
PASSAGES_PER_FACT = 3

UNCERTAINTY_CUE_PATTERN = re.compile(
    r"\b(?:possible|possibly|may|might|could|suspected|suspicion|questionable|"
    r"unclear|uncertain|not sure|likely|appears)\b",
    flags=re.IGNORECASE,
)
PAST_CUE_PATTERN = re.compile(
    r"\b(?:was|were|had|has had|previously|past|history|earlier|ago|used to|"
    r"no longer|in \d{4})\b",
    flags=re.IGNORECASE,
)


class SimplifierError(Exception):
    """A failure the Simplifier can name, instead of a raw SDK or validation traceback."""


class FactRewrite(BaseModel):
    """One plain-language rewrite of one fact, addressed by the fact's id."""

    fact_id: str = Field(description="The fact_id copied exactly from the FACTS block.")
    text: str = Field(
        description=(
            "The patient-friendly wording for this fact. It must stand alone, keep the "
            "subject, assertion, numbers and units, and add nothing the fact does not say."
        )
    )
    ungrounded_terms: List[str] = Field(
        default_factory=list,
        description="Terms explained without a reference block, so no definition was invented.",
    )


class SimplifierReply(BaseModel):
    """What the model returns: one rewrite per fact, plus the reference titles it used."""

    fact_rewrites: List[FactRewrite]
    sources_used: List[str] = Field(
        default_factory=list,
        description="The exact title of every reference block the draft actually used.",
    )


class SourcePassage(BaseModel):
    """A MedlinePlus passage copied from the store, not from the model."""

    title: str
    url: Optional[str] = None
    section: Optional[str] = None
    text: str


class SimplifiedOutput(BaseModel):
    """What simplify() returns: the prose, its provenance, and what needs review."""

    simplified_text: str
    source_metadata: List[SourcePassage]
    atoms_covered: List[int] = Field(
        default_factory=list,
        description="One-based positions of the facts that appear in the prose, for the Verifier.",
    )
    fact_ids_covered: List[str] = Field(
        default_factory=list,
        description="The same facts by id, so a draft can be traced back to an extraction.",
    )
    review_warnings: List[str] = Field(default_factory=list)


INSTRUCTION = """
ROLE
You explain one patient's clinical facts in plain language that the patient can read.
You are not a clinician: you do not diagnose, advise, prescribe, or reassure.

THE ONLY SOURCE OF TRUTH
The facts below are the only source of truth about this patient.
Never add a fact, symptom, medicine, dose, date, number, unit, or cause the facts do not state.
Never soften, upgrade, or correct what a fact says.
Never drop a fact because it is negative, uncertain, historical, or about a family member.

REFERENCE BLOCKS
Reference blocks are MedlinePlus definitions supplied for this draft. They explain words.
They never describe this patient.
- Use a definition only to explain a term that the fact already contains.
- Never copy a number, threshold, frequency, duration, dose, or piece of advice from a block.
- If a term has no block, explain it in the plainest words the facts support, invent no
  definition, and list the term in that entry's ungrounded_terms.

KEEP THE MEANING
- A denied symptom stays denied.
- "possible" or "suspected" never becomes certain.
- A past fact or a stopped medicine stays in the past.
- A relative's condition never becomes the patient's condition.
- Keep numbers and units exactly as written. Do not convert, round, or restate them.

WRITING
- Eighth-grade reading level or below: short sentences, everyday words, active voice.
- Address the patient as "you" only for that patient's own facts; name the relative otherwise.
- Explain a medical term in the same sentence where it first appears.
- Each entry must stand alone. Do not rely on an earlier entry for the subject.

OUTPUT
Return one fact_rewrites entry for every fact, copying its fact_id exactly.
List in sources_used the exact title of every reference block you actually used.
Return only the structured output that matches the schema: no markdown, no commentary.
"""


def simplify_error(message: str) -> SimplifierError:
    """Build the module's error with a message that carries no patient text."""
    return SimplifierError(message)


def normalize_atoms(atoms) -> List[dict]:
    """Accept atoms as dicts from session state or as model objects, and index them."""
    normalized = []
    for position, atom in enumerate(atoms, start=1):
        # ADK state holds dictionaries, while direct callers may pass Pydantic objects.
        # Give both forms the same fields before any later function uses them.
        data = atom.model_dump() if hasattr(atom, "model_dump") else dict(atom)
        # position is this fact's place in the input list, starting at 1.
        # fact_id is its name, such as F007. After extraction drops, these can differ.
        normalized.append(
            {
                "position": position,
                "fact_id": str(data.get("fact_id") or f"F{position:03d}"),
                "text": str(data.get("text", "")).strip(),
                "type": str(data.get("type", "")),
                "assertion": str(data.get("assertion", "")),
                "source_quote": str(data.get("source_quote", "")),
            }
        )
    return normalized


def numbers_in(text: str) -> set:
    """Every numeric value in the text, with O2 and A1C digits excluded."""
    return set(re.findall(NUMBER_PATTERN, text))


def build_grounding(atoms) -> Dict[str, List[dict]]:
    """Fetch the reference passages for every fact, keyed by fact_id.

    A store or embedding failure raises, because a draft built without grounding would
    look the same as a draft whose terms genuinely have no definition.
    """
    # Each fact ID gets its own list, including an empty list for a genuine no-match.
    # An outage is different: stop here so missing references are not silently hidden.
    grounding: Dict[str, List[dict]] = {}
    for atom in normalize_atoms(atoms):
        try:
            grounding[atom["fact_id"]] = retrieve_passages(atom["text"], limit=PASSAGES_PER_FACT)
        except Exception as error:
            raise simplify_error(
                f"MedlinePlus retrieval failed for {atom['fact_id']}; "
                "the draft was not produced."
            ) from error
    return grounding


def format_atoms(atoms) -> str:
    """Render the facts with a number, an id, and the assertion the rewrite must keep."""
    lines = []
    for atom in normalize_atoms(atoms):
        lines.append(
            f"{atom['position']}. {atom['fact_id']} [{atom['type']}/{atom['assertion']}] "
            f"{atom['text']}"
        )
    return "\n".join(lines)


def format_grounding(atoms, grounding: Dict[str, List[dict]]) -> str:
    """Render each fact's own passages, and say plainly when there are none."""
    blocks = []
    for atom in normalize_atoms(atoms):
        passages = grounding.get(atom["fact_id"]) or []
        if passages:
            blocks.append(f"[{atom['fact_id']}]\n{format_passages(passages)}")
        else:
            blocks.append(f"[{atom['fact_id']}]\nNo reference block is available for this fact.")
    return "\n\n".join(blocks)


def supplied_titles(grounding: Dict[str, List[dict]]) -> set:
    """Every article title the draft was allowed to cite."""
    titles = set()
    for passages in grounding.values():
        for passage in passages:
            if passage.get("title"):
                titles.add(str(passage["title"]))
    return titles


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
            response_schema=SimplifierReply,
            temperature=0,
        ),
    )
    return response.text or ""


def parse_reply(reply_text: str) -> Optional[SimplifierReply]:
    """Validate a reply against the schema, returning None when it does not parse."""
    try:
        return SimplifierReply.model_validate_json(reply_text)
    except Exception:
        return None


def assess_fact_rewrite(atom: dict, rewrite: FactRewrite) -> List[str]:
    """The problems this one rewrite has, as sentences a repair can act on."""
    text = rewrite.text.strip()
    if not text:
        return ["the rewrite is empty"]

    # Set subtraction finds values that appear in the rewrite but not in its fact.
    # A value being allowed does not prove its unit, owner or clinical use is correct.
    problems = []
    allowed_numbers = numbers_in(atom["text"] + " " + atom["source_quote"])
    invented = numbers_in(text) - allowed_numbers
    if invented:
        problems.append(f"unsupported quantities in the rewrite: {', '.join(sorted(invented))}")

    # These patterns look for cue words such as "no", "possible" and "previously".
    # They catch selected wording mistakes, not every possible change in meaning.
    has_negation = bool(NEGATION_CUE_PATTERN.search(text))
    fact_has_negation = bool(NEGATION_CUE_PATTERN.search(atom["text"]))
    assertion = atom["assertion"]

    if assertion == "absent" and not has_negation:
        problems.append("lost the absence: the rewrite does not say the fact is absent")
    if assertion == "present" and has_negation and not fact_has_negation:
        problems.append("added a negation the fact does not have")
    if assertion == "possible" and not UNCERTAINTY_CUE_PATTERN.search(text):
        problems.append("lost the uncertainty: a possible fact became certain")
    if assertion == "past" and not PAST_CUE_PATTERN.search(text):
        problems.append("lost the timing: a past fact reads as current")
    if assertion == "family" and not RELATIVE_PATTERN.search(text):
        problems.append("lost the owner: the rewrite does not name the relative")
    if assertion != "family" and RELATIVE_PATTERN.search(text) and not RELATIVE_PATTERN.search(atom["text"]):
        problems.append("added a relative the fact does not have")
    return problems


def assess(reply: SimplifierReply, atoms: List[dict]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Split the reply's problems into per-fact problems and unknown fact ids."""
    # First find repeated IDs. Then check each expected fact, including missing ones.
    # Unknown IDs are reported separately because they do not belong in this draft.
    problems: Dict[str, List[str]] = {}
    seen = set()
    for rewrite in reply.fact_rewrites:
        if rewrite.fact_id in seen:
            problems.setdefault(rewrite.fact_id, []).append("returned more than one rewrite for this fact")
            continue
        seen.add(rewrite.fact_id)

    # A dictionary keeps the last entry for a repeated ID. The duplicate issue above
    # still marks that fact for repair, even if this particular entry reads correctly.
    by_id = {rewrite.fact_id: rewrite for rewrite in reply.fact_rewrites}
    for atom in atoms:
        rewrite = by_id.get(atom["fact_id"])
        if rewrite is None:
            problems.setdefault(atom["fact_id"], []).append("no rewrite was returned for this fact")
            continue
        for message in assess_fact_rewrite(atom, rewrite):
            problems.setdefault(atom["fact_id"], []).append(message)

    known_ids = {atom["fact_id"] for atom in atoms}
    unknown_ids = sorted({rewrite.fact_id for rewrite in reply.fact_rewrites if rewrite.fact_id not in known_ids})
    return problems, unknown_ids


def build_repair_prompt(
    atoms: List[dict],
    grounding: Dict[str, List[dict]],
    reply: SimplifierReply,
    problems: Dict[str, List[str]],
) -> str:
    """Ask for the failing facts only, with the wording to beat and the exact complaint."""
    previous = [
        f"[{rewrite.fact_id}] {rewrite.text}"
        for rewrite in reply.fact_rewrites
        if rewrite.fact_id in problems
    ]
    complaints = [
        f"- {fact_id}: {'; '.join(messages)}"
        for fact_id, messages in sorted(problems.items())
    ]
    return (
        f"FACTS:\n{format_atoms(atoms)}\n\n"
        f"REFERENCE BLOCKS:\n{format_grounding(atoms, grounding)}\n\n"
        "REPAIR MODE\n"
        "You are fixing a draft that failed a meaning check. Fix only the facts listed."
        "\n\nYOUR PREVIOUS WORDING:\n" + "\n".join(previous) +
        "\n\nPROBLEMS:\n" + "\n".join(complaints) +
        "\n\nReturn one fact_rewrites entry for every fact listed above, and no others. "
        "Fix the reported problem and change as little else as possible. Never fix a "
        "complaint by deleting the fact, softening it, or moving it to another person. "
        "Never invent a number. Return only the structured output that matches the schema."
    )


def merge_repair(
    original: SimplifierReply,
    repair: SimplifierReply,
    problems: Dict[str, List[str]],
) -> SimplifierReply:
    """Keep the original rewrite for every fact the repair did not need to touch."""
    # Accept replacements only for facts that failed. A repair must not rewrite facts
    # that already passed. Final assembly below restores the original input order.
    replacements = {
        rewrite.fact_id: rewrite
        for rewrite in repair.fact_rewrites
        if rewrite.fact_id in problems
    }
    kept = [rewrite for rewrite in original.fact_rewrites if rewrite.fact_id not in replacements]
    titles = list(original.sources_used) + list(repair.sources_used)
    return SimplifierReply(fact_rewrites=kept + list(replacements.values()), sources_used=titles)


def prune_sources(sources_used: List[str], grounding: Dict[str, List[dict]]) -> Tuple[List[str], List[str]]:
    """Keep the citations the draft was entitled to make, and report the rest."""
    allowed = supplied_titles(grounding)
    kept: List[str] = []
    unknown: List[str] = []
    for title in sources_used:
        if title in kept:
            continue
        if title in allowed:
            kept.append(title)
        else:
            unknown.append(title)
    return kept, unknown


def collect_sources(titles: List[str], grounding: Dict[str, List[dict]]) -> List[SourcePassage]:
    """Copy the store's passage text for the titles the draft used."""
    # The model names article titles, but the saved text comes from supplied passages.
    # Different chunks of an article may share a title, so deduplicate by title AND text.
    collected: List[SourcePassage] = []
    seen = set()
    for title in titles:
        for passages in grounding.values():
            for passage in passages:
                if str(passage.get("title")) != title:
                    continue
                key = (title, passage.get("text"))
                if key in seen:
                    continue
                seen.add(key)
                collected.append(
                    SourcePassage(
                        title=title,
                        url=passage.get("url"),
                        section=passage.get("section"),
                        text=str(passage.get("text", "")),
                    )
                )
    return collected


def assemble(texts: List[str]) -> str:
    """Join the rewrites into one text, in fact order."""
    joined = " ".join(text.strip() for text in texts if text.strip())
    return " ".join(joined.split())


def simplify(
    atoms,
    grounding: Dict[str, List[dict]],
    send: Optional[Callable[[str, str], str]] = None,
    model: str = MODEL,
) -> SimplifiedOutput:
    """Write the patient text, or fall back to source wording and say which facts did.

    `send` is injected so the tests can run with no API key and no network.
    """
    normalized = normalize_atoms(atoms)
    if not normalized:
        raise simplify_error("Nothing to simplify: no atoms were supplied.")

    # Step 1: request one rewrite per fact, using the passages already fetched.
    # A scripted sender lets tests exercise the same path without an API request.
    sender = send or live_send
    prompt = f"FACTS:\n{format_atoms(normalized)}\n\nREFERENCE BLOCKS:\n{format_grounding(normalized, grounding)}"
    reply = parse_reply(sender(prompt, model))
    if reply is None:
        raise simplify_error("The model reply was not a valid simplification.")

    # Step 2: check IDs and selected meaning rules before accepting the wording.
    problems, unknown_ids = assess(reply, normalized)
    review_warnings: List[str] = []
    if unknown_ids:
        review_warnings.append(f"dropped rewrites for unknown facts: {', '.join(unknown_ids)}")

    if problems:
        # Step 3: allow one correction, naming the exact problems. If it parses,
        # merge the failing facts only and check the combined reply again.
        repair = parse_reply(sender(build_repair_prompt(normalized, grounding, reply, problems), model))
        if repair is not None:
            reply = merge_repair(reply, repair, problems)
            problems, unknown_ids = assess(reply, normalized)
            if unknown_ids:
                review_warnings.append(f"dropped rewrites for unknown facts: {', '.join(unknown_ids)}")

    # Keep the first returned rewrite for each ID during assembly. Duplicate IDs
    # still count as problems unless the repair removed them.
    rewrites = {}
    for rewrite in reply.fact_rewrites:
        rewrites.setdefault(rewrite.fact_id, rewrite)

    texts: List[str] = []
    covered_positions: List[int] = []
    covered_ids: List[str] = []

    for atom in normalized:
        # Step 4: walk the facts, not the model's reply order. Every fact gets text:
        # either its checked rewrite or its original wording with a review warning.
        # Coverage therefore includes source-wording fallbacks, not just easy rewrites.
        fact_id = atom["fact_id"]
        rewrite = rewrites.get(fact_id)
        covered_positions.append(atom["position"])
        covered_ids.append(fact_id)

        if rewrite is None:
            texts.append(atom["text"])
            review_warnings.append(f"{fact_id}: no rewrite returned; source wording used.")
            continue

        for term in rewrite.ungrounded_terms:
            review_warnings.append(f"{fact_id}: '{term}' had no reference block and was paraphrased.")

        if fact_id in problems:
            texts.append(atom["text"])
            review_warnings.append(
                f"{fact_id}: {'; '.join(problems[fact_id])}. Source wording used."
            )
        else:
            texts.append(rewrite.text)

    # Step 5: remove titles that were never supplied. Keeping a supplied title proves
    # availability, not that the model used its passage correctly. That needs review.
    kept_titles, unknown_titles = prune_sources(reply.sources_used, grounding)
    if unknown_titles:
        review_warnings.append(
            f"dropped citations to passages that were not supplied: {', '.join(unknown_titles)}"
        )

    return SimplifiedOutput(
        simplified_text=assemble(texts),
        source_metadata=collect_sources(kept_titles, grounding),
        atoms_covered=covered_positions,
        fact_ids_covered=covered_ids,
        review_warnings=review_warnings,
    )
