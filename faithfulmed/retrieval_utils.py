"""Find MedlinePlus reference passages that help explain medical terms.

The passages come from the saved ChromaDB database built by index_chroma.py.
This file provides three ways to request them:

    retrieve_passages: return passages with their titles, links, and other details.
    retrieve_medical_context: return reference text for the Simplifier.
    get_source_context: return the same reference text for use by a Verifier.

First, look for a known article title, alternate name, or abbreviation.
If one matches, return up to three passages from that article in reading order.
This does not need an API request.

For other queries, ask the embedding service to represent the query as numbers
so it can be compared with saved passages. Keep only close matches that also
contain most of the query's words that are at least three characters long.
These checks help reject unrelated results, including made-up medical terms.

Finding no match is different from a failed lookup. The text tools return a
different message for each case. retrieve_passages raises an error if the lookup
fails, so the calling code can stop instead of continuing without references.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Reading path: retrieve_passages tries a known name first, then similarity search.
# An alias is another name for an article. A passage is one saved chunk of its text.
# Environment settings let a caller choose a store without changing this source file.
REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = Path(os.getenv("FAITHFULMED_CHROMA_PATH") or REPO_ROOT / "chroma")
COLLECTION_NAME = os.getenv("FAITHFULMED_COLLECTION", "medical_docs")
EMBED_MODEL = os.getenv("FAITHFULMED_EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = 3072                    # Use the same number of values as index_chroma.py.
EMBED_TASK_QUERY = "RETRIEVAL_QUERY"
ALIAS_PATH = Path(__file__).with_name("medlineplus_aliases.json")

MAX_PASSAGES = 3                    # Return at most this many passages per lookup.
CANDIDATE_COUNT = 10                # Check this many possible matches before choosing.
DISTANCE_CUTOFF = 0.40              # Lower means closer. Reject scores above this value.
MIN_TOKEN_LENGTH = 3                # Ignore shorter words when checking word overlap.
MIN_CONTAINED_WORD_LENGTH = 8        # Avoid matching short names inside longer queries.
MIN_ALPHANUMERIC_LENGTH = 2          # Names with a digit, such as A1C, can be shorter.

NO_MATCH_MSG = "No relevant MedlinePlus definition found."
RETRIEVAL_ERROR_MSG = (
    "MedlinePlus retrieval failed for this term. "
    "It has no grounding and must not be explained."
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

_genai_client: genai.Client | None = None


def _client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=0.5,
                    max_delay=2.0,
                    http_status_codes=[408, 429, 500, 502, 503, 504],
                )
            ),
        )
    return _genai_client


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    """Open the saved database once. Report an error if the collection is missing."""
    # lru_cache remembers a return value for the same arguments. With no arguments,
    # this function opens the collection once per process instead of once per fact.
    return chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION_NAME)


@lru_cache(maxsize=256)
def _embedding(query: str) -> tuple[float, ...]:
    """Reuse a query's saved number representation to avoid repeating API requests."""
    # An embedding is a list of numbers representing text for similarity comparison.
    # Its length must match the saved passage embeddings. Cache up to 256 queries
    # in this process so repeated identical query strings avoid another API request.
    response = _client().models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type=EMBED_TASK_QUERY,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return tuple(response.embeddings[0].values or [])


def _normalize(term: str) -> str:
    return _NON_ALPHANUMERIC.sub(" ", term.lower()).strip()


def _containable(key: str) -> bool:
    """Decide whether a known name is specific enough to match inside a longer query.

    Allow names with several words, but avoid short single words. Otherwise,
    "blood in urine" could incorrectly match the article titled Blood.
    Allow shorter names with digits so terms such as "a1c" can still match.
    """
    if " " in key:
        return True
    if any(character.isdigit() for character in key):
        return len(key) >= MIN_ALPHANUMERIC_LENGTH
    return len(key) >= MIN_CONTAINED_WORD_LENGTH


@lru_cache(maxsize=1)
def _alias_terms() -> dict[str, str]:
    """Load the list that connects known names and abbreviations to article titles."""
    if not ALIAS_PATH.is_file():
        logger.warning("Alias file missing, falling back to embeddings only: %s", ALIAS_PATH)
        return {}
    return json.loads(ALIAS_PATH.read_text(encoding="utf-8"))


def _known_term(query: str) -> str | None:
    """Find an article by name. Try the full query, then the longest known name within it."""
    normalized = _normalize(query)
    terms = _alias_terms()
    exact = terms.get(normalized)
    if exact is not None:
        return exact

    # Surround both strings with spaces to match whole normalized names, not parts
    # of words. Of the allowed names inside the query, prefer the longest one.
    # This chooses one article, even if the sentence mentions several medical terms.
    padded = f" {normalized} "
    contained = [key for key in terms if _containable(key) and f" {key} " in padded]
    if not contained:
        return None
    return terms[max(contained, key=len)]


def _passage(
    metadata: dict[str, Any], document: str | None, distance: float | None
) -> dict[str, Any]:
    return {
        "text": document or "",
        "title": metadata.get("title") or "MedlinePlus",
        "url": metadata.get("url"),
        "section": metadata.get("section"),
        "distance": distance,
    }


def _article_passages(title: str, limit: int) -> list[dict[str, Any]]:
    """Read up to the requested number of passages from a known article, in order."""
    # A known title bypasses embedding search. Read its chunks in article order,
    # rather than assuming the database returns them in a useful reading order.
    stored = _collection().get(where={"title": title}, include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    rows = sorted(
        zip(documents, metadatas),
        key=lambda row: (row[1] or {}).get("chunk_index") or 1,
    )
    return [_passage(metadata, document, None) for document, metadata in rows][:limit]


def _tokens_present(query: str, passage: str) -> bool:
    """Check whether more than half of the query's words appear in the passage.

    Ignore words shorter than three characters and count repeated words once.
    The similarity score alone can give made-up terms a good match.
    For example, "flurble gribble syndrome" may score well, but only "syndrome"
    appears in the results. The word check rejects that result.
    A normal sentence can still pass when most of its words appear.
    """
    tokens = {
        token
        for token in _TOKEN_PATTERN.findall(query.lower())
        if len(token) >= MIN_TOKEN_LENGTH
    }
    if not tokens:
        return False
    # A set counts each query word once. The formula requires a strict majority:
    # with four query words, at least three must appear somewhere in the passage.
    # This is a substring check, not a test of medical meaning or word boundaries.
    folded = passage.lower()
    present = sum(1 for token in tokens if token in folded)
    return present >= len(tokens) // 2 + 1


def _title_matches_query(query: str, title: str) -> bool:
    """Require a specific shared title word on the fallback search.

    This deliberately loses some synonym matches. Known names and abbreviations
    still use the name list first. A symptom list alone must not select a disease.
    """
    # Common words such as "patient" must not make an unrelated title look relevant.
    # Removing trailing s characters is a simple approximation for plural words.
    ignored = {
        "the", "and", "with", "who", "has", "have", "had", "was", "were",
        "patient", "mother", "father", "sister", "brother", "family", "history",
        "health", "good", "problem", "disease", "syndrome", "noted", "notable",
    }
    query_words = {
        word.rstrip("s") for word in _TOKEN_PATTERN.findall(query.lower())
        if len(word) >= MIN_TOKEN_LENGTH
    } - ignored
    title_words = {
        word.rstrip("s") for word in _TOKEN_PATTERN.findall(title.lower())
        if len(word) >= MIN_TOKEN_LENGTH
    } - ignored
    return bool(query_words & title_words)


def _similarity_passages(query: str, limit: int) -> list[dict[str, Any]]:
    """Find similar passages, check their scores and words, then group them by article.

    Put the closest article first and keep its accepted passages in reading order.
    Return each passage only once, up to the requested limit.
    """
    # Ask for more candidates than we plan to return because some will fail filters.
    # Chroma compares the query's embedding with the saved passage embeddings.
    result = _collection().query(
        query_embeddings=[list(_embedding(query))],
        n_results=CANDIDATE_COUNT,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    # Each accepted row keeps: title, distance, chunk number, metadata, passage text.
    # Lower distance means a closer match. All three filters below must pass.
    accepted: list[tuple[str, float, int, dict[str, Any], str]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        if distance > DISTANCE_CUTOFF:
            continue
        if not _title_matches_query(query, metadata.get("title") or ""):
            continue
        if not _tokens_present(query, f"{metadata.get('title', '')} {document}"):
            continue
        chunk_index = metadata.get("chunk_index") or 1
        accepted.append(
            (metadata.get("title") or "", float(distance), chunk_index, metadata, document)
        )

    # Rank articles by their best accepted match, then put their accepted chunks in
    # reading order. This does not fill in chunks that failed the filters.
    best_distance = {}
    for title, distance, _, _, _ in accepted:
        if title not in best_distance or distance < best_distance[title]:
            best_distance[title] = distance
    accepted.sort(key=lambda row: (best_distance[row[0]], row[2]))

    # A set prevents the same article chunk appearing twice in the returned list.
    # Stop at the caller's limit, which can be smaller than the candidate count.
    passages: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for title, distance, chunk_index, metadata, document in accepted:
        key = (title, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        passages.append(_passage(metadata, document, distance))
        if len(passages) == limit:
            break
    return passages


def format_passages(passages: list[dict[str, Any]]) -> str:
    """Turn passages into readable reference text with a title, passage, and source."""
    blocks = []
    for passage in passages:
        blocks.append(
            "[MedlinePlus Reference]\n"
            f"Title: {passage['title']}\n"
            f"Definition: {passage['text'].strip()}\n"
            f"Source: {passage['url'] or 'MedlinePlus'}"
        )
    return "\n\n".join(blocks)


def retrieve_passages(query: str, limit: int = MAX_PASSAGES) -> list[dict[str, Any]]:
    """Return matching reference passages for a medical term or short sentence.

    An empty list means no match, a blank query, or a limit below one.
    A failed API request or database lookup raises an error instead.
    Each passage includes its text, title, link, section, and match score.
    The score is None for a known-name match because no comparison was needed.
    """
    if not query or not query.strip() or limit < 1:
        return []
    title = _known_term(query)
    if title is not None:
        # A recognized name returns that article directly, even if it has no chunks.
        # This path does not try a different article through similarity search.
        return _article_passages(title, limit)
    return _similarity_passages(query, limit)


def retrieve_medical_context(query: str) -> str:
    """Find MedlinePlus reference text for a medical term or short sentence.

    Args:
        query: A medical term, phrase, or short sentence taken from a note,
            e.g. "acute myocardial infarction", "metformin", or a fact such as
            "The patient reports a mild headache."

    Returns:
        Reference text with titles and sources. NO_MATCH_MSG means no suitable
        passage was found or the query was blank. RETRIEVAL_ERROR_MSG means
        the lookup failed, not that MedlinePlus has no matching information.
    """
    if not query or not query.strip():
        return NO_MATCH_MSG
    try:
        passages = retrieve_passages(query.strip())
    except Exception:
        # The text interface reports failure with a distinct message. The structured
        # retrieve_passages interface lets the error reach its caller instead.
        logger.exception("Retrieval failed for query=%r", query)
        return RETRIEVAL_ERROR_MSG
    if not passages:
        return NO_MATCH_MSG
    return format_passages(passages)


def get_source_context(term: str) -> str:
    """Return the same reference text as retrieve_medical_context, for use by a Verifier."""
    return retrieve_medical_context(term)

# This is a quick manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for q in ["acute myocardial infarction", "metformin", "zorblax quarantine", "flurble gribble syndrome"]:
        print(f"\n=== {q} ===")
        print(retrieve_medical_context(q))
