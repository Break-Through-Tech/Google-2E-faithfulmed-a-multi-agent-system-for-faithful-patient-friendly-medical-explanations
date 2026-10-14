"""
retrieval_utils.py — FaithfulMed runtime retrieval tools.

Two ADK-ready tools, both reading the persistent ChromaDB MedlinePlus
index built by index_chroma.py:

    retrieve_medical_context(query: str) -> str   # RAG tool
    get_source_context(term: str)  -> str         # Verifier

Design rule: FAIL CLOSED. Any error, empty hit, or weak match returns a
sentinel string instead of a definition.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import chromadb
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# config
CHROMA_PATH       = os.getenv("FAITHFULMED_CHROMA_PATH", "/content/chroma_db")
COLLECTION_NAME   = os.getenv("FAITHFULMED_COLLECTION",  "medical_docs")
EMBED_MODEL       = os.getenv("FAITHFULMED_EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM         = 3072          # MUST match index_chroma.py
EMBED_TASK_QUERY  = "RETRIEVAL_QUERY"   # docs were indexed as RETRIEVAL_DOCUMENT

TOP_K             = 3
MAX_HITS          = 3             # cap on refs handed back
DISTANCE_THRESHOLD = 0.40         # cosine distance; > this => no match   (calibrated empirically; adjust if false positive/negative appear)

NO_MATCH_MSG = "No relevant MedlinePlus definition found."

# singletons
_genai_client: genai.Client | None = None


def _client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _genai_client


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(COLLECTION_NAME)

# embedding - takes query via Gemini; returns a 3072-dimension vector
# uses api calls
def _embed_query(text: str) -> list[float]:
    resp = _client().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=EMBED_TASK_QUERY,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return list(resp.embeddings[0].values)


# chroma I/O - sends embedded query to chromadb, retrieves top-k
# returns dict of documents (text), metadatas (title, source), distances (how close match was)
def _query(query: str) -> list[dict[str, Any]]:
    res = _collection().query(
        query_embeddings=[_embed_query(query)],
        n_results=TOP_K,   # top 3 results
        include=["documents", "metadatas", "distances"],
    )
    docs  = res.get("documents",  [[]])[0]
    metas = res.get("metadatas",  [[]])[0]
    dists = res.get("distances",  [[]])[0]
    return [
        {"document": d, "metadata": m or {}, "distance": float(dist)}
        for d, m, dist in zip(docs, metas, dists)
    ]


# formatting - turns raw chroma into readable string
def _format_hit(hit: dict[str, Any]) -> str:
    meta = hit["metadata"]
    title = meta.get("title") or meta.get("name") or "MedlinePlus Entry"
    source = meta.get("source") or meta.get("url") or "MedlinePlus"
    definition = (hit["document"] or "").strip()
    return (
        "[MedlinePlus Reference]\n"
        f"Title: {title}\n"
        f"Definition: {definition}\n"
        f"Source: {source}"
    )

# catches errors, filters results too far away, caps num of results
# if error, returns empty list
def _good_hits(query: str) -> list[dict[str, Any]]:
    try:
        hits = _query(query)
    except Exception:                       # noqa: BLE001 — fail closed
        logger.exception("Retrieval failed for query=%r", query)
        return []
    good = [h for h in hits if h["distance"] <= DISTANCE_THRESHOLD]
    return good[:MAX_HITS]

# TOOLS
# simplifier calls to get context for writing text
def retrieve_medical_context(query: str) -> str:
    """Fetch plain-language MedlinePlus definitions for a clinical term.

    Args:
        query: A single medical term or phrase extracted from a clinical
            note, e.g. "acute myocardial infarction" or "metformin".

    Returns:
        A formatted block of MedlinePlus references (title, definition,
        source), or the string "No relevant MedlinePlus definition found."
        if nothing passes the similarity threshold.
    """
    if not query or not query.strip():
        return NO_MATCH_MSG

    hits = _good_hits(query.strip())
    if not hits:
        return NO_MATCH_MSG
    return "\n\n".join(_format_hit(h) for h in hits)

# verifier will call this tool as ground truth
def get_source_context(term: str) -> str:
    """Fetch the RAW MedlinePlus source text for a term, for the Verifier.

    Identical retrieval to retrieve_medical_context, but intended for
    faithfulness checking: the Verifier compares draft claims against
    this text. Returns "No relevant MedlinePlus definition found." when
    the term is ungrounded — the Verifier should treat that as a signal
    that any claim about this term is unsupported.

    Args:
        term: A term, phrase, or claim-keyword to look up verbatim.

    Returns:
        Formatted raw MedlinePlus reference block, or the no-match sentinel.
    """
    if not term or not term.strip():
        return NO_MATCH_MSG

    hits = _good_hits(term.strip())
    if not hits:
        return NO_MATCH_MSG
    # Raw text; title/source included so the Verifier can cite provenance.
    return "\n\n".join(_format_hit(h) for h in hits)

# smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for q in ["acute myocardial infarction", "metformin", "zorblax quarantine"]:
        print(f"\n=== {q} ===")
        print(retrieve_medical_context(q))