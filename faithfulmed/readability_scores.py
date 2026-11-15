"""Readability estimates for the final explanation.

This module only measures text. The agent that rewrites text lives in
`faithfulmed/agents/readability.py`; both use these scores so there is one definition
of what a reading grade is.
"""
from dataclasses import dataclass

from textstat import textstat

@dataclass(frozen=True)
class ReadabilityScores:
    grade: float
    smog: float

def score_readability(text: str) -> ReadabilityScores:
    """Score the original output using the evaluation harness's Flesch-Kincaid grade level and SMOG index."""
    word_count = textstat.lexicon_count(text, removepunct=True)
    if word_count == 0:
        raise ValueError("The final explanation contains no words to score.")
    return ReadabilityScores(
        grade=textstat.flesch_kincaid_grade(text),
        smog=textstat.smog_index(text)
    )