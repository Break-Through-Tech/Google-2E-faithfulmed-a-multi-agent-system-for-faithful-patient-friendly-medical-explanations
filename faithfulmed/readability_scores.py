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
    """Score a text with the Flesch-Kincaid grade level and the SMOG index.

    These are the definitions the evaluation harness in `notebooks/eval_harness.py`
    reports, so an exported evaluation grade and an app grade are the same number.
    """
    word_count = textstat.lexicon_count(text, removepunct=True)
    if word_count == 0:
        raise ValueError("The final explanation contains no words to score.")
    return ReadabilityScores(
        grade=textstat.flesch_kincaid_grade(text),
        smog=textstat.smog_index(text)
    )