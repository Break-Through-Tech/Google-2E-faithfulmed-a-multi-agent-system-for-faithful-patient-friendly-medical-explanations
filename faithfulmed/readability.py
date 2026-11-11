"""Readability estimates for the final explanation."""

from dataclasses import dataclass

import textstat


@dataclass(frozen=True)
class ReadabilityScores:
    grade: float
    reading_ease: float
    word_count: int


def score_readability(text: str) -> ReadabilityScores:
    """Score the original output using the evaluation harness's Flesch-Kincaid measure."""
    word_count = textstat.lexicon_count(text, removepunct=True)
    if word_count == 0:
        raise ValueError("The final explanation contains no words to score.")
    return ReadabilityScores(
        grade=textstat.flesch_kincaid_grade(text),
        reading_ease=textstat.flesch_reading_ease(text),
        word_count=word_count,
    )
