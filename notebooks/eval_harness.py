"""
FaithfulMed evaluation harness.

Build this out in Week 1-2, BEFORE the agents. You can't improve what you can't measure.
Run: python notebooks/eval_harness.py  (expects data/medaesqa_v1.json — see data/README.md)

Provides:
  - readability_scores(text): Flesch-Kincaid grade, SMOG, jargon density, length
  - meets_readability_target(text): the <= 8th-grade success criterion
  - load_medaesqa(path): loads the gold eval set
  - verifier_agreement(human, verifier): accuracy + Cohen's kappa for Verifier calibration

This is model-agnostic — it scores text, whatever produced it. Extend, don't treat as final.
Each agent owner plugs their agent's outputs into this harness.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from textstat import textstat
from sklearn.metrics import cohen_kappa_score, accuracy_score

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "medaesqa_v1.json"

# The harness runs as a script from notebooks/, so the package root is not on the path.
# Grade and SMOG come from the app's own readability module, which is the only place a
# reading grade is defined; a second formula here is how the harness and the app end up
# reporting different numbers for the same output.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from faithfulmed.readability_scores import score_readability


def readability_scores(text: str) -> dict:
    """Readability metrics for one output.

    Grade and SMOG come from `faithfulmed.readability_scores`, the same scorer the
    pipeline, the Refiner and the Streamlit app use. Jargon density and word count are
    measured here only.

    Text with no words raises ValueError.
    """
    scores = score_readability(text)
    words = textstat.lexicon_count(text, removepunct=True)
    return {
        "flesch_kincaid_grade": scores.grade,
        "smog_index": scores.smog,
        "jargon_density": textstat.difficult_words(text) / words,
        "word_count": words,
    }


def meets_readability_target(text: str, max_grade: float = 8.0) -> bool:
    """Success criterion: <= 8th-grade reading level (Flesch-Kincaid).

    Goes through the same scorer as the app, so the gate cannot pass on a formula the
    app does not use. Blank text raises rather than counting as a pass.
    """
    return score_readability(text).grade <= max_grade


def load_medaesqa(path: Path = DATA) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download medaesqa_v1.json from https://osf.io/ydbzq "
            f"into the data/ folder (see data/README.md)."
        )
    with open(path) as f:
        return json.load(f)


def verifier_agreement(human_labels: list[int], verifier_labels: list[int]) -> dict:
    """Calibrate the Verifier against human faithfulness labels.

    Targets from the success criteria: accuracy >= 0.80, Cohen's kappa >= 0.6.
    Pass 0/1 (or categorical) labels of equal length.
    """
    return {
        "accuracy": accuracy_score(human_labels, verifier_labels),
        "cohen_kappa": cohen_kappa_score(human_labels, verifier_labels),
        "n": len(human_labels),
    }


if __name__ == "__main__":
    demo = ("Your discharge summary says you were prescribed a beta-blocker to manage "
            "hypertension and should follow up with cardiology in two weeks.")
    print("Readability demo:", readability_scores(demo))
    print("Meets <=8th grade:", meets_readability_target(demo))

    # Tiny agreement demo (replace with real Verifier vs. human labels):
    print("Agreement demo:", verifier_agreement([1, 1, 0, 1, 0], [1, 0, 0, 1, 0]))

    try:
        data = load_medaesqa()
        print(f"Loaded MedAESQA: {len(data)} questions.")
    except FileNotFoundError as e:
        print(e)
