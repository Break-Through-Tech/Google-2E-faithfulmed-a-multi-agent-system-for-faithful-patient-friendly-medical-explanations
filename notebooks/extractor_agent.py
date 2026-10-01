"""Starter Extractor for FaithfulMed.

This script does three useful things for the Extractor agent:
1. Loads the MTSamples CSV from the data folder.
2. Pulls out a structured "clinical atoms" view from a sample note.
3. Gives you a clean baseline format to improve with LLM prompting later.

This is intentionally a v1 prototype: deterministic, simple, and easy to extend.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mtsamples - mtsamples.csv"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def get_sections(transcription: str) -> Dict[str, str]:
    """Split a clinical transcription into labeled sections.

    Example output:
    {
        "SUBJECTIVE": "This patient presents with ...",
        "MEDICATIONS": "Ortho Tri-Cyclen and Allegra.",
        "ASSESSMENT": "Allergic rhinitis.",
        "PLAN": "She will try Zyrtec ...",
    }
    """
    if not transcription:
        return {}

    # Match section labels written in all caps, often with a colon.
    pattern = re.compile(
        r"(?i)\b([A-Z][A-Z /-]{2,}?)\s*:\s*|\b([A-Z][A-Z /-]{2,}?)\s*[,\n]"
    )
    matches = list(pattern.finditer(transcription))
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = (match.group(1) or match.group(2) or "").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(transcription)
        value = clean_text(transcription[start:end])
        if label:
            sections[label.upper()] = value

    # Add a fallback of the whole text if no sections were found.
    if not sections:
        sections["FULL_TEXT"] = clean_text(transcription)

    return sections


def extract_clinical_atoms(transcription: str) -> List[Dict[str, str]]:
    """Very simple v1 extraction.

    We are not trying to be perfect yet. This baseline focuses on the most important
    clinically meaningful fact types that the team wants to pass downstream.
    """
    sections = get_sections(transcription)
    atoms: List[Dict[str, str]] = []

    def add_atom(atom_type: str, value: str, source: str = "") -> None:
        if not value:
            return
        atoms.append({
            "type": atom_type,
            "value": clean_text(value),
            "source": clean_text(source),
        })

    # Diagnoses / assessments / impressions
    for key in ["ASSESSMENT", "IMPRESSION", "DIAGNOSIS", "POSTOPERATIVE DIAGNOSIS", "PREOPERATIVE DIAGNOSIS"]:
        if key in sections:
            add_atom("diagnosis", sections[key], key)

    # Medications / current medications
    for key in ["MEDICATIONS", "CURRENT MEDICATIONS", "HOME MEDICATIONS", "CURRENT MEDICATIONS:"]:
        if key in sections:
            add_atom("medication", sections[key], key)

    # Allergies
    for key in ["ALLERGIES", "ALLERGIES TO MEDICATIONS"]:
        if key in sections:
            add_atom("allergy", sections[key], key)

    # Procedures / operations
    for key in ["PROCEDURE", "PROCEDURE PERFORMED", "OPERATION", "OPERATION PERFORMED", "PROCEDURES"]:
        if key in sections:
            add_atom("procedure", sections[key], key)

    # Symptoms / chief complaints / history of present illness
    for key in ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS", "SUBJECTIVE"]:
        if key in sections:
            text = sections[key]
            if len(text) < 300:
                add_atom("symptom_or_history", text, key)

    # Lab values if present
    lab_pattern = re.compile(r"([A-Z][A-Za-z ]+\s*[:=]\s*[0-9]+\.?[0-9]*\s*(?:mg/dL|mmHg|%|mEq/L|mmol/L|units|cm|kg|lbs|bpm|C|F|mm)? )")
    for field_name, value in sections.items():
        for match in lab_pattern.finditer(value):
            add_atom("lab_value", match.group(1), field_name)

    # If we found nothing, keep the whole transcription as the fallback atom.
    if not atoms:
        add_atom("raw_text", transcription, "FULL_TEXT")

    return atoms


def load_mtsamples(path: Path = DATA_PATH) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"MTSamples CSV not found at {path}. "
            "Place the CSV in the data folder before running this script."
        )

    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


if __name__ == "__main__":
    rows = load_mtsamples()
    print(f"Loaded {len(rows)} MTSamples records.")
    print("\nShowing a sample extraction from the first clinical note.\n")

    sample = rows[0]
    transcription = sample.get("transcription", "")
    print("SOURCE SUMMARY:")
    print(sample.get("medical_specialty", "Unknown specialty"), "|", sample.get("sample_name", "Unknown sample"))
    print("\nRAW TRANSCRIPTION (truncated):")
    print(transcription[:800])
    print("\nEXTRACTED ATOMS:")
    atoms = extract_clinical_atoms(transcription)
    print(json.dumps(atoms[:10], indent=2, ensure_ascii=False))
