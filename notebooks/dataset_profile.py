"""Data understanding function helpers for the FaithfulMed datasets.

Two jobs:

1. Parse the XML datasets (MedQuAD, MedlinePlus health topics, MedlinePlus
   definitions) into one flat jsonl retrieval document schema.
2. Run column classification, missingness, univariate
   stats, outlier flags, duplication, and ID check as one pass.

Every run also writes everything it prints to a numbered text file, so the runs
sort in the order they were made. For example:

    notebooks/profile_runs/dataset_profile_full_01.txt
    notebooks/profile_runs/dataset_profile_sample_01.txt
    notebooks/profile_runs/dataset_profile_recon_01.txt

The number continues from the highest one already in the folder. 
Use --out-dir to write elsewhere, 
    --prefix to change the name before the number, and 
    --pad to change the digit count.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

MEDQUAD_DIR = DATA_DIR / "medquad"
MEDLINEPLUS_DIR = DATA_DIR / "medlineplus"
MPL_TOPICS = MEDLINEPLUS_DIR / "health_topics" / "mplus_topics_2026-09-10.xml"
MPL_DEFINITIONS_DIR = MEDLINEPLUS_DIR / "definitions_of_health_terms"

# MedQuAD collection folders that hold QAPairs.
MEDQUAD_COLLECTIONS = [
    "1_CancerGov_QA",
    "2_GARD_QA",
    "3_GHR_QA",
    "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "8_NHLBI_QA_XML",
    "9_CDC_QA",
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
]

# Collections whose answers MedQuAD stripped for copyright reasons.
ANSWER_STRIPPED_COLLECTIONS = {
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}

# The one schema every dataset projects into before it is profiled or indexed.
RETRIEVAL_COLUMNS = [
    "doc_uid",
    "source_dataset",
    "collection",
    "title",
    "section",
    "text",
    "url",
    "language",
    "char_len",
    "word_len",
]

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def normalize_ws(value: str | None) -> str:
    """Collapse the indentation and newlines that XML pretty-printing adds."""
    if not value:
        return ""
    return _WS.sub(" ", value).strip()


def strip_html(value: str | None) -> str:
    """MedlinePlus summaries arrive HTML-escaped; unescape then drop tags."""
    if not value:
        return ""
    return normalize_ws(html.unescape(_TAG.sub(" ", value)))


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


# Four files in 6_NINDS_QA ship in an older MedQuAD schema: a lowercase <doc>
# root, lowercase element names, and a flat <umls> block that the current schema
# nests inside <FocusAnnotations>. Every field the parser reads is present, so
# the names are rewritten in memory instead of directly modify the files.
LEGACY_TAG_NAMES = {
    "doctitle-focus": "Focus",
    "umls": "UMLS",
    "cui": "CUI",
    "semanticType": "SemanticType",
    "semanticGroup": "SemanticGroup",
    "qaPairs": "QAPairs",
    "pair": "QAPair",
    "question": "Question",
    "answer": "Answer",
}
LEGACY_ATTR_NAMES = {"docid": "id", "corpus": "source"}


def upgrade_legacy_medquad(root: ET.Element) -> bool:
    """Reshape an old lowercase MedQuAD document to the current schema.

    Returns True when the document was written in the legacy schema. The
    rewrite is lossless for those files: docid, corpus, url, doctitle-focus, and
    the qid, qtype, and pid attributes all map onto fields the parser already
    reads.

    Their annotation block carries no data. All four files hold a <cui>, a
    <semanticType>, and a <semanticGroup> element with nothing inside, so the
    one-to-one mapping of <cui> to <CUI> below is written from the tag names
    rather than observed on populated values. _text_values drops those blank
    placeholders, which is what keeps has_umls false for these rows.
    """
    if root.tag != "doc":
        return False

    root.tag = "Document"
    for element in root.iter():
        if element is not root:
            element.tag = LEGACY_TAG_NAMES.get(element.tag, element.tag)
    for old_name, new_name in LEGACY_ATTR_NAMES.items():
        if old_name in root.attrib:
            root.set(new_name, root.attrib.pop(old_name))

    # Lift the top-level UMLS block into FocusAnnotations and restore the CUIs
    # and SemanticTypes layers the current schema uses.
    umls = root.find("UMLS")
    if umls is not None:
        position = list(root).index(umls)
        root.remove(umls)

        rebuilt = ET.Element("UMLS")
        cuis = ET.SubElement(rebuilt, "CUIs")
        for child in umls.findall("CUI"):
            cuis.append(child)
        types = ET.SubElement(rebuilt, "SemanticTypes")
        for child in umls.findall("SemanticType"):
            types.append(child)
        group = umls.find("SemanticGroup")
        if group is not None:
            rebuilt.append(group)

        wrapper = ET.Element("FocusAnnotations")
        wrapper.append(rebuilt)
        root.insert(position, wrapper)

    return True


# MedQuAD
def _text_values(parent: ET.Element, path: str) -> list[str]:
    """Every non-empty text value found under a path.

    An empty placeholder element such as <CUI></CUI> and a missing element both
    read back as "". Blank entries are dropped here, because keeping them would
    make a file that carries an empty placeholder look annotated.
    """
    values = []
    for element in parent.findall(path):
        text = normalize_ws(element.text)
        if text:
            values.append(text)
    return values


def parse_medquad_file(path: Path, collection: str) -> list[dict]:
    """Return one record per QAPair in a MedQuAD XML file."""
    root = ET.parse(path).getroot()
    upgrade_legacy_medquad(root)

    doc_id = root.get("id", "")
    focus = normalize_ws(root.findtext("Focus"))

    annotations = root.find("FocusAnnotations")
    category = ""
    synonyms: list[str] = []
    cuis: list[str] = []
    semantic_types: list[str] = []
    semantic_group = ""
    if annotations is not None:
        category = normalize_ws(annotations.findtext("Category"))
        synonyms = _text_values(annotations, "Synonyms/Synonym")
        cuis = _text_values(annotations, "UMLS/CUIs/CUI")
        semantic_types = _text_values(annotations, "UMLS/SemanticTypes/SemanticType")
        semantic_group = normalize_ws(annotations.findtext("UMLS/SemanticGroup"))

    records = []
    for pair in root.findall("QAPairs/QAPair"):
        question_el = pair.find("Question")
        answer_el = pair.find("Answer")
        question = normalize_ws(question_el.text) if question_el is not None else ""
        answer = normalize_ws(answer_el.text) if answer_el is not None else ""
        qid = question_el.get("qid", "") if question_el is not None else ""
        records.append(
            {
                "qa_uid": f"{collection}/{qid}",
                "doc_id": doc_id,
                "collection": collection,
                "source": root.get("source", ""),
                "url": root.get("url", ""),
                "focus": focus,
                "focus_category": category,
                "qtype": question_el.get("qtype", "") if question_el is not None else "",
                "pid": pair.get("pid", ""),
                "qid": qid,
                "question": question,
                "answer": answer,
                "answer_missing": answer == "",
                "answer_chars": len(answer),
                "answer_words": word_count(answer),
                "question_words": word_count(question),
                "has_umls": bool(cuis),
                "n_cuis": len(cuis),
                "first_cui": cuis[0] if cuis else "",
                "semantic_group": semantic_group,
                "n_semantic_types": len(semantic_types),
                "n_synonyms": len(synonyms),
                "file": path.name,
            }
        )
    return records


def load_medquad(
    root: Path = MEDQUAD_DIR,
    limit_per_collection: int | None = None,
) -> pd.DataFrame:
    """Parse every MedQuAD collection into one DataFrame."""
    rows: list[dict] = []
    started = time.time()
    for collection in MEDQUAD_COLLECTIONS:
        folder = root / collection
        if not folder.is_dir():
            print(f"  missing folder, skipped: {collection}", file=sys.stderr)
            continue
        files = sorted(folder.glob("*.xml"))
        if limit_per_collection is not None:
            files = files[:limit_per_collection]
        for path in files:
            rows.extend(parse_medquad_file(path, collection))
        print(f"  {collection}: {len(files)} files", flush=True)
    df = pd.DataFrame(rows)
    print(f"MedQuAD parsed in {time.time() - started:.1f}s -> {len(df):,} QAPairs")
    return df


def medquad_tag_inventory(root: Path = MEDQUAD_DIR, per_collection: int = 40) -> pd.DataFrame:
    """Which XML tags each collection actually uses. Guards the parser."""
    rows = []
    for collection in MEDQUAD_COLLECTIONS:
        folder = root / collection
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.xml"))[:per_collection]:
            tree = ET.parse(path)
            for node in tree.iter():
                rows.append({"collection": collection, "tag": node.tag})
    if not rows:
        return pd.DataFrame(columns=["collection", "tag", "count"])
    return (
        pd.DataFrame(rows)
        .value_counts(["collection", "tag"])
        .rename("count")
        .reset_index()
        .sort_values(["collection", "tag"])
        .reset_index(drop=True)
    )


# MedlinePlus
def parse_medlineplus_topics(path: Path = MPL_TOPICS, limit: int | None = None) -> pd.DataFrame:
    """One row per health topic. XML attributes carry most of the metadata."""
    root = ET.parse(path).getroot()
    topics = list(root.findall("health-topic"))
    if limit is not None:
        topics = topics[:limit]

    rows = []
    for topic in topics:
        summary = strip_html(topic.findtext("full-summary"))
        groups = [normalize_ws(g.text) for g in topic.findall("group")]
        mesh = [normalize_ws(m.text) for m in topic.findall("mesh-heading/descriptor")]
        sites = topic.findall("site")
        language = topic.get("language", "")
        rows.append(
            {
                "doc_uid": f"medlineplus_topics/{topic.get('id', '')}",
                "source_dataset": "medlineplus_topics",
                "collection": "MedlinePlus Health Topics",
                "topic_id": topic.get("id", ""),
                "title": normalize_ws(topic.get("title")),
                "url": topic.get("url", ""),
                "language": language,
                "is_english": language.lower() == "english", # population filter English-only topics.
                "date_created": topic.get("date-created", ""),
                "meta_desc": normalize_ws(topic.get("meta-desc")),
                "also_called": [normalize_ws(a.text) for a in topic.findall("also-called")],
                "text": summary,
                "char_len": len(summary),
                "word_len": word_count(summary),
                "text_missing": summary == "",
                "n_groups": len(groups),
                "groups": groups,
                "n_mesh": len(mesh),
                "mesh": mesh,
                "n_sites": len(sites),
                "primary_institute": normalize_ws(topic.findtext("primary-institute")),
                "n_see_reference": len(topic.findall("see-reference")),
                "other_language": [
                    o.get("vernacular-name", "") for o in topic.findall("other-language")
                ],
            }
        )
    return pd.DataFrame(rows)


def parse_medlineplus_definitions(folder: Path = MPL_DEFINITIONS_DIR) -> pd.DataFrame:
    """One row per glossary term. Every term carries a stray leading '>'."""
    rows = []
    for path in sorted(folder.glob("*.xml")):
        root = ET.parse(path).getroot()
        for group in root.findall("term-group"):
            term = normalize_ws(group.findtext("term")).lstrip(">").strip()
            definition = normalize_ws(group.findtext("definition")).lstrip(">").strip()
            rows.append(
                {
                    "doc_uid": f"medlineplus_definitions/{path.stem}:{term}",
                    "source_dataset": "medlineplus_definitions",
                    "collection": path.stem,
                    "page_title": root.get("title", ""),
                    "title": term,
                    "section": "definition",
                    "text": definition,
                    "url": group.get("reference-url", ""),
                    "language": "English",
                    "reference": group.get("reference", ""),
                    "char_len": len(definition),
                    "word_len": word_count(definition),
                    "text_missing": definition == "",
                }
            )
    return pd.DataFrame(rows)


def load_medlineplus(limit_topics: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    topics = parse_medlineplus_topics(limit=limit_topics)
    definitions = parse_medlineplus_definitions()
    return topics, definitions


def to_retrieval_docs_medquad(df: pd.DataFrame) -> pd.DataFrame:
    """Project MedQuAD rows into the shared retrieval-document schema."""
    out = pd.DataFrame(
        {
            "doc_uid": "medquad/" + df["qa_uid"],
            "source_dataset": "medquad",
            "collection": df["collection"],
            "title": df["focus"],
            "section": df["qtype"],
            "text": (df["question"].str.strip() + "\n" + df["answer"].str.strip()).str.strip(),
            "url": df["url"],
            "language": "English",
        }
    )
    out["char_len"] = out["text"].str.len()
    out["word_len"] = out["text"].map(word_count)
    return out[RETRIEVAL_COLUMNS]


def to_retrieval_docs_medlineplus(topics: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    """Project MedlinePlus topics and glossary terms into the same schema."""
    topic_docs = pd.DataFrame(
        {
            "doc_uid": topics["doc_uid"],
            "source_dataset": topics["source_dataset"],
            "collection": topics["collection"],
            "title": topics["title"],
            "section": "topic-summary",
            "text": topics["text"],
            "url": topics["url"],
            "language": topics["language"],
        }
    )
    definition_docs = pd.DataFrame(
        {
            "doc_uid": definitions["doc_uid"],
            "source_dataset": definitions["source_dataset"],
            "collection": definitions["collection"],
            "title": definitions["title"],
            "section": definitions["section"],
            "text": definitions["text"],
            "url": definitions["url"],
            "language": definitions["language"],
        }
    )
    out = pd.concat([topic_docs, definition_docs], ignore_index=True, sort=False)
    out["char_len"] = out["text"].str.len()
    out["word_len"] = out["text"].map(word_count)
    return out[RETRIEVAL_COLUMNS]


# Profiling


def _n_unique(series: pd.Series) -> int:
    """nunique() raises on list-valued cells, so fall back to the row count."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return -1


def classify_columns(df: pd.DataFrame, id_columns: list[str] | None = None) -> pd.DataFrame:
    """Continuous / integer / ordinal / nominal / text / id, per the course types."""
    id_columns = id_columns or []
    rows = []
    for name in df.columns:
        series = df[name]
        dtype = str(series.dtype)
        is_list = series.map(type).isin([list, dict]).any()
        if name in id_columns:
            kind = "id (join key, never a feature)"
        elif is_list:
            kind = "nominal (multi-value list)"
        elif pd.api.types.is_bool_dtype(series):
            kind = "nominal (binary flag)"
        elif pd.api.types.is_numeric_dtype(series):
            nunique = _n_unique(series)
            is_int = pd.api.types.is_integer_dtype(series)
            if is_int and nunique <= 10 and series.min() >= 0:
                kind = "ordinal or nominal (small integer range)"
            elif is_int:
                kind = "integer"
            else:
                kind = "continuous"
        elif dtype == "object" and _n_unique(series) <= 50:
            kind = "nominal"
        else:
            kind = "nominal (free text)"
        rows.append(
            {
                "column": name,
                "dtype": dtype,
                "course_type": kind,
                "n_unique": _n_unique(series),
                "pct_missing": round(float(series.isna().mean() * 100), 2),
            }
        )
    return pd.DataFrame(rows)


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    blank = (df.astype(str) == "").sum()
    return (
        pd.DataFrame(
            {
                "missing": counts,
                "missing_pct": (counts / len(df) * 100).round(2),
                "blank_or_empty": blank,
            }
        )
        .query("missing > 0 or blank_or_empty > 0")
        .sort_values("missing", ascending=False)
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.describe().T
    out["skew"] = numeric.skew()
    return out[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew"]].round(2)


def iqr_outlier_mask(series: pd.Series, k: float = 2.0) -> pd.Series:
    """Course default: outside Q1 - k*IQR to Q3 + k*IQR, with k = 2."""
    clean = series.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (series < low) | (series > high)


def duplication_report(df: pd.DataFrame, text_column: str, key_columns: list[str]) -> dict:
    """How much of the dataset repeats itself, on keys and on raw text.

    Empty strings are excluded from the text counts: a stripped collection has
    thousands of blank answers, and counting those as duplicates would hide the
    real repetition.
    """
    keys = "+".join(key_columns)
    text = df[text_column].fillna("").str.strip()
    non_empty = text[text != ""]
    return {
        "rows": len(df),
        f"unique_{keys}": int(df[key_columns].drop_duplicates().shape[0]),
        "duplicate_key_rows": int(df.duplicated(subset=key_columns).sum()),
        "rows_with_empty_text": int((text == "").sum()),
        "unique_non_empty_text": int(non_empty.nunique()),
        "duplicate_non_empty_text": int(non_empty.duplicated().sum()),
        "pct_repeated_text": round(100 * non_empty.duplicated().mean(), 2) if len(non_empty) else 0.0,
    }


def profile(df: pd.DataFrame, name: str, id_columns: list[str] | None = None) -> dict:
    """Run the whole Phase 2 checklist and return the pieces as a dict."""
    print(f"\n{'=' * 78}\n{name}: {len(df):,} rows x {df.shape[1]} columns\n{'=' * 78}")
    types = classify_columns(df, id_columns)
    print("\n-- Column classification --")
    print(types.to_string(index=False))

    print("\n-- Missingness --")
    miss = missingness(df)
    print(miss.to_string() if not miss.empty else "  none")

    print("\n-- Numeric summary --")
    summary = numeric_summary(df)
    print(summary.to_string() if not summary.empty else "  no numeric columns")

    print("\n-- Duplication --")
    key = [c for c in ("qa_uid", "doc_uid", "topic_id") if c in df.columns][:1]
    if key:
        text_col = "answer" if "answer" in df.columns else "text"
        for k, v in duplication_report(df, text_col, key).items():
            print(f"  {k}: {v:,}")

    return {"name": name, "types": types, "missing": miss, "numeric": summary}


def medquad_by_collection(df: pd.DataFrame) -> pd.DataFrame:
    """Per-collection row counts, answer coverage, and text volume."""
    grouped = df.groupby("collection").agg(
        files=("file", "nunique"),
        qapairs=("qid", "size"),
        unique_qids=("qid", "nunique"),
        focus_terms=("focus", "nunique"),
        question_types=("qtype", "nunique"),
        answers_present=("answer_missing", lambda s: int((~s).sum())),
        pct_answer_missing=("answer_missing", lambda s: round(100 * s.mean(), 2)),
        median_answer_words=("answer_words", "median"),
        max_answer_words=("answer_words", "max"),
    )
    return grouped.sort_values("qapairs", ascending=False)


# CLI
DEFAULT_OUT_DIR = NOTEBOOK_DIR / "profile_runs"


class _Tee:
    """Write to several streams at once, so a run lands on screen and on disk."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def next_run_path(out_dir: Path, prefix: str, pad: int = 2) -> Path:
    """Next free name in the sequence: prefix_01.txt, prefix_02.txt, ..."""
    out_dir.mkdir(parents=True, exist_ok=True)
    highest = 0
    for path in out_dir.glob(f"{prefix}_*.txt"):
        suffix = path.stem[len(prefix) + 1 :]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return out_dir / f"{prefix}_{highest + 1:0{pad}d}.txt"


def run_profile(args: argparse.Namespace) -> None:
    """Do the work. Everything printed here is captured by main()."""
    if args.recon:
        print(medquad_tag_inventory().to_string(index=False))
        return

    medquad = load_medquad(limit_per_collection=args.sample)
    profile(medquad, "MedQuAD", id_columns=["qa_uid", "doc_id", "qid"])
    print("\n-- By collection --")
    print(medquad_by_collection(medquad).to_string())

    topics, definitions = load_medlineplus(limit_topics=args.sample)
    profile(topics, "MedlinePlus health topics", id_columns=["doc_uid", "topic_id"])
    profile(definitions, "MedlinePlus definitions", id_columns=["doc_uid"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 profile for the FaithfulMed datasets.")
    parser.add_argument("--sample", type=int, default=None, help="files per MedQuAD collection")
    parser.add_argument("--recon", action="store_true", help="MedQuAD tag inventory only")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="folder the numbered run logs are written to",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="filename prefix; defaults to the mode that ran, e.g. dataset_profile_full",
    )
    parser.add_argument("--pad", type=int, default=2, help="digits in the run number")
    args = parser.parse_args()

    mode = "recon" if args.recon else ("sample" if args.sample else "full")
    prefix = args.prefix or f"dataset_profile_{mode}"
    out_path = next_run_path(args.out_dir, prefix, args.pad)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
        print(f"# {out_path.name}")
        print(f"# started : {datetime.now().isoformat(timespec='seconds')}")
        print(f"# mode    : {mode}")
        print(f"# args    : sample={args.sample} recon={args.recon}")
        print(f"# project : {PROJECT_ROOT}")
        print()
        run_profile(args)

    out_path.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nrun log: {out_path}")


if __name__ == "__main__":
    main()
