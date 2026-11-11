"""Calibrate the Verifier against MedAESQA's human citation labels with Cohen's kappa.

MedAESQA experts judged each (answer sentence, cited PubMed abstract) pair as
supporting / contradicting / neutral / not relevant. We give the Verifier the
same pair (abstract = SOURCE, sentence = EXPLANATION) and compare labels.

Run step by step:
    python calibrate.py pairs      # 1. build a sample of pairs from medaesqa_v1.json
    python calibrate.py abstracts  # 2. download the cited abstracts from PubMed
    python calibrate.py run        # 3. run the Verifier on every pair
    python calibrate.py score      # 4. kappa + confusion matrix
"""
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data/medaesqa_v1.json")
OUT = Path("calibration_out")
SENT_KEY = "answer_sentence"   # confirmed from the dataset
PER_LABEL = 60                 # pairs sampled per human label (contradicting is rare, so we balance)
SEED = 0
SECONDS_BETWEEN_CALLS = 4.5    # free tier allows 15 requests per minute; set to 0 on a paid key

# MedAESQA label -> our 3 classes. "not relevant" means the abstract says nothing about the sentence.
HUMAN_MAP = {"supporting": "supported", "contradicting": "contradicted",
             "neutral": "not_supported", "not relevant": "not_supported", "not_relevant": "not_supported"}


def find_sentence_key(sentence_record: dict) -> str:
    """The sentence text is the one string field that is not an id or a label."""
    skip = {"answer_sentence_id", "answer_sentence_relevance", "citation_assessment"}
    for key, value in sentence_record.items():
        if key not in skip and isinstance(value, str):
            return key
    raise KeyError(f"No sentence text found; keys are {list(sentence_record)}. Set SENT_KEY by hand.")


def build_pairs():
    """Step 1: one row per (sentence, cited PMID) with the human label, balanced by label."""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    first = next(iter(data[0]["machine_generated_answers"].values()))["answer_sentences"][0]
    sent_key = SENT_KEY or find_sentence_key(first)
    print("Sentence text key:", sent_key, "| example:", first[sent_key][:80])
    rows = []
    for q in data:
        for system, answer in q["machine_generated_answers"].items():
            for s in answer["answer_sentences"]:
                for c in s.get("citation_assessment") or []:
                    label = HUMAN_MAP.get(str(c.get("evidence_relation")).lower())
                    if label:
                        rows.append({"id": f"{q['question_id']}|{system}|{s['answer_sentence_id']}|{c['cited_pmid']}",
                                     "question_id": q["question_id"], "sentence": re.sub(r"\s*\[[\d,\s]+\]", "", s[sent_key]),   # drop "[22014867]" citation tags
                                     "pmid": str(c["cited_pmid"]), "human": label})
    print("All pairs:", Counter(r["human"] for r in rows))

    random.seed(SEED)
    sample = []
    for label in ("supported", "contradicted", "not_supported"):
        group = [r for r in rows if r["human"] == label]
        sample += random.sample(group, min(PER_LABEL, len(group)))

    # Split by question so the dev set (for prompt tuning) never shares questions with the test set
    questions = sorted({r["question_id"] for r in sample})
    random.shuffle(questions)
    dev_qs = set(questions[: len(questions) // 2])
    for r in sample:
        r["split"] = "dev" if r["question_id"] in dev_qs else "test"

    OUT.mkdir(exist_ok=True)
    (OUT / "pairs.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print("Sampled:", Counter((r["split"], r["human"]) for r in sample))


def fetch_abstracts():
    """Step 2: download each cited abstract once from PubMed and cache it."""
    pairs = json.loads((OUT / "pairs.json").read_text(encoding="utf-8"))
    cache_dir = OUT / "abstracts"
    cache_dir.mkdir(exist_ok=True)
    for pmid in sorted({p["pmid"] for p in pairs}):
        path = cache_dir / f"{pmid}.txt"
        if path.exists():
            continue
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
        path.write_text(urllib.request.urlopen(url).read().decode("utf-8"), encoding="utf-8")
        time.sleep(0.4)   # NCBI allows ~3 requests per second without an API key


def collapse(report) -> str:
    """Turn the Verifier's claim labels for one sentence into one label."""
    labels = [c.label for c in report.claims]
    if "distorted" in labels:
        return "contradicted"
    if labels and all(l == "supported" for l in labels):
        return "supported"
    return "not_supported"   # unsupported, or background (no reference material is given here)


def run_verifier(split=None):
    """Step 3: run verify() on each pair; results are cached so a crash can resume."""
    from faithfulmed.agents.simplifier import SimplifiedOutput
    from faithfulmed.agents.verifier import verify

    pairs = json.loads((OUT / "pairs.json").read_text(encoding="utf-8"))
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    in_split = [p for p in pairs if not split or p["split"] == split]
    todo = [p for p in in_split if p["id"] not in results]
    print(f"{len(in_split) - len(todo)} of {len(in_split)} already done, {len(todo)} to run")
    start = time.time()

    for i, p in enumerate(todo, 1):
        source = (OUT / "abstracts" / f"{p['pmid']}.txt").read_text(encoding="utf-8")
        # The Verifier needs at least one fact; omissions are ignored in calibration.
        atoms = [{"text": p["sentence"], "type": "finding"}]
        # Add any other required SimplifiedOutput fields here if validation complains.
        draft = SimplifiedOutput(simplified_text=p["sentence"], source_metadata=[], atoms_covered=[])
        for attempt in range(3):
            try:
                report = verify(source, atoms, draft)
                results[p["id"]] = {"verifier": collapse(report),
                                    "claims": [c.model_dump() for c in report.claims]}
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:   # rate limited: wait for the quota to reset, then retry
                    print("Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                print("Failed", p["id"], str(e)[:200])
                break
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        # Progress line: count, percent, this pair's result, and time left
        left = (time.time() - start) / i * (len(todo) - i)
        outcome = results[p["id"]]["verifier"] if p["id"] in results else "FAILED"
        print(f"[{i}/{len(todo)}] {i / len(todo):4.0%}  human={p['human']:<13} verifier={outcome:<13} ~{left / 60:.1f} min left")
        time.sleep(SECONDS_BETWEEN_CALLS)


def score(split=None):
    """Step 4: Cohen's kappa, 3-class and binary, plus the confusion matrix."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    pairs = json.loads((OUT / "pairs.json").read_text(encoding="utf-8"))
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    done = [p for p in pairs if p["id"] in results and (not split or p["split"] == split)]
    human = [p["human"] for p in done]
    model = [results[p["id"]]["verifier"] for p in done]

    classes = ["supported", "contradicted", "not_supported"]
    print(f"n = {len(done)}  (split = {split or 'all'})")
    print("3-class kappa:", round(cohen_kappa_score(human, model, labels=classes), 3))
    to_bin = lambda l: "supported" if l == "supported" else "not"
    print("binary kappa (supported vs not):",
          round(cohen_kappa_score([to_bin(h) for h in human], [to_bin(m) for m in model]), 3))
    print("confusion matrix (rows = human, cols = verifier):", classes)
    print(confusion_matrix(human, model, labels=classes))

    disagreements = [{**p, **results[p["id"]]} for p in done if p["human"] != results[p["id"]]["verifier"]]
    (OUT / f"disagreements_{split or 'all'}.json").write_text(json.dumps(disagreements, indent=2), encoding="utf-8")
    print(f"{len(disagreements)} disagreements written to {OUT}/disagreements_{split or 'all'}.json")


if __name__ == "__main__":
    step = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else None   # optional: dev or test
    {"pairs": build_pairs, "abstracts": fetch_abstracts,
     "run": lambda: run_verifier(split), "score": lambda: score(split)}[step]()
