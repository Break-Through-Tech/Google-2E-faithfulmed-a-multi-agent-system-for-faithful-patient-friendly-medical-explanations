# FaithfulMed: A Multi-Agent System for Faithful, Patient-Friendly Medical Explanations

---

### 👥 **Team Members**

| Name             | GitHub Handle  | Contribution                                                             |
|------------------|----------------|--------------------------------------------------------------------------|
| Jenna Hunte      | @jehunte       | Break Through Tech Coach                                                 |
| Claudia Jimenez  | @cjimenez909   | Break Through Tech Fellow                                                |
| Hoang Ngo        | @hoangngo-sudo | Break Through Tech Fellow                                                |
| Sarada Varshini  | @saraa-siva    | Break Through Tech Fellow                                                |
| Katelyn Louie    | @katelynlouie  | Break Through Tech Fellow                                                |
| Tionge Kabwe     | @KettyKay      | Break Through Tech Fellow                                                |
| Milka Zekarias   | @MilkaZek      | Break Through Tech Fellow                                                |
| Samuel Tuffour   | @samueltuffour | Break Through Tech Fellow                                                |
| Jaiden Ha        | @jaidenha      | Break Through Tech Fellow                                                |

---

## 🏗️ **Project Overview**

This project is part of the **Google AI Studio Challenge Program** (Fall 2026), hosted by **Google**, with guidance from Challenge Advisor **Samaneh Kazemi Nafchi** and AI Studio Coach **Jenna Hunte**.

### The Problem

Health information is one of the hardest categories to make truly accessible. Clinical notes, lab reports, and discharge instructions are written for clinicians, not patients. This creates a **health-literacy gap** — patients routinely misunderstand care instructions written above their reading level, leading to poor health outcomes.

Additionally, while generative AI can help translate clinical language into patient-friendly text, **hallucination and unfaithfulness** are critical barriers to patient-facing deployment. A system that invents unsupported medical claims could directly harm patients if trusted.

### Our Solution

**FaithfulMed** combines five specialized agents in a deterministic multi-agent pipeline to produce patient-friendly medical explanations that are simultaneously:

1. **Readable** at ≤8th-grade level (supporting health-literacy goals)
2. **Faithful** to source documents (provably grounded, no hallucinations)
3. **Better than a single-agent baseline** (demonstrating the value of the multi-agent approach)

### Core Research Question

> Does a pipeline of specialized agents produce explanations that are *both* more readable *and* more faithful than a single well-prompted LLM — and **which agents actually matter** for each goal?

### The Multi-Agent Pipeline

```
Clinical Note
     ↓
[Extractor] → Extract clinical "atoms" (facts with types and source quotes)
     ↓
[Simplifier] → Rewrite in plain language using retrieval-augmented generation (RAG)
     ↓
[Verifier] → LLM-as-judge: score faithfulness, flag hallucinations and omissions
     ↓
[Refiner] ↔ [Verifier] → (bounded loop, max 2 iterations) Fix flagged passages
     ↓
[Readability] → Adjust grammar, sentence length, and tone for target grade level
     ↓
Patient-Friendly Explanation
```

---

## 🎯 **Project Highlights**

- **Multi-agent orchestration using Google Agent Development Kit (ADK):** Implements a deterministic DAG (Directed Acyclic Graph) that chains five specialized LLM agents, avoiding open-ended recursion and enabling reproducible traces.
- **Faithfulness verification at scale:** Calibrated an LLM-as-judge Verifier against human-annotated medical QA data (MedAESQA) to achieve ≥80% agreement (Cohen's κ ≥ 0.6).
- **RAG grounding:** Leverages Gemini embeddings and ChromaDB to ground simplifications in medical reference material (MedlinePlus, clinical guidelines).
- **Comprehensive evaluation:** Measures readability (Flesch-Kincaid, SMOG), faithfulness, hallucination rate, and system performance against a single-agent baseline.
- **Ablation studies:** Quantifies the marginal contribution of each agent to readability and faithfulness scores.
- **Model comparison leaderboard:** Benchmarks Gemini (API), open-weight Gemma-2, and external models on cost, latency, openness, and medical accuracy.

---

## 👩🏽‍💻 **Setup and Installation**

### Quick Start

To run the interactive Streamlit pipeline visualizer:

```bash
# 1. Clone the repository
git clone https://github.com/Break-Through-Tech/Google-2E-faithfulmed-a-multi-agent-system-for-faithful-patient-friendly-medical-explanations.git
cd Google-2E-faithfulmed-...

# 2. Create a virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv  # Windows PowerShell
# or
python3.11 -m venv .venv  # macOS/Linux

# 3. Activate and install dependencies
# Windows PowerShell:
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# macOS/Linux:
./.venv/bin/python -m pip install -r requirements.txt

# 4. Create .env file with your Gemini API key
echo GEMINI_API_KEY=your_key_here > .env

# 5. Add the prepared Chroma vector database to chroma/

# 6. Run Streamlit
# Windows PowerShell:
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py

# macOS/Linux:
./.venv/bin/python -m streamlit run streamlit_app.py
```

**Full setup guide with platform-specific instructions:** See [DEMO.md](DEMO.md).

### Requirements

- Python 3.11+
- Google AI Studio API key (free tier available at https://aistudio.google.com)
- Prepared Chroma vector database (request from team)
- Dependencies in `requirements.txt`: `google-adk`, `google-genai`, `chromadb`, `streamlit`, `pandas`, `textstat`

### Important Notes

- The visualizer is an **experimental prototype**, not medical advice.
- Use only **fictional or approved de-identified notes**; submitted text is sent to the configured model provider.
- Do not commit your `.env` file or API keys to the repository.

---

## 📊 **Data & Datasets**

This project uses publicly available, de-identified, synthetic, or expert-curated data. **No PHI/PII is involved.**

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| **MedAESQA** ⭐ | Verifier calibration + faithfulness evaluation | 40 questions, 7,651 evidence excerpts | https://osf.io/ydbzq |
| **MTSamples** | Source clinical documents to simplify | ~5,000 de-identified medical reports | https://www.mtsamples.com/ |
| **PLABA** | Gold pairs for plain-language reference | Professional ↔ lay-language text pairs | https://osf.io/rnpmf/ |
| **MedQuAD** | RAG knowledge base + evaluation | ~47k consumer-health Q&A | https://github.com/abachaa/MedQuAD |
| **Synthea** | Synthetic patient records for stress testing | Generate as needed (FHIR format) | https://synthetichealth.github.io/synthea/ |
| **MedlinePlus Glossary** | Lay-language medical definitions | Public, free-to-use glossary | https://medlineplus.gov/ |

**Key insight:** MedAESQA is the backbone—it provides human-annotated faithfulness labels used to calibrate the Verifier agent and evaluate system performance.

---

## 🧠 **Architecture & Agents**

### Agent Specifications

| Agent | Input | Output | Key Responsibility |
|-------|-------|--------|------------------|
| **Extractor** | Clinical note | List of clinical atoms (facts, types, source quotes) | Structured extraction of medical concepts with evidence anchoring |
| **Simplifier** | Atoms + clinical note | Plain-language draft + reference passages | Rewriting with RAG-grounded terminology and explanations |
| **Verifier** | Draft explanation + source note | Faithfulness score, flagged issues, evidence map | Scoring factual fidelity; detecting hallucinations and omissions |
| **Refiner** | Verification report + source note | Revised explanation | Targeted fixes to flagged passages (bounded loop) |
| **Readability** | Refined explanation | Final patient-facing text | Adjust to target grade level (≤8th grade) without losing accuracy |

### Technology Stack

- **Models:** Google Gemini (API), Gemma-2 (open weights), + external open model (comparison)
- **Orchestration:** Google Agent Development Kit (ADK) — `SequentialAgent`, `LoopAgent`, `LlmAgent`
- **RAG:** Gemini `text-embedding-004` embeddings + ChromaDB vector store
- **Evaluation:** `textstat` (Flesch-Kincaid, SMOG), `scikit-learn` (Cohen's κ)
- **Frontend:** Streamlit (dev prototype)
- **Code structure:** Modular agents in `faithfulmed/agents/`, shared utilities in `faithfulmed/`

---

## 📈 **Results & Evaluation**

### Success Criteria (Target Metrics)

| Criterion | Good | Moderate | Poor |
|-----------|------|----------|------|
| **Readability** | ≥80% outputs at ≤8th-grade level | 50–79% at ≤8th-grade | <50% at ≤8th-grade |
| **Faithfulness** | ≥85% factual fidelity (human-labeled test set) | 70–84% fidelity | <70% fidelity |
| **Hallucination rate** | <10% of outputs with unsupported claims | 10–20% with claims | >20% with claims |
| **Multi-agent vs. baseline** | ≥15% absolute improvement over single-agent | 7–14.99% improvement | 0–6.99% improvement |
| **Verifier agreement** | ≥80% with human annotators (κ ≥ 0.6) | 70–79.99% agreement | <70% agreement |

### Key Findings

*(Results to be completed as experiments conclude; update with figures and visualizations from test runs)*

- **Readability improvements:** [pending ablation]
- **Faithfulness gains:** [pending multi-agent vs. baseline comparison]
- **Hallucination detection:** Verifier performance on held-out test set [pending]
- **Most impactful agent(s):** [pending ablation analysis]
- **Model leaderboard:** Gemini vs. Gemma-2 vs. external model [pending]

### Evaluation Artifacts

- Ablation table: marginal contribution of each agent
- Confusion matrix: Verifier accuracy vs. human labels
- Leaderboard: faithfulness, readability, latency, cost per model
- Error analysis: which content types (lab values, drug names, procedures) are most prone to hallucination?

---

## 🚀 **Getting Started for Developers**

### Project Structure

```
Google-2E-faithfulmed.../
├── README.md                    # This file
├── DEMO.md                      # Detailed setup and troubleshooting
├── Challenge-Project-Overview.md # Full project scope and stretch goals
├── SUCCESS_RUBRIC.md            # Quantitative evaluation criteria
├── requirements.txt             # Python dependencies
├── .env                         # (Create locally: add GEMINI_API_KEY)
├── streamlit_app.py             # Streamlit visualizer
│
├── faithfulmed/                 # Main package
│   ├── __init__.py
│   ├── agent.py                 # ADK agent definition
│   ├── adk_pipeline.py          # Sequential pipeline orchestration
│   ├── retrieval_utils.py       # RAG utilities (Chroma, embeddings)
│   ├── readability.py           # Readability metrics and rewrites
│   ├── pipeline_ui.py           # Streamlit UI helpers
│   ├── medlineplus_aliases.json # Lay-language term mappings
│   │
│   └── agents/                  # Individual agent implementations
│       ├── extractor.py         # Clinical fact extraction
│       ├── simplifier.py        # Plain-language rewriting
│       ├── verifier.py          # Faithfulness scoring + verification
│       ├── refiner.py           # Targeted fixes
│       └── readability_agent.py # Grade-level targeting
│
├── notebooks/                   # Development and analysis
│   ├── eval_harness.py          # Evaluation framework
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_single_agent.ipynb
│   └── ...
│
├── data/                        # Datasets (download/prepare)
│   ├── README.md
│   ├── MedAESQA/
│   ├── MTSamples/
│   └── evaluation/
│
├── evaluation/                  # Results and analysis
│   ├── ablations.csv
│   ├── leaderboard.csv
│   └── ...
│
└── chroma/                      # Vector database (prepared by team)
    └── chroma.sqlite3
```

### Quick Development Setup

1. **Clone and install:**
   ```bash
   git clone ...
   cd Google-2E-...
   py -3.11 -m venv .venv
   & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
   ```

2. **Get API key:**
   - Visit https://aistudio.google.com
   - Create a free Gemini API key
   - Add to `.env`: `GEMINI_API_KEY=your_key`

3. **Get Chroma database:**
   - Request the prepared `chroma/` folder from the team
   - Extract to `chroma/` in repo root (should contain `chroma.sqlite3` + subfolders)

4. **Run a small test:**
   - Start Streamlit: `& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app.py`
   - Paste the synthetic note from [DEMO.md](DEMO.md#6-try-a-small-test)
   - Verify atoms, grounding, draft, and verification report appear

5. **Run ADK Web for agent debugging:**
   ```bash
   & ".\.venv\Scripts\adk.exe" web --host 127.0.0.1 --port 8000
   ```
   - Navigate to `http://127.0.0.1:8000`
   - Select `faithfulmed` and inspect agent traces

### Next Steps

For detailed development guidance:

- **[Challenge-Project-Overview.md](Challenge-Project-Overview.md):** Full project scope, success criteria, milestones, and stretch goals.
- **[DEMO.md](DEMO.md):** Step-by-step local setup, platform-specific commands, troubleshooting.
- **[SUCCESS_RUBRIC.md](SUCCESS_RUBRIC.md):** Evaluation metrics and rubric.

---

## 📝 **License**

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

**Data attribution:**
- MedAESQA: CC BY 4.0 (cite: Deepak et al., 2023)
- PLABA: Research-use; cite Attal et al. (*Scientific Data* 2023)
- MTSamples: De-identified; handle with care per terms of use
- MedQuAD, MedlinePlus: Public resources per NIH/NLM policies

---

## 📚 **Resources & References**

### Core Reading

- **Health Literacy:** AHRQ and CDC health-literacy resources; why ≤8th-grade is the target.
- **LLM Faithfulness:** FaithBench; hallucination detection in abstractive summarization.
- **Multi-Agent Systems:** Google ADK docs; LangGraph, CrewAI (alternative frameworks).
- **Medical QA & Evaluation:** MedAESQA, MedQA, PubMedQA papers; Cohen's κ for inter-rater agreement.

### Key Links

- [Google AI Studio](https://aistudio.google.com) — Free Gemini API key
- [Google Agent Development Kit](https://google.github.io/adk-docs) — ADK Python docs
- [Gemini API Docs](https://ai.google.dev) — Model reference
- [Gemma Open Models](https://ai.google.dev/gemma) — Open-weight LLM alternative
- [ChromaDB Docs](https://docs.trychroma.com/) — Vector database for RAG
- [Textstat Library](https://pypi.org/project/textstat/) — Readability metrics
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces) — Free demo hosting

---

## 🙏 **Acknowledgements**

We gratefully acknowledge:

- **Challenge Advisor:** Samaneh Kazemi Nafchi (Google) for expert guidance, technical review, and project oversight.
- **AI Studio Coach:** Jenna Hunte (Break Through Tech) for mentorship and program support.
- **Google:** For providing the Agent Development Kit, Gemini API access, and Gemma open models; and for framing this challenge at the intersection of trustworthy AI and equitable access.
- **Data providers:** Deepak et al. (MedAESQA), Attal et al. (PLABA), MIT MIMIC (MTSamples), NIH/NLM (MedQuAD, MedlinePlus).
- **Break Through Tech:** For organizing the AI Studio program and supporting early-career researchers in building AI systems for good.

---

## ❓ **Questions & Support**

- **GitHub Issues:** Use the repo's Issues tab for bugs, feature requests, and design discussions.
- **Biweekly Lab Meetings:** Official check-ins during Break Through Tech lab sections (2nd and 4th weeks of each month).
- **Discord:** Ask quick questions in the team's Break Through Tech Discord channel.
- **Email:** For urgent blockers, reach out to the AI Studio Coach first.

---

**Last updated:** November 2026  
**Status:** In active development (October–December 2026)  
**Maintainers:** @samueltuffour, @cjimenez909, @hoangngo-sudo, @saraa-siva, @katelynlouie, @KettyKay, @MilkaZek, @jaidenha
