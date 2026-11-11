# 👋 Getting Started

**Welcome to your AI Studio Challenge Project repository!**

This repo is your team's workspace for Fall 2026 AI Studio. Everything you work on — task planning, notebooks, data, documentation — will live here. Make it your own, use it to collaborate, and consult with your AI Studio Coach and Challenge Advisor if you need help.

If you're not familiar with how GitHub works, or need a refresher, you can find helpful documentation [here](https://docs.github.com/en/get-started).

To start adding project tasks aligned with your Challenge Project's monthly milestones, use the Projects tab [here](https://github.com/Break-Through-Tech/fall-ai-studio-2026-challenge-project-template-repo/projects)

> 💡 **Note:** You can delete this file once your team is up and running; it's just here to help you get oriented.

---

## 📂 What's in This Repository

| File/Folder                                                    | What Is It?                                                                                                     | Creator           |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------- |
| [Challenge-Project-Overview.md](Challenge-Project-Overview.md) | Project details and guidance from your Challenge Advisor to help your team get started                          | Challenge Advisor |
| [data](data)                                                   | Where project dataset files get stored (e.g., `.csv`, `.json`, `.parquet`, `.zip`)                              | Challenge Advisor |
| [README.md](README.md)                                         | Where your team will document your project work; this will become the core of your AI Studio portfolio artifact | Fellows           |
| [notebooks](notebooks)                                         | Where your team's Jupyter notebooks for exploration, analysis, and modeling get stored (e.g., `.ipynb`)         | Fellows           |
| [requirements.txt](requirements.txt)                           | A place to list Python packages your project uses (e.g., `pandas`, `scikit-learn`)                              | Fellows           |
| [gitignore](.gitignore)                                        | A place to list files Git should ignore (e.g., system files like .DS_Store)                                     | Fellows           |

---

## Running the Google ADK Pipeline (Windows)

Use Python 3.12 and a short virtual-environment path to avoid Windows path-length issues when installing ADK.

```powershell
py -3.12 -m venv "$HOME\venvs\faithfulmed-adk"
& "$HOME\venvs\faithfulmed-adk\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your own Google AI Studio API key. Keep `.env` private; it is excluded from Git. In VS Code, select the environment at `$HOME\venvs\faithfulmed-adk\Scripts\python.exe` using **Python: Select Interpreter**.

Start the local ADK interface from the repository root:

```powershell
adk web faithfulmed
```

Then open `http://127.0.0.1:8000` in your browser. Stop the server with `Ctrl+C`.

## Streamlit Pipeline Visualizer

After completing the environment and API key setup above, run this from the
repository root. The integrated Simplifier also needs the team's prepared
Chroma database at `chroma/`, including `chroma.sqlite3` and its companion
folders, with the `medical_docs` collection. Ask a teammate for the complete
database; an empty Chroma database is not sufficient. To use a different
location, set `FAITHFULMED_CHROMA_PATH` to the database directory.

```powershell
python -m streamlit run streamlit_app.py
```

Open the local address printed in the terminal. The app starts with a fictional
example note; enter a note and select **Run pipeline** to see agent outputs as
they arrive. The verifier and refiner may each produce more than one output.
Stop the server with `Ctrl+C`.

The readability section calculates Flesch-Kincaid grade and Flesch reading ease
on the final output, including headings, and compares the grade with the
project's grade 8 target. These automated estimates do not measure clinical
accuracy, medical jargon, or actual patient understanding; short texts can be
less reliable.

The verifier section shows the latest report's faithfulness, additions,
omissions, total facts, hallucination and revision flags, and false coverage.
These are model judgments and do not establish clinical accuracy. If the
Refiner changes the text after the last verifier pass, the app warns that the
report does not assess the final text.

This visualizer is not a production medical system.
Use only fictional or approved de-identified notes: submitted text is sent to
the configured model provider. A working API key, network access, and model
quota are required. If a stage fails, the app shows the error and any partial
outputs rather than marking the run as successful.
