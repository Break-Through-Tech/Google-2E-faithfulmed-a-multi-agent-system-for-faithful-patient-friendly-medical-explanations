# Run FaithfulMed in ADK Web

Use this guide to run the agents locally in your browser. ADK Web is for development, not patient use or production.

## Demo

https://github.com/user-attachments/assets/fda67231-eaf5-4f94-ba2d-e0eb15db6356

## 1. Open the project

Get the team's agreed code version and open the `Google-2E` folder in VS Code.
Open **Terminal**. Run the commands below from `Google-2E`, the folder containing `requirements.txt` and `faithfulmed/`.
Do not run them from inside `faithfulmed/` or `notebooks/`.

## 2. Install dependencies

Create a virtual environment once. It keeps this project's packages separate from other projects.
Python 3.11 is a reasonable starting choice. Install the versions in `requirements.txt`, not a separate latest ADK version.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### Git Bash on Windows

Use forward slashes and the `Scripts` folder:

```bash
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Windows path-length errors

If installation fails with `No such file or directory` for a very long path, the repository's long folder name is exceeding the Windows path limit.
Either clone the repository to a short path such as `C:\fm`, or create the environment in a short location and use its executables instead:

```powershell
py -3.11 -m venv C:\fm-venv
& "C:\fm-venv\Scripts\python.exe" -m pip install -r requirements.txt
& "C:\fm-venv\Scripts\adk.exe" web --host 127.0.0.1 --port 8000
```

If you already have a working environment, reuse it and adjust the executable paths below.
For example, an environment one folder above the repository uses `..\.venv\Scripts\adk.exe` on Windows.
These commands do not require environment activation.

## 3. Add your API key

Create a local `.env` file in `Google-2E` with these entries:

```dotenv
GEMINI_API_KEY=replace_with_your_own_key
```

Enter your own Google AI Studio key locally. Do not commit it, paste it into chat, or include it in screenshots.

## 4. Get the retrieval database

Ask the team for the prepared Chroma database and extract the entire folder into `Google-2E/chroma/`.
Keep `chroma.sqlite3` and its companion folders together.
A tiny `chroma.sqlite3` (a few hundred KB) with no collections is an empty database that Chroma created automatically. The prepared one is about 31 MB and contains `medical_docs` with 2,403 embeddings (3072 dimensions). The database is separate from the Python source code.

Expected layout:

```text
Google-2E/
  .env
  requirements.txt
  chroma/
    chroma.sqlite3
    ...other folders...
  faithfulmed/
    __init__.py
    agent.py
    adk_pipeline.py
    retrieval_utils.py
    medlineplus_aliases.json
    agents/
```

The default collection is `medical_docs`. Do not create an empty database or rebuild the index just to launch ADK Web.
For a database stored elsewhere, set `FAITHFULMED_CHROMA_PATH` to its absolute folder path before starting the server.

## 5. Start ADK Web

From `Google-2E`, use the command for your operating system.

### Windows PowerShell

```powershell
& ".\.venv\Scripts\adk.exe" web --host 127.0.0.1 --port 8000
```

### macOS or Linux

```bash
./.venv/bin/adk web --host 127.0.0.1 --port 8000
```

### Git Bash on Windows

```bash
./.venv/Scripts/adk.exe web --host 127.0.0.1 --port 8000
```

Keep the terminal running. Open `http://127.0.0.1:8000` and select **faithfulmed** in the agent menu.
If port 8000 is busy, use `--port 8001` and open the matching address.
Press **Ctrl+C** in the server terminal when you finish.

## 6. Try a small test

Start a **new session for each different note**. The pipeline can reuse `source_note` already stored in a session.
Paste the following synthetic note as the message. You do not need to provide agent instructions.

```text
The patient reports a mild headache. The patient denies nausea.
Blood pressure was 120/80. The patient's father has diabetes.
```

Look at the events and session state:

- `atoms` should contain extracted facts with source quotes and IDs.
- `grounding` should contain reference lists. An empty list means no matching passage, not necessarily a broken lookup.
- `draft` should contain the Simplifier's original text, source metadata and review warnings.
- `text` should contain the current patient-facing explanation.
- `verdict` should contain the Verifier's report if the run reaches that step.

Check that nausea stays absent, diabetes stays with the father, and `120/80` stays unchanged.
Do not rely only on the last chat bubble. Read `text` and the latest events to understand the outcome.
A completed run is not proof of medical accuracy. A clean Verifier report is a model judgment, not a human label.

The current Readability node is a placeholder rewrite step. It does not measure its own reading grade.
The Verifier/Refiner loop has a limit, so its last rewrite may not receive another verification before the run ends.

## 7. Test with a real MTSamples note

Use this after the small test works. This echocardiogram report is MTSamples row `1656` in the team's development split.
It comes from `data/evaluation/baseline_50_full.csv`. Use development notes for troubleshooting, not held-out test notes.

Start a new session and paste only the note below. The original comma-separated formatting is intentional.

```text
EXAM:, Echocardiogram.,INDICATION: , Aortic stenosis.,INTERPRETATION: , Transthoracic echocardiogram was performed of adequate technical quality. Left ventricle reveals concentric hypertrophy with normal size and dimensions and normal function. Ejection fraction is 60% without any obvious wall motion abnormality. Left atrium and right side chambers are of normal size and dimensions. Aortic root has normal diameter.,Mitral and tricuspid valves are structurally normal except for minimal annular calcification. Valvular leaflet excursion is adequate. Aortic valve reveals annular calcification. Fibrocalcific valve leaflets with decreased excursion. Atrial and ventricular septum are intact. Pericardium is intact without any effusion. No obvious intracardiac mass or thrombi noted.,Doppler reveals mild mitral regurgitation, mild-to-moderate tricuspid regurgitation. Estimated pulmonary pressure of 48. Systolic consistent with mild-to-moderate pulmonary hypertension. Peak velocity across the aortic valve is 3.0 with a peak gradient of 37, mean gradient of 19, valve area calculated at 1.1 sq. cm consistent with moderate aortic stenosis.,IN SUMMARY:,1. Concentric hypertrophy of the left ventricle with normal function.,2. Doppler study as above, most pronounced being moderate aortic stenosis, valve area of 1.1 sq. cm.
```

### What to check

Compare the note with both `draft.simplified_text` and the final `text`.
Wording and atom counts can vary. Do not require a word-for-word match to an earlier run.

| Source detail | Successful result |
| --- | --- |
| Ejection fraction `60%` | Keeps `60%` and does not confuse it with another measurement. |
| Estimated pulmonary pressure `48` | Keeps `48` without inventing a unit the note does not supply. |
| Peak velocity `3.0`, peak gradient `37`, mean gradient `19` | Keeps each value attached to its original measurement. |
| Valve area `1.1 sq. cm` | Keeps the value and unit unchanged. |
| Moderate aortic stenosis | Keeps the diagnosis and its moderate severity. |
| Mild mitral regurgitation and mild-to-moderate tricuspid regurgitation | Keeps the two findings and their different severity levels. |
| Mild-to-moderate pulmonary hypertension | Does not replace it with a different severity or condition. |
| No obvious wall motion abnormality, no effusion, no obvious intracardiac mass or thrombi | Does not turn absent findings into present findings or remove the word "obvious" where it limits certainty. |
| Reference passages | Uses supplied passages to explain terms, not to invent patient findings, treatment or advice. |

Check `draft.review_warnings` for terms without reference support or facts that used source wording.
Some terms may have no matching passage in the current database. Inspect those cases rather than expecting every fact to have a reference.

If `verdict.needs_revision` is true, inspect its reasons and whether the Refiner addresses them.
Check the last event as well: reaching the loop limit does not establish that the last text passed verification.
Readability passes only when a separate measurement of the final text gives Flesch-Kincaid grade 8 or below.
Record missing facts, changed values, unsupported additions and run errors for the relevant agent owner.

## Common problems

| Problem | What to check |
| --- | --- |
| `adk` is not recognized | Use the environment's full executable path shown above. Confirm dependency installation completed. |
| PowerShell blocks activation | Activation is unnecessary with these commands. Call the executables directly. |
| `faithfulmed` is missing from the menu | Start in `Google-2E`. Confirm `faithfulmed/agent.py` and `__init__.py` exist. Read import errors in the terminal. |
| Missing key or authentication error | Check your local `.env` and existing environment variables. Restart the server after changing credentials. |
| Missing collection or retrieval error (`Collection [medical_docs] does not exist`, or `MedlinePlus retrieval failed for F001`) | Check the Chroma folder, companion files and `medical_docs` collection. Use the team's prepared database. |
| `NameError: readability` when the agent loads | Pull the latest code. The Sequential agent must list `readability_agent`. |
| 429 or 503 response | The API is rate-limited or unavailable. Pause and retry later. Do not repeatedly submit the same note. |
| `ExtractionValidationError` | The reply failed source checks. Read the named checks in the error. It is not automatically an installation problem. |
| `SimplifierError` | Read the message to distinguish a retrieval failure from a draft problem. |
| A new message processes the old note | Start a new session. The previous session may still contain `source_note`. |

ADK can store sessions and artifacts under `faithfulmed/.adk/`.
Keep credentials, session data and note text out of commits (this is already gitignored). Use synthetic or team-approved de-identified examples when sharing screenshots.

## Reference

Official ADK Python quickstart: `https://google.github.io/adk-docs/get-started/python/`

To inspect the options supported by your installed version, run the same ADK executable with `web --help`.