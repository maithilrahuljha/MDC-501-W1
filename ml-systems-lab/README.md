# From Notebook to Production — the hands-on lab package

Companion lab for the chapter **"From Notebook to Production: Architecting Real-World ML Systems."**
B.Tech, one 3-hour session (or two 90-minute sessions), Python 3.10+.

> **New to terminals, Git, or Python?** Stop reading this file.
> Open **`instructions.txt`** and follow it like a recipe. It assumes you know nothing,
> explains every click, and has a troubleshooting section. This README is for people
> who already know what `pip` is.

---

## The four ideas, and where each one lives

| # | Idea | Where it happens | What the student sees |
|---|------|------------------|-----------------------|
| 1 | A notebook is not a system | `lab1_break_notebook/` | One good request and three crashes: missing column, wrong type, unseen category — each mapped to the pillar that would have prevented it |
| 2 | Static vs dynamic training | `lab3_drift/` | A plot where the frozen model decays from ~80% to ~50% after the world changes, while the retrained one recovers to ~89% |
| 3 | Batch vs online inference | `lab4_batch_vs_online/` | The same model, the same 10,000 rows: milliseconds in batch, ~40 seconds as 10,000 HTTP calls |
| 4 | Store, share, deploy | GitHub + `hf_space/` + `streamlit_space/` | A public URL anyone can open, and a CI pipeline that re-tests the work on every push |

## The five pillars used everywhere in this package

Every failure message names one of these. Students can repeat them from memory:

1. **Data validation** — check what arrives before the model touches it.
2. **Testing** — prove it works automatically, every time.
3. **Versioning** — know which data, code and model produced this answer.
4. **Serving contract** — a documented schema and a latency budget.
5. **Monitoring** — log what really happens, and alert when it drifts.

## Folder map

```
ml-systems-lab/
├── README.md                  <- you are here
├── requirements.txt           <- every dependency, pinned
├── instructions.txt           <- the plain-English guide (start here)
├── pyproject.toml             <- ruff + pytest config used by CI
├── lab1_break_notebook/       <- Idea 1
│   ├── train.py               <- generates data, trains, saves model + metadata
│   ├── app.py                 <- FastAPI: /predict (naive) and /predict_validated (same model, with a contract)
│   ├── test_requests.py       <- good request + 3 bad ones, explained
│   └── README.md
├── lab3_drift/                <- Idea 2
│   ├── drift_experiment.py    <- static vs dynamic under simulated drift (runs in Colab too)
│   └── README.md
├── lab4_batch_vs_online/      <- Idea 3
│   ├── online.py              <- FastAPI, one customer at a time
│   ├── batch.py               <- 10,000 rows in one call
│   ├── benchmark.py           <- times both, prints one table
│   └── README.md
├── lab6_vertex/               <- managed pipelines (optional, cloud)
│   └── pipeline.py            <- --mode local (MLflow) or --mode vertex (Vertex AI)
├── hf_space/                  <- deploy target, Plan A (Hugging Face Spaces)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md              <- the Space's own metadata card
├── streamlit_space/           <- Plan B deploy: ONE app, FOUR labs (sidebar: Lab 1 / 3 / 4 / 6)
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── tests/                     <- 18 tests, all offline, ~4 seconds
└── .github/workflows/ci.yml   <- lint + test + a real server smoke test (+ commented-out deploy job)
```

The repo root also has `.github/workflows/ci.yml` — the same workflow, because GitHub
only reads `.github/` at the **root** of a repository.

## Quickstart (for the impatient instructor)

```bash
cd ml-systems-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Lab 1** — two terminals:
```bash
cd lab1_break_notebook && python train.py && python app.py    # terminal 1
cd lab1_break_notebook && python test_requests.py             # terminal 2
```

**Lab 3**
```bash
cd lab3_drift && python drift_experiment.py     # writes outputs/drift_accuracy.png
```

**Lab 4** — two terminals:
```bash
cd lab4_batch_vs_online && python online.py       # terminal 1
cd lab4_batch_vs_online && python batch.py && python benchmark.py   # terminal 2
```

**Lab 6**
```bash
cd lab6_vertex && python pipeline.py --mode local    # free, writes ./mlruns
cd lab6_vertex && python pipeline.py --mode vertex   # prints a friendly setup message if no GCP credentials
```

**Tests + lint** (what CI runs):
```bash
pytest tests -v
ruff check .
```

## Design decisions worth defending in class

- **Everything is synthetic and generated in the script.** No downloads, no logins, no
  licences, no student who cannot start because the internet is slow. It also means the
  results are reproducible from a seed.
- **The failures are loud on purpose.** `app.py` returns the raw error text and prints a
  banner. Real services must not do this; the lab makes the failure impossible to miss.
- **`/predict` is deliberately broken, `/predict_validated` is the fix.** Both use the
  same fitted pipeline. The only difference is a contract — which is the whole lesson.
- **Load testing is a ~40-line asyncio/httpx script, not locust.** The question here is
  "what does per-request overhead cost?", not "how many users survive?". A script the
  student can read in one screen beats a tool with a dashboard. Locust is the right
  answer to a different question, and the file says so.
- **MLflow for Lab 6's local mode, Vertex AI for cloud mode, one file.** The mapping
  between them is written in the comments, so the cloud half is optional and free.

## Requirements and known platform notes

- Tested on Python 3.11 with the pins in `requirements.txt` (Windows, macOS, Linux).
- Lab 1 and Lab 4 bind to `127.0.0.1` — they are local teaching services, not deployments.
- **Hugging Face Spaces free tier changed in 2026:** creating Gradio/Docker Spaces on
  free CPU hardware now usually requires a paid PRO plan; free accounts in good standing
  may still run up to two Gradio Spaces on ZeroGPU. That is why `streamlit_space/` exists
  as a genuinely free Plan B. `instructions.txt` Part H walks through both, in order.
- **`streamlit_space/app.py` is a multi-lab dashboard**, not a single model form: a
  sidebar switches between Lab 1 (the three real failures), Lab 3 (a smaller drift
  run), Lab 4 (batch vs single-row, in-process and labelled as such) and Lab 6
  (a real training run plus the MLflow/Vertex map). A deployed app runs exactly one
  entrypoint file, so the tabs live in one file rather than across five Spaces.

## Licence and data

Course material for MDC-501. Code is MIT. All data is synthetic; no real customer
records are used anywhere in this repository.
