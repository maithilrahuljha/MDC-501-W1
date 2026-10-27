# MDC-501-W1 — From Notebook to Production

Hands-on lab package for the chapter **"From Notebook to Production: Architecting
Real-World ML Systems."**

Everything lives in [`ml-systems-lab/`](ml-systems-lab/).

| If you are… | Open this |
|---|---|
| A student who has never used a terminal | **[`ml-systems-lab/instructions.txt`](ml-systems-lab/instructions.txt)** — plain English, every click, troubleshooting |
| An instructor who knows Python | [`ml-systems-lab/README.md`](ml-systems-lab/README.md) — folder map, quickstart, design decisions |

## What the four labs teach

1. **`lab1_break_notebook/`** — a model in a notebook is not a system. One good
   request and three crashes (missing column, wrong type, unseen category), each
   mapped to the pillar that would have prevented it.
2. **`lab3_drift/`** — static vs dynamic training. A frozen model decays from ~80%
   to ~50% accuracy after the world changes; a retrained one recovers to ~89%.
   Prints and saves the accuracy-vs-sample-number plot.
3. **`lab4_batch_vs_online/`** — the same model, 10,000 rows, two ways: one
   vectorised call (milliseconds) vs 10,000 sequential HTTP requests (~40 seconds).
   Prints the comparison table and average online latency.
4. **GitHub + `hf_space/` (or `streamlit_space/`)** — store, share and publish the
   work at a public URL, with a CI workflow that lint-checks and tests every push.

Also included: `lab6_vertex/pipeline.py`, a minimal managed-pipeline example that
runs locally on MLflow (`--mode local`) or on Vertex AI (`--mode vertex`) from the
same file, with the open-source equivalent of every cloud call in a comment.

All data is synthetic and generated inside the scripts. No logins, no downloads,
no API keys. Python 3.10+, dependencies pinned in
[`ml-systems-lab/requirements.txt`](ml-systems-lab/requirements.txt).
