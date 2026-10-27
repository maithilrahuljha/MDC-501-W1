# Lab 1 — Break the Notebook

**Idea:** a model in a notebook is not a production system.
**Time:** 35–45 minutes. **Pillars:** 1 (data validation), 2 (testing), 4 (serving contract), 5 (monitoring).

## What you get

| File | What it is |
|------|------------|
| `train.py` | Generates 4,000 synthetic telecom customers, trains a scikit-learn pipeline, saves `artifacts/model.joblib` and `artifacts/model_metadata.json` |
| `app.py` | A FastAPI service with **two** doors: `/predict` (trusts everyone, crashes) and `/predict_validated` (same model, enforces a contract) |
| `test_requests.py` | Sends one good request and three bad ones, prints what each failure means and which pillar prevents it |

## Run it

Two terminals. This is not decoration — the service must keep running while the
second terminal talks to it.

```bash
# terminal 1
cd lab1_break_notebook
python train.py
python app.py

# terminal 2
cd lab1_break_notebook
python test_requests.py
```

## What you should see

```
Health check      : HTTP 200 -> {"status":"ok","model_version":"1.0.0",...}

CASE 1 - VALID REQUEST ........................ /predict 200   /predict_validated 200
CASE 2 - (a) MISSING COLUMN ................... /predict 500   /predict_validated 422
CASE 3 - (b) WRONG DATA TYPE ................. /predict 500   /predict_validated 422
CASE 4 - (c) UNSEEN CATEGORY ................. /predict 500   /predict_validated 422

RESULT: the service behaved exactly as Lab 1 intends.
```

The exact error strings, if you want to check them by hand:

| Failure | The naive door says |
|---------|--------------------|
| (a) missing column | `ValueError: columns are missing: {'monthly_charges'}` |
| (b) wrong type | `ValueError: could not convert string to float: 'twelve'` |
| (c) unseen category | `ValueError: Found unknown categories ['lifetime'] in column 0 during transform` |

The fixed door answers `HTTP 422` with a sentence a human can act on, e.g.
`monthly_charges: Field required`.

## Why each failure happens — say this out loud in class

- **(a) Missing column.** `pd.DataFrame([payload])` builds whatever the caller sent.
  scikit-learn then looks for `monthly_charges` and it is not there.
  **Pillar 1 (data validation)** is the fix: reject incomplete rows at the door.
- **(b) Wrong type.** `"twelve"` is a string. The model needs a number. JSON has types,
  but `dict` does not enforce them.
  **Pillar 4 (serving contract)** is the fix: publish the schema, enforce it at the edge.
- **(c) Unseen category.** `OneHotEncoder(handle_unknown="error")` is the scikit-learn
  default, and it is doing its job: it refuses to guess. The real problem is that the
  world invented a new contract type and nobody noticed.
  **Pillar 1** blocks it; **Pillar 5 (monitoring)** is how you find out it happened at all.

All three would also have been caught by **Pillar 2 (testing)** — one test per failure
mode, run on every push. See `../tests/test_lab1_contract.py`.

## Try this next (5-minute extensions)

1. Open `http://127.0.0.1:8000/docs` and send the same bad requests from the browser.
   The fixed endpoint's form will physically not let you submit nonsense.
2. Change `handle_unknown="error"` to `"ignore"` in `train.py`, retrain, restart the app,
   and re-run the tests. The unseen category stops crashing — and quietly becomes a
   all-zeros row. Now it is *silently wrong* instead of *loudly broken*. Which is worse?
3. Add a fourth failure of your own. Send `monthly_charges: -50` and see what happens.
   Why is "the model accepted it" not the same as "the answer is valid"?

## Common problems

- **`ModuleNotFoundError: No module named 'fastapi'`** → you installed into a different
  Python. Run `python -m pip install -r ../requirements.txt` with the same `python` you
  use to run the script.
- **`[app.py] I cannot find artifacts/model.joblib`** → run `python train.py` first, in
  this folder.
- **`Port 8000 is already in use`** → an old `app.py` is still running. Close that
  window, or run `PORT=8001 python app.py` and set `BASE_URL` at the top of
  `test_requests.py` to match.
- **`Could not reach the service`** → terminal 1 is not running, or it is still starting.
