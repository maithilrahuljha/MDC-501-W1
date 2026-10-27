# Lab 4 — Batch vs Online Inference

**Idea:** the same trained model can be called in two shapes, and the shape — not the
model — decides your cost, your latency and your architecture.
**Time:** 30–40 minutes. **Pillars:** 2 (testing), 4 (serving contract), 5 (monitoring).

## The three files

| File | What it does |
|------|--------------|
| `online.py` | FastAPI service. One customer per HTTP `POST /predict`. Validated with pydantic. |
| `batch.py` | Reads a CSV of customers, scores all of them in **one** `predict_proba` call, writes a predictions CSV. |
| `benchmark.py` | Runs both on the same 10,000 rows, prints one comparison table. |

`online.py` exposes `load_or_train_model()`, which reuses Lab 1's `model.joblib` if it
exists. Both halves therefore use **byte-identical weights** — that is what makes the
comparison honest.

## Run it

Two terminals:

```bash
# terminal 1
cd lab4_batch_vs_online
python online.py

# terminal 2
cd lab4_batch_vs_online
python batch.py
python benchmark.py            # add --rows 2000 for a faster classroom run
```

`benchmark.py` takes about 40 seconds at 10,000 rows (it sends 10,000 real HTTP
requests, one at a time, on purpose).

## What you should see

```
METHOD                                ROWS    TOTAL (s)   PER ROW (ms)
------------------------------------------------------------------------
Batch (1 call, in-process)          10,000        0.004         0.0004
Online (sequential HTTP)            10,000       39.321         3.9321
------------------------------------------------------------------------

  Average latency per online request : 3.92 ms
  Median  latency per online request : 3.85 ms
  95th percentile (p95) latency      : 4.73 ms
  Online requests per second         : 254
  ONLINE IS SLOWER BY A FACTOR OF    : 10,164.8x
```

Your numbers will differ with your machine. The **ratio will not**: roughly four
orders of magnitude, because the model arithmetic is microseconds and the HTTP
machinery is milliseconds.

## How to read the ratio honestly

The factor is huge and the *lesson is not* "online is bad". Per request, 3.9 ms is
excellent — that is the number your user feels, and it is 20× under a 50 ms budget.
The lesson is:

- Batch does 10,000 answers at ~0.4 microseconds each. It cannot put a 3.9 ms answer
  in front of a human *now*.
- Online pays ~4 ms of overhead per answer, forever, and needs a process, a port, a
  health check, a deployment and an on-call rota.
- **Batch wins on cost. Online wins on freshness.** Choose by asking "when does the
  answer need to exist?", not "what is cool?".

## Why a script instead of locust?

Locust answers *"how many concurrent users can this survive?"* — a load question.
This lab answers *"what does per-request overhead cost?"* — a comparison question.
A 40-line `httpx` loop prints one honest table that a first-year student can read.
Load testing with locust is a good *next* lab; it is the wrong tool for this one.
(The file header says the same thing, for the student who wonders.)

## Try this next

1. `python benchmark.py --rows 20000` — does the ratio change? (It should barely move.)
2. Open `http://127.0.0.1:8000/docs`, expand `POST /predict`, click **Try it out**, send
   `contract_type: "lifetime"`. Watch the 422. That is Pillar 4 protecting the model.
3. Run `python online.py` in two terminals on different ports (`PORT=8001`) and think
   about what you would need to do to serve 1,000 users per second. (Hint: more than
   one process, and a health check that fails loudly.)
4. Compare `outputs/predictions_*.csv` with Lab 1's model. Identical probabilities for
   identical rows — the `tests/` suite asserts exactly this.

## Common problems

- **`[OFFLINE] ... while contacting http://127.0.0.1:8000`** → `online.py` is not
  running, or you started it in a different folder/port.
- **`Address already in use`** → Lab 1's `app.py` is still on port 8000. Close it, or
  use `PORT=8001 python online.py` and `python benchmark.py --port 8001`.
- **The benchmark is slow** → it is doing exactly as many round trips as you asked for.
  Use `--rows 2000` during class and `--rows 10000` for the homework screenshot.
