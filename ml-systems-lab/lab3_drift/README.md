# Lab 3 — Static vs Dynamic Training (a data-drift experiment)

**Idea:** a model trained once is a model that gets worse on its own.
**Time:** 25–35 minutes. **Pillars:** 3 (versioning), 5 (monitoring).

## The experiment in one paragraph

We simulate 100 days of customers, 100 per day. On **day 40** the world changes:
a competitor launches, prices rise, and the *relationship* between inputs and churn
flips (long tenure stops protecting people; support calls start dominating). Two
identical logistic-regression models watch this happen. The **static** one is trained
once on the first 2,000 customers and frozen. The **dynamic** one is retrained every
500 customers on the most recent 2,000. We score every row with whichever model is
live at that moment.

## Run it

**Locally**

```bash
cd lab3_drift
python drift_experiment.py
```

**In Colab (the easy path)**

1. Go to `https://colab.research.google.com`
2. `File` → `Upload notebook`… or simply drag `drift_experiment.py` into the file panel,
   then run it with `!python drift_experiment.py`. (Simplest: create a new notebook,
   paste the file's contents into one cell, press the play button. The script detects
   Jupyter and shows the plot inline.)
3. No installs needed — Colab already has pandas, numpy, scikit-learn and matplotlib.

## What you should see

```
AVERAGE ACCURACY, BEFORE vs AFTER THE WORLD CHANGES
      period  rows  static_accuracy  dynamic_accuracy
before drift  2000            0.798              0.80
after drift   4000            0.545              0.83

THE MOMENT OF TRUTH
  Static model accuracy first drops below 70% at SAMPLE INDEX 4,098.
  That is day 40 of the simulation, at an accuracy of 69.5%.
```

Exact numbers move a little with the seed; the *shape* does not. The plot
(`outputs/drift_accuracy.png`) is the deliverable: a red line that falls off a cliff,
a blue line that stumbles for one retrain and then recovers.

Files written: `outputs/drift_accuracy.png`, `drift_results.csv`,
`drift_summary.csv`, `drift_report.json`.

## Vocabulary this lab earns you

- **Data drift (covariate shift):** the inputs change — higher bills, a different
  contract mix.
- **Concept drift:** the *rule* changes — the same inputs now mean something else.
  This lab injects both at the same moment, which is realistic.
- **Static training:** fit once, freeze, ship. Cheap, reproducible, rots.
- **Dynamic training:** refit on a schedule or a trigger. Fresher, more expensive,
  and it needs the labels to come back (here they arrive instantly — in real life
  churn takes 30–90 days, which is the hard part).
- **Retraining window:** how much recent history you refit on. Too small = noisy;
  too big = slow to react. We use 2,000 rows. That is a *design decision*, not a default.

## Knobs to argue about in class

```bash
python drift_experiment.py --drift-day 60            # change arrives later
python drift_experiment.py --retrain-every 2000      # lazier retraining: watch the blue line limp
python drift_experiment.py --window 500              # forgetful retraining: noisier, faster to react
python drift_experiment.py --days 200 --per-day 50   # longer history, same total rows
```

The question worth 10 minutes of argument: **how would you decide, in production,
that today is the day to retrain?** (Hint: you cannot see accuracy — you do not have
today's labels. You watch the *inputs*: a drift detector on feature distributions,
and/or the distribution of predicted probabilities.)

## Why this is Pillar 3 and Pillar 5

- **Pillar 3 (versioning):** the dynamic model is a different artifact every 500 rows.
  If you cannot say *which* version answered a given request, you cannot debug it.
  That is what `model_version` in Lab 1's responses is for.
- **Pillar 5 (monitoring):** nothing in this experiment crashes or alerts. Accuracy
  just quietly halves. Only a monitor watching a proxy metric would have caught it.

## Common problems

- **A window flashes and disappears (Windows)** → that is fine, the plot was already
  saved to `outputs/`. Open the PNG.
- **`ModuleNotFoundError: No module named 'matplotlib'`** → `python -m pip install -r ../requirements.txt`.
- **`static model never fell below 70%`** → your drift was too gentle for that seed.
  Try `--drift-day 30`.
- **In Colab, no plot appears** → run the file as a notebook (one cell) rather than
  `!python`, so the inline display is available.
