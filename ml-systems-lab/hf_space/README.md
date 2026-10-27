---
title: Churn Predictor - MDC-501 Lab
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Will this customer leave?

A small, honest machine-learning service, built for the chapter
**"From Notebook to Production: Architecting Real-World ML Systems"**.

Four inputs go in, one churn probability comes out.

## What is actually happening

1. On startup, `app.py` generates 6,000 synthetic customers and trains a
   logistic regression model. It takes about two seconds.
2. Every time you click **Predict**, the app builds one row of data, runs it
   through the same scikit-learn pipeline, and shows you the probability.
3. Every prediction is printed in the Space's log (**Logs** tab). That log line
   is the first tiny step toward monitoring.

## Why this is a *service* and not a notebook

- **A contract.** The browser can only send what the form allows: a whole
  number of months, a numeric bill, a slider for calls, and one of three
  contract types. A notebook accepts anything; this cannot send nonsense.
- **A version.** `MODEL_VERSION` is written into every answer.
- **A health signal.** The footer tells you which model is live.
- **A public address.** Anyone with the link can use it, from any device.

## Things to try

- Set tenure to 2 months, bill to $120, four support calls, contract
  "monthly". Watch the probability jump.
- Set tenure to 60 months, bill to $30, zero calls, "two year". Watch it fall.
- Try to type a negative bill. The form will not let you. That is Pillar 4
  (a serving contract) doing its job before any Python runs.

## Local run

```bash
pip install -r requirements.txt
python app.py
```
then open the address it prints (usually http://127.0.0.1:7860).

Built for MDC-501. No real customer data is used anywhere in this repository.
