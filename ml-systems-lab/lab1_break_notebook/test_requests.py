"""
WHAT: Sends one good and three bad requests to both Lab 1 endpoints and explains every failure in plain English.
HOW:  start app.py in one terminal, then in a second terminal: python test_requests.py
OK:   prints "RESULT: the service behaved exactly as Lab 1 intends" and exits with code 0.
"""

# -----------------------------------------------------------------------------
# You need TWO terminals for this file:
#   terminal 1:  python app.py          (leave it running)
#   terminal 2:  python test_requests.py
# If you get "Could not reach the service", terminal 1 is not running.
# -----------------------------------------------------------------------------

import json
import sys

import httpx

BASE_URL = "http://127.0.0.1:8000"

GOOD = {
    "tenure_months": 12,
    "monthly_charges": 79.5,
    "support_calls": 2,
    "contract_type": "monthly",
}

# The three ways a notebook model breaks in production. Same model, three doors.
CASES = [
    {
        "title": "CASE 1 - VALID REQUEST (this one must work)",
        "body": GOOD,
        "expect_naive_status": 200,
        "plain_english": "Every field is present, every type is right, the category exists.",
        "pillar": "None needed. This is what a healthy request looks like.",
    },
    {
        "title": "CASE 2 - FAILURE (a): MISSING COLUMN",
        "body": {k: v for k, v in GOOD.items() if k != "monthly_charges"},
        "expect_naive_status": 500,
        "plain_english": (
            "A whole column called monthly_charges was not sent at all. "
            "The model was handed a row of data with a hole in it and blew up."
        ),
        "pillar": "PILLAR 1 - DATA VALIDATION (reject incomplete rows before scoring).",
    },
    {
        "title": "CASE 3 - FAILURE (b): WRONG DATA TYPE",
        "body": {**GOOD, "tenure_months": "twelve"},
        "expect_naive_status": 500,
        "plain_english": (
            "The number 12 arrived as the word 'twelve'. A human understands it; "
            "the model does not. The service never checked the type."
        ),
        "pillar": "PILLAR 4 - SERVING CONTRACT (publish and enforce the expected types).",
    },
    {
        "title": "CASE 4 - FAILURE (c): UNSEEN CATEGORY",
        "body": {**GOOD, "contract_type": "lifetime"},
        "expect_naive_status": 500,
        "plain_english": (
            "A brand-new contract type called 'lifetime' appeared. The model was "
            "trained only on monthly, one_year and two_year, so it has no idea."
        ),
        "pillar": (
            "PILLAR 1 - DATA VALIDATION (block unknown values) plus "
            "PILLAR 5 - MONITORING (notice that a new category showed up in the wild)."
        ),
    },
]

LINE = "=" * 74


def short(text: str, limit: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def explain_422(response: httpx.Response) -> str:
    """Turn FastAPI's validation error list into one friendly line each."""
    try:
        problems = response.json().get("detail", [])
    except ValueError:
        return short(response.text)
    if not isinstance(problems, list):
        return short(response.text)
    lines = []
    for problem in problems:
        where = ".".join(str(part) for part in problem.get("loc", []) if part != "body")
        lines.append(f"{where or 'body'}: {problem.get('msg', 'invalid')}")
    return " | ".join(lines) if lines else short(response.text)


def call(client: httpx.Client, path: str, body: dict) -> httpx.Response:
    try:
        return client.post(BASE_URL + path, json=body)
    except httpx.HTTPError as exc:
        print(f"\n[OFFLINE] Could not reach {BASE_URL}{path} ({type(exc).__name__}).")
        print("          Fix: open a second terminal, cd into lab1_break_notebook,")
        print("          run 'python app.py', wait for 'service starting', then try again.")
        sys.exit(1)


def main() -> int:
    print(LINE)
    print("LAB 1 - Breaking the notebook on purpose")
    print(f"Target service: {BASE_URL}")
    print(LINE)

    results = []
    with httpx.Client(timeout=10.0) as client:
        try:
            health = client.get(BASE_URL + "/health")
            print(f"\nHealth check      : HTTP {health.status_code} -> {short(health.text, 120)}")
            if health.status_code != 200:
                print("The service is up but not healthy. Scroll up in the app.py window.")
        except httpx.HTTPError:
            print(f"\n[OFFLINE] Nothing is answering at {BASE_URL}.")
            print("          Fix: in terminal 1 run 'python app.py' and leave it open.")
            return 1

        for index, case in enumerate(CASES, start=1):
            print()
            print("-" * 74)
            print(f"[{index}/{len(CASES)}] {case['title']}")
            print("-" * 74)
            print("Request body sent:")
            print(json.dumps(case["body"], indent=2))
            print(f"What it means: {case['plain_english']}")

            naive = call(client, "/predict", case["body"])
            print(f"\n  POST /predict            -> HTTP {naive.status_code}")
            if naive.status_code == 200:
                print(f"  Answer: {short(naive.text, 160)}")
            else:
                try:
                    error = naive.json()
                    print(f"  The service said (naive): {error.get('error', '?')}: {short(error.get('detail', ''), 160)}")
                except ValueError:
                    print(f"  The service said (naive): {short(naive.text, 160)}")
                print(f"  Pillar that would have prevented it: {case['pillar']}")

            validated = call(client, "/predict_validated", case["body"])
            print(f"\n  POST /predict_validated  -> HTTP {validated.status_code}   (the same model, with a contract)")
            if validated.status_code == 200:
                print(f"  Answer: {short(validated.text, 160)}")
            else:
                print(f"  The service said (fixed): {short(explain_422(validated), 200)}")
                print("  -> Rejected politely. The model never ran. Nothing crashed.")

            results.append(
                {
                    "case": case["title"],
                    "naive": naive.status_code,
                    "validated": validated.status_code,
                    "naive_ok": naive.status_code == case["expect_naive_status"],
                    "validated_ok": validated.status_code == (200 if case["expect_naive_status"] == 200 else 422),
                }
            )

    print()
    print(LINE)
    print("SUMMARY")
    print(LINE)
    print(f"{'case':<48}{'naive':>8}{'fixed':>8}  as expected?")
    for row in results:
        ok = "yes" if (row["naive_ok"] and row["validated_ok"]) else "NO"
        print(f"{row['case'][:46]:<48}{row['naive']:>8}{row['validated']:>8}  {ok}")

    all_good = all(row["naive_ok"] and row["validated_ok"] for row in results)
    print()
    if all_good:
        print("RESULT: the service behaved exactly as Lab 1 intends.")
        print("        One request worked. Three broke loudly. All three are fixable")
        print("        with a contract, which is what /predict_validated already does.")
        print()
        print("Now look back at the app.py terminal: you will see a big traceback")
        print("block for every failure. That traceback is what a real user would")
        print("never see -- and what would wake you up at 3 a.m.")
        return 0

    print("RESULT: something unexpected happened (see the 'NO' rows above).")
    print("        Usually this means a dependency version changed. Compare your")
    print("        requirements.txt with the one in this package.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
