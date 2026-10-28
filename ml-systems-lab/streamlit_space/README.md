# Deploy target, Plan B — Streamlit Community Cloud (free, no card)

This folder is the backup road to a public URL, used when Hugging Face asks for a
paid plan. It is also the *fastest* road: Streamlit logs in with the GitHub account
you already made in Part G.

- `app.py` — the same model, the same four inputs, a different set of widgets.
- `requirements.txt` — what the server installs before it starts. **Read the comment
  at the top**: ranges here, pins in `../requirements.txt`, and the reason why.

## Deploy in five steps (details in `../instructions.txt`, Part H, Plan B)

1. Put `app.py`, `requirements.txt` and this `README.md` in your GitHub repository,
   in a folder named `streamlit_space`.
2. Go to `https://share.streamlit.io` and click **Continue with GitHub**.
3. Click **Create app** (or **New app**).
4. Choose your repository, branch `main`, and main file path `streamlit_space/app.py`.
   In **Advanced settings**, set **Python 3.12**.
5. Click **Deploy**. A normal build is 2–4 minutes and you get a `*.streamlit.app` address.

## "It has been in the oven for 30 minutes" — what that actually means

"In the oven" is Streamlit's build page. It is doing three things: creating a
container, running `pip install -r requirements.txt`, and starting your script.
That should take **2–4 minutes**. Anything past ~10 minutes means stuck, not slow.
Here is the decision tree, in order:

| What you see | What it means | Fix |
|---|---|---|
| Log keeps scrolling, package names going by | It is genuinely still installing | Wait. It is downloading ~80 MB of wheels |
| Stuck on `Building wheel for numpy` (or pandas / scikit-learn) | No prebuilt wheel for that Python, so it is **compiling from source**. That is the 30-minute bake | Advanced settings → Python **3.12** → save → reboot. This is why `requirements.txt` here uses ranges, not pins |
| No new log lines for several minutes | The build is wedged or the queue starved it | ⋮ menu → **Reboot app**. Still stuck → delete the app and recreate it |
| `Error running app` / `ModuleNotFoundError` | Build finished, but the script cannot find a package | `requirements.txt` is not beside `app.py`, or is missing an entry |
| `file not found` | The **Main file path** does not match the repo | Open the repo on GitHub, click `app.py`, copy what follows the repo name: `streamlit_space/app.py` |

**Where the log lives:** open your app, then look for **"Manage app"** in the lower
right corner (sometimes behind a small **>** chevron). That panel is the build log,
and it is the only honest answer to "why is it taking so long?".

**What a healthy start looks like.** Once the script runs, our startup banner writes
one line to that log:

```
[startup] python 3.12.7 | streamlit 1.40.2 | scikit-learn 1.5.x | pandas 2.2.x | numpy 2.1.x
```

If you can see that line, the deploy worked; anything after it is app logic, not
platform problems.

## Notes

- The app trains its model on first load (about two seconds) and caches it with
  `@st.cache_resource`, so the second visitor waits for nothing.
- Community Cloud apps sleep after a long quiet period. The first visit after a sleep
  shows a "waking up" page — that is normal, and it is a real production concern too.
- Everything is synthetic. No secrets, no API keys, no files to upload besides these.
- Deploy from `main`, not from a short-lived branch: if that branch is ever deleted,
  the app breaks.
