# Deploy target, Plan B — Streamlit Community Cloud (free, no card)

This folder is the backup road to a public URL, used when Hugging Face asks for a
paid plan. It is also the *fastest* road: Streamlit logs in with the GitHub account
you already made in Part G.

- `app.py` — the same model, the same four inputs, a different set of widgets.
- `requirements.txt` — what the server installs before it starts.

## Deploy in five steps (details in `../instructions.txt`, Part H, Plan B)

1. Put `app.py`, `requirements.txt` and this `README.md` in your GitHub repository,
   in a folder named `streamlit_space`.
2. Go to `https://share.streamlit.io` and click **Continue with GitHub**.
3. Click **Create app** (or **New app**).
4. Choose your repository, branch `main`, and main file path `streamlit_space/app.py`.
5. Click **Deploy**. In a few minutes you get a `*.streamlit.app` address.

## Notes

- The app trains its model on first load (about two seconds) and caches it with
  `@st.cache_resource`, so the second visitor waits for nothing.
- Community Cloud apps sleep after a long quiet period. The first visit after a sleep
  shows a "waking up" page — that is normal, and it is a real production concern too.
- Everything is synthetic. No secrets, no API keys, no files to upload besides these.
