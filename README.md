# ✍️ Quote Next-Word Predictor

Streamlit app around an LSTM next-word model trained on a quotes dataset.

## Project structure
```
app.py             Streamlit UI
predictor.py       model loading, tokenization, prediction, text generation
smoke_test.py      quick check that the model works
requirements.txt   dependencies (pinned)
model/
  lstm_model.keras the trained model (inference-only, ~7 MB)
  vocab.json       tokenizer vocabulary + max_len
.streamlit/config.toml
```

## Run locally (use Python 3.12)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smoke_test.py             # optional check
streamlit run app.py             # opens http://localhost:8501
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo** (`git init`, `git add .`, `git commit`, `git push`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings** and select **Python 3.12** (runtime.txt / .python-version are ignored).
5. Click **Deploy**. The first build takes a few minutes (TensorFlow is large).

## Troubleshooting
- Build fails on install → confirm Python 3.12 was selected; to change it, delete the app and redeploy.
- `ModuleNotFoundError` → the package is missing from requirements.txt.
- App sleeps after inactivity on the free tier → just click "wake up".
- Optional Docker run: `docker build -t quote-app . && docker run -p 8501:8501 quote-app`
