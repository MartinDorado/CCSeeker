#!/usr/bin/env bash
set -euxo pipefail

# 0) Create & activate venv
python3 -m venv .venv
. .venv/bin/activate

# 1) Fast installs (avoid building from source when possible)
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: -r requirements.txt

# 2) Optional: show env var presence (no secrets printed)
python - <<'PY'
import os
print("YOUTUBE_API_KEY set:", bool(os.getenv("YOUTUBE_API_KEY")))
print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
PY

# 3) Sanity check: imports (no network calls)
python - <<'PY'
import streamlit, pandas, google.generativeai, googleapiclient.discovery
print("Imports OK:", streamlit.__version__)
PY

# IMPORTANT: do NOT start Streamlit here. Let this script finish & exit.
