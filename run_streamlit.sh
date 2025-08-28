#!/usr/bin/env bash
set -eux
. .venv/bin/activate
# In Codex: keep flags so you can expose the port
streamlit run app_seed_gemini_hardened.py --server.address 0.0.0.0 --server.port 8501
