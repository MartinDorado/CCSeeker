# Repository Guidelines

## Project Structure
- `app_seed_gemini_hardened.py` — production-like Streamlit app (primary entrypoint).
- `app.py` — dev variant for quick UI experiments.
- `seed_topics_hardened.py` — topic seeding helpers.
- `.streamlit/config.toml` — Streamlit server settings.
- `requirements.txt` — Python deps.
- `run_streamlit.sh` — launches hardened app on `0.0.0.0:8501`.

## Dev Setup & Run
**WSL/Linux recommended.** Use a Linux-side venv to avoid NTFS issues.

```bash
# one-time
mkdir -p ~/.venvs && python3 -m venv ~/.venvs/ccseeker
. ~/.venvs/ccseeker/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary=:all: -r requirements.txt
printf "YOUTUBE_API_KEY=...\nGEMINI_API_KEY=...\n" > .env

# every session
cd "/mnt/c/Users/marti/OneDrive/Escritorio/Martin/Formacion/AI/CCSeeker"
. ~/.venvs/ccseeker/bin/activate
streamlit run app_seed_gemini_hardened.py  # http://localhost:8501
Testing (manual for now)
“Keywords” and “Channel-as-Seed” return channels.

Filters (subscribers, country, months) change results.

AI Summary requires GEMINI_API_KEY; outreach drafts generate in chosen language.

Verbose logs: streamlit run app.py --logger.level=debug.

Coding & Commits
Python 3.x, PEP 8, 4 spaces; snake_case for modules/functions/vars, PascalCase for classes.

Keep helpers small/testable (API calls, transforms).

Commits: imperative & scoped, e.g., feat: add Spanish outreach option.

PRs: purpose, summary, screenshots/GIFs, repro steps, linked issues. Don’t break run_streamlit.sh.

Security & Configuration
Never commit secrets. Keep them in .env. (.env and .venv/ are ignored by .gitignore.)

Enforce LF endings for scripts via .gitattributes:

arduino
Copy code
*.sh text eol=lf
Handle API errors/quotas gracefully; avoid hard-coding API keys anywhere in code.

Using Codex (CLI/IDE)
Start in repo root. Approvals: Auto (workspace only) to reduce prompts.

Use venv ~/.venvs/ccseeker (not .venv under /mnt/c).

Common task: “Activate venv, install -r requirements.txt, run hardened app on 8501.”


