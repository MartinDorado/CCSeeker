Review $ARGUMENTS for security vulnerabilities. If no file is specified, review the current branch diff (`git diff main`) and any files in `app/core/` and `app/main.py` touched by the diff.

Focus on:
1. API key exposure or leakage — especially the `_get_secret()` pattern in `app/main.py` and any new BYOK key handling
2. User input validation — anywhere user-supplied strings reach YouTube or Gemini API calls
3. OAuth token handling — storage, expiry, refresh flow, session scope
4. Secrets in logs or error messages — `st.error()`, `print()`, exception tracebacks
5. Unauthorized API usage — ensure user-supplied keys are scoped per-session and never persisted server-side

Flag each issue with: file path, line number, risk level (high/medium/low), and a one-line fix recommendation.
