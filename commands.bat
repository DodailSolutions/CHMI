@echo off
REM ─────────────────────────────────────────────────────────────
REM  CHMI — commands.bat  (Windows)
REM  Run this file from the project root directory.
REM  Usage:  double-click or  > commands.bat
REM ─────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

REM ── Resolve project root (folder containing this .bat) ──────
set ROOT_DIR=%~dp0
set VENV_DIR=%ROOT_DIR%.venv
set SECRETS=%ROOT_DIR%.streamlit\secrets.toml
set EXAMPLE=%ROOT_DIR%.streamlit\secrets.toml.example
set LOG=%ROOT_DIR%streamlit_log.txt

echo [setup] Project root: %ROOT_DIR%

REM ── 1. Create venv if missing ────────────────────────────────
if not exist "%VENV_DIR%" (
    echo [setup] Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM ── 2. Activate venv ─────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"
echo [setup] Python: & python --version

REM ── 3. Install dependencies ──────────────────────────────────
echo [setup] Installing requirements...
pip install --quiet --upgrade pip
pip install --quiet -r "%ROOT_DIR%requirements.txt"

REM ── 4. Check secrets file ────────────────────────────────────
if not exist "%SECRETS%" (
    echo.
    echo WARNING: .streamlit\secrets.toml not found.
    echo          Copying example — fill in SMS_AUTH_KEY and SMS_AUTH_TOKEN.
    copy "%EXAMPLE%" "%SECRETS%"
    echo.
)

REM ── 5. Run healthcheck ───────────────────────────────────────
echo [setup] Running healthcheck...
python "%ROOT_DIR%healthcheck.py"

REM ── 6. Launch app ────────────────────────────────────────────
echo.
echo Starting CHMI on http://localhost:8501
echo Logs → %LOG%
echo Press Ctrl+C to stop.
echo.
streamlit run "%ROOT_DIR%app.py" ^
    --server.port 8501 ^
    --server.address localhost ^
    --server.enableCORS false ^
    --server.enableXsrfProtection false ^
    --server.headless true ^
    >> "%LOG%" 2>&1
