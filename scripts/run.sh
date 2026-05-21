#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  CHMI — run.sh  (macOS / Linux)
#  Usage:  bash scripts/run.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT_DIR/.venv"
SECRETS="$ROOT_DIR/.streamlit/secrets.toml"
EXAMPLE="$ROOT_DIR/.streamlit/secrets.toml.example"

# ── 1. Create venv if missing ────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# ── 2. Activate venv ────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[setup] Python: $(python --version)"

# ── 3. Install / upgrade dependencies ───────────────────────
echo "[setup] Installing requirements..."
pip install --quiet --upgrade pip
pip install --quiet -r "$ROOT_DIR/requirements.txt"

# ── 4. Check secrets file ───────────────────────────────────
if [ ! -f "$SECRETS" ]; then
  echo ""
  echo "⚠️  .streamlit/secrets.toml not found."
  echo "    Copy the example template and add your credentials:"
  echo "    cp .streamlit/secrets.toml.example .streamlit/secrets.toml"
  echo ""
  cp "$EXAMPLE" "$SECRETS"
  echo "[setup] Blank secrets.toml created — fill in SMS_AUTH_KEY and SMS_AUTH_TOKEN."
  echo ""
fi

# ── 5. Run healthcheck ───────────────────────────────────────
echo "[setup] Running healthcheck..."
python "$ROOT_DIR/healthcheck.py" || {
  echo "⚠️  Healthcheck reported issues. Check output above before continuing."
}

# ── 6. Launch app ────────────────────────────────────────────
echo ""
echo "🚀 Starting CHMI on http://localhost:8501"
echo "   Press Ctrl+C to stop."
echo ""
streamlit run "$ROOT_DIR/app.py" \
  --server.port 8501 \
  --server.address localhost \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.headless true
