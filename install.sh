#!/usr/bin/env bash
# APEX Life OS — one-command installer for VPS / bare-metal servers.
#
# Usage:
#   ./install.sh                 # install into ./.venv
#   ./install.sh --with-service  # also install a systemd service (requires sudo)
#
# Requirements: Python 3.11+ with the venv module.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
WITH_SERVICE=0

for arg in "$@"; do
    case "$arg" in
        --with-service) WITH_SERVICE=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (see --help)" >&2
            exit 1
            ;;
    esac
done

# --- Find a suitable Python (3.11+) -----------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.11+ is required but was not found." >&2
    echo "On Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y python3 python3-venv" >&2
    exit 1
fi

echo "==> Using $($PYTHON --version) at $(command -v $PYTHON)"

# --- Create the virtual environment and install -----------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "==> Installing apex-life-os"
"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install "$REPO_DIR"

echo "==> Verifying installation"
"$VENV_DIR/bin/apex" --version

# --- Optional systemd service ------------------------------------------------
if [ "$WITH_SERVICE" -eq 1 ]; then
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "Error: systemd not available; skipping service installation." >&2
        exit 1
    fi
    SERVICE_USER="${SERVICE_USER:-$(id -un)}"
    echo "==> Installing systemd service (user: $SERVICE_USER)"
    sed -e "s|{{REPO_DIR}}|$REPO_DIR|g" \
        -e "s|{{VENV_DIR}}|$VENV_DIR|g" \
        -e "s|{{SERVICE_USER}}|$SERVICE_USER|g" \
        "$REPO_DIR/deploy/apex.service" | sudo tee /etc/systemd/system/apex.service >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now apex.service
    echo "==> Service installed. Check status with: sudo systemctl status apex"
fi

echo
echo "APEX Life OS installed successfully."
echo "Activate the environment:  source $VENV_DIR/bin/activate"
echo "Run the daemon manually:   $VENV_DIR/bin/apex daemon"
echo "Run one cycle:             $VENV_DIR/bin/apex cycle"
