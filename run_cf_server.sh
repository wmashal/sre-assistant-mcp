#!/bin/bash

# SRE Assistant CF Server Launcher
# This script ensures the correct Python environment is used

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment and run the server
source "$SCRIPT_DIR/venv/bin/activate"
exec python "$SCRIPT_DIR/sre_assistant_cf.py"