#!/bin/sh
set -e

echo "[START.SH] Purple agent starting..." >&2
echo "[START.SH] Working directory: $(pwd)" >&2
echo "[START.SH] Files in /app:" >&2
ls -la /app/ >&2

echo "[START.SH] Python version:" >&2
python --version >&2

echo "[START.SH] Checking if run_a2a.py exists:" >&2
if [ -f "/app/run_a2a.py" ]; then
    echo "[START.SH] run_a2a.py found" >&2
else
    echo "[START.SH] ERROR: run_a2a.py NOT FOUND" >&2
    exit 1
fi

echo "[START.SH] Starting Python script with args: $@" >&2
exec python /app/run_a2a.py "$@"
