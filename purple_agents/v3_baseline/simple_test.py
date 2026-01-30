#!/usr/bin/env python
"""Minimal test - no external imports."""
import sys
import os
import time

# Write to stderr immediately
sys.stderr.write("=== SIMPLE_TEST.PY STARTED ===\n")
sys.stderr.flush()

# Print info
sys.stderr.write(f"Python version: {sys.version}\n")
sys.stderr.write(f"Working dir: {os.getcwd()}\n")
sys.stderr.write(f"Command line args: {sys.argv}\n")
sys.stderr.write(f"PATH: {os.environ.get('PATH', 'NOT SET')}\n")
sys.stderr.flush()

# List files
sys.stderr.write("\nFiles in /app:\n")
try:
    for f in os.listdir('/app'):
        sys.stderr.write(f"  {f}\n")
except Exception as e:
    sys.stderr.write(f"  Error listing /app: {e}\n")
sys.stderr.flush()

# Keep alive with simple loop
sys.stderr.write("\n=== Starting keepalive loop ===\n")
sys.stderr.flush()

counter = 0
while True:
    counter += 1
    sys.stderr.write(f"[{counter}] Still running...\n")
    sys.stderr.flush()
    time.sleep(2)
