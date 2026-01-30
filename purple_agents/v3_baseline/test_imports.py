#!/usr/bin/env python
"""Test script to check if all imports work."""
import sys

print("=== Testing imports ===", flush=True, file=sys.stderr)

try:
    print("Importing sys...", flush=True, file=sys.stderr)
    import sys
    print("OK", flush=True, file=sys.stderr)
except Exception as e:
    print(f"FAILED: {e}", flush=True, file=sys.stderr)
    sys.exit(1)

try:
    print("Importing argparse...", flush=True, file=sys.stderr)
    import argparse
    print("OK", flush=True, file=sys.stderr)
except Exception as e:
    print(f"FAILED: {e}", flush=True, file=sys.stderr)
    sys.exit(1)

try:
    print("Importing asyncio...", flush=True, file=sys.stderr)
    import asyncio
    print("OK", flush=True, file=sys.stderr)
except Exception as e:
    print(f"FAILED: {e}", flush=True, file=sys.stderr)
    sys.exit(1)

try:
    print("Importing a2a.server.agent_execution...", flush=True, file=sys.stderr)
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    print("OK", flush=True, file=sys.stderr)
except Exception as e:
    print(f"FAILED: {e}", flush=True, file=sys.stderr)
    sys.exit(1)

try:
    print("Importing a2a.server.apps...", flush=True, file=sys.stderr)
    from a2a.server.apps import A2AStarletteApplication
    print("OK", flush=True, file=sys.stderr)
except Exception as e:
    print(f"FAILED: {e}", flush=True, file=sys.stderr)
    sys.exit(1)

print("=== All imports successful ===", flush=True, file=sys.stderr)
print("Starting infinite loop to keep container alive...", flush=True, file=sys.stderr)

# Keep container running
import time
while True:
    print("Still alive...", flush=True, file=sys.stderr)
    time.sleep(5)
