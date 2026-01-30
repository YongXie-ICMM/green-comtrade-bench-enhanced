#!/usr/bin/env python
"""Debug A2A imports step by step while providing HTTP healthcheck endpoint."""
import sys
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Write to stderr immediately
sys.stderr.write("=== DEBUG_A2A.PY STARTED ===\n")
sys.stderr.flush()

# Print basic info
sys.stderr.write(f"Python version: {sys.version}\n")
sys.stderr.write(f"Working dir: {os.getcwd()}\n")
sys.stderr.write(f"Command line args: {sys.argv}\n")
sys.stderr.flush()


# Simple HTTP handler for agent card
class AgentCardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write(f"[HTTP] {format % args}\n")
        sys.stderr.flush()

    def do_GET(self):
        if self.path == "/.well-known/agent-card.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            card = {
                "name": "purple-comtrade-baseline-v2",
                "version": "2.0.0-debug",
                "description": "Debug version - testing A2A imports",
                "url": "http://purple-comtrade-baseline-v2:9009",
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "capabilities": {"streaming": False},
                "skills": []
            }
            self.wfile.write(json.dumps(card).encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_http_server():
    """Run simple HTTP server on port 9009."""
    server = HTTPServer(("0.0.0.0", 9009), AgentCardHandler)
    sys.stderr.write("[HTTP] Server started on port 9009\n")
    sys.stderr.flush()
    server.serve_forever()


# Start HTTP server in background thread
http_thread = threading.Thread(target=run_http_server, daemon=True)
http_thread.start()
sys.stderr.write("[MAIN] HTTP server thread started\n")
sys.stderr.flush()

# Give server time to start
time.sleep(1)

# Now test A2A imports one by one
sys.stderr.write("\n=== Testing A2A imports ===\n")
sys.stderr.flush()

def test_import(module_name, from_module=None):
    """Test importing a module and report result."""
    try:
        if from_module:
            sys.stderr.write(f"Importing from {from_module}: {module_name}... ")
            exec(f"from {from_module} import {module_name}")
        else:
            sys.stderr.write(f"Importing {module_name}... ")
            __import__(module_name)
        sys.stderr.write("OK\n")
        sys.stderr.flush()
        return True
    except Exception as e:
        sys.stderr.write(f"FAILED: {type(e).__name__}: {e}\n")
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return False


# Test standard library imports first
test_import("argparse")
test_import("asyncio")
test_import("concurrent.futures")
test_import("json")
test_import("logging")
test_import("pathlib")

# Test pip installed packages
test_import("uvicorn")
test_import("pydantic")
test_import("fastapi")
test_import("requests")

# Test A2A packages step by step
sys.stderr.write("\n=== Testing A2A package imports ===\n")
sys.stderr.flush()

test_import("a2a")
test_import("a2a.types")
test_import("AgentCard", "a2a.types")
test_import("AgentSkill", "a2a.types")
test_import("AgentCapabilities", "a2a.types")
test_import("TaskState", "a2a.types")

test_import("a2a.server")
test_import("a2a.server.agent_execution")
test_import("AgentExecutor", "a2a.server.agent_execution")
test_import("RequestContext", "a2a.server.agent_execution")

test_import("a2a.server.apps")
test_import("A2AStarletteApplication", "a2a.server.apps")

test_import("a2a.server.events")
test_import("EventQueue", "a2a.server.events")

test_import("a2a.server.request_handlers")
test_import("DefaultRequestHandler", "a2a.server.request_handlers")

test_import("a2a.server.tasks")
test_import("InMemoryTaskStore", "a2a.server.tasks")
test_import("TaskUpdater", "a2a.server.tasks")

test_import("a2a.utils")
test_import("new_agent_text_message", "a2a.utils")
test_import("new_task", "a2a.utils")

test_import("a2a.utils.errors")
test_import("ServerError", "a2a.utils.errors")

sys.stderr.write("\n=== Import testing complete ===\n")
sys.stderr.write("Keeping container alive for health checks...\n")
sys.stderr.flush()

# Keep alive
counter = 0
while True:
    counter += 1
    if counter % 30 == 0:  # Log every 30 seconds
        sys.stderr.write(f"[{counter}] Still running...\n")
        sys.stderr.flush()
    time.sleep(1)
