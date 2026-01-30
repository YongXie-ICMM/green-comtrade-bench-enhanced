"""
Baseline Purple Agent - CLI Entrypoint

Usage:
    Server mode (AgentBeats runner):
        python3 run.py --host 0.0.0.0 --port 9009 --card-url http://purple-agent:9009

    Local mode (run single task):
        python3 run.py --task-id T1_single_page
        python3 run.py --task-id T7_totals_trap --mock-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Server mode imports (lazy)
def run_server(host: str, port: int, card_url: str) -> None:
    """Start FastAPI server for AgentBeats runner."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="Purple Comtrade Baseline v2")

    AGENT_CARD = {
        "name": "purple-comtrade-baseline-v2",
        "description": "Baseline Purple agent for Green Comtrade Bench v2",
        "version": "2.0.0",
        "url": "http://purple-comtrade-baseline-v2:9009/a2a/rpc",
        "endpoints": {
            "rpc": "/a2a/rpc",
            "health": "/healthz",
        },
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "comtrade.bench.run",
                "name": "run",
                "description": "Run benchmark tasks via /run",
                "tags": ["comtrade", "benchmark", "a2a"],
            }
        ],
    }

    @app.get("/")
    async def root():
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/agent-card")
    async def agent_card_simple():
        return JSONResponse(content=AGENT_CARD)

    @app.get("/.well-known/agent-card.json")
    async def agent_card():
        return JSONResponse(content=AGENT_CARD)

    @app.post("/run")
    async def run_task(request: Request):
        """Handle task run request from AgentBeats runner."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        
        task_id = body.get("task_id", "unknown")
        return {
            "ok": True,
            "message": "purple agent server up",
            "task_id": task_id,
            "agent": "purple-comtrade-baseline-v2",
        }

    @app.post("/a2a/rpc")
    async def a2a_rpc(request: Request):
        """Handle A2A JSON-RPC requests."""
        from purple_agent import PurpleAgent
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        import logging
        import sys

        # Configure logging to stdout
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
        logger = logging.getLogger("purple_rpc")

        logger.info("Handler invoked")
        try:
            body = await request.json()
            logger.info(f"Received request: method={body.get('method')}, id={body.get('id')}")
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}")
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32700, "message": "Parse error"}
            })

        method = body.get("method", "")
        rpc_id = body.get("id", "1")
        params = body.get("params", {})
        logger.info(f"method={method}, rpc_id={rpc_id}")

        # Handle tasks/send method for task execution
        if method == "tasks/send":
            message = params.get("message", {})
            parts = message.get("parts", [])
            logger.info(f"message keys={list(message.keys())}")
            logger.info(f"parts count={len(parts)}")

            # Extract task_id from message
            task_request = None
            for i, part in enumerate(parts):
                logger.info(f"Processing part {i}: keys={list(part.keys()) if isinstance(part, dict) else 'not a dict'}")
                if isinstance(part, dict) and part.get("kind") == "text":
                    try:
                        task_request = json.loads(part.get("text", "{}"))
                        logger.info(f"Extracted task_request: {task_request}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to parse text as JSON: {e}")
                        pass

            if not task_request or "task_id" not in task_request:
                logger.error(f"task_id not found in request")
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: task_id not found"
                    }
                })

            task_id = task_request["task_id"]
            mock_url = task_request.get("mock_url", "http://mock-comtrade:8000")
            output_dir = task_request.get("output_dir", f"/workspace/purple_output/{task_id}")

            # Run task in background thread
            agent = PurpleAgent()
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                success = await loop.run_in_executor(
                    executor,
                    agent.run,
                    task_id,
                    output_dir,
                    mock_url
                )

            if success:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "task": {
                            "id": f"task-{task_id}",
                            "status": {
                                "state": "completed",
                                "message": {
                                    "parts": [
                                        {
                                            "kind": "text",
                                            "text": f"Task {task_id} completed successfully"
                                        }
                                    ]
                                }
                            },
                            "artifacts": [
                                {
                                    "name": "result",
                                    "parts": [
                                        {
                                            "kind": "text",
                                            "text": json.dumps({
                                                "task_id": task_id,
                                                "status": "completed",
                                                "output_dir": output_dir
                                            })
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                })
            else:
                return JSONResponse(content={
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32603,
                        "message": f"Task {task_id} execution failed"
                    }
                })

        # Default response for other methods
        logger.info(f"Returning default response for method: {method}")
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "ok": True,
                "message": "purple agent a2a endpoint",
                "method": method,
            }
        })

    print(f"Starting purple agent server on {host}:{port}")
    print(f"Agent card URL: {card_url}")
    uvicorn.run(app, host=host, port=port)


def run_local(task_id: str, output_dir: str, mock_url: str) -> int:
    """Run single task locally and exit."""
    from purple_agent import PurpleAgent

    agent = PurpleAgent()
    success = agent.run(
        task_id=task_id,
        output_dir=output_dir,
        mock_url=mock_url,
    )
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline Purple Agent for green-comtrade-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Server mode args (default: start server on 0.0.0.0:9009)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="Server port (default: 9009)",
    )
    parser.add_argument(
        "--card-url",
        default=None,
        help="Agent card base URL for endpoint discovery",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        default=True,
        help="Run in server mode (default: True)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run in local task mode",
    )
    
    # Local mode args
    parser.add_argument(
        "--task-id",
        default="T1_single_page",
        help="Task ID to run (default: T1_single_page)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: _purple_output/<task_id>/)",
    )
    parser.add_argument(
        "--mock-url",
        default="http://localhost:8000",
        help="Mock service URL (default: http://localhost:8000)",
    )
    
    # Parse known args, ignore unknown (for compose compatibility)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")
    
    # Local mode if --local flag is set
    if args.local:
        if args.output_dir is None:
            args.output_dir = f"_purple_output/{args.task_id}"
        return run_local(args.task_id, args.output_dir, args.mock_url)
    
    # Server mode (default)
    port = int(os.getenv("PORT", str(args.port)))
    card_url = args.card_url or f"http://localhost:{port}"
    run_server(args.host, port, card_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())

