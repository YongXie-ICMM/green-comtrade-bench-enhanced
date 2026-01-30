"""
Baseline Purple Agent - A2A Server Implementation

Usage:
    python run_a2a.py --host 0.0.0.0 --port 9009 --card-url http://purple-agent:9009
"""

import sys
sys.stderr.write("[STARTUP] Starting run_a2a.py...\n")
sys.stderr.flush()

print("[STARTUP] Importing modules...", flush=True)

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from pydantic import BaseModel

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, AgentSkill, AgentCapabilities, TaskState
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.types import InvalidParamsError

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("purple-agent")


class TaskRequest(BaseModel):
    """Request to run a benchmark task."""
    task_id: str
    mock_url: str = "http://mock-comtrade:8000"
    output_dir: str = None


class PurpleExecutor(AgentExecutor):
    """A2A AgentExecutor for Purple Comtrade Baseline."""

    def __init__(self):
        pass

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute task request."""
        # Get user input (message content)
        request_text = context.get_user_input()
        logger.info(f"Received request: {request_text[:200]}...")

        # Parse as TaskRequest
        try:
            request_data = json.loads(request_text)
            task_request = TaskRequest(**request_data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse TaskRequest: {e}")
            raise ServerError(error=InvalidParamsError(message=f"Invalid TaskRequest format: {e}"))

        # Set default output_dir if not provided
        if task_request.output_dir is None:
            task_request.output_dir = f"/workspace/purple_output/{task_request.task_id}"

        # Create task
        msg_obj = context.message
        if msg_obj:
            task = new_task(msg_obj)
            await event_queue.enqueue_event(task)
        else:
            raise ServerError(error=InvalidParamsError(message="Missing message in context"))

        # Create task updater
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting task {task_request.task_id}")
        )

        # Run task in background thread
        try:
            from purple_agent import PurpleAgent
            agent = PurpleAgent()
            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                success = await loop.run_in_executor(
                    executor,
                    agent.run,
                    task_request.task_id,
                    task_request.output_dir,
                    task_request.mock_url
                )

            if success:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Task {task_request.task_id} completed successfully")
                )
                await updater.complete()
            else:
                await updater.failed(new_agent_text_message(f"Task {task_request.task_id} failed"))
                raise ServerError(error=InvalidParamsError(message=f"Task execution failed"))

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            await updater.failed(new_agent_text_message(f"Task failed: {e}"))
            raise

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel is not supported."""
        from a2a.types import UnsupportedOperationError
        raise ServerError(error=UnsupportedOperationError())


def create_agent_card(agent_url: str) -> AgentCard:
    """Create agent card for purple baseline."""
    skill = AgentSkill(
        id="comtrade.bench.run",
        name="run",
        description="Run benchmark tasks",
        tags=["comtrade", "benchmark", "a2a"]
    )

    return AgentCard(
        name="purple-comtrade-baseline-v2",
        version="2.0.0",
        description="Baseline Purple agent for Green Comtrade Bench v2",
        url=agent_url,
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill]
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Purple Comtrade Baseline (A2A)")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9009, help="Server port")
    parser.add_argument("--card-url", default=None, help="External agent URL")

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignoring unknown args: {unknown}")

    # Determine agent URL
    agent_url = args.card_url or f"http://{args.host}:{args.port}"

    # Create executor
    executor = PurpleExecutor()

    # Create agent card
    agent_card = create_agent_card(agent_url)

    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # Create A2A server
    a2a_server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build app
    app = a2a_server.build()

    logger.info(f"Starting Purple Comtrade Baseline on {args.host}:{args.port}")
    logger.info(f"Agent URL: {agent_url}")

    # Run server
    config = uvicorn.Config(app, host=args.host, port=args.port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        print("Starting purple agent A2A server...", flush=True)
        asyncio.run(main())
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
