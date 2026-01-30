FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies - use a2a-sdk (not a2a-server!)
RUN pip install --no-cache-dir requests fastapi uvicorn pydantic httpx "a2a-sdk>=0.3.5" starlette "sse-starlette>=1.6.5"

# Copy application code
COPY *.py ./
COPY start.sh ./
RUN chmod +x start.sh

# Create output directory
RUN mkdir -p /workspace/purple_output

# Test if the Python script can be imported (will fail build if there's a syntax error)
RUN python -m py_compile run_a2a.py && echo "Python syntax check passed"

# Default environment variables
ENV MOCK_URL=http://mock-comtrade:8000
ENV TASK_ID=T1_single_page
ENV OUTPUT_DIR=/workspace/purple_output

# Expose server port
EXPOSE 9009

# Entrypoint: run the purple agent with A2A Server SDK (args passed through from runner)
# ENTRYPOINT receives args from docker-compose command field
ENTRYPOINT ["python", "-u", "/app/run_a2a.py"]
CMD []
