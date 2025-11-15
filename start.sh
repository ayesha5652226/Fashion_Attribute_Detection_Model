#!/bin/bash

# Render MUST NOT restart multiple servers.
# Ensure we run only one uvicorn instance.

echo "Starting FastAPI app..."

# Export PORT from Render
export PORT=${PORT:-10000}

# Start uvicorn WITHOUT reload (reload causes infinite restarts on Render)
uvicorn app:app --host 0.0.0.0 --port $PORT
