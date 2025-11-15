#!/bin/bash

echo "Starting FastAPI app..."

export PORT=${PORT:-10000}

# Correct module name (fashion_app.py → app:app)
uvicorn app:app --host 0.0.0.0 --port $PORT
