#!/usr/bin/env bash
set -o errexit

# unzip model if needed
if [ -f "model/model_quantized_scripted.zip" ] && [ ! -f "model/model_quantized_scripted.pt" ]; then
  unzip model/model_quantized_scripted.zip -d model/
fi

# start fastapi
uvicorn app:app --host 0.0.0.0 --port $PORT
