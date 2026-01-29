#!/usr/bin/env bash

set -a
source .env.local 2>/dev/null || true
set +a

source .venv/Scripts/activate

export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama2

uvicorn app.main:app --reload
