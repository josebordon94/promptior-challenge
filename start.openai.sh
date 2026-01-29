#!/usr/bin/env bash

set -a
source .env.openai
set +a

source .venv/Scripts/activate

export LLM_PROVIDER=openai
export OPENAI_MODEL=gpt-4o-mini

uvicorn app.main:app --reload
