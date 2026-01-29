#!/usr/bin/env bash

source .venv/Scripts/activate
uvicorn app.main:app --reload