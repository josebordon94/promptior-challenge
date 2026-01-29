import os
from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "local")

if APP_ENV == "openai":
    load_dotenv(".env.openai")
else:
    load_dotenv(".env.local")

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
