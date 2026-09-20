import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434/v1",
)

client = OpenAI(
    api_key=OLLAMA_API_KEY,
    base_url=OLLAMA_BASE_URL,
)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()