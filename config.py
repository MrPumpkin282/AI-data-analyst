import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY") if hasattr(st, "secrets") else None
NVIDIA_API_KEY = NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. On Streamlit Cloud, add it under "
        "App settings -> Secrets. For local dev, add it to a .env file."
    )

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)

client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
)

NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()