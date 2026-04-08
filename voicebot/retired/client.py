"""
client.py
──────────
Shared AsyncOpenAI singleton.

Why a singleton?
  AsyncOpenAI() opens an HTTP/2 connection pool to api.openai.com.
  If agent.py and turn_detector.py each instantiate their own client,
  the process holds two separate connection pools to the same service —
  wasted file descriptors and TCP connections.

  Importing from this module gives every caller the same pool.

Usage:
    from voicebot.client import openai_client
"""

from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_client = AsyncOpenAI()
