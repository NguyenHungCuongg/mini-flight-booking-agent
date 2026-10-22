"""The real LLM, configured only from .env (OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def make_model():
    load_dotenv()
    base_url = os.environ["OPENAI_BASE_URL"]
    if "googleapis.com" in base_url:
        # Gemini rejects unknown body fields (400); it turns thinking off with reasoning_effort.
        thinking_off = {"reasoning_effort": "none"}
    else:
        # Qwen thinking off: OpenRouter reads "reasoning", the UIT vLLM server reads "chat_template_kwargs".
        thinking_off = {"extra_body": {"reasoning": {"enabled": False},
                                       "chat_template_kwargs": {"enable_thinking": False}}}
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        base_url=base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_retries=3,          # 429 / timeout / 5xx: retried with backoff by the client
        timeout=120,
        **thinking_off,
    )
