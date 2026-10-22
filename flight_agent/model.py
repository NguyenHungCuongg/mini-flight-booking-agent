"""The real LLM, configured only from .env (OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def make_model():
    load_dotenv()
    base_url = os.environ["OPENAI_BASE_URL"]
    if "googleapis.com" in base_url:
        # Gemini 3 needs each tool call's thought_signature sent back on the next turn. ChatOpenAI drops it
        # (400 "missing a thought_signature"), so Gemini goes through Google's own client, which keeps it.
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.environ["OPENAI_MODEL"],
            api_key=os.environ["OPENAI_API_KEY"],
            thinking_level="low",   # Gemini 3 cannot turn thinking off; "low" is the fastest
            max_retries=3,
            timeout=120,
            # temperature left at Gemini 3's default 1.0: Google warns lower values can cause loops
        )
    return ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],
        base_url=base_url,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
        max_retries=3,          # 429 / timeout / 5xx: retried with backoff by the client
        timeout=120,
        # Qwen thinking off: OpenRouter reads "reasoning", the UIT vLLM server reads "chat_template_kwargs".
        extra_body={"reasoning": {"enabled": False}, "chat_template_kwargs": {"enable_thinking": False}},
    )
