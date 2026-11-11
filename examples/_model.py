"""The chat model the examples use: $LC_MODEL, default Claude Sonnet."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def model() -> BaseChatModel:
    """Return the chat model named in $LC_MODEL.

    Any LangChain chat model works, for example "google_genai:gemini-3.8-flash"
    (pip install langchain-google-genai, GOOGLE_API_KEY) or "openai:gpt-5".
    """
    return init_chat_model(os.environ.get("LC_MODEL", "anthropic:claude-sonnet-5-5"))
