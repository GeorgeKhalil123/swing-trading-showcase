"""The LLM layer: one backend protocol, one runtime that validates, retries once and caches."""

from swingcore.llm.base import AgentCall, Backend, LLMError, LLMReply, LLMTimeout
from swingcore.llm.fake import FakeBackend
from swingcore.llm.runtime import AgentFailed, AgentResult, Runtime, extract_json

__all__ = [
    "AgentCall",
    "AgentFailed",
    "AgentResult",
    "Backend",
    "FakeBackend",
    "LLMError",
    "LLMReply",
    "LLMTimeout",
    "Runtime",
    "extract_json",
]
