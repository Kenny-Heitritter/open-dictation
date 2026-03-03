"""LLM service for agent mode.

Local-first: defaults to Ollama (native API with think:false for
reasoning models like qwen3).
Cloud optional: set OPENAI_API_KEY or GROQ_API_KEY to use cloud providers.

Follows shuo's streaming pattern: tokens are yielded as they arrive
so the TTS pipeline can begin synthesis immediately.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Callable, List, Optional

import urllib.request


# ── Configuration ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. You are speaking out loud, so "
    "avoid markdown, bullet points, code blocks, or other formatting "
    "that doesn't work in speech. Be conversational and natural. "
    "Match your response length to the question — short answers for "
    "simple questions, longer answers when the user asks for detail, "
    "stories, explanations, or recitations."
)


def _get_backend() -> tuple:
    """Determine which LLM backend to use.

    Returns (backend_type, base_url, api_key, model).
    backend_type is 'ollama', 'groq', or 'openai'.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model_override = os.environ.get("LLM_MODEL", "")

    if groq_key:
        return (
            "groq",
            "https://api.groq.com/openai/v1",
            groq_key,
            model_override or "llama-3.3-70b-versatile",
        )
    if openai_key:
        return (
            "openai",
            "https://api.openai.com/v1",
            openai_key,
            model_override or "gpt-4o-mini",
        )
    # Default: Ollama local (native API for think:false support)
    return (
        "ollama",
        ollama_url,
        "",
        model_override or "llama3.2",
    )


class LLMService:
    """Streaming LLM chat completion service.

    Maintains conversation history across turns (like shuo's Agent).
    Tokens are delivered via callback for immediate TTS pipelining.

    Uses native Ollama API (with think:false) for local models,
    OpenAI-compatible API for cloud backends.
    """

    def __init__(
        self,
        on_token: Callable[[str], None],
        on_done: Callable[[], None],
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._backend_type, self._base_url, self._api_key, self._model = (
            _get_backend()
        )
        self._on_token = on_token
        self._on_done = on_done

        self._history: List[dict] = [
            {"role": "system", "content": system_prompt},
        ]
        self._task: Optional[threading.Thread] = None
        self._cancelled = False
        self._partial_response = ""

        # For cloud backends, use the openai SDK
        self._openai_client = None
        if self._backend_type in ("groq", "openai"):
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(
                    base_url=self._base_url, api_key=self._api_key
                )
            except ImportError:
                raise ImportError(
                    "openai package required for cloud LLM backends. "
                    "Install with: pip install openai"
                )

        backend_name = {
            "ollama": "Ollama",
            "groq": "Groq",
            "openai": "OpenAI",
        }[self._backend_type]
        sys.stdout.write(f"[LLM] Backend: {backend_name} ({self._model})\n")
        sys.stdout.flush()

    def start(self, user_message: str) -> None:
        """Begin streaming a response to the user's message."""
        self._cancelled = False
        self._partial_response = ""
        self._history.append({"role": "user", "content": user_message})
        self._task = threading.Thread(target=self._generate, daemon=True)
        self._task.start()

    def cancel(self) -> None:
        """Cancel in-flight generation."""
        self._cancelled = True
        if self._partial_response:
            self._history.append({
                "role": "assistant",
                "content": self._partial_response + "...",
            })
            self._partial_response = ""

    def _generate(self) -> None:
        """Stream tokens from the LLM."""
        try:
            if self._backend_type == "ollama":
                self._generate_ollama()
            else:
                self._generate_openai()
        except Exception as e:
            sys.stdout.write(f"[LLM] Error: {e}\n")
            sys.stdout.flush()
            if not self._cancelled:
                self._on_done()

    def _generate_ollama(self) -> None:
        """Stream via native Ollama API with think:false."""
        url = f"{self._base_url}/api/chat"
        payload = json.dumps({
            "model": self._model,
            "messages": self._history,
            "stream": True,
            "think": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                if self._cancelled:
                    break
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                token = msg.get("content", "")
                if token:
                    self._partial_response += token
                    self._on_token(token)
                if chunk.get("done"):
                    break

        if not self._cancelled:
            self._history.append({
                "role": "assistant",
                "content": self._partial_response,
            })
            self._on_done()

    def _generate_openai(self) -> None:
        """Stream via OpenAI-compatible API (Groq, OpenAI)."""
        if not self._openai_client:
            return

        stream = self._openai_client.chat.completions.create(
            model=self._model,
            messages=self._history,
            max_tokens=500,
            temperature=0.7,
            stream=True,
        )

        for chunk in stream:
            if self._cancelled:
                break
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token = delta.content
                self._partial_response += token
                self._on_token(token)

        if not self._cancelled:
            self._history.append({
                "role": "assistant",
                "content": self._partial_response,
            })
            self._on_done()
