"""
Single interface over Ollama (local) and OpenAI API, chosen via
config/llm.yaml. react_loop.py should only ever call this module,
never Ollama/OpenAI SDKs directly — this is what lets us swap
backend with a config change and no code edits elsewhere.

Backend SDKs are imported lazily inside each client, so the OpenAI package
being absent never breaks an Ollama run and vice versa.

``EchoClient`` is a scripted backend for tests: it replays a fixed list of
replies. The ReAct loop needs to be testable without a model server, and a
deterministic transcript is also the only way to assert that Phase 2 injection
changes agent behaviour rather than just changing sampling noise.
"""

from __future__ import annotations

import abc
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_LLM_CONFIG_PATH = "config/llm.yaml"
DEFAULT_TIMEOUT = 120


@dataclass
class LLMResponse:
    """One completion. ``raw`` keeps the backend payload for logging."""

    text: str
    model: str = ""
    backend: str = ""
    finish_reason: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.text or self.tool_calls)


class LLMClient(abc.ABC):
    """Backend-agnostic completion interface."""

    backend: str = ""

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options

    @abc.abstractmethod
    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat-style message list and return the reply."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} backend={self.backend!r} model={self.model!r}>"


class OllamaClient(LLMClient):
    """Local Ollama server over its HTTP chat API."""

    backend = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = DEFAULT_TIMEOUT,
        supports_tools: Optional[bool] = None,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        self.host = host.rstrip("/")
        self.timeout = timeout
        #: None means "not yet known" — discovered on first use and cached.
        self.supports_tools = supports_tools

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to /api/chat, surfacing the server's own error text."""
        import requests

        try:
            response = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 - connection-level failure
            raise LLMError(
                f"Ollama request to {self.host} failed: {type(exc).__name__}: {exc}. "
                f"Is `ollama serve` running and has `{self.model}` been pulled?"
            ) from exc

        if response.status_code >= 400:
            # The body carries the actual reason. Reporting only the status code
            # turns a one-line fix into a debugging session.
            detail = (response.text or "").strip()[:500]
            raise LLMError(
                f"Ollama returned HTTP {response.status_code} for model "
                f"{self.model!r}: {detail or '(no response body)'}"
            )
        return response.json()

    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        options = {**self.options, **kwargs}
        if options:
            payload["options"] = options

        # Many Ollama models have no tool-calling template and reject the
        # request outright ("does not support tools"). The ReAct loop parses
        # ACTION: out of plain text and does not need native tool calls, so
        # tools are an optimisation, not a requirement — never a reason to fail.
        send_tools = bool(tools) and self.supports_tools is not False
        if send_tools:
            # Ollama mirrors OpenAI's function-tool shape.
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        try:
            body = self._post(payload)
            if send_tools and self.supports_tools is None:
                self.supports_tools = True
        except LLMError as exc:
            if not (send_tools and "does not support tools" in str(exc)):
                raise
            warnings.warn(
                f"{self.model} has no tool-calling template; falling back to "
                f"text-parsed tool calls for the rest of this session.",
                stacklevel=2,
            )
            self.supports_tools = False
            payload.pop("tools", None)
            body = self._post(payload)

        message = body.get("message", {}) or {}
        return LLMResponse(
            text=message.get("content", "") or "",
            model=body.get("model", self.model),
            backend=self.backend,
            finish_reason=body.get("done_reason", ""),
            tool_calls=_normalize_tool_calls(message.get("tool_calls")),
            usage={
                k: body[k]
                for k in ("prompt_eval_count", "eval_count", "total_duration")
                if k in body
            },
            raw=body,
        )


class OpenAIClient(LLMClient):
    """OpenAI chat completions. The API key comes from the environment only."""

    backend = "openai"

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = DEFAULT_TIMEOUT,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        self.api_key_env = api_key_env
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise LLMError(
                f"{self.api_key_env} is not set. config/llm.yaml deliberately "
                f"stores no secrets; export the key in the environment."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMError(
                "the `openai` package is not installed; `pip install openai` or "
                "set backend: \"ollama\" in config/llm.yaml."
            ) from exc

        client = OpenAI(api_key=api_key, timeout=self.timeout)
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            **self.options,
            **kwargs,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        try:
            completion = client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI request failed: {type(exc).__name__}: {exc}") from exc

        choice = completion.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            backend=self.backend,
            finish_reason=choice.finish_reason or "",
            tool_calls=_normalize_tool_calls(
                [
                    {
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        }
                    }
                    for c in (choice.message.tool_calls or [])
                ]
            ),
            usage=(completion.usage.model_dump() if completion.usage else {}),
            raw=completion.model_dump(),
        )


class EchoClient(LLMClient):
    """
    Scripted backend for tests. Returns ``replies`` in order, then repeats the
    last one. Records every message list it was given as ``.calls``, so a test
    can assert exactly what text reached the model — which is how Phase 2 will
    verify that injected telemetry actually entered the context.
    """

    backend = "echo"

    def __init__(self, replies: Optional[Sequence[str]] = None, model: str = "echo") -> None:
        super().__init__(model)
        self.replies = list(replies or [""])
        self.calls: List[List[Dict[str, str]]] = []

    def complete(
        self,
        messages: Sequence[Dict[str, str]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return LLMResponse(
            text=self.replies[index], model=self.model, backend=self.backend
        )


class LLMError(RuntimeError):
    """Backend failure, raised with enough context to act on."""


def _normalize_tool_calls(calls: Any) -> List[Dict[str, Any]]:
    """Flatten either backend's tool-call shape to ``{name, arguments}``."""
    out: List[Dict[str, Any]] = []
    for call in calls or []:
        function = call.get("function", call) if isinstance(call, dict) else {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        out.append({"name": function.get("name", ""), "arguments": arguments})
    return out


def load_llm_config(path: str = DEFAULT_LLM_CONFIG_PATH) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_BACKENDS = {"ollama": OllamaClient, "openai": OpenAIClient, "echo": EchoClient}


def build_llm_client(config: Optional[Dict[str, Any]] = None, **overrides: Any) -> LLMClient:
    """
    Build the client named by ``config['backend']``.

    The backend's own block supplies its arguments, so switching between local
    and hosted models is a one-line config change with no code edit, as the
    module docstring promises.
    """
    config = load_llm_config() if config is None else config
    backend = str(config.get("backend", "ollama")).lower()
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown LLM backend {backend!r}; expected one of {sorted(_BACKENDS)}. "
            f"Set `backend` in config/llm.yaml."
        )
    settings = dict(config.get(backend, {}) or {})
    settings.pop("api_key", None)  # never accept a key from the config file
    settings.update(overrides)
    if backend != "echo" and "model" not in settings:
        raise ValueError(f"config/llm.yaml `{backend}` block must specify a model.")
    return _BACKENDS[backend](**settings)


__all__ = [
    "EchoClient",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "OllamaClient",
    "OpenAIClient",
    "build_llm_client",
    "load_llm_config",
]
