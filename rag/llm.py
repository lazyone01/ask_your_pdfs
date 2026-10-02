"""Chat LLM backends (Ollama default; Groq, OpenAI-compatible APIs and Anthropic optional) behind one interface."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from .config import GenerationConfig

log = logging.getLogger(__name__)

# Some local models (qwen3, deepseek-r1) emit reasoning in <think> tags; never show it as the answer.
# "Thinking-only" builds (e.g. qwen3:4b = Qwen3-4B-Thinking-2507) ignore think=False and emit the
# reasoning with only a closing </think> tag, so we also cut everything before the last </think>.
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_END = "</think>"


def strip_reasoning(text: str) -> str:
    text = _THINK.sub("", text)
    if _THINK_END in text:
        text = text.rsplit(_THINK_END, 1)[1]
    return text.strip()


def _visible_so_far(text: str, may_think: bool) -> str | None:
    """Part of a partial stream that is safe to show; None = still (possibly) inside reasoning."""
    if _THINK_END in text:
        return text.rsplit(_THINK_END, 1)[1].lstrip()
    if may_think or text.lstrip().startswith("<think>"):
        return None
    return text


class LLMError(RuntimeError):
    """User-presentable LLM failure (server down, model missing, rate limit...)."""


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict = field(default_factory=dict)


class LLM(ABC):
    answer_model: str
    rewrite_model: str

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict],
        model: str,
        max_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse: ...

    def status(self) -> tuple[bool, str]:
        """(ok, message) for the UI health indicator."""
        return True, "ok"


class OllamaLLM(LLM):
    def __init__(self, cfg: GenerationConfig):
        import ollama

        self.cfg = cfg.ollama
        self.answer_model = self.cfg.answer_model
        self.rewrite_model = self.cfg.rewrite_model
        self._ollama = ollama
        self.client = ollama.Client(host=self.cfg.base_url, timeout=self.cfg.timeout_seconds)
        self._thinker: dict[str, bool] = {}  # model -> has "thinking" capability

    def _is_thinker(self, model: str) -> bool:
        if model not in self._thinker:
            try:
                caps = self.client.show(model).capabilities or []
                self._thinker[model] = "thinking" in caps
            except Exception:
                return False  # unknown (server down / model missing); chat() reports the real error
        return self._thinker[model]

    def status(self) -> tuple[bool, str]:
        try:
            installed = {m.model for m in self.client.list().models}
        except Exception:
            return False, f"Ollama is not reachable at {self.cfg.base_url}. Start the Ollama app."
        missing = [
            m for m in {self.answer_model, self.rewrite_model}
            if m not in installed and f"{m}:latest" not in installed
        ]
        if missing:
            return False, "Model not downloaded. Run: " + "; ".join(f"ollama pull {m}" for m in missing)
        return True, f"Ollama ready ({self.answer_model})"

    def chat(self, system, messages, model, max_tokens, on_token=None) -> LLMResponse:
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            options={
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                "num_predict": max_tokens,
            },
            stream=True,
        )
        thinker = self._is_thinker(model)
        if thinker:
            kwargs["think"] = False  # hybrid models (original qwen3) skip reasoning: much faster on CPU
        try:
            return self._run(kwargs, model, on_token, thinker)
        except self._ollama.ResponseError as e:
            if "think" in str(e).lower() and "think" in kwargs:
                kwargs.pop("think")  # server rejected the flag; retry without it
                return self._run(kwargs, model, on_token, thinker)
            if e.status_code == 404:
                raise LLMError(f"Ollama model '{model}' is not installed. Run: ollama pull {model}") from e
            raise LLMError(f"Ollama error: {e.error}") from e
        except ConnectionError as e:
            raise LLMError(f"Cannot reach Ollama at {self.cfg.base_url}. Is the Ollama app running?") from e
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower():
                raise LLMError(
                    f"Ollama timed out after {self.cfg.timeout_seconds:.0f}s. Try a smaller model, "
                    "lower retrieval.top_k, or raise generation.ollama.timeout_seconds."
                ) from e
            if "connect" in str(e).lower():
                raise LLMError(f"Cannot reach Ollama at {self.cfg.base_url}. Is the Ollama app running?") from e
            raise

    def _run(self, kwargs: dict, model: str, on_token, may_think: bool) -> LLMResponse:
        parts: list[str] = []
        usage: dict = {}
        emitted = 0  # characters of visible answer already streamed to on_token
        for chunk in self.client.chat(**kwargs):
            # Reasoning sent in the separate `thinking` field is simply ignored.
            delta = chunk.message.content or ""
            if delta:
                parts.append(delta)
                if on_token:
                    visible = _visible_so_far("".join(parts), may_think)
                    if visible is not None and len(visible) > emitted:
                        on_token(visible[emitted:])
                        emitted = len(visible)
            if chunk.done:
                usage = {"input_tokens": chunk.prompt_eval_count, "output_tokens": chunk.eval_count}
        text = strip_reasoning("".join(parts))
        if on_token and len(text) > emitted:  # held back because the model might have been reasoning
            on_token(text[emitted:])
        return LLMResponse(text, model, usage)


class AnthropicLLM(LLM):
    def __init__(self, cfg: GenerationConfig):
        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("generation.provider is 'anthropic' but ANTHROPIC_API_KEY is not set in .env")
        self.cfg = cfg.anthropic
        self.answer_model = self.cfg.answer_model
        self.rewrite_model = self.cfg.rewrite_model
        self._anthropic = anthropic
        # The SDK retries 429/5xx/connection errors with exponential backoff.
        self.client = anthropic.Anthropic(max_retries=self.cfg.max_retries, timeout=self.cfg.timeout_seconds)

    def chat(self, system, messages, model, max_tokens, on_token=None) -> LLMResponse:
        try:
            with self.client.messages.stream(
                model=model, system=system, messages=messages, max_tokens=max_tokens
            ) as stream:
                for delta in stream.text_stream:
                    if on_token:
                        on_token(delta)
                final = stream.get_final_message()
        except self._anthropic.RateLimitError as e:
            raise LLMError("Anthropic rate limit hit even after retries; wait a minute and try again.") from e
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError("Cannot reach the Anthropic API (network problem).") from e
        text = "".join(b.text for b in final.content if b.type == "text").strip()
        usage = {"input_tokens": final.usage.input_tokens, "output_tokens": final.usage.output_tokens}
        return LLMResponse(text, model, usage)


class GroqLLM(LLM):
    """Groq's hosted open models (free tier, no card). Used by the Streamlit Cloud deployment."""

    def __init__(self, cfg: GenerationConfig):
        import groq

        if not os.getenv("GROQ_API_KEY"):
            raise LLMError("generation.provider is 'groq' but GROQ_API_KEY is not set (.env or app secrets)")
        self.cfg = cfg.groq
        self.answer_model = self.cfg.answer_model
        self.rewrite_model = self.cfg.rewrite_model
        self._groq = groq
        # The SDK retries 429/5xx/connection errors with exponential backoff.
        self.client = groq.Groq(max_retries=self.cfg.max_retries, timeout=self.cfg.timeout_seconds)

    def status(self) -> tuple[bool, str]:
        return True, f"Groq ready ({self.answer_model})"

    def chat(self, system, messages, model, max_tokens, on_token=None) -> LLMResponse:
        parts: list[str] = []
        usage: dict = {}
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, *messages],
                max_tokens=max_tokens,
                temperature=self.cfg.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    if on_token:
                        on_token(delta)
                u = getattr(getattr(chunk, "x_groq", None), "usage", None)
                if u:
                    usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}
        except self._groq.RateLimitError as e:
            raise LLMError("Groq free-tier rate limit reached; wait a minute and try again.") from e
        except self._groq.APIStatusError as e:
            raise LLMError(f"Groq API error {e.status_code}: {e.message}") from e
        except self._groq.APIConnectionError as e:
            raise LLMError("Cannot reach the Groq API (network problem).") from e
        return LLMResponse(strip_reasoning("".join(parts)), model, usage)


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible chat endpoint (Gemini, Cerebras, OpenRouter...). Used by the cloud deployment."""

    def __init__(self, cfg: GenerationConfig):
        import openai

        self.cfg = cfg.openai_compat
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise LLMError(
                f"generation.provider is 'openai_compat' ({self.cfg.name}) but {self.cfg.api_key_env} "
                "is not set (.env or app secrets)"
            )
        self.answer_model = self.cfg.answer_model
        self.rewrite_model = self.cfg.rewrite_model
        self._openai = openai
        # The SDK retries 429/5xx/connection errors with exponential backoff.
        self.client = openai.OpenAI(
            api_key=key, base_url=self.cfg.base_url,
            max_retries=self.cfg.max_retries, timeout=self.cfg.timeout_seconds,
        )
        self._use_reasoning = bool(self.cfg.reasoning_effort)

    def status(self) -> tuple[bool, str]:
        return True, f"{self.cfg.name} ready ({self.answer_model})"

    def chat(self, system, messages, model, max_tokens, on_token=None) -> LLMResponse:
        kwargs = dict(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
            temperature=self.cfg.temperature,
            stream=True,
        )
        if self._use_reasoning:
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort
        name = self.cfg.name
        try:
            try:
                return self._run(kwargs, model, on_token)
            except self._openai.BadRequestError as e:
                if "reasoning" not in str(e).lower() or "reasoning_effort" not in kwargs:
                    raise
                self._use_reasoning = False  # this model/endpoint doesn't take the flag
                kwargs.pop("reasoning_effort")
                return self._run(kwargs, model, on_token)
        except self._openai.RateLimitError as e:
            raise LLMError(f"{name} free-tier rate limit reached; wait a minute and try again.") from e
        except self._openai.APIStatusError as e:
            raise LLMError(f"{name} API error {e.status_code}: {e.message}") from e
        except self._openai.APIConnectionError as e:
            raise LLMError(f"Cannot reach the {name} API (network problem).") from e

    def _run(self, kwargs: dict, model: str, on_token) -> LLMResponse:
        parts: list[str] = []
        usage: dict = {}
        for chunk in self.client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
                if on_token:
                    on_token(delta)
            if getattr(chunk, "usage", None):
                usage = {"input_tokens": chunk.usage.prompt_tokens, "output_tokens": chunk.usage.completion_tokens}
        return LLMResponse(strip_reasoning("".join(parts)), model, usage)


def get_llm(cfg: GenerationConfig) -> LLM:
    if cfg.provider == "ollama":
        return OllamaLLM(cfg)
    if cfg.provider == "groq":
        return GroqLLM(cfg)
    if cfg.provider == "openai_compat":
        return OpenAICompatLLM(cfg)
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg)
    raise ValueError(f"Unknown generation provider: {cfg.provider}")
