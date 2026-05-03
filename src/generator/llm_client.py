from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Generator

import yaml
from openai import OpenAI

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - imported lazily at runtime
    genai = None  # type: ignore[assignment]


_openai_client: OpenAI | None = None
_google_configured = False


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "config.yaml").exists():
            return parent
    return current.parents[2]


def _load_config() -> dict[str, Any]:
    config_path = _find_project_root() / "config.yaml"
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _infer_provider(model_name: str, provider_raw: Any) -> str:
    provider = str(provider_raw).strip().lower() if provider_raw is not None else ""
    if provider:
        return provider
    return "google" if "gemini" in model_name.lower() else "openai"


def _generation_settings() -> tuple[str, str, float, int]:
    cfg = _load_config()
    generation = cfg.get("generation")
    generation = generation if isinstance(generation, dict) else {}

    model_name = str(generation.get("model", "gpt-4o-mini"))
    provider = _infer_provider(model_name, generation.get("provider"))

    temperature = float(generation.get("temperature", 0.3))
    max_tokens = int(generation.get("max_tokens", 1000))
    return model_name, provider, temperature, max_tokens


def _init_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing. Please set it in your environment.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _init_google() -> None:
    global _google_configured
    if _google_configured:
        return
    if genai is None:
        raise ValueError("google.generativeai is not available. Please install google-genai.")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Please set it in your environment.")
    genai.configure(api_key=api_key)
    _google_configured = True


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _to_gemini_payload(messages: list[dict], json_mode: bool) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    history: list[dict] = []

    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = _safe_text(m.get("content", "")).strip()
        if not content:
            continue

        if role == "system":
            system_parts.append(content)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        history.append({"role": gemini_role, "parts": [{"text": content}]})

    if json_mode:
        json_hint = "请只返回合法的 JSON 对象，不要包含任何其他内容或 markdown 格式。"
        if history and history[-1]["role"] == "user":
            prev_text = _safe_text(history[-1]["parts"][0].get("text", ""))
            history[-1]["parts"][0]["text"] = f"{prev_text}\n\n{json_hint}"
        else:
            history.append({"role": "user", "parts": [{"text": json_hint}]})

    system_instruction = "\n\n".join(system_parts).strip() or None
    return system_instruction, history


def _chat_openai(
    model_name: str,
    messages: list[dict],
    stream: bool,
    json_mode: bool,
    temperature: float,
    max_tokens: int,
) -> str | Generator[str, None, None]:
    client = _init_openai_client()
    if stream:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        def _gen() -> Generator[str, None, None]:
            for chunk in resp:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        return _gen()

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return _safe_text(resp.choices[0].message.content if resp.choices else "").strip()


def _chat_google(
    model_name: str,
    messages: list[dict],
    stream: bool,
    json_mode: bool,
    temperature: float,
    max_tokens: int,
) -> str | Generator[str, None, None]:
    _init_google()
    system_instruction, history = _to_gemini_payload(messages, json_mode=bool(json_mode and not stream))

    model = genai.GenerativeModel(  # type: ignore[union-attr]
        model_name=model_name,
        system_instruction=system_instruction,
    )

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode and not stream:
        generation_config["response_mime_type"] = "application/json"

    if stream:
        resp = model.generate_content(history, stream=True, generation_config=generation_config)

        def _gen() -> Generator[str, None, None]:
            for chunk in resp:
                text = _safe_text(getattr(chunk, "text", ""))
                if text:
                    yield text

        return _gen()

    resp = model.generate_content(history, generation_config=generation_config)
    return _safe_text(getattr(resp, "text", "")).strip()


def chat_complete(
    messages: list[dict],
    stream: bool = False,
    json_mode: bool = False,
    model_name: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str | Generator[str, None, None]:
    cfg_model, cfg_provider, cfg_temp, cfg_max_tokens = _generation_settings()
    effective_model = model_name or cfg_model
    effective_provider = _infer_provider(effective_model, provider if provider is not None else cfg_provider)
    effective_temp = cfg_temp if temperature is None else float(temperature)
    effective_max_tokens = cfg_max_tokens if max_tokens is None else int(max_tokens)

    if effective_provider == "openai":
        return _chat_openai(
            model_name=effective_model,
            messages=messages,
            stream=stream,
            json_mode=json_mode,
            temperature=effective_temp,
            max_tokens=effective_max_tokens,
        )
    if effective_provider == "google":
        return _chat_google(
            model_name=effective_model,
            messages=messages,
            stream=stream,
            json_mode=json_mode,
            temperature=effective_temp,
            max_tokens=effective_max_tokens,
        )
    raise ValueError(
        f"Unsupported provider: {effective_provider}. Expected 'openai' or 'google'."
    )

