import os
from urllib.parse import urlparse


def is_local_openai_compatible_base_url(base_url: str | None) -> bool:
    if not base_url:
        return True
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def resolve_openai_api_key(base_url: str | None, env_var: str = "OPENAI_API_KEY") -> str:
    return os.environ.get(env_var, "EMPTY")


def resolve_openai_compatible_base_url(explicit_base_url: str | None, port: int | None) -> str:
    if explicit_base_url:
        return explicit_base_url
    if port is None:
        return "https://api.openai.com/v1"
    return f"http://localhost:{port}/v1"


def apply_completion_limit(api_kwargs: dict, base_url: str | None, max_tokens: int) -> dict:
    if urlparse(base_url or '').hostname == 'api.openai.com':
        api_kwargs["max_completion_tokens"] = max_tokens
    else:
        api_kwargs["max_tokens"] = max_tokens
    return api_kwargs


def is_remote_gpt5_family(base_url: str | None, model_name: str | None) -> bool:
    if is_local_openai_compatible_base_url(base_url):
        return False
    name = (model_name or "").lower()
    return name.startswith("gpt-5")
