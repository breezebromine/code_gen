"""Audited model client; simulator workers do not import this module."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from code_gen import api_config
from code_gen._io import PROJECT_ROOT, write_json


class ModelError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError("Model endpoint redirects are not accepted")


class ChatClient:
    def __init__(self, model=None, base_url=None, timeout=None, api_key=None):
        model = api_config.MODEL if model is None else model
        base_url = api_config.BASE_URL if base_url is None else base_url
        timeout = api_config.TIMEOUT_SECONDS if timeout is None else timeout
        if api_key is None:
            api_key = api_config.API_KEY.strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not api_key.strip():
                key_file = PROJECT_ROOT / ".codegen_api_key"
                api_key = key_file.read_text().strip() if key_file.is_file() else ""
        self.api_key = api_key
        if not self.api_key.strip():
            raise ModelError("Set API_KEY in code_gen/api_config.py or set DEEPSEEK_API_KEY")
        if not 1 <= timeout <= 300:
            raise ModelError("API timeout must be between 1 and 300 seconds")
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
            raise ModelError("Model endpoint must be an HTTPS API prefix")
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    @classmethod
    def for_vision(cls):
        key = api_config.VISION_API_KEY.strip() or os.environ.get("CODEGEN_VISION_API_KEY", "").strip() or None
        return cls(model=api_config.VISION_MODEL, base_url=api_config.VISION_BASE_URL, api_key=key)

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self.api_key, "[REDACTED]")
        if isinstance(value, list):
            return [self._redact(x) for x in value]
        if isinstance(value, dict):
            return {
                k: self._redact(v)
                for k, v in value.items()
                if k.lower() not in {"authorization", "api_key", "headers", "password", "secret"}
            }
        return value

    def complete(self, messages, directory, max_tokens=4096):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        if urlsplit(self.base_url).hostname == "api.deepseek.com":
            payload["thinking"] = {"type": "disabled"}
        write_json(directory / "request.json", self._redact(payload))
        req = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        try:
            with build_opener(_NoRedirect).open(req, timeout=self.timeout) as response:
                raw = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            error = {"type": type(exc).__name__, "status_code": getattr(exc, "code", None)}
            write_json(directory / "error.json", error)
            raise ModelError(f"Model request failed: {error}") from None
        write_json(directory / "response.json", self._redact(raw))
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
        ):
            raise ModelError("Model did not return one complete final response")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelError("Model returned no final text")
        return self._redact(content)
