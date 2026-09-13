from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import jsonschema

from core.errors import ExternalServiceError
from core.io_utils import load_json
from core.isolation import assert_validation_safe_value


def _load_env_value(repo_root: Path, name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return None
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            candidate = candidate.strip()
            if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
                candidate = candidate[1:-1]
            return candidate or None
    return None


class AgentClient:
    def __init__(self, repo_root: Path, selection: dict):
        self.repo_root = repo_root.resolve()
        catalog = load_json(self.repo_root / "configs" / "agent_models.json")
        model_name = selection["model"]
        effort = selection["reasoning_effort"]
        if model_name not in catalog["models"]:
            raise ExternalServiceError(f"AGENT_MODEL_NOT_CATALOGED: {model_name}")
        model = catalog["models"][model_name]
        if effort not in model["reasoning_efforts"]:
            raise ExternalServiceError(f"AGENT_EFFORT_NOT_ALLOWED: {effort}")
        provider = catalog["providers"][model["provider"]]
        key = _load_env_value(self.repo_root, provider["api_key_env"])
        if not key:
            raise ExternalServiceError(f"AGENT_API_KEY_MISSING: {provider['api_key_env']}")
        base_url = provider.get("base_url") or _load_env_value(self.repo_root, provider["base_url_env"])
        if not base_url:
            raise ExternalServiceError("AGENT_BASE_URL_MISSING")
        self.model_name = model_name
        self.model_id = model["model_id"]
        self.reasoning_effort = effort
        self.api = provider["api"]
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    @staticmethod
    def _response_text(payload: dict) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        parts: list[str] = []
        for item in payload.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content", []) if isinstance(item, dict) else []:
                if isinstance(content, dict) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        if parts:
            return "".join(parts)
        choices = payload.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, str):
                return content
        raise ExternalServiceError("AGENT_RESPONSE_HAS_NO_TEXT")

    def complete_json(self, system: str, user_context: dict, schema: dict, schema_name: str) -> dict:
        assert_validation_safe_value(user_context)
        user = "Return only JSON matching the supplied schema.\n\nCONTEXT:\n" + json.dumps(user_context, ensure_ascii=False, sort_keys=True)
        if self.api == "responses":
            url = self.base_url + "/responses"
            request = {
                "model": self.model_id,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "reasoning": {"effort": self.reasoning_effort},
                "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema}},
            }
        elif self.api == "chat_completions":
            url = self.base_url + "/chat/completions"
            request = {
                "model": self.model_id,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "reasoning_effort": self.reasoning_effort,
                "response_format": {"type": "json_object"},
            }
        else:
            raise ExternalServiceError(f"AGENT_API_UNSUPPORTED: {self.api}")

        last_error = "unknown"
        for attempt in range(3):
            try:
                with httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0)) as client:
                    response = client.post(url, headers=self._headers, json=request)
                if response.status_code >= 400:
                    last_error = f"HTTP_{response.status_code}: {response.text[:500]}"
                    if response.status_code < 500 and response.status_code != 429:
                        break
                else:
                    text = self._response_text(response.json())
                    result = json.loads(text)
                    jsonschema.Draft202012Validator(schema).validate(result)
                    assert_validation_safe_value(result)
                    return result
            except (httpx.HTTPError, ValueError, jsonschema.ValidationError) as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            if attempt < 2:
                time.sleep(2 ** attempt)
        raise ExternalServiceError(f"AGENT_CALL_FAILED[{self.model_name}/{self.reasoning_effort}]: {last_error}")
