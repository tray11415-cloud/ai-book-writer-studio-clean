"""Local compatibility proxy for SSE-style OpenAI-like upstreams."""
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable

import requests
from dotenv import load_dotenv
from env_utils import get_dotenv_path


load_dotenv(get_dotenv_path())

HOST = os.getenv("COMPAT_PROXY_HOST", "127.0.0.1")
PORT = int(os.getenv("COMPAT_PROXY_PORT", "8000"))
UPSTREAM_BASE_URL = os.getenv(
    "UPSTREAM_OPENAI_BASE_URL",
    "https://www.gpt4novel.com/api/xiaoshuoai/ext/v1",
).rstrip("/")
UPSTREAM_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("LLM_API_KEY")
    or ""
)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "nalang-turbo-0826")
DZMM_MODELS = [
    "x-apex-surge-0505-16k",
    "x-apex-surge-0505",
    "x-apex-sigma-0621-16k",
    "x-apex-sigma-0621",
    "nalang-turbo-0826",
    "nalang-medium-0826",
    "nalang-max-0826",
    "nalang-xl-0826",
    "nalang-max-0826-16k",
    "nalang-xl-0826-16k",
]


def _extract_sse_text(chunks: Iterable[str]) -> str:
    """Aggregate SSE delta content into a single assistant message."""
    parts: list[str] = []
    for raw_line in chunks:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        delta = (data.get("choices") or [{}])[0].get("delta") or {}
        content = delta.get("content")
        if content:
            parts.append(content)

    return "".join(parts)


def _extract_message_text(raw: str) -> str:
    """Pull the assistant text out of either a JSON completion or an SSE stream.

    DZMM returns a standard non-streaming JSON body for stream=False requests, so
    try that first; only fall back to SSE delta aggregation if the body is not a
    plain completion JSON.
    """
    raw = raw.strip()
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if content is None:
            content = (choice.get("delta") or {}).get("content")
        if isinstance(content, str) and content:
            return content
    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
        pass
    return _extract_sse_text(raw.splitlines())


class CompatProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _get_upstream_api_key(self) -> str:
        if UPSTREAM_API_KEY:
            return UPSTREAM_API_KEY
        auth_header = self.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header.split(" ", 1)[1].strip()
        return ""

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
                            "object": "model",
                            "created": int(datetime.now(tz=timezone.utc).timestamp()),
                            "owned_by": "compat-proxy",
                        }
                        for model in DZMM_MODELS
                    ],
                }
            )
            return

        if self.path in {"/health", "/v1/health"}:
            self._send_json({"status": "ok"})
            return

        self._send_json({"error": f"Unsupported path: {self.path}"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json({"error": f"Unsupported path: {self.path}"}, status=404)
            return

        try:
            request_body = self._read_json_body()
            request_body["stream"] = False

            upstream_response = requests.post(
                f"{UPSTREAM_BASE_URL}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._get_upstream_api_key()}",
                },
                json=request_body,
                timeout=180,
            )
            upstream_response.raise_for_status()

            upstream_text = upstream_response.content.decode("utf-8", errors="replace")
            text = _extract_message_text(upstream_text)
            now_ts = int(datetime.now(tz=timezone.utc).timestamp())
            response_payload = {
                "id": f"chatcmpl-proxy-{now_ts}",
                "object": "chat.completion",
                "created": now_ts,
                "model": request_body.get("model", DEFAULT_MODEL),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            self._send_json(response_payload)
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            self._send_json({"error": detail}, status=502)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format: str, *args) -> None:
        """Keep the proxy quiet during normal runs."""
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), CompatProxyHandler)
    print(f"Compat proxy listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
