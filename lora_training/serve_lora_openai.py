from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# PyTorch 2.0+ memory optimization that reduces fragmentation on CUDA. Only set
# it when the user has not provided their own value (setdefault), so custom or
# older setups can override or disable it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_NAME = "qwen3-4b-novel-lora"
ADAPTER_DIR = Path("lora_training/lora_output/qwen3_4b_novel_lora")

# Largest request body we will read, to avoid memory exhaustion from a malicious
# or buggy client declaring a huge Content-Length. 10MB is ample for JSON chat.
MAX_REQUEST_BODY = 10 * 1024 * 1024


def _int_env(name: str, default: int) -> int:
    """Read an int env var, falling back to a default if it is unset or invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[lora-server] [WARN] Invalid {name}={raw!r}, using default {default}", flush=True)
        return default


MAX_INPUT_TOKENS = _int_env("LORA_MAX_INPUT_TOKENS", 4096)
MAX_NEW_TOKENS = _int_env("LORA_MAX_NEW_TOKENS", 12000)
_MODEL_LOCK = threading.Lock()
_TOKENIZER = None
_MODEL = None


def validate_adapter_dir(adapter_dir: Path) -> None:
    """Raise a clear error if the LoRA adapter directory is missing required files."""
    if not adapter_dir.exists():
        raise RuntimeError(f"LoRA adapter directory not found: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(f"LoRA adapter config (adapter_config.json) not found in {adapter_dir}")
    weights = (adapter_dir / "adapter_model.safetensors").exists() or (
        adapter_dir / "adapter_model.bin"
    ).exists()
    if not weights:
        raise RuntimeError(
            f"LoRA adapter weights (adapter_model.safetensors/.bin) not found in {adapter_dir}"
        )


def load_model() -> tuple[Any, Any]:
    global _TOKENIZER, _MODEL
    with _MODEL_LOCK:
        if _TOKENIZER is not None and _MODEL is not None:
            return _TOKENIZER, _MODEL

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[lora-server] Loading base model: {MODEL_ID}", flush=True)
        print(f"[lora-server] Loading LoRA adapter: {ADAPTER_DIR}", flush=True)

        # Fail fast with a clear message if the adapter is missing, instead of a
        # cryptic file-not-found deep inside peft on the first request.
        validate_adapter_dir(ADAPTER_DIR)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
        # is_trainable=False keeps the adapter in inference mode (adapters must
        # remain unmerged on disk for this load path to work correctly).
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
        model.eval()

        _TOKENIZER = tokenizer
        _MODEL = model
        print("[lora-server] Model ready.", flush=True)
        return _TOKENIZER, _MODEL


def make_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:  # noqa: BLE001
        # Surface why the chat template failed (e.g. transformers version skew)
        # before falling back to manual formatting.
        print(f"[lora-server] [WARN] apply_chat_template failed: {exc}; using manual formatting", flush=True)
        lines = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)


def clean_generation(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def generate_reply(payload: dict[str, Any]) -> str:
    import torch

    tokenizer, model = load_model()
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("Invalid request: 'messages' must be a non-empty list.")
    prompt = make_prompt(tokenizer, messages)

    # Initialize before the try so the finally cleanup never references unbound
    # names if tokenization or generation raises early.
    inputs = None
    output_ids = None

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Validate/convert client-supplied generation params defensively: a malformed
    # value falls back to the server default rather than crashing the request.
    raw_max_tokens = payload.get("max_tokens")
    try:
        max_new_tokens = int(raw_max_tokens or MAX_NEW_TOKENS)
    except (ValueError, TypeError):
        max_new_tokens = MAX_NEW_TOKENS
    if isinstance(raw_max_tokens, (int, float)) and raw_max_tokens < 0:
        print(f"[lora-server] [WARN] Negative max_tokens {raw_max_tokens}, clamped to 1", flush=True)
    max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS))

    try:
        temperature = float(payload.get("temperature") or 0.8)
    except (ValueError, TypeError):
        temperature = 0.8
    try:
        top_p = float(payload.get("top_p") or 0.9)
    except (ValueError, TypeError):
        top_p = 0.9
    # repetition_penalty is configurable via frequency_penalty for OpenAI-API
    # compatibility; it must be positive for model.generate().
    try:
        repetition_penalty = float(payload.get("frequency_penalty") or 1.08)
    except (ValueError, TypeError):
        repetition_penalty = 1.08
    if repetition_penalty <= 0:
        repetition_penalty = 1.08
    do_sample = temperature > 0.01

    try:
        clear_cuda_cache()
        model.eval()  # ensure inference mode even if the model was reloaded
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return clean_generation(tokenizer.decode(generated, skip_special_tokens=True))
    finally:
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        clear_cuda_cache()


class LoraHandler(BaseHTTPRequestHandler):
    server_version = "QwenLoraOpenAI/0.1"

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        # Content-Length may be missing or non-numeric from buggy/malicious
        # clients; default to 0 (empty body) rather than crashing.
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        # Defense in depth: clamp the read so a huge declared length cannot
        # exhaust memory. do_POST already rejects oversized requests with 413.
        length = min(max(length, 0), MAX_REQUEST_BODY)
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_NAME,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        if self.path.rstrip("/") in {"", "/", "/health"}:
            self._send_json(
                200,
                {
                    "ok": True,
                    "model": MODEL_NAME,
                    "max_input_tokens": MAX_INPUT_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            )
            return
        self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": {"message": "Not found"}})
            return

        # Reject oversized requests up front (413) before reading the body, so a
        # huge declared Content-Length cannot be used to exhaust memory.
        try:
            declared_length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            declared_length = 0
        if declared_length > MAX_REQUEST_BODY:
            self.log_message(
                "Rejected oversized request: Content-Length=%s exceeds limit %s",
                declared_length,
                MAX_REQUEST_BODY,
            )
            self._send_json(413, {"error": {"message": "Request body too large."}})
            return

        try:
            payload = self._read_json()
            if payload.get("stream"):
                self.log_message(
                    "Rejected streaming request: model=%s, Content-Length=%s",
                    payload.get("model"),
                    self.headers.get("Content-Length"),
                )
                self._send_json(400, {"error": {"message": "Streaming is not supported by the local LoRA server."}})
                return
            content = generate_reply(payload)
            now_ts = int(time.time())
            response = {
                "id": f"chatcmpl-lora-{now_ts}",
                "object": "chat.completion",
                "created": now_ts,
                "model": payload.get("model") or MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            self._send_json(200, response)
        except RuntimeError as exc:
            clear_cuda_cache()
            message = str(exc)
            if "out of memory" in message.lower():
                message = (
                    "CUDA out of memory. The local LoRA request was too large for 16GB VRAM. "
                    "Try smaller chunks or lower max_tokens."
                )
            self._send_json(500, {"error": {"message": message}})
        except Exception as exc:  # noqa: BLE001
            clear_cuda_cache()
            self._send_json(500, {"error": {"message": str(exc)}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[lora-server] {self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    global ADAPTER_DIR, MODEL_NAME

    parser = argparse.ArgumentParser(description="Serve a Qwen LoRA adapter through a small OpenAI-compatible API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    ADAPTER_DIR = args.adapter.resolve()
    MODEL_NAME = args.model_name

    # Fail fast at startup (before binding the port) if the adapter is missing,
    # rather than only surfacing the error on the first generation request.
    try:
        validate_adapter_dir(ADAPTER_DIR)
    except RuntimeError as exc:
        print(f"[lora-server] [WARN] {exc}", flush=True)
        print("[lora-server] [WARN] The server will start, but generation requests will fail until the adapter exists.", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), LoraHandler)
    print(f"[lora-server] Listening on http://{args.host}:{args.port}/v1", flush=True)
    print("[lora-server] The model loads lazily on first generation request.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
