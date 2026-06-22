from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_NAME = "qwen3-4b-novel-lora"
ADAPTER_DIR = Path("lora_training/lora_output/qwen3_4b_novel_lora")
MAX_INPUT_TOKENS = int(os.getenv("LORA_MAX_INPUT_TOKENS", "4096"))
MAX_NEW_TOKENS = int(os.getenv("LORA_MAX_NEW_TOKENS", "12000"))
_MODEL_LOCK = threading.Lock()
_TOKENIZER = None
_MODEL = None


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
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
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
    except Exception:
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
    prompt = make_prompt(tokenizer, messages)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    max_new_tokens = int(payload.get("max_tokens") or MAX_NEW_TOKENS)
    max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS))
    temperature = float(payload.get("temperature") or 0.8)
    top_p = float(payload.get("top_p") or 0.9)
    do_sample = temperature > 0.01

    try:
        clear_cuda_cache()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return clean_generation(tokenizer.decode(generated, skip_special_tokens=True))
    finally:
        try:
            del inputs
            del output_ids
        except Exception:
            pass
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
        length = int(self.headers.get("Content-Length", "0") or "0")
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

        try:
            payload = self._read_json()
            if payload.get("stream"):
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
    server = ThreadingHTTPServer((args.host, args.port), LoraHandler)
    print(f"[lora-server] Listening on http://{args.host}:{args.port}/v1", flush=True)
    print("[lora-server] The model loads lazily on first generation request.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
