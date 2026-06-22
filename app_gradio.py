"""Gradio story studio inspired by unlimited story writer workflows."""
from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from chapter_craft_skill import DEFAULT_GOAL as CHAPTER_CRAFT_DEFAULT_GOAL
from chapter_craft_skill import analyze_chapter_craft
from compat_proxy import HOST as PROXY_HOST
from compat_proxy import PORT as PROXY_PORT
from compat_proxy import UPSTREAM_BASE_URL as PROXY_UPSTREAM_BASE_URL
from compat_proxy import main as run_compat_proxy
from config import get_config
from env_utils import get_dotenv_path
from lora_runtime import LORA_BASE_URL, LORA_MODEL_NAME, ensure_lora_server_running, is_lora_base_url
from plot_ideation_skill import DEFAULT_PLOT_GOAL
from plot_ideation_skill import generate_plot_ideation
from scene_technique_skill import DEFAULT_LIBRARY_GOAL, DEFAULT_TECHNIQUE_GOAL
from scene_technique_skill import aggregate_scene_techniques
from scene_technique_skill import distill_novel_to_technique_finder
from report_technique_distiller import DEFAULT_REPORT_DISTILL_GOAL, TECHNIQUE_LOAD_MODES
from report_technique_distiller import distill_full_report_to_agent_library
from report_technique_distiller import load_distilled_library_to_agent_fields
from report_technique_distiller import load_latest_distilled_library_to_agent_fields
from skill_technique_review import REVIEW_CHOICES, review_skill_and_technique
from technique_library_builder import ALL_CATEGORY_CHOICES, BOOK_LIBRARY_LOAD_MODES
from technique_library_builder import DEFAULT_BOOK_LIBRARY_GOAL
from technique_library_builder import add_references_to_shelf
from technique_library_builder import build_integrated_technique_book_library
from technique_library_builder import build_integrated_technique_book_library_from_shelf
from technique_library_builder import clear_reference_shelf
from technique_library_builder import load_saved_reference_shelf
from technique_library_builder import load_latest_technique_book_to_agent_fields
from technique_library_builder import load_technique_book_to_agent_fields
from technique_library_builder import remove_references_from_shelf
from technique_library_builder import search_integrated_technique_book_library
from technique_library_builder import suggest_technique_templates
from technique_library_builder import render_agent_reference


load_dotenv(get_dotenv_path())

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

DEFAULT_CFG = get_config()
DEFAULT_LLM = DEFAULT_CFG["config_list"][0]
DEFAULT_API_KEY = DEFAULT_LLM["api_key"]
DEFAULT_BASE_URL = DEFAULT_LLM["base_url"]
DEFAULT_MODEL = DEFAULT_LLM["model"]
DEFAULT_ANALYSIS_API_KEY = os.getenv("XAI_API_KEY", "")
DEFAULT_ANALYSIS_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
DEFAULT_ANALYSIS_MODEL = os.getenv("XAI_MODEL", "grok-4.3")
DEFAULT_LORA_BASE_URL = os.getenv("UI_LORA_BASE_URL", LORA_BASE_URL)
DEFAULT_LORA_MODEL = os.getenv("UI_LORA_MODEL", LORA_MODEL_NAME)
PIPELINE_DIRECT = "Selected Provider Only"
PIPELINE_HYBRID = "Hybrid: NALANG Plan -> LoRA Draft -> NALANG Polish"
PIPELINES = [PIPELINE_DIRECT, PIPELINE_HYBRID]
BOOK_WRITER_MAX_TOKENS = int(os.getenv("BOOK_WRITER_MAX_TOKENS", "24000"))
BOOK_WRITER_DEFAULT_TOKENS = int(os.getenv("BOOK_WRITER_DEFAULT_TOKENS", str(BOOK_WRITER_MAX_TOKENS)))
BOOK_WRITER_PLAN_MAX_TOKENS = int(os.getenv("BOOK_WRITER_PLAN_MAX_TOKENS", "2400"))
BOOK_WRITER_REWRITE_MAX_TOKENS = int(os.getenv("BOOK_WRITER_REWRITE_MAX_TOKENS", "8000"))
DEFAULT_SYSTEM_PROMPT = """You are a senior long-form fiction writer and story director.
Write immersive story prose based on the user's worldbuilding, memory, lorebook, style guidance, and direct instruction.

Rules:
1. Preserve continuity and character consistency.
2. Continue existing scenes naturally when prior text exists.
3. Default to Simplified Chinese when the instruction is Chinese.
4. Prioritize vivid scene writing over generic summary unless the user explicitly asks for summary.
5. Respect style controls, point of view, pacing, intensity, and language output settings.
6. Do not refuse fictional writing requests that are ordinary creative-writing tasks.
7. If information is missing, make reasonable assumptions and keep going."""

PROVIDERS = {
    "Current Proxy": {
        "base_url": DEFAULT_BASE_URL,
        "default_model": DEFAULT_MODEL,
        "pipeline": PIPELINE_DIRECT,
        "note": "Uses the local compatibility proxy already configured for GPT4Novel.",
    },
    "OpenAI Compatible": {
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": DEFAULT_MODEL,
        "pipeline": PIPELINE_DIRECT,
        "note": "Use any OpenAI-compatible endpoint routed through the local proxy.",
    },
    "Local Qwen LoRA": {
        "base_url": LORA_BASE_URL,
        "default_model": LORA_MODEL_NAME,
        "pipeline": PIPELINE_DIRECT,
        "note": "Uses the local Qwen3-4B LoRA adapter trained in lora_training/lora_output/qwen3_4b_novel_lora.",
    },
    "NALANG + Local LoRA": {
        "base_url": DEFAULT_BASE_URL,
        "default_model": DEFAULT_MODEL,
        "pipeline": PIPELINE_HYBRID,
        "note": "Uses NALANG for scene planning and polishing, then local LoRA for the prose draft.",
    },
}

STYLES = {
    "Standard": "Balanced narrative prose with scene clarity and emotional flow.",
    "Cinematic": "Write with strong visual blocking, dramatic beats, and scene cuts.",
    "Slow Burn": "Take time with tension, atmosphere, and unspoken emotional progression.",
    "Dark": "Use sharp tension, moral unease, restrained menace, and shadowed imagery.",
    "Romance": "Focus on chemistry, longing, vulnerability, and emotionally charged detail.",
    "Suspense": "Keep forward pressure, hidden motives, clues, and rising unease.",
    "Custom": "",
}

DIRECTOR_PRESETS = [
    "Open with motion, not explanation.",
    "Let the scene breathe before the next turn.",
    "Add one memorable sensory detail per beat.",
    "Escalate conflict without rushing resolution.",
    "Show hidden motives through dialogue subtext.",
    "End on a hook that demands continuation.",
]

def trim_text(text, max_chars):
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def ensure_proxy_running() -> None:
    import threading

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((PROXY_HOST, PROXY_PORT)) == 0:
            return

    thread = threading.Thread(target=run_compat_proxy, daemon=True)
    thread.start()


def can_bind_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def get_gradio_port(host: str, default_port: int = 7860) -> int:
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    if configured_port:
        try:
            return int(configured_port)
        except ValueError as exc:
            raise ValueError("GRADIO_SERVER_PORT must be a number.") from exc

    for port in range(default_port, 7900):
        if can_bind_port(host, port):
            return port

    raise OSError("No empty Gradio port found in range 7860-7899.")


def get_client(api_key: str, base_url: str) -> OpenAI:
    if is_lora_base_url(base_url):
        ensure_lora_server_running()
    elif is_proxy_upstream(base_url):
        # gpt4novel / NALANG (DZMM) streams SSE and is not OpenAI-compatible for
        # non-streaming calls: the OpenAI SDK silently returns the raw text body
        # instead of a ChatCompletion, which later blows up as
        # "'str' object has no attribute 'choices'". The local compat proxy
        # aggregates that SSE into a standard chat.completion JSON body, so route
        # the call through it transparently even when the user typed the raw URL.
        ensure_proxy_running()
        base_url = f"http://{PROXY_HOST}:{PROXY_PORT}/v1"
    elif is_local_compat_proxy(base_url):
        ensure_proxy_running()
    return OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=900)


def is_local_compat_proxy(base_url: str) -> bool:
    parsed = urlparse((base_url or "").strip())
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"} and parsed.port == PROXY_PORT


def is_proxy_upstream(base_url: str) -> bool:
    """True when base_url points directly at the SSE upstream the compat proxy fronts.

    Hitting that endpoint directly with the OpenAI SDK returns a raw text body
    (an SSE stream) rather than JSON, so such calls must go through the proxy.
    """
    host = (urlparse((base_url or "").strip()).hostname or "").lower()
    upstream_host = (urlparse((PROXY_UPSTREAM_BASE_URL or "").strip()).hostname or "").lower()
    return bool(host) and bool(upstream_host) and host == upstream_host


def get_book_output_dir() -> Path:
    output_dir = Path.cwd() / "book_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_generation_log_dir() -> Path:
    log_dir = get_book_output_dir() / "generation_logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_model_config_path() -> Path:
    return get_book_output_dir() / "model_config.json"


def model_config_payload(
    writing_api_key,
    writing_base_url,
    writing_model,
    pipeline_mode,
    analysis_api_key,
    analysis_base_url,
    analysis_model,
    lora_base_url,
    lora_model,
) -> dict[str, Any]:
    return {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "writing": {
            "api_key": writing_api_key or "",
            "base_url": writing_base_url or "",
            "model": writing_model or "",
            "pipeline": pipeline_mode or PIPELINE_HYBRID,
        },
        "analysis": {
            "api_key": analysis_api_key or "",
            "base_url": analysis_base_url or "",
            "model": analysis_model or "",
        },
        "lora": {
            "base_url": lora_base_url or DEFAULT_LORA_BASE_URL,
            "model": lora_model or DEFAULT_LORA_MODEL,
        },
    }


def save_model_config(
    writing_api_key,
    writing_base_url,
    writing_model,
    pipeline_mode,
    analysis_api_key,
    analysis_base_url,
    analysis_model,
    lora_base_url,
    lora_model,
) -> str:
    payload = model_config_payload(
        writing_api_key,
        writing_base_url,
        writing_model,
        pipeline_mode,
        analysis_api_key,
        analysis_base_url,
        analysis_model,
        lora_base_url,
        lora_model,
    )
    path = get_model_config_path()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"[OK] Model config saved: {path}"


def load_model_config():
    path = get_model_config_path()
    if not path.exists():
        return (
            DEFAULT_API_KEY,
            DEFAULT_BASE_URL,
            DEFAULT_MODEL,
            PIPELINE_HYBRID,
            DEFAULT_ANALYSIS_API_KEY,
            DEFAULT_ANALYSIS_BASE_URL,
            DEFAULT_ANALYSIS_MODEL,
            DEFAULT_LORA_BASE_URL,
            DEFAULT_LORA_MODEL,
            f"[INFO] No saved model config found. Using defaults: {path}",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    writing = payload.get("writing", {})
    analysis = payload.get("analysis", {})
    lora = payload.get("lora", {})
    return (
        writing.get("api_key", DEFAULT_API_KEY),
        writing.get("base_url", DEFAULT_BASE_URL),
        writing.get("model", DEFAULT_MODEL),
        writing.get("pipeline", PIPELINE_HYBRID),
        analysis.get("api_key", DEFAULT_ANALYSIS_API_KEY),
        analysis.get("base_url", DEFAULT_ANALYSIS_BASE_URL),
        analysis.get("model", DEFAULT_ANALYSIS_MODEL),
        lora.get("base_url", DEFAULT_LORA_BASE_URL),
        lora.get("model", DEFAULT_LORA_MODEL),
        f"[OK] Model config loaded: {path}",
    )


def serialize_for_log(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_log(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return serialize_for_log(model_dump())
    dict_fn = getattr(value, "dict", None)
    if callable(dict_fn):
        return serialize_for_log(dict_fn())
    return str(value)


def autosave_generation(
    *,
    updated_story: str,
    new_part: str,
    thought: str,
    request_payload: dict[str, Any],
    response_payload: Any,
) -> None:
    timestamp = datetime.now()
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    output_dir = get_book_output_dir()
    log_dir = get_generation_log_dir()

    latest_story_path = output_dir / "full_story_latest.txt"
    snapshot_story_path = output_dir / f"full_story_{stamp}.txt"
    latest_part_path = output_dir / "latest_continuation.txt"
    snapshot_part_path = output_dir / f"latest_continuation_{stamp}.txt"
    request_path = log_dir / f"request_{stamp}.json"
    response_path = log_dir / f"response_{stamp}.json"
    thought_path = log_dir / f"thought_{stamp}.txt"
    jsonl_path = log_dir / "request_response_log.jsonl"

    latest_story_path.write_text(updated_story, encoding="utf-8")
    snapshot_story_path.write_text(updated_story, encoding="utf-8")
    latest_part_path.write_text(new_part, encoding="utf-8")
    snapshot_part_path.write_text(new_part, encoding="utf-8")
    thought_path.write_text(thought, encoding="utf-8")

    serialized_request = serialize_for_log(request_payload)
    serialized_response = serialize_for_log(response_payload)

    request_path.write_text(json.dumps(serialized_request, ensure_ascii=False, indent=2), encoding="utf-8")
    response_path.write_text(json.dumps(serialized_response, ensure_ascii=False, indent=2), encoding="utf-8")

    log_entry = {
        "saved_at": timestamp.isoformat(timespec="seconds"),
        "full_story_path": str(snapshot_story_path),
        "latest_full_story_path": str(latest_story_path),
        "continuation_path": str(snapshot_part_path),
        "latest_continuation_path": str(latest_part_path),
        "thought_path": str(thought_path),
        "request_path": str(request_path),
        "response_path": str(response_path),
        "request": serialized_request,
        "response": serialized_response,
    }
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def fetch_all_models(api_key, base_url):
    try:
        client = get_client(api_key, base_url)
        models = sorted({item.id for item in client.models.list().data if getattr(item, "id", None)})
        if not models:
            models = [DEFAULT_MODEL]
    except Exception:
        models = [DEFAULT_MODEL]
    return gr.update(choices=models, value=models[0] if models else DEFAULT_MODEL)


def test_api_connection(api_key, base_url, model_name):
    if not model_name:
        return "[ERROR] Model name is empty."
    try:
        client = get_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Reply with: connection ok"}],
            max_tokens=30,
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()
        return f"[OK] Connected.\nModel: {model_name}\nReply: {content}"
    except Exception as exc:
        return f"[ERROR] {exc}"


def add_empty_row(data, width):
    rows = list(data or [])
    rows.append([""] * width)
    return rows


def normalize_rows(data, width):
    rows = []
    for row in data or []:
        normalized = [str(cell or "").strip() for cell in list(row)[:width]]
        normalized += [""] * (width - len(normalized))
        if any(normalized):
            rows.append(normalized)
    return rows


def rows_to_block(title, rows, labels):
    lines = [title]
    for row in rows:
        paired = [f"{labels[idx]}: {value}" for idx, value in enumerate(row) if value]
        if paired:
            lines.append("- " + " | ".join(paired))
    return "\n".join(lines) if len(lines) > 1 else ""


def infer_output_language(lang):
    mapping = {
        "简体中文": "Use Simplified Chinese.",
        "繁体中文": "Use Traditional Chinese.",
        "English": "Use English.",
        "日本語": "Use Japanese.",
    }
    return mapping.get(lang, "Use Simplified Chinese.")


def infer_style_instruction(style_name, custom_style):
    if style_name == "Custom" and custom_style.strip():
        return custom_style.strip()
    return STYLES.get(style_name, STYLES["Standard"])


def build_director_note(custom_director):
    if custom_director.strip():
        return custom_director.strip()
    return random.choice(DIRECTOR_PRESETS)


def build_reference_context(background, roles, lore, memory, style_dna, style_samples, chronicle, technique_library):
    sections = []

    if background.strip():
        sections.append("World / Background:\n" + trim_text(background, 500))
    if memory.strip():
        sections.append("Story Memory:\n" + trim_text(memory, 400))
    if chronicle.strip():
        sections.append("Story Chronicle:\n" + trim_text(chronicle, 500))

    roles_rows = normalize_rows(roles, 3)[:5]
    if roles_rows:
        sections.append(rows_to_block("Characters", roles_rows, ["Name", "Role", "Traits"]))

    lore_rows = normalize_rows(lore, 2)[:6]
    if lore_rows:
        sections.append(rows_to_block("Lorebook", lore_rows, ["Keyword", "Content"]))

    if style_dna.strip():
        sections.append("Style DNA:\n" + trim_text(style_dna, 450))
    if style_samples.strip():
        sections.append("Style Samples:\n" + trim_text(style_samples, 450))
    if technique_library.strip():
        sections.append(
            "Technique Library:\n"
            + trim_text(
                technique_library,
                3500,
            )
        )

    return "\n\n".join(section for section in sections if section.strip())


def build_story_prompt(
    background,
    roles,
    lore,
    memory,
    current_story,
    instruction,
    style_name,
    custom_style,
    pov,
    ling_texture,
    pacing,
    intensity,
    output_lang,
    para_density,
    dialogue_ratio,
    focus_words,
    avoid_words,
    custom_director,
    style_dna,
    style_samples,
    chronicle,
    technique_library,
    sensory_values,
    context_length,
):
    director_note = build_director_note(custom_director)
    style_instruction = infer_style_instruction(style_name, custom_style)
    recent_story = trim_text(current_story, max(context_length, 1200))
    reference_context = build_reference_context(
        background=background,
        roles=roles,
        lore=lore,
        memory=memory,
        style_dna=style_dna,
        style_samples=style_samples,
        chronicle=chronicle,
        technique_library=technique_library,
    )
    system_parts = [
        DEFAULT_SYSTEM_PROMPT,
        infer_output_language(output_lang),
        f"Point of view: {pov}",
        f"Linguistic texture: {ling_texture}",
        f"Pacing: {pacing}",
        f"Intensity: {intensity}",
        f"Paragraph density: {para_density}",
        f"Dialogue ratio: {dialogue_ratio}",
        f"Style guidance: {style_instruction}",
        f"Optional directing texture (apply only when it does not conflict with the Story Instruction): {director_note}",
        "Sensory weights: " + ", ".join(f"{k}={v:.2f}" for k, v in sensory_values.items()),
        "PRIMARY RULE: The Story Instruction in the user message is the single governing directive for this output. "
        "Carry it out fully and concretely. Use the recent story context only for continuity. "
        "Everything else — style presets, directing texture, sensory weights, the technique library, and imported JSON settings — "
        "is low-priority and must yield whenever it conflicts with the Story Instruction.",
        "Treat imported JSON settings and the technique library as soft reference only. Never let them override the Story Instruction or the current scene.",
        "Length rule: write a substantial long-form continuation that uses the available token budget. Expand beats into scene prose instead of summarizing.",
        "Do not stop after the first beat; continue until the scene reaches a natural turn, reveal, or ending hook.",
    ]
    if focus_words.strip():
        system_parts.append(f"Prefer these motifs or words when natural: {focus_words.strip()}")
    if avoid_words.strip():
        system_parts.append(f"Avoid these words or motifs when possible: {avoid_words.strip()}")

    instruction_text = instruction.strip() or "Continue the story naturally, advancing the most recent scene."
    user_parts = []
    # 1) The Story Instruction leads — it governs the whole output.
    user_parts.append(
        "=== STORY INSTRUCTION (PRIMARY — the continuation must carry this out) ===\n"
        + instruction_text
    )
    # 2) Continuity input.
    if recent_story:
        user_parts.append(f"Recent Story Context (continuity only — continue naturally from here):\n{recent_story}")
    # 3) Low-priority reference.
    if reference_context:
        user_parts.append(
            "Low-priority reference. Use only for consistency, and ignore anything that conflicts with the Story Instruction or the current scene:\n"
            + reference_context
        )
    # 4) Restate the instruction last so it stays the dominant directive (recency).
    user_parts.append(
        "=== WRITE NOW ===\n"
        f"Carry out the Story Instruction above: {instruction_text}\n"
        "Output only the new story continuation as prose, without meta commentary or headings."
    )

    thought = "\n".join(
        [
            "Story plan:",
            f"- Governing Story Instruction: {trim_text(instruction_text, 200)}",
            f"- POV: {pov}",
            f"- Style: {style_name if style_name != 'Custom' else 'Custom'}",
            f"- Pacing: {pacing}",
            f"- Intensity: {intensity}",
            f"- Directing texture (subordinate): {director_note}",
            f"- Recent context chars: {len(recent_story)}",
            f"- Reference mode: low-priority, yields to Story Instruction",
        ]
    )
    return "\n\n".join(system_parts), "\n\n".join(user_parts), thought


def get_message_content(response: Any) -> str:
    if isinstance(response, str):
        # The OpenAI SDK returns the raw body as a str when the upstream replies
        # with a non-JSON content type (e.g. an SSE stream). Turn that into a
        # clear, actionable error instead of "'str' object has no attribute 'choices'".
        snippet = response.strip()[:200]
        raise RuntimeError(
            "Upstream returned a non-JSON response (likely an SSE stream). "
            f"Set the Base URL to the local compat proxy (http://{PROXY_HOST}:{PROXY_PORT}/v1) "
            f"instead of the provider's raw endpoint. Response began: {snippet!r}"
        )
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(f"Upstream response contained no choices: {response!r}"[:300])
    return (choices[0].message.content or "").strip()


def generate_direct_response(
    *,
    client: OpenAI,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    freq_penalty: float | None = None,
    pres_penalty: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if freq_penalty is not None:
        kwargs["frequency_penalty"] = freq_penalty
    if pres_penalty is not None:
        kwargs["presence_penalty"] = pres_penalty
    return client.chat.completions.create(**kwargs)


def generate_hybrid_continuation(
    *,
    remote_client: OpenAI,
    remote_model: str,
    lora_base_url: str,
    lora_model: str,
    system_prompt: str,
    user_prompt: str,
    instruction: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    freq_penalty: float,
    pres_penalty: float,
) -> tuple[str, dict[str, Any], str]:
    planner_messages = [
        {
            "role": "system",
            "content": (
                "You are a fiction story director. Create a detailed long-form scene plan only. "
                "The Story Instruction inside the writing request is the primary objective — "
                "build every beat to carry it out. Do not write final prose. Preserve continuity, "
                "clarify beats, emotional turn, conflict, and ending hook. Include enough beats for a substantial continuation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"System guidance:\n{system_prompt}\n\n"
                f"Writing request:\n{user_prompt}\n\n"
                "Return only a scene plan with expanded beats and continuity cautions."
            ),
        },
    ]
    plan_response = generate_direct_response(
        client=remote_client,
        model_name=remote_model,
        messages=planner_messages,
        temperature=min(float(temperature), 0.75),
        top_p=top_p,
        max_tokens=min(max(max_tokens // 3, 800), BOOK_WRITER_PLAN_MAX_TOKENS),
    )
    plan = get_message_content(plan_response)

    lora_client = get_client("not-needed", lora_base_url or DEFAULT_LORA_BASE_URL)
    lora_messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\nYou are now the local LoRA prose writer. Follow the plan, but write natural story prose. "
                "Write a substantial long-form scene, expand each beat, and use the available length budget. "
                "Output only the new scene text."
            ),
        },
        {
            "role": "user",
            "content": (
                user_prompt
                + "\n\nNALANG scene plan:\n"
                + plan
                + "\n\nWrite the scene in the trained LoRA style. Do not explain the plan. "
                "Do not compress this into a short excerpt."
            ),
        },
    ]
    draft_response = generate_direct_response(
        client=lora_client,
        model_name=lora_model or DEFAULT_LORA_MODEL,
        messages=lora_messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        freq_penalty=freq_penalty,
        pres_penalty=pres_penalty,
    )
    draft = get_message_content(draft_response)

    editor_messages = [
        {
            "role": "system",
            "content": (
                "You are a continuity editor and prose polisher. Improve coherence, transitions, rhythm, "
                "and instruction-following. Keep the local LoRA draft's core style and events. "
                "Do not shorten the draft; preserve its length and expand thin transitions when useful. "
                "Do not add analysis or commentary; output only polished story prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Primary Story Instruction (the final continuation must carry this out):\n{instruction.strip() or 'Continue the story naturally.'}\n\n"
                f"Scene plan:\n{plan}\n\n"
                f"Local LoRA draft:\n{draft}\n\n"
                "Polish this into the final continuation without condensing it. "
                "Make sure the result fully and concretely fulfills the Primary Story Instruction above; "
                "adjust anything in the draft that drifts from it."
            ),
        },
    ]
    polish_response = generate_direct_response(
        client=remote_client,
        model_name=remote_model,
        messages=editor_messages,
        temperature=min(float(temperature), 0.8),
        top_p=top_p,
        max_tokens=max_tokens,
        freq_penalty=freq_penalty,
        pres_penalty=pres_penalty,
    )
    final_text = get_message_content(polish_response) or draft
    response_payload = {
        "pipeline": PIPELINE_HYBRID,
        "remote_model": remote_model,
        "lora_base_url": lora_base_url or DEFAULT_LORA_BASE_URL,
        "lora_model": lora_model or DEFAULT_LORA_MODEL,
        "plan": plan,
        "draft": draft,
        "plan_response": plan_response,
        "draft_response": draft_response,
        "polish_response": polish_response,
    }
    thought_extra = "\n\nHybrid pipeline:\n- NALANG planned the scene.\n- Local LoRA wrote the prose draft.\n- NALANG polished continuity and flow.\n\nScene plan:\n" + plan
    return final_text, response_payload, thought_extra


def generate_continuation(
    background,
    roles,
    lore,
    current_story,
    instruction,
    style_name,
    custom_style,
    temperature,
    freq_penalty,
    pres_penalty,
    top_p,
    max_tokens,
    context_length,
    pov,
    system_prompt_override,
    v_weight,
    a_weight,
    o_weight,
    t_weight,
    g_weight,
    ling_texture,
    pacing,
    intensity,
    focus_words,
    avoid_words,
    custom_director,
    output_lang,
    para_density,
    dialogue_ratio,
    memory,
    style_dna,
    style_samples,
    chronicle,
    technique_library,
    pipeline_mode,
    api_key,
    base_url,
    model_name,
    lora_base_url,
    lora_model,
    history_state=None,
):
    ensure_proxy_running()
    client = get_client(api_key, base_url)
    system_prompt, user_prompt, thought = build_story_prompt(
        background=background,
        roles=roles,
        lore=lore,
        memory=memory,
        current_story=current_story,
        instruction=instruction,
        style_name=style_name,
        custom_style=custom_style,
        pov=pov,
        ling_texture=ling_texture,
        pacing=pacing,
        intensity=intensity,
        output_lang=output_lang,
        para_density=para_density,
        dialogue_ratio=dialogue_ratio,
        focus_words=focus_words,
        avoid_words=avoid_words,
        custom_director=custom_director,
        style_dna=style_dna,
        style_samples=style_samples,
        chronicle=chronicle,
        technique_library=technique_library,
        sensory_values={
            "visual": v_weight,
            "auditory": a_weight,
            "olfactory": o_weight,
            "tactile": t_weight,
            "gustatory": g_weight,
        },
        context_length=int(context_length),
    )
    if system_prompt_override.strip():
        system_prompt = system_prompt_override.strip() + "\n\n" + system_prompt

    request_payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "base_url": base_url,
        "generation_params": {
            "pipeline_mode": pipeline_mode,
            "lora_base_url": lora_base_url,
            "lora_model": lora_model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "frequency_penalty": freq_penalty,
            "presence_penalty": pres_penalty,
            "context_length": int(context_length),
        },
        "story_controls": {
            "style_name": style_name,
            "custom_style": custom_style,
            "pov": pov,
            "ling_texture": ling_texture,
            "pacing": pacing,
            "intensity": intensity,
            "output_lang": output_lang,
            "para_density": para_density,
            "dialogue_ratio": dialogue_ratio,
            "focus_words": focus_words,
            "avoid_words": avoid_words,
            "custom_director": custom_director,
            "technique_library_chars": len(technique_library or ""),
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if pipeline_mode == PIPELINE_HYBRID:
        new_part, response, thought_extra = generate_hybrid_continuation(
            remote_client=client,
            remote_model=model_name,
            lora_base_url=lora_base_url,
            lora_model=lora_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            instruction=instruction,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
            freq_penalty=freq_penalty,
            pres_penalty=pres_penalty,
        )
        thought += thought_extra
    else:
        response = generate_direct_response(
            client=client,
            model_name=model_name,
            messages=request_payload["messages"],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            freq_penalty=freq_penalty,
            pres_penalty=pres_penalty,
        )
        new_part = get_message_content(response)
    updated_story = (current_story.strip() + "\n\n" + new_part).strip() if current_story.strip() else new_part
    updated_history = list(history_state or [])
    updated_history.append(current_story or "")
    latest_output = new_part
    try:
        autosave_generation(
            updated_story=updated_story,
            new_part=new_part,
            thought=thought,
            request_payload=request_payload,
            response_payload=response,
        )
    except Exception:
        logging.exception("Failed to autosave generation artifacts.")
    return updated_story, updated_history, latest_output, thought


def undo_last_step(history_state):
    history = list(history_state or [])
    if not history:
        return "", [], "Nothing to undo."
    restored = history.pop()
    return restored, history, "Undo complete."


def clear_story():
    return "", [], ""


def save_project(
    background,
    roles,
    lore,
    full_story,
    memory,
    style_dna,
    style_samples,
    chronicle,
    technique_library,
):
    payload = {
        "background": background,
        "roles": normalize_rows(roles, 3),
        "lore": normalize_rows(lore, 2),
        "full_story": full_story,
        "memory": memory,
        "style_dna": style_dna,
        "style_samples": style_samples,
        "chronicle": chronicle,
        "technique_library": technique_library,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_dir = get_book_output_dir()
    filename = output_dir / f"story_studio_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(filename)


def load_project(file_obj):
    if file_obj is None:
        return "", [["", "", ""]], [["", ""]], "", "", "", "", "", ""
    file_path = getattr(file_obj, "name", file_obj)
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return (
        payload.get("background", ""),
        payload.get("roles", [["", "", ""]]) or [["", "", ""]],
        payload.get("lore", [["", ""]]) or [["", ""]],
        payload.get("full_story", ""),
        payload.get("memory", ""),
        payload.get("style_dna", ""),
        payload.get("style_samples", ""),
        payload.get("chronicle", ""),
        payload.get("technique_library", ""),
    )


def analyze_style_dna(files, api_key, base_url, model_name):
    if not files:
        return "No files uploaded.", ""
    excerpts = []
    for file_obj in files[:10]:
        file_path = getattr(file_obj, "name", file_obj)
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            excerpts.append(text[:2000])
        except Exception:
            continue
    if not excerpts:
        return "Could not read the uploaded text files.", ""

    prompt = f"""Analyze the writing style of the following samples.
Return two sections:
STYLE_DNA:
- concise bullet points covering rhythm, diction, imagery, dialogue handling, and emotional texture

FEW_SHOT:
- 2 or 3 short imitative sample fragments that capture the style without copying verbatim

Samples:
{chr(10).join(excerpts)}"""
    client = get_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1200,
    )
    content = (response.choices[0].message.content or "").strip()
    dna_match = re.search(r"STYLE_DNA:\s*(.*?)(?:FEW_SHOT:|$)", content, re.S)
    shot_match = re.search(r"FEW_SHOT:\s*(.*)$", content, re.S)
    dna = dna_match.group(1).strip() if dna_match else content
    shots = shot_match.group(1).strip() if shot_match else ""
    return dna, shots


def analyze_story_chronicle(files, api_key, base_url, model_name):
    if not files:
        return "No files uploaded."
    excerpts = []
    for file_obj in files[:12]:
        file_path = getattr(file_obj, "name", file_obj)
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            excerpts.append(text[:2500])
        except Exception:
            continue
    if not excerpts:
        return "Could not read the uploaded text files."

    prompt = f"""You are organizing a long-running fictional story.
Create a story chronicle with these sections:
1. Current recap
2. Character status
3. Open threads and foreshadowing
4. Risks / continuity cautions
5. Suggested next directions

Source material:
{chr(10).join(excerpts)}"""
    client = get_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=1800,
    )
    return (response.choices[0].message.content or "").strip()


def rewrite_with_style(
    style_files,
    target_text,
    instruction,
    output_lang,
    api_key,
    base_url,
    model_name,
    target_length,
):
    if not target_text.strip():
        return "Target text is empty."

    style_context = ""
    if style_files:
        parts = []
        for file_obj in style_files[:8]:
            file_path = getattr(file_obj, "name", file_obj)
            try:
                parts.append(Path(file_path).read_text(encoding="utf-8", errors="ignore")[:1800])
            except Exception:
                continue
        if parts:
            style_context = "\n\nReference style samples:\n" + "\n\n".join(parts)

    prompt = f"""Rewrite the target text according to the requested instruction and style reference.
Output language: {output_lang}
Target maximum length: about {target_length} characters.
Instruction: {instruction or 'Preserve meaning but improve style and readability.'}
{style_context}

Target text:
{target_text}"""
    client = get_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=min(max(target_length // 2, 800), BOOK_WRITER_REWRITE_MAX_TOKENS),
    )
    return (response.choices[0].message.content or "").strip()


def apply_provider(provider_name):
    provider = PROVIDERS.get(provider_name, PROVIDERS["Current Proxy"])
    return provider["base_url"], provider["default_model"], provider.get("pipeline", PIPELINE_DIRECT)


def add_references_to_shelf_ui(source_files, source_paths, pasted_source, pasted_label, shelf_state, max_chars):
    status, preview, state, choices = add_references_to_shelf(
        source_files,
        source_paths,
        pasted_source,
        pasted_label,
        shelf_state,
        max_chars,
    )
    return status, preview, state, gr.update(choices=choices, value=choices)


def remove_references_from_shelf_ui(selected_labels, shelf_state):
    status, preview, state, choices = remove_references_from_shelf(selected_labels, shelf_state)
    return status, preview, state, gr.update(choices=choices, value=choices)


def clear_reference_shelf_ui():
    status, preview, state, choices = clear_reference_shelf()
    return status, preview, state, gr.update(choices=choices, value=[])


def load_saved_reference_shelf_ui():
    status, preview, state, choices = load_saved_reference_shelf()
    return status, preview, state, gr.update(choices=choices, value=choices)


# --- Auto technique-template watcher (Interactive Writing) ---
TEMPLATE_WATCH_TAIL_CHARS = 1600


def _template_watch_topics(value) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 4


def _template_watch_text(story, instruction) -> str:
    """Detection text = the Story Instruction (weighted) + the tail of the in-progress story,
    so the topic the writer is currently on rises to the top."""
    tail = (story or "")[-TEMPLATE_WATCH_TAIL_CHARS:]
    instr = (instruction or "").strip()
    return (instr + "\n" + instr + "\n" + tail).strip()


def watch_templates_ui(story, instruction, book_state, max_topics):
    text = _template_watch_text(story, instruction)
    markdown, _ = suggest_technique_templates(text, book_state or "", "", _template_watch_topics(max_topics), 1)
    return markdown


def watch_templates_if_auto(story, instruction, book_state, max_topics, auto_on):
    if not auto_on:
        return gr.update()
    return watch_templates_ui(story, instruction, book_state, max_topics)


def apply_templates_ui(story, instruction, book_state, max_topics, current_technique_library):
    text = _template_watch_text(story, instruction)
    _, chosen = suggest_technique_templates(text, book_state or "", "", _template_watch_topics(max_topics), 1)
    if not chosen:
        return current_technique_library, "未偵測到描寫主題，沒有可套用的模板。"
    block = render_agent_reference(chosen, "", "Auto-detected topics")
    base = (current_technique_library or "").strip()
    header = "# Auto Technique Templates (依當前主題自動偵測)"
    merged = ((base + "\n\n") if base else "") + header + "\n\n" + block
    topics = "、".join(sorted({f"{card.category}/{card.subcategory}" for card in chosen}))
    return merged, f"已把 {len(chosen)} 張技法模板（{topics}）套用到 Technique Library。"


MANUAL_FILENAME = "AI_Book_Writer_Studio_面板說明書.md"


def load_manual_markdown() -> str:
    """Read the studio manual so it can be shown inside the app (說明書 tab)."""
    for candidate in (Path(__file__).resolve().parent / MANUAL_FILENAME, Path.cwd() / MANUAL_FILENAME):
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        except OSError:
            continue
    return (
        f"找不到說明書檔案 `{MANUAL_FILENAME}`。\n\n"
        "請確認它和 `app_gradio.py` 放在同一個資料夾，再按「重新載入說明書」。"
    )


with gr.Blocks(title="AI Book Writer Studio") as demo:
    state_history = gr.State([])
    lora_api_key_state = gr.State("not-needed")
    distilled_library_state = gr.State("")
    distilled_memory_state = gr.State("")
    distilled_director_state = gr.State("")
    technique_reference_shelf_state = gr.State("")
    technique_book_state = gr.State("")

    gr.Markdown(
        """
        # AI Book Writer Studio

        一個把**參考蒐集 → 深度技法分析 → 寫作 → 改寫 → 檢閱**整合在同一工作台的小說創作系統。

        ### 統整工作流（依分頁順序）

        | 階段 | 分頁 | 你在這裡做什麼 |
        |---|---|---|
        | ① 設定 | `1. 設定 Core Settings` | 設好寫作/分析/LoRA 模型，填世界觀、角色、記憶 |
        | ② 蒐集參考 | `2. 參考書架 Reference Library` | **隨時**上傳或刪除小說、報告當參考；這是全域共用的參考來源 |
        | ③ 深度分析 | `5. 章節技法分析` ＋ `7. 深度技法書庫` | 把參考拆解成「眼睛怎麼寫、喝酒怎麼寫」級別的深度技法卡 |
        | ④ 灌進寫作腦 | `7. 深度技法書庫` 的搜尋載入 ／ `9. 技法回灌與檢閱` | 把技法卡載入寫作 AGENT 的 Technique Library |
        | ⑤ 寫作 | `3. 寫作 Interactive Writing` | 用載入的技法生成續寫 |
        | ⑥ 打磨 | `4. 改寫`、`6. 劇情編排`、`9. 檢閱` | 改寫風格、編排劇情、檢閱技法品質 |

        > 📌 **參考書架是全域的**：在 `2. 參考書架` 加入或刪減的小說參考，會自動供 `7. 深度技法書庫` 建庫使用，不必每次重新上傳。
        > 📖 不熟悉操作？打開最後一個分頁 **`說明書 Manual`**，裡面有逐面板說明與「如何寫身體部位／如何寫動作」的深度解說。
        """
    )

    with gr.Tab("1. 設定 Core Settings"):
        with gr.Row():
            provider_select = gr.Dropdown(
                choices=list(PROVIDERS.keys()),
                value="NALANG + Local LoRA",
                label="Writing Provider Preset",
            )
            refresh_models_btn = gr.Button("Refresh Models", variant="secondary")
            test_conn_btn = gr.Button("Test Writing Connection", variant="secondary")
        with gr.Row():
            api_key_input = gr.Textbox(label="Writing API Key (NALANG)", value=DEFAULT_API_KEY, type="password")
            base_url_input = gr.Textbox(label="Writing Base URL", value=DEFAULT_BASE_URL)
        with gr.Row():
            model_name_input = gr.Textbox(label="Writing Model Name", value=DEFAULT_MODEL)
            model_quick_select = gr.Dropdown(
                choices=[DEFAULT_MODEL],
                value=DEFAULT_MODEL,
                label="Writing Model Quick Select",
            )
        pipeline_mode_input = gr.Dropdown(
            choices=PIPELINES,
            value=PIPELINE_HYBRID,
            label="Writing Pipeline",
        )
        with gr.Accordion("Analysis / Grok Routing", open=True):
            with gr.Row():
                analysis_api_key_input = gr.Textbox(label="Analysis API Key (Grok)", value=DEFAULT_ANALYSIS_API_KEY, type="password")
                analysis_base_url_input = gr.Textbox(label="Analysis Base URL", value=DEFAULT_ANALYSIS_BASE_URL)
            with gr.Row():
                analysis_model_input = gr.Textbox(label="Analysis Model Name", value=DEFAULT_ANALYSIS_MODEL)
                test_analysis_conn_btn = gr.Button("Test Analysis Grok", variant="secondary")
            analysis_test_output = gr.Markdown()
        with gr.Accordion("LoRA Routing", open=True):
            with gr.Row():
                lora_base_url_input = gr.Textbox(label="LoRA Base URL", value=DEFAULT_LORA_BASE_URL)
                lora_model_input = gr.Textbox(label="LoRA Model Name", value=DEFAULT_LORA_MODEL)
                test_lora_conn_btn = gr.Button("Test LoRA", variant="secondary")
            lora_test_output = gr.Markdown()
        with gr.Row():
            load_model_config_btn = gr.Button("Load Model Config", variant="secondary")
            save_model_config_btn = gr.Button("Save Model Config", variant="primary")
        model_config_status = gr.Markdown()
        system_prompt_input = gr.Textbox(label="System Prompt Override", value=DEFAULT_SYSTEM_PROMPT, lines=8)
        test_conn_output = gr.Markdown()

        background_input = gr.Textbox(label="World / Story Background", lines=8, placeholder="Worldbuilding, core premise, major locations...")
        memory_input = gr.Textbox(label="Story Memory", lines=8, placeholder="Long-term memory that should always stay in prompt...")

        with gr.Accordion("Characters", open=True):
            roles_input = gr.Dataframe(
                headers=["Name", "Role", "Traits"],
                value=[["", "", ""]],
                type="array",
                interactive=True,
                wrap=True,
                label="Character Table",
            )
            add_role_btn = gr.Button("Add Character Row", size="sm", variant="secondary")

        with gr.Accordion("Lorebook", open=False):
            lore_input = gr.Dataframe(
                headers=["Keyword", "Content"],
                value=[["", ""]],
                type="array",
                interactive=True,
                wrap=True,
                label="Lore Entries",
            )
            add_lore_btn = gr.Button("Add Lore Row", size="sm", variant="secondary")

        with gr.Accordion("Style DNA", open=False):
            style_files = gr.File(label="Upload Style Samples (.txt)", file_count="multiple", file_types=[".txt"])
            dna_btn = gr.Button("Analyze Style DNA", variant="primary")
            with gr.Row():
                style_dna_output = gr.Textbox(label="Style DNA", lines=8)
                style_samples_output = gr.Textbox(label="Few-Shot Samples", lines=8)

        with gr.Accordion("Story Chronicle", open=False):
            chronicle_files = gr.File(label="Upload Story Files (.txt)", file_count="multiple", file_types=[".txt"])
            chronicle_btn = gr.Button("Build Story Chronicle", variant="primary")
            chronicle_output = gr.Textbox(label="Chronicle", lines=12)

        with gr.Accordion("Technique Library", open=False):
            technique_library_input = gr.Textbox(
                label="Technique Library",
                lines=12,
                placeholder="Distilled craft rules loaded from full_report.md or Technique Finder...",
            )

    with gr.Tab("2. 參考書架 Reference Library"):
        gr.Markdown(
            "### 全域參考書架 — 隨時新增／刪減小說參考\n\n"
            "把多本小說 TXT、`full_report.md`、Technique Finder JSON 放進這個書架。"
            "這裡的內容是 **全域共用** 的：`7. 深度技法書庫` 會直接從這個書架建立深度技法卡，"
            "不必每次重新上傳。\n\n"
            "你不需要一次準備好所有小說——隨時可以回來 **新增、刪除、清空或重新載入** 已保存的書架。"
            "書架會保存到 `book_output/technique_reference_shelf.json`。"
        )
        with gr.Accordion("Reference Shelf — Add / Remove Novel References Anytime", open=True):
            with gr.Row():
                with gr.Column(scale=3):
                    book_source_files = gr.File(
                        label="Add Source Files (.md / .txt / .json)",
                        file_count="multiple",
                        file_types=[".md", ".txt", ".json"],
                    )
                    book_source_paths = gr.Textbox(
                        label="Add File / Folder Paths (one per line)",
                        lines=4,
                        placeholder=r"C:\path\to\full_report.md",
                    )
                    book_pasted_label = gr.Textbox(label="Pasted Source Label", placeholder="例如：風華神女錄技法報告、某本小說片段")
                    book_pasted_source = gr.Textbox(label="Pasted Source Text", lines=8)
                with gr.Column(scale=2):
                    book_max_chars = gr.Number(label="Max Chars / Source", value=65000, precision=0)
                    book_reference_select = gr.Dropdown(
                        choices=[],
                        value=[],
                        multiselect=True,
                        label="Current References",
                    )
                    with gr.Row():
                        book_add_refs_btn = gr.Button("Add To Reference Shelf", variant="primary")
                        book_load_shelf_btn = gr.Button("Load Saved Shelf", variant="secondary")
                    with gr.Row():
                        book_remove_refs_btn = gr.Button("Remove Selected References", variant="secondary")
                        book_clear_shelf_btn = gr.Button("Clear Shelf", variant="stop")
            book_shelf_status = gr.Textbox(label="Reference Shelf Status", lines=5, interactive=False)
            book_shelf_preview = gr.Markdown(label="Reference Shelf Preview")
        gr.Markdown(
            "整理好參考後，到 `7. 深度技法書庫` 按 **Build Technique Book From Reference Shelf** 開始深度分析。"
        )

    with gr.Tab("3. 寫作 Interactive Writing"):
        with gr.Row():
            with gr.Column(scale=3):
                full_story_box = gr.Textbox(label="Full Story", lines=24, interactive=True)
                with gr.Row():
                    undo_btn = gr.Button("Undo", variant="secondary")
                    clear_btn = gr.Button("Clear Story", variant="stop")
            with gr.Column(scale=2):
                style_dropdown = gr.Dropdown(list(STYLES.keys()), value="Standard", label="Story Style")
                custom_style_input = gr.Textbox(label="Custom Style", lines=3, placeholder="Used when style is Custom")
                with gr.Row():
                    temp_slider = gr.Slider(0.1, 2.0, value=0.9, step=0.1, label="Temperature")
                    top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                with gr.Row():
                    freq_slider = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Frequency Penalty")
                    pres_slider = gr.Slider(0.0, 2.0, value=0.4, step=0.1, label="Presence Penalty")
                len_slider = gr.Slider(
                    200,
                    BOOK_WRITER_MAX_TOKENS,
                    value=min(BOOK_WRITER_DEFAULT_TOKENS, BOOK_WRITER_MAX_TOKENS),
                    step=100,
                    label="Max Tokens",
                )

                with gr.Accordion("Global & Advanced", open=False):
                    with gr.Tab("Art & Texture"):
                        ling_texture_input = gr.Dropdown(
                            ["Poetic", "Hard-boiled", "Flowery", "Raw", "Gothic", "Clean and Light", "Dark Fairytale"],
                            value="Poetic",
                            label="Linguistic Texture",
                        )
                        pacing_input = gr.Dropdown(
                            ["Slow Burn", "Standard", "Fast-paced", "Freeze-frame"],
                            value="Standard",
                            label="Narrative Pacing",
                        )
                        intensity_input = gr.Dropdown(
                            ["Mild", "Emotional", "Intense", "Explicit", "Extreme"],
                            value="Emotional",
                            label="Intensity",
                        )
                        pov_dropdown = gr.Dropdown(
                            ["Third Person Limited", "Third Person Omniscient", "First Person", "Second Person"],
                            value="Third Person Limited",
                            label="Point of View",
                        )
                    with gr.Tab("Sensory Weights"):
                        v_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Visual")
                        a_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Auditory")
                        o_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Olfactory")
                        t_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Tactile")
                        g_slider = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Gustatory")
                    with gr.Tab("Format & Directing"):
                        output_lang_input = gr.Dropdown(["简体中文", "繁体中文", "English", "日本語"], value="简体中文", label="Output Language")
                        para_density_input = gr.Dropdown(
                            ["Standard Paragraphs", "Dialogue Dense", "Long-form Descriptive", "Poetic Line Breaks"],
                            value="Standard Paragraphs",
                            label="Paragraph Density",
                        )
                        dialogue_ratio_input = gr.Dropdown(
                            ["Low Dialogue", "Balanced", "High Dialogue"],
                            value="Balanced",
                            label="Dialogue Ratio",
                        )
                        focus_words_input = gr.Textbox(label="Focus Words", placeholder="moonlight, sweat, static, rain...")
                        avoid_words_input = gr.Textbox(label="Avoid Words", placeholder="love, forever, destiny...")
                        custom_director_input = gr.Textbox(label="Director Note Override", placeholder="Override the random director note")
                        context_length_slider = gr.Slider(500, 8000, value=3500, step=500, label="Context Window Hint")

                instruction = gr.Textbox(label="Story Instruction (Director)", lines=5, placeholder="What should happen next? 這條故事指令主導本次輸出。")
                generate_btn = gr.Button("Generate Continuation", variant="primary")
                with gr.Accordion("AI Thought Process", open=False):
                    thought_output = gr.Markdown("...")
                latest_output = gr.Markdown("...")

                with gr.Accordion("📋 相關技法模板（遇到主題自動檢視該怎麼寫）", open=True):
                    gr.Markdown(
                        "當進行中的文章或故事指令出現特定描寫主題（眼睛、喝酒、拔劍、壓抑、章尾鉤子…），"
                        "這裡會自動跳出對應的「該怎麼寫」技法模板。"
                    )
                    with gr.Row():
                        auto_template_toggle = gr.Checkbox(
                            value=True,
                            label="自動偵測（生成後與修改指令時自動更新）",
                        )
                        template_watch_topics = gr.Number(label="最多主題數", value=4, precision=0)
                    with gr.Row():
                        template_refresh_btn = gr.Button("立即檢視相關技法模板", variant="secondary")
                        template_apply_btn = gr.Button("套用到寫作參考", variant="secondary")
                    template_apply_status = gr.Markdown()
                    template_watch_box = gr.Markdown(
                        "在這裡會自動顯示『眼睛怎麼寫、喝酒怎麼寫、拔劍怎麼寫…』等技法模板。"
                        "先在 `7. 深度技法書庫` 建一次書庫，模板會更貼合你的參考小說。"
                    )

    with gr.Tab("4. 改寫 Rewrite / Style Transfer"):
        with gr.Row():
            with gr.Column():
                rewrite_style_files = gr.File(label="Style Reference Files", file_count="multiple", file_types=[".txt"])
                rewrite_instruction = gr.Textbox(label="Rewrite Instruction", lines=2, placeholder="Make it colder, more lyrical, more cinematic...")
                rewrite_lang_input = gr.Dropdown(["简体中文", "繁体中文", "English", "日本語"], value="简体中文", label="Output Language")
                rewrite_len_slider = gr.Slider(500, 12000, value=3000, step=250, label="Target Length")
            with gr.Column():
                target_text_input = gr.Textbox(label="Target Draft", lines=16, placeholder="Paste the text to rewrite...")
        rewrite_btn = gr.Button("Rewrite", variant="primary")
        rewrite_output = gr.Textbox(label="Rewrite Output", lines=18, interactive=True)

    with gr.Tab("5. 章節技法分析 Chapter Craft Analysis"):
        with gr.Row():
            with gr.Column():
                craft_txt_file = gr.File(label="Novel TXT", file_count="single", file_types=[".txt"])
                craft_url_input = gr.Textbox(label="Chapter Directory URL", placeholder="https://example.com/book/index.html")
            with gr.Column():
                craft_goal_input = gr.Textbox(label="/goal", value=CHAPTER_CRAFT_DEFAULT_GOAL, lines=4)
                with gr.Row():
                    craft_limit_input = gr.Number(label="Chapter Limit (0 = Full Book)", value=0, precision=0)
                    craft_max_chars_input = gr.Number(label="Max Chars / Chapter / Auto Chunk", value=12000, precision=0)
                craft_dry_run_input = gr.Checkbox(label="Dry Run", value=True)
        craft_pasted_text = gr.Textbox(label="Pasted Novel Text", lines=12)
        craft_analyze_btn = gr.Button("Analyze Chapter Craft", variant="primary")
        craft_status = gr.Textbox(label="Status", lines=4, interactive=False)
        craft_preview = gr.Markdown()
        craft_report_file = gr.File(label="Full Report")

    with gr.Tab("6. 劇情編排 Plot Ideation"):
        gr.Markdown("Planning route: Grok extracts craft and strategy, NALANG builds structure and final plan, LoRA contributes scene-level writing instincts.")
        with gr.Row():
            with gr.Column(scale=3):
                plot_premise_input = gr.Textbox(label="Story Seed / Premise", lines=8, placeholder="主角、世界、核心衝突、你想要的劇情方向...")
                plot_goal_input = gr.Textbox(label="/goal", value=DEFAULT_PLOT_GOAL, lines=4)
                plot_reference_text = gr.Textbox(label="Reference Text", lines=10, placeholder="可貼參考小說片段，系統會先蒸餾技法 DNA，再用於劇情編排。")
            with gr.Column(scale=2):
                plot_reference_file = gr.File(label="Reference TXT", file_count="single", file_types=[".txt"])
                plot_reference_url = gr.Textbox(label="Reference Chapter Directory URL", placeholder="https://example.com/book/index.html")
                plot_genre_tone = gr.Textbox(label="Genre / Tone", placeholder="仙俠、宮廷、懸疑、黑暗浪漫、慢熱...")
                plot_arc_mode = gr.Dropdown(
                    ["章節連載", "三幕式", "起承轉合", "單卷篇章", "多線群像", "懸疑揭謎"],
                    value="章節連載",
                    label="Arrangement Mode",
                )
                plot_output_lang = gr.Dropdown(["繁體中文", "简体中文", "English", "日本語"], value="繁體中文", label="Output Language")
                with gr.Row():
                    plot_target_chapters = gr.Number(label="Target Chapters", value=12, precision=0)
                    plot_reference_limit = gr.Number(label="Reference Chapter Limit", value=3, precision=0)
                plot_max_reference_chars = gr.Number(label="Max Reference Chars", value=9000, precision=0)
                plot_dry_run = gr.Checkbox(label="Dry Run", value=True)
        plot_generate_btn = gr.Button("Generate Plot Ideation", variant="primary")
        plot_status = gr.Textbox(label="Status", lines=4, interactive=False)
        plot_preview = gr.Markdown()
        plot_report_file = gr.File(label="Plot Report")

    with gr.Tab("7. 深度技法書庫 Deep Technique Book"):
        gr.Markdown(
            "### 深度技法書庫 — 把參考拆解成「眼睛怎麼寫、喝酒怎麼寫」級別的技法卡\n\n"
            "從 `2. 參考書架` 的參考建立可搜尋的階層技法書庫，搜出精確技法，再載入寫作 AGENT。"
            "每張技法卡都會深入到 **部位／動作解剖、句法節奏、用詞調色盤、感官分層、弱寫→強寫對照**，"
            "不是淺淺幾句帶過。"
        )
        gr.Markdown(
            "📚 **參考來源在 `2. 參考書架 Reference Library` 分頁管理**（可隨時新增／刪減小說、報告）。"
            "在那裡整理好參考後，回到這裡按下方的 **Build** 即可建庫。"
        )

        with gr.Accordion("2. Build Deep Technique Book From Shelf", open=True):
            with gr.Row():
                with gr.Column(scale=3):
                    book_goal_input = gr.Textbox(label="/goal", value=DEFAULT_BOOK_LIBRARY_GOAL, lines=4)
                with gr.Column(scale=2):
                    book_output_lang = gr.Dropdown(
                        ["繁體中文", "簡體中文", "English", "日本語"],
                        value="繁體中文",
                        label="Output Language",
                    )
                    book_max_entries = gr.Number(label="Max Entries / Subcategory", value=3, precision=0)
                    book_dry_run = gr.Checkbox(label="Dry Run / Local Categorizer", value=True)
            book_build_btn = gr.Button("Build Technique Book From Reference Shelf", variant="primary")
            book_build_status = gr.Textbox(label="Build Status", lines=6, interactive=False)
            book_preview = gr.Markdown(label="Deep Technique Book Preview")
            with gr.Row():
                book_md_file = gr.File(label="Technique Book Markdown")
                book_json_file = gr.File(label="Technique Book JSON")

        with gr.Accordion("3. Search Technique Book / Load To Writing AGENT", open=True):
            with gr.Row():
                book_query = gr.Textbox(label="Search Query", placeholder="眼睛、嘴巴、手、腰身、喝酒、拔劍、壓抑、章尾鉤子...")
                book_category_filter = gr.Dropdown(
                    ALL_CATEGORY_CHOICES,
                    value="All",
                    label="Category / Subcategory",
                )
            with gr.Row():
                book_json_path = gr.Textbox(
                    label="Existing Library JSON Path",
                    placeholder="Optional. Blank uses current session library or latest saved library.",
                )
                book_result_limit = gr.Number(label="Search / Load Result Limit", value=12, precision=0)
            book_load_mode = gr.Dropdown(
                BOOK_LIBRARY_LOAD_MODES,
                value=BOOK_LIBRARY_LOAD_MODES[0],
                label="Load Mode",
            )
            with gr.Row():
                book_search_btn = gr.Button("Search Technique Book", variant="secondary")
                book_load_btn = gr.Button("Load Search Results To Writing AGENT", variant="secondary")
                book_load_latest_btn = gr.Button("Load Latest Book Library To Writing AGENT", variant="secondary")
            book_search_status = gr.Textbox(label="Search Status", lines=4, interactive=False)
            book_agent_load_status = gr.Textbox(label="Writing AGENT Load Status", lines=4, interactive=False)
            book_search_results = gr.Markdown(label="Deep Search Results")

        with gr.Accordion("Legacy Tool: Distill One Novel To Technique Finder", open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    library_novel_file = gr.File(label="Novel TXT", file_count="single", file_types=[".txt"])
                    library_novel_text = gr.Textbox(label="Pasted Novel Text", lines=10)
                    library_goal_input = gr.Textbox(label="/goal", value=DEFAULT_LIBRARY_GOAL, lines=3)
                with gr.Column(scale=2):
                    library_novel_url = gr.Textbox(label="Novel Chapter Directory URL", placeholder="https://example.com/book/index.html")
                    library_output_lang = gr.Dropdown(["繁體中文", "简体中文", "English", "日本語"], value="繁體中文", label="Output Language")
                    with gr.Row():
                        library_chapter_limit = gr.Number(label="Chapter Limit (0 = Full Book)", value=0, precision=0)
                        library_cards_per_chapter = gr.Number(label="Cards / Chapter", value=3, precision=0)
                    library_max_chars = gr.Number(label="Max Chars / Chapter / Auto Chunk", value=9000, precision=0)
                    library_dry_run = gr.Checkbox(label="Dry Run", value=True)
            library_btn = gr.Button("Distill Novel To Technique Finder", variant="primary")
            library_status = gr.Textbox(label="Library Status", lines=5, interactive=False)
            library_preview = gr.Markdown()
            library_report_file = gr.File(label="Technique Finder Library")

        gr.Markdown("Use the fields below when you want a focused sheet for one exact scene/action/situation.")
        with gr.Row():
            with gr.Column(scale=3):
                technique_scene_input = gr.Textbox(label="Specific Scene", lines=2, placeholder="王城正殿、雨夜城門、破敗藥鋪、密室審問...")
                technique_action_input = gr.Textbox(label="Specific Action", lines=2, placeholder="侍女接近昏迷者、拔劍、遞信、跪拜、暗中下毒...")
                technique_situation_input = gr.Textbox(label="Specific Situation", lines=3, placeholder="祕密刺殺前、久別重逢卻不能相認、真相即將暴露、權力壓迫...")
                technique_effect_input = gr.Textbox(label="Desired Reader Effect", lines=2, placeholder="壓迫、曖昧、危險、悲涼、懸疑、莊嚴、失控...")
                technique_goal_input = gr.Textbox(label="/goal", value=DEFAULT_TECHNIQUE_GOAL, lines=4)
            with gr.Column(scale=2):
                technique_reference_file = gr.File(label="Reference TXT", file_count="single", file_types=[".txt"])
                technique_reference_url = gr.Textbox(label="Reference Chapter Directory URL", placeholder="https://example.com/book/index.html")
                technique_reference_text = gr.Textbox(label="Reference Text", lines=8)
                technique_output_lang = gr.Dropdown(["繁體中文", "简体中文", "English", "日本語"], value="繁體中文", label="Output Language")
                with gr.Row():
                    technique_reference_limit = gr.Number(label="Reference Chapter Limit", value=3, precision=0)
                    technique_max_chars = gr.Number(label="Max Reference Chars", value=9000, precision=0)
                technique_dry_run = gr.Checkbox(label="Dry Run", value=True)
        technique_btn = gr.Button("Aggregate Techniques", variant="primary")
        technique_status = gr.Textbox(label="Status", lines=5, interactive=False)
        technique_preview = gr.Markdown()
        technique_report_file = gr.File(label="Technique Report")

    with gr.Tab("8. 存讀檔 Save / Load"):
        save_btn = gr.Button("Save Project", variant="primary")
        save_file = gr.Textbox(label="Saved File")
        load_btn = gr.File(label="Load Project JSON", file_count="single", file_types=[".json"])
        load_msg = gr.Markdown()

    with gr.Tab("9. 技法回灌與檢閱 Skill / Technique Review"):
        with gr.Accordion("full_report.md -> Compact Technique Library -> Writing AGENT", open=True):
            with gr.Row():
                with gr.Column(scale=3):
                    report_distill_file = gr.File(label="full_report.md", file_count="single", file_types=[".md", ".txt", ".json"])
                    report_distill_path = gr.Textbox(
                        label="full_report Path",
                        placeholder=r"C:\path\to\full_report.md",
                    )
                    report_distill_text = gr.Textbox(label="Pasted full_report Text", lines=8)
                    report_distill_goal = gr.Textbox(label="/goal", value=DEFAULT_REPORT_DISTILL_GOAL, lines=3)
                with gr.Column(scale=2):
                    report_distill_lang = gr.Dropdown(["繁體中文", "簡體中文", "English", "日本語"], value="繁體中文", label="Output Language")
                    with gr.Row():
                        report_distill_max_source = gr.Number(label="Max Report Chars", value=45000, precision=0)
                        report_distill_max_library = gr.Number(label="Max Library Chars", value=9000, precision=0)
                    report_distill_dry_run = gr.Checkbox(label="Dry Run / Local Extract", value=True)
                    report_load_mode = gr.Dropdown(
                        TECHNIQUE_LOAD_MODES,
                        value=TECHNIQUE_LOAD_MODES[0],
                        label="Load Mode",
                    )
            report_distill_btn = gr.Button("Distill full_report To Compact Library", variant="primary")
            with gr.Row():
                report_load_btn = gr.Button("Load Distilled Library To Writing AGENT", variant="secondary")
                report_load_latest_btn = gr.Button("Load Latest Saved Library To Writing AGENT", variant="secondary")
            report_distill_status = gr.Textbox(label="Distill Status", lines=6, interactive=False)
            report_agent_load_status = gr.Textbox(label="Writing AGENT Load Status", lines=3, interactive=False)
            report_distill_preview = gr.Markdown(label="Compact Library Preview")
            report_distill_file_output = gr.File(label="Compact Technique Library File")

        with gr.Row():
            review_target_input = gr.Dropdown(
                choices=REVIEW_CHOICES,
                value=REVIEW_CHOICES[0],
                label="Review Target",
            )
            review_max_chars_input = gr.Number(label="Preview Chars", value=22000, precision=0)
        review_custom_path_input = gr.Textbox(
            label="Custom Markdown / JSON Path",
            placeholder=r"C:\path\to\full_report.md",
        )
        review_btn = gr.Button("Review Skill / Technique", variant="primary")
        review_status = gr.Textbox(label="Review Status", lines=7, interactive=False)
        review_preview = gr.Markdown(label="Review Preview")
        review_file = gr.File(label="Review Report")

    with gr.Tab("10. 說明書 Manual"):
        gr.Markdown(
            "### 使用說明書 — 逐面板說明、深度分析原理與操作流程\n\n"
            "下面是完整說明書（與專案根目錄的 "
            f"`{MANUAL_FILENAME}` 同步）。其中特別解說了 **「如何寫身體部位／如何寫動作」** 的深度分析原理。"
            "若你在外部編輯了說明書檔案，按下方按鈕即可重新載入。"
        )
        manual_reload_btn = gr.Button("重新載入說明書 Reload Manual", variant="secondary")
        manual_display = gr.Markdown(value=load_manual_markdown())

    manual_reload_btn.click(load_manual_markdown, outputs=manual_display, api_name=False)

    provider_select.change(apply_provider, inputs=provider_select, outputs=[base_url_input, model_name_input, pipeline_mode_input], api_name=False)
    add_role_btn.click(lambda d: add_empty_row(d, 3), inputs=roles_input, outputs=roles_input, api_name=False)
    add_lore_btn.click(lambda d: add_empty_row(d, 2), inputs=lore_input, outputs=lore_input, api_name=False)
    model_quick_select.change(lambda selected: str(selected), inputs=model_quick_select, outputs=model_name_input, api_name=False)
    refresh_models_btn.click(fetch_all_models, inputs=[api_key_input, base_url_input], outputs=model_quick_select, api_name=False)
    test_conn_btn.click(test_api_connection, inputs=[api_key_input, base_url_input, model_name_input], outputs=test_conn_output, api_name=False)
    test_analysis_conn_btn.click(test_api_connection, inputs=[analysis_api_key_input, analysis_base_url_input, analysis_model_input], outputs=analysis_test_output, api_name=False)
    test_lora_conn_btn.click(test_api_connection, inputs=[lora_api_key_state, lora_base_url_input, lora_model_input], outputs=lora_test_output, api_name=False)
    save_model_config_btn.click(
        save_model_config,
        inputs=[
            api_key_input,
            base_url_input,
            model_name_input,
            pipeline_mode_input,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
            lora_base_url_input,
            lora_model_input,
        ],
        outputs=model_config_status,
        api_name=False,
    )
    load_model_config_btn.click(
        load_model_config,
        outputs=[
            api_key_input,
            base_url_input,
            model_name_input,
            pipeline_mode_input,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
            lora_base_url_input,
            lora_model_input,
            model_config_status,
        ],
        api_name=False,
    )
    dna_btn.click(analyze_style_dna, inputs=[style_files, analysis_api_key_input, analysis_base_url_input, analysis_model_input], outputs=[style_dna_output, style_samples_output], api_name=False)
    chronicle_btn.click(analyze_story_chronicle, inputs=[chronicle_files, analysis_api_key_input, analysis_base_url_input, analysis_model_input], outputs=chronicle_output, api_name=False)

    generate_btn.click(
        generate_continuation,
        inputs=[
            background_input, roles_input, lore_input, full_story_box, instruction,
            style_dropdown, custom_style_input,
            temp_slider, freq_slider, pres_slider, top_p_slider, len_slider,
            context_length_slider, pov_dropdown, system_prompt_input,
            v_slider, a_slider, o_slider, t_slider, g_slider,
            ling_texture_input, pacing_input, intensity_input,
            focus_words_input, avoid_words_input, custom_director_input,
            output_lang_input, para_density_input, dialogue_ratio_input, memory_input,
            style_dna_output, style_samples_output, chronicle_output, technique_library_input,
            pipeline_mode_input,
            api_key_input, base_url_input, model_name_input,
            lora_base_url_input, lora_model_input,
            state_history,
        ],
        outputs=[full_story_box, state_history, latest_output, thought_output],
        api_name=False,
    ).then(
        watch_templates_if_auto,
        inputs=[full_story_box, instruction, technique_book_state, template_watch_topics, auto_template_toggle],
        outputs=template_watch_box,
        api_name=False,
    )

    template_refresh_btn.click(
        watch_templates_ui,
        inputs=[full_story_box, instruction, technique_book_state, template_watch_topics],
        outputs=template_watch_box,
        api_name=False,
    )
    instruction.change(
        watch_templates_if_auto,
        inputs=[full_story_box, instruction, technique_book_state, template_watch_topics, auto_template_toggle],
        outputs=template_watch_box,
        api_name=False,
    )
    template_apply_btn.click(
        apply_templates_ui,
        inputs=[full_story_box, instruction, technique_book_state, template_watch_topics, technique_library_input],
        outputs=[technique_library_input, template_apply_status],
        api_name=False,
    )

    undo_btn.click(undo_last_step, inputs=state_history, outputs=[full_story_box, state_history, latest_output], api_name=False)
    clear_btn.click(clear_story, outputs=[full_story_box, state_history, latest_output], api_name=False)

    rewrite_btn.click(
        rewrite_with_style,
        inputs=[rewrite_style_files, target_text_input, rewrite_instruction, rewrite_lang_input, api_key_input, base_url_input, model_name_input, rewrite_len_slider],
        outputs=rewrite_output,
        api_name=False,
    )

    craft_analyze_btn.click(
        analyze_chapter_craft,
        inputs=[
            craft_txt_file,
            craft_pasted_text,
            craft_url_input,
            craft_goal_input,
            craft_limit_input,
            craft_max_chars_input,
            craft_dry_run_input,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
        ],
        outputs=[craft_status, craft_preview, craft_report_file],
        api_name=False,
    )

    plot_generate_btn.click(
        generate_plot_ideation,
        inputs=[
            plot_reference_file,
            plot_reference_text,
            plot_reference_url,
            plot_premise_input,
            plot_goal_input,
            plot_genre_tone,
            plot_arc_mode,
            plot_target_chapters,
            plot_output_lang,
            plot_reference_limit,
            plot_max_reference_chars,
            plot_dry_run,
            background_input,
            roles_input,
            lore_input,
            memory_input,
            style_dna_output,
            chronicle_output,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
            api_key_input,
            base_url_input,
            model_name_input,
            lora_base_url_input,
            lora_model_input,
        ],
        outputs=[plot_status, plot_preview, plot_report_file],
        api_name=False,
    )

    book_add_refs_btn.click(
        add_references_to_shelf_ui,
        inputs=[
            book_source_files,
            book_source_paths,
            book_pasted_source,
            book_pasted_label,
            technique_reference_shelf_state,
            book_max_chars,
        ],
        outputs=[
            book_shelf_status,
            book_shelf_preview,
            technique_reference_shelf_state,
            book_reference_select,
        ],
        api_name=False,
    )

    book_load_shelf_btn.click(
        load_saved_reference_shelf_ui,
        outputs=[
            book_shelf_status,
            book_shelf_preview,
            technique_reference_shelf_state,
            book_reference_select,
        ],
        api_name=False,
    )

    book_remove_refs_btn.click(
        remove_references_from_shelf_ui,
        inputs=[
            book_reference_select,
            technique_reference_shelf_state,
        ],
        outputs=[
            book_shelf_status,
            book_shelf_preview,
            technique_reference_shelf_state,
            book_reference_select,
        ],
        api_name=False,
    )

    book_clear_shelf_btn.click(
        clear_reference_shelf_ui,
        outputs=[
            book_shelf_status,
            book_shelf_preview,
            technique_reference_shelf_state,
            book_reference_select,
        ],
        api_name=False,
    )

    book_build_btn.click(
        build_integrated_technique_book_library_from_shelf,
        inputs=[
            technique_reference_shelf_state,
            book_goal_input,
            book_output_lang,
            book_max_entries,
            book_dry_run,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
        ],
        outputs=[
            book_build_status,
            book_preview,
            book_md_file,
            book_json_file,
            technique_book_state,
        ],
        api_name=False,
    )

    book_search_btn.click(
        search_integrated_technique_book_library,
        inputs=[
            book_query,
            book_category_filter,
            book_json_path,
            technique_book_state,
            book_result_limit,
        ],
        outputs=[book_search_status, book_search_results],
        api_name=False,
    )

    book_load_btn.click(
        load_technique_book_to_agent_fields,
        inputs=[
            book_query,
            book_category_filter,
            book_json_path,
            technique_book_state,
            technique_library_input,
            memory_input,
            instruction,
            book_load_mode,
            book_result_limit,
        ],
        outputs=[technique_library_input, memory_input, instruction, book_agent_load_status],
        api_name=False,
    )

    book_load_latest_btn.click(
        load_latest_technique_book_to_agent_fields,
        inputs=[
            technique_library_input,
            memory_input,
            instruction,
            book_load_mode,
            book_result_limit,
        ],
        outputs=[technique_library_input, memory_input, instruction, book_agent_load_status],
        api_name=False,
    )

    library_btn.click(
        distill_novel_to_technique_finder,
        inputs=[
            library_novel_file,
            library_novel_text,
            library_novel_url,
            library_goal_input,
            library_output_lang,
            library_chapter_limit,
            library_cards_per_chapter,
            library_max_chars,
            library_dry_run,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
        ],
        outputs=[library_status, library_preview, library_report_file],
        api_name=False,
    )

    technique_btn.click(
        aggregate_scene_techniques,
        inputs=[
            technique_reference_file,
            technique_reference_text,
            technique_reference_url,
            technique_scene_input,
            technique_action_input,
            technique_situation_input,
            technique_effect_input,
            technique_goal_input,
            technique_output_lang,
            technique_reference_limit,
            technique_max_chars,
            technique_dry_run,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
        ],
        outputs=[technique_status, technique_preview, technique_report_file],
        api_name=False,
    )

    report_distill_btn.click(
        distill_full_report_to_agent_library,
        inputs=[
            report_distill_file,
            report_distill_path,
            report_distill_text,
            report_distill_goal,
            report_distill_lang,
            report_distill_max_source,
            report_distill_max_library,
            report_distill_dry_run,
            analysis_api_key_input,
            analysis_base_url_input,
            analysis_model_input,
        ],
        outputs=[
            report_distill_status,
            report_distill_preview,
            report_distill_file_output,
            distilled_library_state,
            distilled_memory_state,
            distilled_director_state,
        ],
        api_name=False,
    )

    report_load_btn.click(
        load_distilled_library_to_agent_fields,
        inputs=[
            distilled_library_state,
            distilled_memory_state,
            distilled_director_state,
            technique_library_input,
            memory_input,
            instruction,
            report_load_mode,
        ],
        outputs=[technique_library_input, memory_input, instruction, report_agent_load_status],
        api_name=False,
    )

    report_load_latest_btn.click(
        load_latest_distilled_library_to_agent_fields,
        inputs=[
            technique_library_input,
            memory_input,
            instruction,
            report_load_mode,
        ],
        outputs=[technique_library_input, memory_input, instruction, report_agent_load_status],
        api_name=False,
    )

    review_btn.click(
        review_skill_and_technique,
        inputs=[review_target_input, review_custom_path_input, review_max_chars_input],
        outputs=[review_status, review_preview, review_file],
        api_name=False,
    )

    save_btn.click(
        save_project,
        inputs=[
            background_input,
            roles_input,
            lore_input,
            full_story_box,
            memory_input,
            style_dna_output,
            style_samples_output,
            chronicle_output,
            technique_library_input,
        ],
        outputs=save_file,
        api_name=False,
    )

    load_btn.upload(
        load_project,
        inputs=load_btn,
        outputs=[
            background_input,
            roles_input,
            lore_input,
            full_story_box,
            memory_input,
            style_dna_output,
            style_samples_output,
            chronicle_output,
            technique_library_input,
        ],
        api_name=False,
    ).then(lambda: "Project loaded.", outputs=load_msg, api_name=False)

    demo.load(
        load_saved_reference_shelf_ui,
        outputs=[
            book_shelf_status,
            book_shelf_preview,
            technique_reference_shelf_state,
            book_reference_select,
        ],
        api_name=False,
    )


def main() -> None:
    server_name = "127.0.0.1"
    server_port = get_gradio_port(server_name)
    ensure_proxy_running()
    demo.queue(default_concurrency_limit=1)
    print(f"Starting Gradio studio on http://{server_name}:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=True,
        show_api=False,
        quiet=True,
    )


if __name__ == "__main__":
    main()
