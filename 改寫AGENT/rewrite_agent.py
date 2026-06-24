"""Chunked full-text rewrite agent using local LoRA, external API, or both.

This agent supports three pillars:

1. 深化故事脈絡 (story-context engine): a rolling, model-maintained digest of
   characters / setting / events / open threads / already-used imagery is kept
   across chunks and injected into every rewrite + polish prompt, so the rewrite
   stays coherent over a long manuscript instead of treating each chunk in
   isolation.
2. 降低全篇重複性 (repetition control): dedup compares paragraphs through a
   script- and punctuation-agnostic canonical key (OpenCC + punctuation strip),
   so simplified-vs-traditional duplicate paste-ins are caught, and near-duplicate
   paragraphs / stuttered sentences are removed (not merely reported).
3. 增補 / 改寫 / 刪減 (operation modes): an explicit operation dimension drives
   whether a chunk is expanded, plainly rewritten, or condensed, with matching
   prompt rules and dynamic token budgets.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, ClassVar, Iterable

from openai import OpenAI


logger = logging.getLogger(__name__)

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
OUTPUT_DIR = AGENT_DIR / "output"
INPUT_DIR = AGENT_DIR / "input"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compat_proxy import HOST as PROXY_HOST  # noqa: E402
from compat_proxy import PORT as PROXY_PORT  # noqa: E402
from compat_proxy import main as run_compat_proxy  # noqa: E402
from config import get_config  # noqa: E402
from lora_runtime import LORA_BASE_URL, LORA_MODEL_NAME, ensure_lora_server_running, is_lora_base_url  # noqa: E402

# Cross-chunk anti-repetition. The module sits at the repo root (already added to
# sys.path above) and depends only on the stdlib, so the import is safe; if it is
# ever unavailable we degrade gracefully rather than break the rewrite engine.
try:
    import repetition_guard  # noqa: E402
except Exception as exc:  # noqa: BLE001 - optional enhancement, never fatal.
    repetition_guard = None  # type: ignore[assignment]
    logger.warning("repetition_guard unavailable, cross-chunk dedup disabled: %s", exc)


MODE_API = "外部 API"
MODE_LORA = "本地 LoRA"
MODE_HYBRID = "混合：LoRA改寫 -> API校稿"
MODE_FULL_HYBRID = "完整混合：API規劃 -> LoRA改寫 -> API校稿"
MODES = [MODE_HYBRID, MODE_LORA, MODE_API, MODE_FULL_HYBRID]

STRENGTHS = ["強改寫", "標準改寫", "保守潤色", "重構改寫"]

OP_REWRITE = "改寫"
OP_EXPAND = "增補擴寫"
OP_CONDENSE = "精簡刪減"
OPERATIONS = [OP_REWRITE, OP_EXPAND, OP_CONDENSE]

# Selectable rewrite models on the DZMM endpoint (NALANG first, then x-apex).
# NALANG returns SSE and must go through the local compat proxy; x-apex returns
# JSON and works through the proxy too (it normalizes both).
NALANG_MODELS = [
    "nalang-xl-0826-16k",
    "nalang-xl-16k",
    "nalang-turbo-1115-16k",
    "nalang-max-0826-16k",
    "nalang-v17-2",
    "nalang-xl-0826-10k",
    "nalang-turbo-1115-10k",
    "nalang-turbo-1115",
    "nalang-turbo-0826-16k",
    "nalang-turbo-0826-10k",
    "nalang-medium-0826-16k",
    "nalang-medium-0826-10k",
    "nalang-max-0826-10k",
    "nalang-max-0826",
    "nalang-xl-0826",
    "nalang-medium-0826",
    "nalang-turbo-0826",
]
APEX_MODELS = [
    "x-apex-surge-0505-16k",
    "x-apex-surge-0505",
    "x-apex-sigma-0621-16k",
    "x-apex-sigma-0621",
]
DZMM_MODELS = NALANG_MODELS + APEX_MODELS

LORA_SAFE_MAX_TOKENS = 900
# Expansion needs more headroom than the safe default, or 增補擴寫 collapses into
# plain 改寫 on the pure-LoRA path. Still bounded to avoid local-GPU OOM; lower
# the UI/CLI max_tokens if your card runs out of memory.
LORA_EXPAND_MAX_TOKENS = 1400


DEFAULT_CFG = get_config()
DEFAULT_LLM = DEFAULT_CFG["config_list"][0]
# Non-secret defaults are fine to keep at module scope. The API key is NOT baked
# in as a dataclass default or CLI default any more: it is resolved lazily from the
# environment / config only when a client is actually built (see resolve_api_key).
# DEFAULT_API_KEY remains exported because the Gradio UI (改寫AGENT/app.py) reads it
# to pre-fill the key field; it is just no longer used as a default argument value.
DEFAULT_API_KEY = DEFAULT_LLM["api_key"]
DEFAULT_BASE_URL = DEFAULT_LLM["base_url"]
DEFAULT_MODEL = DEFAULT_LLM["model"]


def resolve_api_key(explicit: str = "") -> str:
    """Return the API key to use, preferring an explicit override.

    Reads from the live config/env on each call instead of holding the secret in
    a long-lived module constant, so the key is only materialized when needed.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    return get_config()["config_list"][0].get("api_key", "") or ""


@dataclass
class RewriteSettings:
    mode: str = MODE_HYBRID
    instruction: str = ""
    rewrite_strength: str = "強改寫"
    operation: str = OP_REWRITE
    output_language: str = "繁體中文"
    # Empty by default: the key is resolved from env/config at client-build time
    # (resolve_api_key) rather than baked into the dataclass as a secret default.
    api_key: str = ""
    api_base_url: str = DEFAULT_BASE_URL
    api_model: str = DEFAULT_MODEL
    chunk_chars: int = 900
    max_tokens: int = 900
    temperature: float = 0.75
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    style_reference: str = ""
    continuity_notes: str = ""
    use_story_context: bool = True
    chapter_continuity: bool = True
    dedup_similarity: float = 0.9
    # Phase-2 inputs derived from a Grok diagnosis (analysis_agent.Diagnosis).
    # Kept as plain strings/lists so rewrite_agent has no dependency on the
    # analysis module (the glue layer populates these from a loaded Diagnosis).
    diagnosis_spine_seed: str = ""
    diagnosis_brief: str = ""
    diagnosis_window_problems: list[str] = field(default_factory=list)


@dataclass
class RewriteResult:
    output_text: str
    output_path: Path
    log_path: Path
    report_path: Path
    chunks_total: int
    mode: str


@dataclass
class StoryContext:
    """A rolling, model-maintained digest of the story so far.

    The digest is internal scaffolding: it is fed into rewrite / polish prompts
    to keep long manuscripts coherent and to remind the model which imagery has
    already been used (so it stops re-describing the same thing). It is never
    written into the final text.
    """

    digest: str = ""
    updates: int = 0

    def render(self) -> str:
        if not self.digest.strip():
            return ""
        return "\n\n目前故事脈絡（內部參考，嚴禁輸出到正文）：\n" + self.digest.strip()


@dataclass
class NarrativeSpine:
    """The 主敘事模組: a stable, slowly-evolving backbone for the whole book.

    Where StoryContext is the volatile recent-state layer (what just happened),
    the spine is the persistent narrative module: premise / main plotline, the
    cast and their arcs, world rules, central conflict, and a terse per-chapter
    beat list. It is refreshed at chapter boundaries (or every few chunks) rather
    than every chunk, so it stays consistent and cheap, and it is injected into
    every prompt to keep chapters connected. ``handoff`` carries the explicit
    "what the next chapter must continue" note — the continuity system's bridge.
    """

    # Only the most recent beats are ever rendered, so the list is also bounded to
    # this length on append (see add_beat) to keep memory and context size flat for
    # very long (hundreds-of-chapter) manuscripts.
    MAX_BEATS: ClassVar[int] = 12

    bible: str = ""
    chapter_beats: list[str] = field(default_factory=list)
    handoff: str = ""
    updates: int = 0

    def add_beat(self, beat: str) -> None:
        """Append a chapter beat, keeping the list bounded to MAX_BEATS."""
        self.chapter_beats.append(beat)
        if len(self.chapter_beats) > self.MAX_BEATS:
            del self.chapter_beats[: len(self.chapter_beats) - self.MAX_BEATS]

    def render(self) -> str:
        if not (self.bible.strip() or self.chapter_beats or self.handoff.strip()):
            return ""
        parts = ["\n\n主敘事模組（全書骨幹，內部參考，嚴禁輸出到正文）："]
        if self.bible.strip():
            parts.append(self.bible.strip())
        if self.chapter_beats:
            # Cap the rendered beats block so a long backbone can never balloon the
            # injected context past the model's window.
            beats_text = "\n".join(self.chapter_beats[-self.MAX_BEATS:])[:1200]
            parts.append("各章脈絡：\n" + beats_text)
        if self.handoff.strip():
            parts.append("接續要點（下一段／下一章必須延續或回應）：\n" + self.handoff.strip())
        return "\n".join(parts)

    def echo_context(self) -> str:
        """Raw backbone text (no labels) for leak-stripping of model output."""
        return "\n".join(
            part for part in [self.bible, "\n".join(self.chapter_beats), self.handoff] if part.strip()
        )


TERMINAL_CHARS = set("。！？!?」』”’）》】]")
TRUNCATED_TAIL_CHARS = set("，,、；;：:—-")
PROMPT_ONLY_LINES = {
    "繁体中文",
    "繁體中文",
    "简体中文",
    "簡體中文",
    "正文",
    "最終正文",
    "最终正文",
}
PROMPT_LINE_PREFIXES = (
    "段落：",
    "段落:",
    "目前段落",
    "輸出語言",
    "输出语言",
    "改寫總要求",
    "全篇改寫要求",
    "硬性規則",
    "待改寫原文",
    "原文段落",
    "本地 LoRA 草稿",
    "本地LoRA草稿",
    "請輸出",
    "请输出",
    "工作資訊",
    "內部資訊",
    "内部信息",
    "目前故事脈絡",
    "故事脈絡",
    "本段操作",
    "本段改寫計畫",
    "主敘事模組",
    "全書骨幹",
    "各章脈絡",
    "接續要點",
    "目前章節",
    "本章一句話",
    "改寫診斷",
    "本段對應問題",
    "連貫要點",
    "全書診斷出",
)


def ensure_proxy_running() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((PROXY_HOST, PROXY_PORT)) == 0:
            return

    thread = threading.Thread(target=run_compat_proxy, daemon=True)
    thread.start()


def _is_proxy_url(base_url: str) -> bool:
    """True only when base_url points at the local compat proxy host:port."""
    normalized = (base_url or "").strip().lower().rstrip("/")
    return f"{PROXY_HOST}:{PROXY_PORT}".lower() in normalized


def get_client(api_key: str, base_url: str) -> OpenAI:
    if is_lora_base_url(base_url):
        ensure_lora_server_running()
    elif _is_proxy_url(base_url):
        # Only auto-start the local proxy when we are actually pointed at it.
        # A direct remote endpoint (e.g. DZMM) is reached straight through the SDK.
        ensure_proxy_running()
    return OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=900)


def read_text_file(path: str | Path) -> str:
    file_path = Path(path)
    encodings = ["utf-8-sig", "utf-8", "gb18030", "big5"]
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="replace")


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# Canonical key: script- and punctuation-agnostic, used for all dedup compares.
# ---------------------------------------------------------------------------

_CANON_CONVERTER = None
_CANON_TRIED = False
_PUNCT_RE = re.compile(
    r"[\s，。！？；：、…—－\-·.,!?;:\"'`“”‘’「」『』（）()《》〈〉【】\[\]{}~〜～*※•]"
)


def _get_canon_converter():
    """Lazily build the OpenCC converter used to canonicalize for comparison.

    We normalize toward Taiwan-traditional (``s2twp``). Unlike ``t2s`` (which
    leaves variant pairs such as 著/着 untouched and so fails to collapse twins),
    ``s2twp`` maps BOTH a simplified string and its traditional twin to the same
    traditional form, which is exactly what dedup comparison needs. It is also
    idempotent on already-traditional input, so applying it as a comparison key
    is stable regardless of the source script.
    """
    global _CANON_CONVERTER, _CANON_TRIED
    if _CANON_TRIED:
        return _CANON_CONVERTER
    _CANON_TRIED = True
    try:
        from opencc import OpenCC

        try:
            _CANON_CONVERTER = OpenCC("s2twp")
        except Exception:  # noqa: BLE001
            _CANON_CONVERTER = OpenCC("s2tw")
    except Exception:  # noqa: BLE001
        _CANON_CONVERTER = None
    return _CANON_CONVERTER


def canonical_key(text: str) -> str:
    """Normalize text for similarity comparison.

    Collapses simplified/traditional differences (so a simplified block and its
    traditional twin compare equal) and strips whitespace + punctuation so minor
    punctuation drift does not defeat dedup.
    """
    converter = _get_canon_converter()
    normalized = text
    if converter is not None:
        try:
            normalized = converter.convert(text)
        except Exception:  # noqa: BLE001
            normalized = text
    return _PUNCT_RE.sub("", normalized)


def normalize_blank_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def is_segment_marker(line: str) -> bool:
    compact = compact_text(line).replace("：", ":").replace("／", "/")
    if compact.startswith("目前段落"):
        compact = compact.replace("目前段落", "段落", 1)
    if not compact.startswith("段落"):
        return False
    rest = compact[len("段落") :]
    if rest.startswith(":"):
        rest = rest[1:]
    return bool(re.fullmatch(r"\d+/\d+", rest))


def is_prompt_residue_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in {"```", "```txt", "```text", "```markdown"}:
        return True
    if stripped in PROMPT_ONLY_LINES:
        return True
    if is_segment_marker(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in PROMPT_LINE_PREFIXES) and len(stripped) <= 120:
        return True
    return False


def clean_model_output(text: str) -> tuple[str, dict[str, int]]:
    removed_prompt_lines = 0
    kept_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        if is_prompt_residue_line(line):
            removed_prompt_lines += 1
            continue
        kept_lines.append(line.rstrip())

    cleaned = normalize_blank_lines("\n".join(kept_lines))
    return cleaned, {"removed_prompt_lines": removed_prompt_lines}


def looks_truncated(text: str) -> bool:
    stripped = compact_text(text)
    if len(stripped) < 80:
        return False
    if stripped[-1] in TERMINAL_CHARS:
        return False
    if stripped[-1] in TRUNCATED_TAIL_CHARS:
        return True
    return not any(stripped.endswith(char) for char in TERMINAL_CHARS)


def trim_repeated_prefix(text: str, previous_tail: str, min_overlap: int = 80, max_overlap: int = 700) -> tuple[str, int]:
    current = text.lstrip()
    previous = previous_tail.strip()
    if not current or not previous:
        return text, 0

    max_size = min(max_overlap, len(previous), len(current))
    for size in range(max_size, min_overlap - 1, -1):
        suffix = previous[-size:]
        if current.startswith(suffix):
            return current[size:].lstrip(), size

    first_paragraph, _, rest = current.partition("\n\n")
    if len(compact_text(first_paragraph)) >= min_overlap and first_paragraph in previous[-2500:]:
        return rest.lstrip(), len(first_paragraph)
    return text, 0


def clean_rewrite_candidate(text: str, previous_tail: str) -> tuple[str, dict[str, int]]:
    cleaned, stats = clean_model_output(text)
    cleaned, trimmed_chars = trim_repeated_prefix(cleaned, previous_tail)
    if trimmed_chars:
        stats["trimmed_repeated_prefix_chars"] = trimmed_chars
    cleaned = normalize_blank_lines(cleaned)
    stats["looks_truncated_after_cleanup"] = int(looks_truncated(cleaned))
    return cleaned, stats


def convert_output_language(text: str, output_language: str) -> tuple[str, str]:
    if "繁" not in output_language:
        return text, "skipped"
    try:
        from opencc import OpenCC

        try:
            converter = OpenCC("s2twp")
            return converter.convert(text), "opencc s2twp"
        except Exception:
            converter = OpenCC("s2tw")
            return converter.convert(text), "opencc s2tw"
    except Exception as exc:  # noqa: BLE001
        return text, f"opencc unavailable: {exc}"


# ---------------------------------------------------------------------------
# Dedup engine.
# ---------------------------------------------------------------------------


def _similar(key_a: str, key_b: str, threshold: float) -> bool:
    """Cheap-then-exact similarity test on canonical keys."""
    if not key_a or not key_b:
        return False
    longer = max(len(key_a), len(key_b))
    if longer == 0:
        return False
    # length gate: very different lengths cannot exceed the ratio threshold.
    if min(len(key_a), len(key_b)) / longer < threshold:
        return False
    matcher = SequenceMatcher(None, key_a, key_b)
    if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
        return False
    return matcher.ratio() >= threshold


def deduplicate_paragraphs(
    text: str,
    similarity: float = 0.9,
    min_key_chars: int = 24,
    window: int = 80,
) -> tuple[str, dict[str, int]]:
    """Remove exact (global) and near-duplicate (recent-window) paragraphs.

    Exact duplicates are matched anywhere in the document through the canonical
    key, which is why a simplified paste of an earlier traditional paragraph is
    caught. Near-duplicates are matched only against a sliding window of recently
    kept paragraphs to keep this close to linear on long manuscripts.
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    kept: list[str] = []
    kept_keys: list[str] = []
    removed_exact = 0
    removed_near = 0

    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            continue
        key = canonical_key(stripped)
        if len(key) < min_key_chars:
            kept.append(stripped)
            kept_keys.append(key)
            continue
        # Compare only against a recent window (nearest first). Both exact and
        # near duplicates are bounded by distance, so an intentional verbatim
        # refrain / epigraph repeated far apart is preserved, while a local
        # re-emit of the previous section's tail is removed.
        dup = None
        for prev_key in reversed(kept_keys[-window:]):
            if len(prev_key) < min_key_chars:
                continue
            if prev_key == key:
                dup = "exact"
                break
            if _similar(prev_key, key, similarity):
                dup = "near"
                break
        if dup == "exact":
            removed_exact += 1
            continue
        if dup == "near":
            removed_near += 1
            continue
        kept.append(stripped)
        kept_keys.append(key)

    result = "\n\n".join(part for part in kept if part).strip()
    return result, {
        "removed_exact_paragraphs": removed_exact,
        "removed_near_paragraphs": removed_near,
    }


def remove_repeated_long_lines(
    text: str,
    similarity: float = 0.85,
    min_chars: int = 30,
    window: int = 50,
) -> tuple[str, int]:
    """Remove a long line that repeats (verbatim or near-verbatim) a recent one.

    Targets the most common artifact in chunked rewriting: the model re-emits the
    tail of the previous section at the head of the next one — often with the
    script flipped (simplified vs traditional) and minor rewording. Block-level
    dedup misses this because the surrounding paragraph blocks differ in length;
    line-level comparison on canonical keys catches it. Short lines (dialogue,
    refrains) are left untouched.
    """
    removed = 0
    recent_keys: list[str] = []
    out_lines: list[str] = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        key = canonical_key(stripped)
        # Distance-bounded: only a repeat within the recent window is removed, so
        # a far-apart intentional echo (refrain, bookended line) is preserved.
        # The window advances on EVERY kept non-blank line (short lines included)
        # so distance reflects real text length, not just the count of long lines.
        if len(key) >= min_chars and any(
            len(prev) >= min_chars and (prev == key or _similar(prev, key, similarity))
            for prev in recent_keys[-window:]
        ):
            removed += 1
            continue
        recent_keys.append(key)
        out_lines.append(line)
    return "\n".join(out_lines), removed


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?…])")


def collapse_repeated_sentences(text: str, similarity: float = 0.95) -> tuple[str, int]:
    """Drop a sentence that near-repeats the sentence immediately before it.

    Conservative on purpose (adjacent only, high threshold) so that intentional
    repetition for rhythm is left alone while LLM stutters are removed.
    """
    removed = 0
    out_paragraphs: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s]
        if len(sentences) <= 1:
            out_paragraphs.append(paragraph)
            continue
        kept_sentences: list[str] = []
        prev_key = ""
        for sentence in sentences:
            key = canonical_key(sentence)
            if key and len(key) >= 12 and prev_key and _similar(prev_key, key, similarity):
                removed += 1
                continue
            kept_sentences.append(sentence)
            prev_key = key
        out_paragraphs.append("".join(kept_sentences))
    return "\n\n".join(out_paragraphs), removed


_BULLET_STRIP = "-•·–—*●○~ \t　"


def strip_context_echo(text: str, digest: str, similarity: float = 0.92, min_chars: int = 12) -> tuple[str, int]:
    """Drop output lines that near-verbatim echo a line of the injected digest.

    The story digest is internal scaffolding injected into prompts; if a draft
    model parrots its bullets into the body, only the label line was being
    filtered. This removes body lines that closely match a digest line. The
    threshold is deliberately high so genuine prose that merely paraphrases the
    summary is preserved (paraphrase is legitimate content, not leakage).
    """
    if not digest.strip():
        return text, 0
    digest_keys: list[str] = []
    for dline in digest.splitlines():
        key = canonical_key(dline.strip().strip(_BULLET_STRIP))
        if len(key) >= min_chars:
            digest_keys.append(key)
    if not digest_keys:
        return text, 0

    removed = 0
    out_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        key = canonical_key(stripped.strip(_BULLET_STRIP))
        if len(key) >= min_chars and any(dk == key or _similar(dk, key, similarity) for dk in digest_keys):
            removed += 1
            continue
        out_lines.append(line)
    return "\n".join(out_lines), removed


def scan_duplicate_windows(text: str) -> dict[str, int]:
    """Report (do not remove) residual repetition, using canonical keys."""
    lines = [line for line in text.splitlines() if compact_text(line)]
    long_line_seen: set[str] = set()
    long_line_dups = 0
    normalized_lines: list[str] = []
    for line in lines:
        key = canonical_key(line)
        normalized_lines.append(key)
        if len(key) < 60:
            continue
        if key in long_line_seen:
            long_line_dups += 1
        else:
            long_line_seen.add(key)

    window_seen: set[str] = set()
    window_dups = 0
    for index in range(max(0, len(normalized_lines) - 2)):
        key = "".join(normalized_lines[index : index + 3])
        if len(key) < 150:
            continue
        if key in window_seen:
            window_dups += 1
        else:
            window_seen.add(key)
    return {
        "exact_long_line_duplicates": long_line_dups,
        "three_line_window_duplicates": window_dups,
    }


def postprocess_full_output(text: str, settings: RewriteSettings) -> tuple[str, dict[str, object]]:
    cleaned, clean_stats = clean_model_output(text)
    # Unify the script first: for a 繁體 target this maps simplified and
    # traditional twins to the same form, so the dedup passes below can actually
    # recognize them as duplicates.
    converted, converter = convert_output_language(cleaned, settings.output_language)
    deduped, dedup_stats = deduplicate_paragraphs(converted, similarity=settings.dedup_similarity)
    line_similarity = max(0.80, settings.dedup_similarity - 0.05)
    deduped, removed_lines = remove_repeated_long_lines(deduped, similarity=line_similarity)
    deduped, collapsed_sentences = collapse_repeated_sentences(deduped)
    final_text = normalize_blank_lines(deduped)

    validation_patterns = {
        "segment_markers_remaining": r"段落[:：]?\s*\d+\s*[/／]\s*\d+",
        "progress_logs_remaining": r"(?:正在|完成)第\s*\d+\s*[/／]\s*\d+\s*段",
        "prompt_language_remaining": r"繁[体體]中文|輸出語言|输出语言",
        "html_like_remaining": r"<[^>]+>",
    }
    validation = {name: len(re.findall(pattern, final_text)) for name, pattern in validation_patterns.items()}
    validation.update(scan_duplicate_windows(final_text))
    validation["looks_truncated_final"] = int(looks_truncated(final_text))

    report: dict[str, object] = {
        "postprocess": clean_stats,
        "removed_exact_paragraphs": dedup_stats["removed_exact_paragraphs"],
        "removed_near_paragraphs": dedup_stats["removed_near_paragraphs"],
        "removed_repeated_lines": removed_lines,
        "collapsed_repeated_sentences": collapsed_sentences,
        "language_conversion": converter,
        "canonical_opencc_active": _get_canon_converter() is not None,
        "validation": validation,
        "final_lines": len(final_text.splitlines()),
        "final_chars": len(final_text),
    }
    return final_text, report


def write_quality_report(path: Path, report: dict[str, object]) -> None:
    validation = report.get("validation", {})
    lines = [
        "Rewrite QA report",
        "=================",
        "",
        "Postprocess",
        f"- Removed prompt residue lines: {report.get('postprocess', {}).get('removed_prompt_lines', 0)}",
        f"- Removed exact duplicate paragraphs: {report.get('removed_exact_paragraphs', 0)}",
        f"- Removed near-duplicate paragraphs: {report.get('removed_near_paragraphs', 0)}",
        f"- Removed repeated long lines: {report.get('removed_repeated_lines', 0)}",
        f"- Collapsed repeated sentences: {report.get('collapsed_repeated_sentences', 0)}",
        f"- Language conversion: {report.get('language_conversion', 'unknown')}",
        f"- Canonical OpenCC active (simp/trad dedup): {report.get('canonical_opencc_active', 'unknown')}",
        f"- Chapters detected: {report.get('chapters_detected', 'unknown')}",
        f"- Story context updates: {report.get('story_context_updates', 0)}",
        f"- Narrative spine updates: {report.get('spine_updates', 0)}",
        f"- Final lines: {report.get('final_lines', 0)}",
        f"- Final chars: {report.get('final_chars', 0)}",
        "",
        "Validation",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8-sig")


def load_reference_text(files: Iterable[str | Path] | None, max_chars: int = 6000) -> str:
    if not files:
        return ""

    parts: list[str] = []
    remaining = max_chars
    for file_path in files:
        if remaining <= 0:
            break
        try:
            text = read_text_file(file_path).strip()
        except Exception:
            continue
        if not text:
            continue
        piece = text[:remaining]
        parts.append(piece)
        remaining -= len(piece)
    return "\n\n".join(parts)


def get_strength_rules(rewrite_strength: str) -> str:
    if rewrite_strength == "保守潤色":
        return (
            "改寫強度：保守潤色。保留原段落功能與事件順序，但仍要重寫句子，"
            "避免只做簡繁轉換或同義詞替換。"
        )
    if rewrite_strength == "重構改寫":
        return (
            "改寫強度：重構改寫。可以調整段落切入角度、描寫順序、句式節奏與場景推進，"
            "只保留核心事件、人物關係與必要設定。"
        )
    if rewrite_strength == "標準改寫":
        return (
            "改寫強度：標準改寫。保留劇情資訊，但必須重組語句、節奏與描寫方式，"
            "讓成品像重新寫過，而不是原文翻譯。"
        )
    return (
        "改寫強度：強改寫。保留核心劇情與人物關係，但要大幅重寫句式、段落節奏、"
        "描寫角度、比喻和轉場。不要逐句跟隨原文。"
    )


def get_operation_rules(operation: str) -> str:
    if operation == OP_EXPAND:
        return (
            "本段操作：增補擴寫。在不改變主線事件與人物關係的前提下，擴充感官細節、"
            "環境氛圍、人物心理與動作層次，讓段落更飽滿。成品篇幅應明顯多於原文"
            "（約 1.3 至 1.7 倍），但新增內容必須服務既有劇情，不可硬湊字數、"
            "不可重複既有描寫、不可加入會破壞後文的新設定或新角色。"
        )
    if operation == OP_CONDENSE:
        return (
            "本段操作：精簡刪減。保留所有關鍵事件、轉折與必要資訊，刪去冗詞、重複描寫、"
            "過度鋪陳與無資訊量的句子，讓敘事更緊湊有力。成品篇幅應明顯少於原文"
            "（約 0.5 至 0.7 倍），但不可遺漏劇情、人物動機或重要伏筆。"
        )
    return (
        "本段操作：改寫。維持與原文相近的篇幅與資訊量，專注重組句式、敘事節奏與"
        "描寫角度，使其像重新寫過。"
    )


def operation_token_scale(operation: str) -> float:
    if operation == OP_EXPAND:
        return 1.7
    if operation == OP_CONDENSE:
        # Lower the ceiling so the budget pushes toward the 0.5-0.7x target
        # instead of leaving the model free to expand. The 300-token floor in
        # resolve_max_tokens keeps it from going too low.
        return 0.65
    return 1.0


def resolve_max_tokens(settings: RewriteSettings, base: int | None = None, cap: int | None = None) -> int:
    base_tokens = settings.max_tokens if base is None else base
    scaled = int(round(base_tokens * operation_token_scale(settings.operation)))
    scaled = max(scaled, 300)
    if cap is not None:
        scaled = min(scaled, cap)
    return scaled


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[。！？!?；;])", paragraph)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(buffer) + len(sentence) <= max_chars:
            buffer += sentence
            continue
        if buffer:
            chunks.append(buffer.strip())
        while len(sentence) > max_chars:
            chunks.append(sentence[:max_chars].strip())
            sentence = sentence[max_chars:]
        buffer = sentence
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def split_text(text: str, max_chars: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    max_chars = max(600, int(max_chars))
    # Split into "scene segments" first: a run of 3+ newlines marks a deliberate
    # scene break, which we treat as a hard chunk boundary so the structural cue is
    # never silently merged into the middle of a chunk. Within a scene, paragraphs
    # are split on ordinary blank lines as before.
    scene_segments = re.split(r"\n{3,}", normalized)
    # (unit_text, starts_scene): the flag is True for the first unit after a scene
    # break, telling the packer to flush the current chunk at that boundary.
    units: list[tuple[str, bool]] = []
    for seg_index, segment in enumerate(scene_segments):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", segment) if part.strip()]
        for para_index, paragraph in enumerate(paragraphs):
            for sub_index, piece in enumerate(split_long_paragraph(paragraph, max_chars)):
                starts_scene = seg_index > 0 and para_index == 0 and sub_index == 0
                units.append((piece, starts_scene))

    chunks: list[str] = []
    buffer = ""
    for unit, starts_scene in units:
        # A scene break flushes the current chunk so it does not straddle the break.
        if starts_scene and buffer:
            chunks.append(buffer)
            buffer = ""
        candidate = (buffer + "\n\n" + unit).strip() if buffer else unit
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        buffer = unit
    if buffer:
        chunks.append(buffer)
    return chunks


# Chapter heading detection: "第N章/回/卷/節...", 序章/楔子/尾聲/番外, "Chapter N".
_CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9零〇一二三四五六七八九十百千兩两]+\s*[章回卷折幕節节集部篇]"
    r"|卷\s*[0-9零〇一二三四五六七八九十百千]+"
    r"|序章|序幕|楔子|引子|前言|尾聲|尾声|終章|终章|後記|后记|番外(?:篇)?|外傳|外传"
    r"|Chapter\s+\d+|CHAPTER\s+\d+|Prologue|Epilogue"
    r")",
    re.IGNORECASE,
)


def is_chapter_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return False
    return bool(_CHAPTER_RE.match(stripped))


def split_into_chapters(text: str) -> list[tuple[str, str]]:
    """Split a manuscript into (title, body) chapters by heading lines.

    Text before the first heading becomes an untitled leading chapter. When no
    heading is found, returns a single untitled chapter holding the whole text,
    so chapterless manuscripts behave exactly as before.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    chapters: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []
    started = False
    for line in lines:
        if is_chapter_heading(line):
            if started or any(part.strip() for part in current_body):
                chapters.append((current_title, current_body))
            current_title = line.strip()
            current_body = []
            started = True
        else:
            current_body.append(line)
    if started or any(part.strip() for part in current_body):
        chapters.append((current_title, current_body))
    if not chapters:
        return [("", normalized.strip())]
    return [(title, "\n".join(body).strip()) for title, body in chapters]


@dataclass
class Chunk:
    text: str
    chapter_index: int
    chapter_title: str
    is_chapter_start: bool
    is_chapter_end: bool
    index_in_chapter: int


def split_text_with_chapters(text: str, max_chars: int) -> tuple[list[Chunk], int]:
    """Chunk the text chapter by chapter, tagging each chunk with its chapter.

    Returns (chunks, chapter_count). The chapter title is folded into the start
    of each chapter so it is rewritten in context, and chunk boundaries never
    cross a chapter boundary.
    """
    chapters = split_into_chapters(text)
    result: list[Chunk] = []
    for chapter_index, (title, body) in enumerate(chapters):
        chapter_text = (title + "\n\n" + body).strip() if title else body
        pieces = split_text(chapter_text, max_chars)
        if not pieces:
            continue
        last = len(pieces) - 1
        for j, piece in enumerate(pieces):
            result.append(
                Chunk(
                    text=piece,
                    chapter_index=chapter_index,
                    chapter_title=title,
                    is_chapter_start=(j == 0),
                    is_chapter_end=(j == last),
                    index_in_chapter=j,
                )
            )
    return result, len(chapters)


def is_transient_api_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "502",
            "bad gateway",
            "cloudflare",
            "timeout",
            "timed out",
            "connection reset",
            "temporarily",
            "service unavailable",
        ]
    )


def api_extra_body(settings: RewriteSettings) -> dict[str, object] | None:
    """Non-standard params for OpenAI-compatible backends (e.g. DZMM).

    DZMM/gpt4novel accepts ``repetition_penalty`` which also helps suppress
    repetition. Sent via the SDK's ``extra_body`` so the standard schema is
    untouched. Only applied to the external-API path (the local LoRA path keeps
    its known-good request shape).
    """
    if settings.repetition_penalty and settings.repetition_penalty > 0:
        return {"repetition_penalty": settings.repetition_penalty}
    return None


def _create_completion(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    extra_body: dict[str, object] | None,
    retries: int,
):
    """Call chat.completions.create with retry + transient-error backoff.

    Shared by the main request and the length-continuation request so both get
    identical retry semantics instead of the continuation call being unguarded.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,  # DZMM defaults to SSE when omitted; force a JSON body.
                extra_body=extra_body,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised unless transient + retries left.
            last_exc = exc
            if attempt >= retries or not is_transient_api_error(exc):
                raise
            time.sleep(3 * (attempt + 1))
    # Unreachable in practice (the loop returns or raises), but keep a definite
    # error so the function never falls through to return None.
    raise last_exc or RuntimeError("Model request failed.")


def _response_content(response) -> tuple[str, str | None]:
    """Safely pull (content, finish_reason) out of a completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Model response contained no choices.")
    choice = choices[0]
    if choice is None:
        raise ValueError("Model response choice was empty.")
    message = getattr(choice, "message", None)
    content = (getattr(message, "content", None) or "") if message is not None else ""
    finish_reason = getattr(choice, "finish_reason", None)
    return content.strip(), finish_reason


def chat_text(
    *,
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    settings: RewriteSettings,
    max_tokens: int | None = None,
    temperature: float | None = None,
    retries: int = 2,
    continue_on_length: bool = False,
    extra_body: dict[str, object] | None = None,
) -> str:
    resolved_temperature = settings.temperature if temperature is None else temperature
    resolved_max_tokens = max_tokens or settings.max_tokens
    response = _create_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=resolved_temperature,
        top_p=settings.top_p,
        max_tokens=resolved_max_tokens,
        extra_body=extra_body,
        retries=retries,
    )
    content, finish_reason = _response_content(response)
    if not content:
        # An empty body would silently drop a chunk / digest downstream; surface it
        # so callers (which catch and fall back) can react instead of producing a gap.
        raise ValueError(f"Model returned empty content (finish_reason={finish_reason}).")

    if continue_on_length and content and (finish_reason == "length" or looks_truncated(content)):
        continuation_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "剛才輸出停在半句。請只從最後一句之後自然接續，把本段收束到完整句子。"
                    "不要重複已輸出內容，不要輸出段落標籤、語言標籤或說明。"
                ),
            },
        ]
        try:
            continuation = _create_completion(
                client=client,
                model=model,
                messages=continuation_messages,
                temperature=resolved_temperature,
                top_p=settings.top_p,
                max_tokens=min(resolved_max_tokens, 400),
                extra_body=extra_body,
                retries=retries,
            )
            extra, _ = _response_content(continuation)
            extra, _ = clean_model_output(extra)
            if extra:
                # The continuation often re-emits the truncated tail before
                # continuing. Trim that overlap so the sentence is stitched
                # seamlessly instead of leaving a duplicated fragment.
                extra, overlap = trim_repeated_prefix(extra, content, min_overlap=8, max_overlap=400)
                if extra:
                    separator = "" if overlap else "\n"
                    content = normalize_blank_lines(content + separator + extra)
        except Exception as exc:  # noqa: BLE001 - fall back to the truncated body.
            logger.warning("Continuation request failed: %s", exc)
    return content


def build_rewrite_user_prompt(
    *,
    source_chunk: str,
    chunk_index: int,
    chunks_total: int,
    settings: RewriteSettings,
    previous_tail: str,
    story_context: str = "",
    plan: str = "",
) -> str:
    style_block = ""
    if settings.style_reference.strip():
        style_block = "\n\n參考風格，不要照抄，只吸收語感：\n" + settings.style_reference.strip()[:5000]

    notes_block = ""
    if settings.continuity_notes.strip():
        notes_block = "\n\n全篇設定與連貫要求：\n" + settings.continuity_notes.strip()

    context_block = story_context if story_context else ""

    previous_block = ""
    if previous_tail.strip():
        previous_block = "\n\n上一段改寫結尾，用於銜接，不要重複：\n" + previous_tail.strip()[-600:]

    plan_block = ""
    if plan.strip():
        plan_block = "\n\n本段改寫計畫：\n" + plan.strip()

    return f"""你正在做長篇小說的全篇改寫。
工作資訊（嚴禁輸出到正文）：第 {chunk_index} 段，共 {chunks_total} 段；目標語言為 {settings.output_language}。
改寫總要求（嚴禁原樣輸出）：{settings.instruction.strip() or "保留事件與人物關係，提升文筆、節奏、畫面感與可讀性。"}
{get_strength_rules(settings.rewrite_strength)}
{get_operation_rules(settings.operation)}
{notes_block}{context_block}{style_block}{previous_block}{plan_block}

硬性規則：
1. 只輸出改寫後正文，不要解釋，不要列點。
2. 不要連續照搬原文超過 12 個中文字；專有名詞、稱謂、固定設定例外。
3. 不要只做簡繁轉換、同義詞替換或局部潤飾；必須重組句式與敘事節奏。
4. 保留原文主要事件、角色關係、必要資訊與場景功能。
5. 嚴格遵守本段操作（改寫／增補／刪減）的篇幅與重點要求。
6. 嚴格遵守主敘事模組（全書骨幹）：人物、稱謂、設定、主線與世界觀必須一致，不可與骨幹矛盾。
7. 善用故事脈絡與「接續要點」，自然延續本章前文與前一章結尾；不要重複已標記用過的描寫與意象。
8. 若有「改寫診斷」，依其指出的問題與本段對應問題一併修正（連貫矛盾、人物失常、重複冗長、邏輯硬傷等），但不可改變主線事件。
9. 不要新增會破壞主線或後文連貫的大設定、新角色或新支線。
10. 若原文是章節標題或短段落，保留其功能並自然改寫。
11. 不得輸出「段落」「目前段落」「輸出語言」「繁體中文」「故事脈絡」「主敘事模組」「接續要點」「改寫診斷」等提示標籤。
12. 不要重複上一段結尾，不要重啟已經完成的動作。
13. 如果上下文不足，依主敘事模組保守銜接，不要憑空捏造與骨幹衝突的內容。
14. 結尾必須停在完整句子，不要停在半句、逗號或未完成引號。

待改寫原文：
{source_chunk}"""


def plan_chunk(
    *,
    api_client: OpenAI,
    source_chunk: str,
    chunk_index: int,
    chunks_total: int,
    settings: RewriteSettings,
    previous_tail: str,
    story_context: str = "",
) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是長篇小說改寫導演。只產生簡短改寫計畫，不寫正文。",
        },
        {
            "role": "user",
            "content": build_rewrite_user_prompt(
                source_chunk=source_chunk,
                chunk_index=chunk_index,
                chunks_total=chunks_total,
                settings=settings,
                previous_tail=previous_tail,
                story_context=story_context,
            )
            + "\n\n請輸出本段改寫計畫，包含保留事件、情緒走向、銜接注意事項，以及依本段操作要增補或刪減的重點。",
        },
    ]
    return chat_text(
        client=api_client,
        model=settings.api_model,
        messages=messages,
        settings=settings,
        max_tokens=450,
        temperature=min(settings.temperature, 0.55),
        extra_body=api_extra_body(settings),
    )


def rewrite_with_lora(
    *,
    lora_client: OpenAI,
    source_chunk: str,
    chunk_index: int,
    chunks_total: int,
    settings: RewriteSettings,
    previous_tail: str,
    story_context: str = "",
    plan: str = "",
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是本地 LoRA 小說改寫器。吸收訓練出的語感，專注輸出自然、連貫、具畫面感的正文。"
                "不要分析，不要解釋。"
            ),
        },
        {
            "role": "user",
            "content": build_rewrite_user_prompt(
                source_chunk=source_chunk,
                chunk_index=chunk_index,
                chunks_total=chunks_total,
                settings=settings,
                previous_tail=previous_tail,
                story_context=story_context,
                plan=plan,
            ),
        },
    ]
    lora_cap = LORA_EXPAND_MAX_TOKENS if settings.operation == OP_EXPAND else LORA_SAFE_MAX_TOKENS
    return chat_text(
        client=lora_client,
        model=LORA_MODEL_NAME,
        messages=messages,
        settings=settings,
        max_tokens=resolve_max_tokens(settings, cap=lora_cap),
        continue_on_length=True,
    )


def rewrite_with_api(
    *,
    api_client: OpenAI,
    source_chunk: str,
    chunk_index: int,
    chunks_total: int,
    settings: RewriteSettings,
    previous_tail: str,
    story_context: str = "",
    plan: str = "",
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "你是長篇小說改寫作者。保持故事資訊與人物連貫，將原文改寫為成熟、流暢、可直接連載的正文。"
                "只輸出正文。"
            ),
        },
        {
            "role": "user",
            "content": build_rewrite_user_prompt(
                source_chunk=source_chunk,
                chunk_index=chunk_index,
                chunks_total=chunks_total,
                settings=settings,
                previous_tail=previous_tail,
                story_context=story_context,
                plan=plan,
            ),
        },
    ]
    return chat_text(
        client=api_client,
        model=settings.api_model,
        messages=messages,
        settings=settings,
        max_tokens=resolve_max_tokens(settings),
        continue_on_length=True,
        extra_body=api_extra_body(settings),
    )


def polish_with_api(
    *,
    api_client: OpenAI,
    source_chunk: str,
    draft: str,
    chunk_index: int,
    chunks_total: int,
    settings: RewriteSettings,
    previous_tail: str,
    story_context: str = "",
) -> str:
    context_block = ("\n" + story_context.strip() + "\n") if story_context.strip() else ""
    messages = [
        {
            "role": "system",
            "content": (
                "你是長篇小說校稿與潤色編輯。保留草稿的風格和事件，只修正連貫、節奏、語句與前後銜接，"
                "並消除與前文重複的描寫。只輸出最終正文。"
            ),
        },
        {
            "role": "user",
            "content": f"""工作資訊（嚴禁輸出到正文）：第 {chunk_index} 段，共 {chunks_total} 段；目標語言為 {settings.output_language}。
全篇改寫要求（嚴禁原樣輸出）：{settings.instruction.strip() or "保留事件並提升文筆。"}
{get_strength_rules(settings.rewrite_strength)}
{get_operation_rules(settings.operation)}
{context_block}
上一段改寫結尾：
{previous_tail[-600:] if previous_tail else ""}

原文段落：
{source_chunk}

本地 LoRA 草稿：
{draft}

請輸出校稿後的最終正文。維持本段操作的篇幅要求；不得輸出段落標籤、語言標籤或說明；不得重複上一段結尾或故事脈絡中已用過的描寫；結尾必須是完整句。""",
        },
    ]
    return chat_text(
        client=api_client,
        model=settings.api_model,
        messages=messages,
        settings=settings,
        max_tokens=resolve_max_tokens(settings),
        temperature=min(settings.temperature, 0.7),
        continue_on_length=True,
        extra_body=api_extra_body(settings),
    )


def update_story_context(
    *,
    client: OpenAI | None,
    model: str,
    max_tokens: int,
    settings: RewriteSettings,
    story: StoryContext,
    rewritten: str,
    chunk_index: int,
    chunks_total: int,
) -> None:
    """Refresh the rolling digest after a chunk. Best-effort: never fatal."""
    if not settings.use_story_context or client is None or not rewritten.strip():
        return
    messages = [
        {
            "role": "system",
            "content": (
                "你是長篇小說的連續性編輯。維護一份精簡的故事脈絡摘要，供後續段落保持一致並避免重複。"
                "只輸出摘要本身。"
            ),
        },
        {
            "role": "user",
            "content": f"""目前故事脈絡摘要（可能為空）：
{story.digest or "（尚無）"}

剛完成第 {chunk_index}/{chunks_total} 段，內容如下：
{rewritten[-1600:]}

請整合上述內容，更新並輸出新的故事脈絡摘要，使用 {settings.output_language}，控制在 500 字內，條列：
- 主要人物與當前狀態、關係
- 目前場景、地點、時間線
- 已發生的關鍵事件（累積式精簡，不要逐段流水帳）
- 尚未解決的伏筆或懸念
- 已大量使用、後文應避免重複的描寫與意象
只輸出摘要，不要加任何說明或標題。""",
        },
    ]
    try:
        digest = chat_text(
            client=client,
            model=model,
            messages=messages,
            settings=settings,
            max_tokens=max_tokens,
            temperature=min(settings.temperature, 0.35),
        )
        digest, _ = clean_model_output(digest)
        digest = digest.strip()
        if digest:
            story.digest = digest[:1600]
            story.updates += 1
    except Exception as exc:  # noqa: BLE001 - best-effort digest refresh, never fatal.
        logger.error("Story context update failed: %s", exc, exc_info=True)


_SPINE_SECTION_RE = re.compile(r"【\s*(全書骨幹|本章一句話|接續要點)\s*】")


def _parse_spine_output(text: str) -> tuple[str, str, str]:
    """Parse the three labelled sections out of a spine-update response.

    Lenient: if the model ignored the format and returned unstructured text, the
    whole thing is treated as the backbone so nothing is lost.
    """
    parts = _SPINE_SECTION_RE.split(text.replace("\r\n", "\n"))
    sections = {"全書骨幹": "", "本章一句話": "", "接續要點": ""}
    matched = False
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        if name in sections:
            sections[name] = parts[i + 1].strip()
            matched = True
    bible = sections["全書骨幹"].strip()
    if not matched and text.strip():
        bible = text.strip()
    return bible, sections["本章一句話"].strip(), sections["接續要點"].strip()


def update_narrative_spine(
    *,
    client: OpenAI | None,
    model: str,
    settings: RewriteSettings,
    spine: NarrativeSpine,
    chapter_index: int,
    chapter_title: str,
    chapter_text: str,
    rolling_digest: str,
) -> None:
    """Refresh the 主敘事模組 backbone + handoff. Best-effort: never fatal.

    Called at chapter boundaries (or every few chunks for chapterless texts), so
    it stays consistent and cheap rather than churning every chunk.
    """
    if client is None or not chapter_text.strip():
        return
    title = f"《{chapter_title}》" if chapter_title else ""
    messages = [
        {
            "role": "system",
            "content": (
                "你是長篇小說的總編輯，負責維護全書骨幹與章節之間的連貫，不寫正文。"
                "只輸出指定格式的結構化內容。"
            ),
        },
        {
            "role": "user",
            "content": f"""目前的全書骨幹（可能為空）：
{spine.bible or "（尚無）"}

最近完成的是第 {chapter_index + 1} 章{title}，其改寫後內容（摘錄）：
{chapter_text[-2200:]}

目前滾動脈絡（近期細節，可能為空）：
{rolling_digest or "（無）"}

請依下列格式輸出三個區塊，使用 {settings.output_language}：
【全書骨幹】
更新後的全書骨幹，累積精簡，控制在 700 字內，條列：核心前提與主線方向、主要角色與其目標/關係/稱謂、世界觀與重要設定、中心衝突與主題、尚未解決的主要伏筆。可改寫舊內容使其更準確，但不可遺漏既定設定。
【本章一句話】
用一句話總結本章在主線上推進了什麼。
【接續要點】
本章結尾的地點、時間與人物當下狀態，以及下一章開頭必須延續或回應的事，最多 3 條。
只輸出這三個區塊，不要其他說明。""",
        },
    ]
    try:
        out = chat_text(
            client=client,
            model=model,
            messages=messages,
            settings=settings,
            max_tokens=1000,
            temperature=min(settings.temperature, 0.35),
            extra_body=api_extra_body(settings),
        )
        bible, beat, handoff = _parse_spine_output(out)
        if bible:
            if len(bible) > 2200:
                logger.warning("Spine bible truncated from %d to 2200 chars", len(bible))
            spine.bible = bible[:2200]
        if beat:
            spine.add_beat(f"第{chapter_index + 1}章：{beat}")
        if len(handoff) > 900:
            logger.warning("Spine handoff truncated from %d to 900 chars", len(handoff))
        spine.handoff = handoff[:900]
        spine.updates += 1
    except Exception as exc:  # noqa: BLE001 - best-effort backbone refresh, never fatal.
        logger.error("Narrative spine update failed: %s", exc, exc_info=True)


def rewrite_chunk(
    *,
    source_chunk: str,
    chunk_index: int,
    chunks_total: int,
    settings: RewriteSettings,
    api_client: OpenAI | None,
    lora_client: OpenAI | None,
    previous_tail: str,
    story_context: str = "",
) -> tuple[str, dict[str, object]]:
    info: dict[str, object] = {
        "mode": settings.mode,
        "operation": settings.operation,
        # Always present so every chunk log entry has a uniform shape, even when
        # the spine does not refresh on this chunk (or its refresh fails).
        "spine_updates": 0,
    }

    def finalize(text: str, *, text_key: str, cleanup_key: str) -> str:
        """Clean a candidate, record the result + stats in ``info``, return text.

        Centralizes the clean-and-log step the four mode paths share; the caller
        keeps its own return / control flow, so semantics are unchanged.
        """
        cleaned, cleanup = clean_rewrite_candidate(text, previous_tail)
        info[text_key] = cleaned
        info[cleanup_key] = cleanup
        return cleaned

    if settings.mode == MODE_LORA:
        if lora_client is None:
            raise RuntimeError("Local LoRA client is not available.")
        rewritten = rewrite_with_lora(
            lora_client=lora_client,
            source_chunk=source_chunk,
            chunk_index=chunk_index,
            chunks_total=chunks_total,
            settings=settings,
            previous_tail=previous_tail,
            story_context=story_context,
        )
        rewritten = finalize(rewritten, text_key="draft", cleanup_key="cleanup")
        return rewritten, info

    if settings.mode == MODE_API:
        if api_client is None:
            raise RuntimeError("External API client is not available.")
        rewritten = rewrite_with_api(
            api_client=api_client,
            source_chunk=source_chunk,
            chunk_index=chunk_index,
            chunks_total=chunks_total,
            settings=settings,
            previous_tail=previous_tail,
            story_context=story_context,
        )
        rewritten = finalize(rewritten, text_key="draft", cleanup_key="cleanup")
        return rewritten, info

    if api_client is None or lora_client is None:
        raise RuntimeError("Hybrid mode needs both external API and local LoRA.")

    plan = ""
    if settings.mode == MODE_FULL_HYBRID:
        try:
            plan = plan_chunk(
                api_client=api_client,
                source_chunk=source_chunk,
                chunk_index=chunk_index,
                chunks_total=chunks_total,
                settings=settings,
                previous_tail=previous_tail,
                story_context=story_context,
            )
            info["plan"] = plan
        except Exception as exc:  # noqa: BLE001 - planning is optional; fall back to no plan.
            logger.error("Plan generation failed: %s", exc, exc_info=True)
            info["plan_error"] = str(exc)
            plan = ""

    draft = rewrite_with_lora(
        lora_client=lora_client,
        source_chunk=source_chunk,
        chunk_index=chunk_index,
        chunks_total=chunks_total,
        settings=settings,
        previous_tail=previous_tail,
        story_context=story_context,
        plan=plan,
    )
    draft = finalize(draft, text_key="draft", cleanup_key="draft_cleanup")
    try:
        final = polish_with_api(
            api_client=api_client,
            source_chunk=source_chunk,
            draft=draft,
            chunk_index=chunk_index,
            chunks_total=chunks_total,
            settings=settings,
            previous_tail=previous_tail,
            story_context=story_context,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to the LoRA draft on polish failure.
        logger.error("Polish API call failed: %s", exc, exc_info=True)
        info["polish_error"] = str(exc)
        info["fallback"] = "Used local LoRA draft because external API polish failed."
        return draft, info
    final = finalize(final, text_key="final", cleanup_key="final_cleanup")
    return final or draft, info


def full_rewrite(
    *,
    source_text: str,
    settings: RewriteSettings,
    progress: Callable[[str], None] | None = None,
) -> RewriteResult:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks, chapter_count = split_text_with_chapters(source_text, settings.chunk_chars)
    if not chunks:
        raise ValueError("改寫目標是空的。")

    api_client = None
    lora_client = None
    if settings.mode in {MODE_API, MODE_HYBRID, MODE_FULL_HYBRID}:
        api_client = get_client(resolve_api_key(settings.api_key), settings.api_base_url)
    if settings.mode in {MODE_LORA, MODE_HYBRID, MODE_FULL_HYBRID}:
        lora_client = get_client("not-needed", LORA_BASE_URL)

    # The digest and the narrative spine are maintained ONLY by the external API.
    # A fiction LoRA prompted to "summarize" tends to emit narrative continuation,
    # which would pollute the injected context. So in pure-LoRA mode we keep the
    # static seed (continuity notes) but skip model-based context updates.
    summary_client, summary_model, summary_tokens = api_client, settings.api_model, 600

    story = StoryContext()
    spine = NarrativeSpine()
    # Seed the backbone from (a) the Grok diagnosis spine, then (b) author notes.
    seed_parts: list[str] = []
    if settings.diagnosis_spine_seed.strip():
        seed_parts.append(settings.diagnosis_spine_seed.strip())
    if settings.continuity_notes.strip():
        seed_parts.append("（作者提供的固定設定）\n" + settings.continuity_notes.strip())
    if seed_parts:
        spine.bible = "\n\n".join(seed_parts)

    # For chapterless manuscripts the spine would otherwise never refresh, so also
    # refresh it on a chunk interval.
    spine_interval = 6
    chapter_continuity_active = settings.chapter_continuity and summary_client is not None

    started_at = datetime.now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"rewrite_{timestamp}.txt"
    log_path = OUTPUT_DIR / f"rewrite_{timestamp}.json"
    report_path = OUTPUT_DIR / f"rewrite_{timestamp}_qa.txt"

    outputs: list[str] = []
    chunk_logs: list[dict[str, object]] = []
    chapter_outputs: list[str] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if chunk.is_chapter_start:
            chapter_outputs = []
        chapter_label = ""
        if chapter_count > 1 or chunk.chapter_title:
            title = f"「{chunk.chapter_title}」" if chunk.chapter_title else ""
            position = "（本章開頭）" if chunk.is_chapter_start else ("（本章結尾）" if chunk.is_chapter_end else "")
            chapter_label = (
                f"\n\n目前章節（內部參考，嚴禁輸出到正文）：第 {chunk.chapter_index + 1} 章{title}，"
                f"本章第 {chunk.index_in_chapter + 1} 段{position}。請延續本章前文與前一章結尾。"
            )

        # Grok-diagnosis brief: global problems to fix + this chunk's local problems.
        diagnosis_block = ""
        if settings.diagnosis_brief.strip():
            diagnosis_block = "\n\n改寫診斷（內部參考，嚴禁輸出到正文）：\n" + settings.diagnosis_brief.strip()
            window_problems = settings.diagnosis_window_problems
            if window_problems:
                w = len(window_problems)
                widx = min(w - 1, (idx - 1) * w // max(1, total))
                local = window_problems[widx].strip()
                if local:
                    diagnosis_block += "\n本段對應問題（請一併修正）：\n" + local[:800]

        # Combined internal context: diagnosis + backbone (主敘事模組) + chapter info + rolling digest.
        story_context = "".join(
            part for part in [diagnosis_block, spine.render(), chapter_label, story.render()] if part
        )

        # Cross-chunk anti-repetition: mine phrases the model has already overused
        # across everything written so far and ask it to avoid them in this chunk.
        # Best-effort and bounded; never fatal.
        if repetition_guard is not None and outputs:
            try:
                prior_text = "\n".join(outputs)[-12000:]
                overused = repetition_guard.extract_overused_phrases(prior_text, top_k=10)
                avoid_directive = repetition_guard.build_avoid_directive(overused)
                if avoid_directive:
                    story_context += "\n\n" + avoid_directive
            except Exception as exc:  # noqa: BLE001 - enhancement only, never fatal.
                logger.warning("repetition_guard phrase extraction failed: %s", exc)

        if progress:
            ch = f"第{chunk.chapter_index + 1}章 " if (chapter_count > 1 or chunk.chapter_title) else ""
            progress(f"正在改寫{ch}第 {idx}/{total} 段，原文字數 {len(chunk.text)}...")
        previous_tail = outputs[-1] if outputs else ""
        try:
            rewritten, info = rewrite_chunk(
                source_chunk=chunk.text,
                chunk_index=idx,
                chunks_total=total,
                settings=settings,
                api_client=api_client,
                lora_client=lora_client,
                previous_tail=previous_tail,
                story_context=story_context,
            )
        except Exception as exc:  # noqa: BLE001 - keep partial output instead of aborting the whole run.
            logger.error("Chunk %d/%d rewrite failed: %s", idx, total, exc, exc_info=True)
            rewritten = f"[第 {idx}/{total} 段改寫失敗：{exc}]"
            info = {
                "mode": settings.mode,
                "operation": settings.operation,
                "spine_updates": 0,
                "chunk_error": str(exc),
            }
        info["chapter_index"] = chunk.chapter_index
        echo_context = "\n".join(
            part for part in [settings.diagnosis_brief, spine.echo_context(), story.digest] if part.strip()
        )
        if echo_context.strip():
            rewritten, echoed = strip_context_echo(rewritten, echo_context)
            if echoed:
                info["stripped_context_echo_lines"] = echoed
        rewritten = rewritten.strip()
        outputs.append(rewritten)
        chapter_outputs.append(rewritten)
        output_path.write_text("\n\n".join(outputs).strip() + "\n", encoding="utf-8-sig")

        if settings.use_story_context and summary_client is not None:
            if progress:
                progress(f"更新故事脈絡（第 {idx}/{total} 段）...")
            update_story_context(
                client=summary_client,
                model=summary_model,
                max_tokens=summary_tokens,
                settings=settings,
                story=story,
                rewritten=rewritten,
                chunk_index=idx,
                chunks_total=total,
            )
            info["story_digest_chars"] = len(story.digest)

        # Refresh the narrative backbone + handoff at chapter ends (or on interval).
        if chapter_continuity_active and (chunk.is_chapter_end or idx % spine_interval == 0):
            if progress:
                progress(f"更新主敘事模組（第 {chunk.chapter_index + 1} 章）...")
            update_narrative_spine(
                client=summary_client,
                model=summary_model,
                settings=settings,
                spine=spine,
                chapter_index=chunk.chapter_index,
                chapter_title=chunk.chapter_title,
                chapter_text="\n\n".join(chapter_outputs),
                rolling_digest=story.digest,
            )
            info["spine_updates"] = spine.updates

        chunk_logs.append(
            {
                "index": idx,
                "chapter_index": chunk.chapter_index,
                "source_chars": len(chunk.text),
                "output_chars": len(rewritten),
                "info": info,
            }
        )
        if progress:
            progress(f"完成第 {idx}/{total} 段，目前累積 {sum(len(part) for part in outputs)} 字。")

    if progress:
        progress("正在執行最終清理與重複掃描...")
    raw_output_text = "\n\n".join(outputs).strip()
    output_text, postprocess_report = postprocess_full_output(raw_output_text, settings)
    postprocess_report["story_context_updates"] = story.updates
    postprocess_report["spine_updates"] = spine.updates
    postprocess_report["chapters_detected"] = chapter_count
    output_path.write_text(output_text.strip() + "\n", encoding="utf-8-sig")
    write_quality_report(report_path, postprocess_report)
    log_payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "settings": asdict(settings),
        "chunks_total": total,
        "chapters_detected": chapter_count,
        "output_path": str(output_path),
        "quality_report_path": str(report_path),
        "postprocess_report": postprocess_report,
        "final_story_digest": story.digest,
        "final_narrative_spine": {
            "bible": spine.bible,
            "chapter_beats": spine.chapter_beats,
            "handoff": spine.handoff,
        },
        "chunks": chunk_logs,
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return RewriteResult(
        output_text=output_text,
        output_path=output_path,
        log_path=log_path,
        report_path=report_path,
        chunks_total=len(chunks),
        mode=settings.mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-text rewrite agent.")
    parser.add_argument("target", help="Path to the source text file.")
    parser.add_argument("--instruction", default="", help="Rewrite requirements.")
    parser.add_argument("--strength", choices=STRENGTHS, default="強改寫")
    parser.add_argument("--operation", choices=OPERATIONS, default=OP_REWRITE, help="改寫 / 增補擴寫 / 精簡刪減")
    parser.add_argument("--mode", choices=MODES, default=MODE_HYBRID)
    parser.add_argument("--language", default="繁體中文")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="外部 API 重複懲罰（DZMM 建議 1.05）。")
    parser.add_argument("--api-key", default="", help="外部 API 金鑰；留空則從環境變數／設定讀取。")
    parser.add_argument("--api-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-model", default=DEFAULT_MODEL)
    parser.add_argument("--style-file", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--no-story-context", action="store_true", help="關閉跨段落故事脈絡摘要。")
    parser.add_argument("--no-chapter-continuity", action="store_true", help="關閉主敘事模組與章節連貫系統。")
    parser.add_argument("--diagnosis", default="", help="Grok 改寫診斷書 JSON 路徑（用於 seed 骨幹與注入問題修正）。")
    parser.add_argument("--dedup-similarity", type=float, default=0.9, help="近似重複段落判定門檻 (0-1)。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style_reference = load_reference_text(args.style_file)
    diag_spine_seed = ""
    diag_brief = ""
    diag_window_problems: list[str] = []
    if args.diagnosis:
        # Lazy import so rewrite_agent has no module-level dependency on analysis_agent.
        from analysis_agent import load_diagnosis  # noqa: PLC0415

        diagnosis = load_diagnosis(args.diagnosis)
        diag_spine_seed = diagnosis.spine_seed
        diag_brief = diagnosis.rewrite_brief()
        diag_window_problems = diagnosis.window_problems
        print(f"已套用診斷：{args.diagnosis}（{len(diagnosis.problems)} 個問題，{diagnosis.window_count} 段）")
    settings = RewriteSettings(
        mode=args.mode,
        instruction=args.instruction,
        rewrite_strength=args.strength,
        operation=args.operation,
        output_language=args.language,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        api_model=args.api_model,
        chunk_chars=args.chunk_chars,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        style_reference=style_reference,
        continuity_notes=args.notes,
        use_story_context=not args.no_story_context,
        chapter_continuity=not args.no_chapter_continuity,
        dedup_similarity=args.dedup_similarity,
        diagnosis_spine_seed=diag_spine_seed,
        diagnosis_brief=diag_brief,
        diagnosis_window_problems=diag_window_problems,
    )
    source_text = read_text_file(args.target)
    result = full_rewrite(source_text=source_text, settings=settings, progress=print)
    print(f"完成：{result.output_path}")
    print(f"紀錄：{result.log_path}")


if __name__ == "__main__":
    main()
