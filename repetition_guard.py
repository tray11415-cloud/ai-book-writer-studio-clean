"""Cross-response repetition control + long-form memory for story generation.

The novel model (nalang / local LoRA) tends to fall back on the same descriptive
phrases, metaphors, and scene "套路" every time it is called again. ``frequency_penalty``
and ``presence_penalty`` only act *within a single completion*, so they reduce
intra-response repetition but do nothing about the same clichés reappearing across
separate continuations. That cross-response recycling is what makes the
"横向重复比例" stay high.

This module is pure-Python (no extra API calls). It provides:

1. ``extract_overused_phrases`` - scan the *whole* story so far and find phrases that
   recur often, so we can hand the model an explicit "stop reusing these wordings" list.
2. ``repetition_ratio`` - measure how much a new continuation overlaps the prior story
   (the横向重复比例 metric, 0..1).
3. ``repeated_spans`` - the concrete recycled phrases inside a continuation, used to push
   a stronger avoid-list on a regeneration pass.
4. ``build_avoid_directive`` - format those into a prompt block.
5. ``build_longform_memory`` - extractive digest of earlier content that has scrolled out
   of the recent-context window, so long stories keep continuity.

All phrase detection works on character n-grams for CJK text and word n-grams for
mostly-latin text, so it handles Chinese (no spaces) and English alike.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Iterable

# --- tunables (env-overridable) ------------------------------------------------
#
# Tuning guide (all overridable via the BOOK_WRITER_REPETITION_* env vars below):
#
#   RATIO_NGRAM       Overlap-measurement window. ~8 CJK chars captures a short clause;
#                     for latin text the code internally halves this (so ~4 words). Larger
#                     -> only long verbatim copies count as overlap (fewer, stricter hits);
#                     smaller -> more sensitive but noisier.
#   PHRASE_NGRAM      Cliché-mining window. ~10 CJK chars ~ a 3-4 "word" Chinese phrase;
#                     5-7 is a better starting point for English. Too small flags single
#                     nouns/names; too large misses shorter recurring tics.
#   PHRASE_MIN_COUNT  Times a phrase must recur to count as overused. 3 balances catching
#                     real clichés against false positives; raise it for very long stories.
#   PHRASE_TOP_K      How many overused phrases to feed back as the avoid-list.
#   PHRASE_MIN_SPAN_CJK  Hard floor on emitted CJK ban length, so short names never surface.
#   RATIO_THRESHOLD   Overlap above this triggers a regeneration. 0.30 suits loosely-written
#                     English; tightly-written Chinese naturally overlaps more on function
#                     words, so this may read as "too sensitive" and can be raised.
#   MAX_RETRIES       Regeneration attempts when the overlap ratio is too high.

GUARD_ENABLED = os.getenv("BOOK_WRITER_REPETITION_GUARD", "1").strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
# n-gram size (CJK characters) used to measure overlap between a new continuation
# and the prior story.
RATIO_NGRAM = int(os.getenv("BOOK_WRITER_REPETITION_NGRAM", "8"))
# n-gram size used to mine recurring "套路" phrases out of the whole story.
PHRASE_NGRAM = int(os.getenv("BOOK_WRITER_REPETITION_PHRASE_NGRAM", "10"))
# A phrase must recur at least this many times across the story to count as overused.
PHRASE_MIN_COUNT = int(os.getenv("BOOK_WRITER_REPETITION_MIN_COUNT", "3"))
# How many overused phrases to feed back as an avoid-list.
PHRASE_TOP_K = int(os.getenv("BOOK_WRITER_REPETITION_TOP_K", "14"))
# Minimum length (CJK characters) a recycled / overused span must reach before it is
# emitted as a ban. Kept above a single short noun / character name (2-4 chars) so the
# guard never tells the model to stop writing a name. Multi-beat wordings clear this.
PHRASE_MIN_SPAN_CJK = int(os.getenv("BOOK_WRITER_REPETITION_MIN_SPAN_CJK", "6"))
# If a fresh continuation's overlap ratio exceeds this, regenerate with a stronger ban.
RATIO_THRESHOLD = float(os.getenv("BOOK_WRITER_REPETITION_THRESHOLD", "0.30"))
# Max number of regeneration retries when the overlap ratio is too high.
MAX_RETRIES = int(os.getenv("BOOK_WRITER_REPETITION_MAX_RETRIES", "1"))

_CJK_RE = re.compile(r"[一-鿿㐀-䶿぀-ヿ]")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?…\n])")
_PUNCT_RE = re.compile(r"[\s，。、；：！？!?…—\-~·.,:;\"'“”‘’()（）《》【】\[\]{}<>/\\|*#]+")


def _is_cjk_text(text: str) -> bool:
    """True when the text is mostly CJK (so char n-grams beat word n-grams)."""
    sample = text[:4000]
    if not sample:
        return True
    cjk = len(_CJK_RE.findall(sample))
    return cjk >= max(8, len(sample) * 0.20)


def _strip_ws(text: str) -> str:
    return _WS_RE.sub("", text or "")


def _char_ngrams(text: str, n: int) -> list[str]:
    s = _strip_ws(text)
    if n <= 0 or len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _word_ngrams(text: str, n: int) -> list[str]:
    words = _WORD_RE.findall((text or "").lower())
    if n <= 0 or len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _ngrams(text: str, n: int, cjk: bool) -> list[str]:
    return _char_ngrams(text, n) if cjk else _word_ngrams(text, max(2, n // 2))


def _punct_ratio(text: str) -> float:
    if not text:
        return 1.0
    stripped = _PUNCT_RE.sub("", text)
    return 1.0 - (len(stripped) / len(text))


def _merge_member_spans(text: str, member: set[str], n: int, cjk: bool) -> list[str]:
    """Merge runs of n-grams that are all in ``member`` into maximal, readable spans.

    Shared by ``repeated_spans`` (member = prior story's n-grams) and
    ``extract_overused_phrases`` (member = the story's own recurring n-grams) so both
    return whole recycled sentences instead of 1-char-shifted shingles.
    """
    spans: list[str] = []
    if cjk:
        s = _strip_ws(text)
        if len(s) < n:
            return []
        hit = [s[i : i + n] in member for i in range(len(s) - n + 1)]
        i = 0
        while i < len(hit):
            if hit[i]:
                j = i
                while j < len(hit) and hit[j]:
                    j += 1
                spans.append(s[i : (j - 1) + n])
                i = j
            else:
                i += 1
        return spans

    words = _WORD_RE.findall((text or "").lower())
    step = max(2, n // 2)
    # Need at least one full window of words; otherwise there are no n-grams to match.
    if len(words) < step:
        return []
    last_start = len(words) - step  # last index a full `step`-word window can start at
    i = 0
    while i <= last_start:
        if " ".join(words[i : i + step]) in member:
            j = i
            while j <= last_start and " ".join(words[j : j + step]) in member:
                j += 1
            # `j - 1` is the last matched window start; clamp the end so the slice can
            # never run past the word list even if the off-by-one logic shifts.
            span = " ".join(words[i : min((j - 1) + step, len(words))])
            if span:
                spans.append(span)
            i = j
        else:
            i += 1
    return spans


def _rank_spans(spans: Iterable[str], *, min_len: int, max_spans: int) -> list[str]:
    """Dedupe spans, drop ones nested in a longer kept span, longest/most-frequent first."""
    counts = Counter(sp for sp in spans if len(sp) >= min_len and _punct_ratio(sp) <= 0.5)
    ordered = sorted(counts, key=lambda sp: (-len(sp), -counts[sp]))
    kept: list[str] = []
    for span in ordered:
        if any(span in bigger for bigger in kept):
            continue
        kept.append(span)
        if len(kept) >= max_spans:
            break
    return kept


def repetition_ratio(new_text: str, prior_text: str, *, n: int | None = None) -> float:
    """Fraction of the new text's n-grams that already occur in the prior story.

    ``1.0`` means the continuation is almost entirely recycled phrasing; ``0.0`` means
    every n-gram is new. Some baseline overlap is normal (names, function words), so
    judge changes against a baseline rather than the absolute number.
    """
    if not new_text or not prior_text:
        return 0.0
    cjk = _is_cjk_text(new_text)
    n = n or (RATIO_NGRAM if cjk else 4)
    new_grams = _ngrams(new_text, n, cjk)
    if not new_grams:
        return 0.0
    prior_grams = set(_ngrams(prior_text, n, cjk))
    if not prior_grams:
        return 0.0
    hits = sum(1 for g in new_grams if g in prior_grams)
    return hits / len(new_grams)


def repeated_spans(
    new_text: str,
    prior_text: str,
    *,
    n: int | None = None,
    max_spans: int = 24,
    min_span_len: int | None = None,
) -> list[str]:
    """Return the concrete phrases in ``new_text`` that also appear in the prior story.

    Adjacent overlapping n-gram hits are merged into longer, human-readable spans so the
    result reads like real recycled sentences ("她的心跳漏了一拍") rather than fragments.
    """
    if not new_text or not prior_text:
        return []
    cjk = _is_cjk_text(new_text)
    n = n or (RATIO_NGRAM if cjk else 4)
    prior_set = set(_ngrams(prior_text, n, cjk))
    if not prior_set:
        return []
    spans = _merge_member_spans(new_text, prior_set, n, cjk)
    # Floor the CJK span length above a single short noun / character name so a recycled
    # name on its own is never surfaced as a ban (callers may still override).
    if min_span_len is not None:
        min_len = min_span_len
    elif cjk:
        min_len = max(n, PHRASE_MIN_SPAN_CJK)
    else:
        min_len = max(2, n // 2) + 2
    return _rank_spans(spans, min_len=min_len, max_spans=max_spans)


def extract_overused_phrases(
    text: str,
    *,
    n: int | None = None,
    min_count: int = PHRASE_MIN_COUNT,
    top_k: int = PHRASE_TOP_K,
    min_span_len: int | None = None,
) -> list[str]:
    """Mine phrases that recur across the whole story - the model's overused clichés.

    Uses a fairly long n-gram so single nouns / character names (which are short and
    *should* repeat) are not flagged; only repeated multi-beat wordings surface. Nested
    phrases are collapsed into their longest representative form.

    The emitted spans are additionally floored at ``min_span_len`` characters (CJK) /
    words-derived length (latin). The CJK floor defaults to ``max(n, PHRASE_MIN_SPAN_CJK)``
    so that even with a small ``n`` a single 2-4 char noun or character name can never be
    handed back as a ban — only genuinely multi-beat wordings clear the bar.
    """
    if not text:
        return []
    cjk = _is_cjk_text(text)
    n = n or (PHRASE_NGRAM if cjk else 5)
    grams = _ngrams(text, n, cjk)
    if not grams:
        return []
    counts = Counter(grams)
    # n-grams that recur often enough to be considered the model's own clichés.
    repeated = {gram for gram, c in counts.items() if c >= min_count}
    if not repeated:
        return []
    # Merge those recurring shingles back into maximal, readable phrases.
    spans = _merge_member_spans(text, repeated, n, cjk)
    if min_span_len is not None:
        min_len = min_span_len
    elif cjk:
        # Floor above a single short name/noun so names are never emitted as bans.
        min_len = max(n, PHRASE_MIN_SPAN_CJK)
    else:
        min_len = max(2, n // 2) + 2
    return _rank_spans(spans, min_len=min_len, max_spans=top_k)


def build_avoid_directive(
    overused: Iterable[str],
    recycled: Iterable[str] | None = None,
    *,
    cjk: bool = True,
) -> str:
    """Build a prompt block that bans recycled wordings (names/proper nouns exempt)."""
    overused = [p.strip() for p in overused if p and p.strip()]
    recycled = [p.strip() for p in (recycled or []) if p and p.strip()]
    # Drop recycled spans already covered by an overused phrase to avoid noise.
    recycled = [r for r in recycled if not any(r in o or o in r for o in overused)]
    if not overused and not recycled:
        return ""

    if cjk:
        lines = [
            "【避免重复·硬性要求】下列是前文已反复出现的描写句式与措辞。"
            "本次续写必须改用全新的意象、句式与表达，禁止照抄或仅做轻微改写。"
            "（人物名、地名、专有名词不在此限，可照常使用；要避免的是“写法”，不是“人/物”。）",
        ]
        if overused:
            lines.append("已被用滥的措辞：")
            lines.extend(f"  - {p}" for p in overused)
        if recycled:
            lines.append("刚刚这一段与前文高度雷同、必须重写的句段：")
            lines.extend(f"  - {p}" for p in recycled)
        lines.append("请针对同样的情节，换一种没用过的角度、感官与节奏来写。")
        return "\n".join(lines)

    lines = [
        "[ANTI-REPETITION — HARD RULE] The phrasings below already recur in earlier text. "
        "This continuation must use fresh imagery, sentence shapes, and wording, and must not "
        "copy or lightly reword them. (Character names and proper nouns are exempt — avoid the "
        "*phrasing*, not the people or objects.)",
    ]
    if overused:
        lines.append("Overused wordings:")
        lines.extend(f"  - {p}" for p in overused)
    if recycled:
        lines.append("Spans from the latest draft that are too close to earlier text:")
        lines.extend(f"  - {p}" for p in recycled)
    return "\n".join(lines)


def build_longform_memory(
    current_story: str,
    recent_window_chars: int,
    *,
    max_chars: int = 900,
    max_points: int = 18,
) -> str:
    """Extractive digest of content that has scrolled out of the recent-context window.

    Long stories only feed the model the last ``recent_window_chars`` of prose, so anything
    earlier is invisible and gets re-introduced or contradicted. This pulls the leading
    sentence of earlier paragraphs into a compact "what happened earlier" anchor so the
    model keeps continuity without re-reading (and re-echoing) the full prose.
    """
    story = (current_story or "").strip()
    if not story:
        return ""
    # "Older" content is everything before the recent-context window. If the window is
    # as large as (or larger than) the whole story, nothing has scrolled out yet; if the
    # window is 0, the entire story counts as older.
    window_size = max(recent_window_chars, 0)
    if window_size == 0:
        older = story
    elif len(story) > window_size:
        older = story[:-window_size]
    else:
        older = ""
    older = older.strip()
    if len(older) < 200:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", older) if p.strip()]
    points: list[str] = []
    for para in paragraphs:
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        if not sentences:
            continue
        lead = sentences[0]
        if len(lead) > 120:
            lead = lead[:120].rstrip() + "…"
        points.append(lead)

    if not points:
        return ""

    # If there are many paragraphs, sample evenly so the digest spans the whole earlier arc.
    if len(points) > max_points:
        step = len(points) / max_points
        points = [points[int(i * step)] for i in range(max_points)]

    digest: list[str] = []
    used = 0
    for point in points:
        if used + len(point) > max_chars:
            break
        digest.append(point)
        used += len(point)
    if not digest:
        return ""
    return "\n".join(f"- {p}" for p in digest)
