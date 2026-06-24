"""Phase 1: Grok-powered manuscript analysis & diagnosis.

Produces a reviewable 改寫診斷書 (rewrite diagnosis) that Phase 2 (the rewrite
engine in rewrite_agent.py) consumes to seed the narrative backbone and to inject
"problems that must be fixed" into each rewrite chunk.

This reuses the *design* of the studio's analysis features — Story Chronicle's
"脈絡 + 連貫風險" structure and Chapter Craft's multi-specialist breakdown — but
as a clean, programmatic, OpenAI-compatible module that targets Grok (xAI). It
does not depend on the Gradio studio.

Pipeline: windows -> per-window specialists (parallel) -> global synthesis ->
Diagnosis (+ markdown report + json).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from openai import OpenAI

logger = logging.getLogger(__name__)

from rewrite_agent import (
    AGENT_DIR,
    OUTPUT_DIR,
    clean_model_output,
    is_transient_api_error,
    normalize_blank_lines,
    read_text_file,
    split_into_chapters,
)


# Grok / xAI defaults (analysis route), same env keys the studio uses.
def grok_defaults() -> tuple[str, str, str]:
    key = os.getenv("XAI_API_KEY", "")
    base = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    model = os.getenv("XAI_MODEL", "grok-4.3")
    return key, base, model


@dataclass
class AnalysisSettings:
    api_key: str = ""
    base_url: str = "https://api.x.ai/v1"
    model: str = "grok-4.3"
    analysis_chunk_chars: int = 5000
    language: str = "繁體中文"
    max_workers: int = 4
    temperature: float = 0.3
    instruction: str = ""  # optional user focus, e.g. "特別注意人物稱謂前後矛盾"


# Specialist lenses applied to every analysis window (Chapter Craft style).
SPECIALISTS = [
    {
        "key": "continuity",
        "name": "脈絡連貫診斷師",
        "focus": (
            "追蹤本段的事件、時間線、人物當前狀態與稱謂、地點，並對照前文摘要，"
            "找出連貫風險：時間矛盾、人物行為/性格前後不一致、稱謂或設定衝突、"
            "未回收或自相矛盾的伏筆、空間或邏輯漏洞。逐條列出，標明涉及的人物或設定。"
        ),
    },
    {
        "key": "structure",
        "name": "結構節奏分析師",
        "focus": (
            "分析本段的敘事功能、因果推進、起承轉合與節奏：哪裡拖沓、哪裡資訊量不足或過載、"
            "場景轉換是否突兀、懸念與張力是否到位。指出結構與節奏上的問題。"
        ),
    },
    {
        "key": "character",
        "name": "人物動機分析師",
        "focus": (
            "分析本段主要人物的慾望、阻力、選擇與關係張力，對白與行動是否符合人物，"
            "情緒轉折是否有鋪墊。指出人物動機薄弱、行為突兀或關係處理失衡之處。"
        ),
    },
    {
        "key": "problems",
        "name": "問題獵手",
        "focus": (
            "只挑出本段具體、可改寫修正的問題：重複描寫與冗詞、流水帳、出戲、設定硬傷、"
            "邏輯不通、對白生硬、感官失衡、文筆弱點。每條盡量附上可定位的線索（如該句開頭幾字）。"
        ),
    },
]


def build_grok_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=(api_key or "not-needed").strip(), base_url=base_url.strip().rstrip("/"), timeout=900)


def grok_chat(
    *,
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    retries: int = 2,
) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= retries or not is_transient_api_error(exc):
                raise
            time.sleep(3 * (attempt + 1))
    raise last_exc or RuntimeError("Grok request failed.")


@dataclass
class AnalysisWindow:
    index: int
    chapter_index: int
    chapter_title: str
    text: str
    char_start: int
    char_end: int


def split_analysis_windows(text: str, chunk_chars: int) -> list[AnalysisWindow]:
    """Split into larger analysis windows (chapter-aware), tracking char offsets.

    Analysis works on bigger units than the 900-char rewrite chunks so Grok sees
    enough context and the call count stays sane.
    """
    requested_chars = int(chunk_chars)
    chunk_chars = max(1500, requested_chars)
    if requested_chars < 1500:
        logger.warning(
            "analysis_chunk_chars=%s below minimum; clamped to %s.", requested_chars, chunk_chars
        )
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chapters = split_into_chapters(normalized)
    windows: list[AnalysisWindow] = []
    # `cursor` is the running character offset into the *original normalized text*.
    # We advance it by each chapter's full length (incl. the title + separators we
    # synthesise) so window char_start/char_end stay aligned with the source.
    cursor = 0
    win_index = 0
    for chapter_index, (title, body) in enumerate(chapters):
        chapter_text = (title + "\n\n" + body).strip() if title else body.strip()
        if not chapter_text:
            cursor += len(body) + (len(title) + 2 if title else 0)
            continue
        paragraphs = [p for p in re.split(r"\n\s*\n", chapter_text) if p.strip()]
        # Offset of the start of `chapter_text` within the original text.
        chapter_cursor = cursor
        buffer = ""
        for para in paragraphs:
            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) <= chunk_chars or not buffer:
                buffer = candidate
                continue
            windows.append(
                AnalysisWindow(
                    win_index, chapter_index, title, buffer, chapter_cursor, chapter_cursor + len(buffer)
                )
            )
            win_index += 1
            # Advance past this window's text plus the "\n\n" separator before the next paragraph.
            chapter_cursor += len(buffer) + 2
            buffer = para
        if buffer.strip():
            windows.append(
                AnalysisWindow(
                    win_index, chapter_index, title, buffer, chapter_cursor, chapter_cursor + len(buffer)
                )
            )
            win_index += 1
        # Advance the global cursor past this whole chapter (title + separator + body).
        cursor += len(body) + (len(title) + 2 if title else 0)
    return windows


def analyze_window(
    *,
    client: OpenAI,
    settings: AnalysisSettings,
    window: AnalysisWindow,
    total_windows: int,
    prior_summary: str,
) -> dict[str, str]:
    """Run all specialists on one window (in parallel). Returns key -> finding."""
    extra = f"\n額外關注：{settings.instruction.strip()}" if settings.instruction.strip() else ""
    chapter_hint = f"（第 {window.chapter_index + 1} 章{('「' + window.chapter_title + '」') if window.chapter_title else ''}）"

    def run(spec: dict) -> tuple[str, str]:
        system = (
            f"你是長篇小說的{spec['name']}，只做分析與診斷，不改寫、不續寫、不複述原文。"
            f"使用{settings.language}，輸出精簡條列，每條一個重點。"
        )
        user = (
            f"任務：{spec['focus']}{extra}\n\n"
            f"這是全書第 {window.index + 1}/{total_windows} 個分析段 {chapter_hint}。\n\n"
            f"前文摘要（僅供對照，可能為空）：\n{prior_summary or '（無）'}\n\n"
            f"本段原文：\n{window.text}\n\n"
            f"請只輸出本段的診斷重點（條列，無前言、無結語）。若本段在該面向沒有明顯問題，輸出「無明顯問題」。"
        )
        try:
            out = grok_chat(
                client=client,
                model=settings.model,
                system=system,
                user=user,
                temperature=settings.temperature,
                max_tokens=1200,
            )
            out, _ = clean_model_output(out)
            return spec["key"], out.strip()
        except Exception as exc:  # noqa: BLE001
            return spec["key"], f"[分析失敗：{exc}]"

    findings: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, settings.max_workers)) as pool:
        for key, value in pool.map(run, SPECIALISTS):
            findings[key] = value
    return findings


def _rolling_summary(client: OpenAI, settings: AnalysisSettings, prior: str, window: AnalysisWindow) -> str:
    """Cheap rolling 'story so far' so the continuity lens has cross-window context."""
    try:
        out = grok_chat(
            client=client,
            model=settings.model,
            system=f"你是小說連續性編輯，維護精簡的劇情進度摘要。使用{settings.language}，只輸出摘要。",
            user=(
                f"目前進度摘要（可能為空）：\n{prior or '（無）'}\n\n"
                f"接著發生（第 {window.index + 1} 段）：\n{window.text[-1800:]}\n\n"
                f"請更新並輸出 300 字內的進度摘要：主要人物與當前狀態、已發生關鍵事件、未解伏筆。只輸出摘要。"
            ),
            temperature=0.2,
            max_tokens=600,
        )
        out, _ = clean_model_output(out)
        return out.strip()[:1500] or prior
    except Exception:  # noqa: BLE001
        return prior


@dataclass
class Diagnosis:
    overall_analysis: str = ""
    continuity_diagnosis: str = ""
    problems: list[dict] = field(default_factory=list)  # {location, severity, problem, fix}
    spine_seed: str = ""
    continuity_notes: str = ""
    window_count: int = 0
    window_problems: list[str] = field(default_factory=list)  # per-window concatenated problem text
    model: str = ""

    def rewrite_brief(self) -> str:
        """A compact global brief injected into every rewrite chunk."""
        lines = []
        if self.continuity_notes.strip():
            lines.append("連貫要點：\n" + self.continuity_notes.strip())
        if self.problems:
            top = self.problems[:14]
            probs = "\n".join(
                f"- {p.get('problem','').strip()} → 改寫方向：{p.get('fix','').strip()}" for p in top if p.get("problem")
            )
            if probs:
                lines.append("全書診斷出、改寫時必須一併修正的問題：\n" + probs)
        return "\n\n".join(lines)


def _decode_first_object(text: str) -> dict | None:
    """Return the first complete JSON object found in `text`, or None.

    Scans forward to each '{' and uses JSONDecoder.raw_decode, which stops at the
    first valid JSON boundary (handling embedded braces and trailing junk).
    """
    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None
        try:
            obj, _end = decoder.raw_decode(text[start:])
        except ValueError:
            search_from = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        search_from = start + 1


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response.

    Handles fenced (```json ... ```), unfenced, missing/malformed fences, and
    responses with surrounding prose. Tries fence content first, then the raw text.
    """
    text = text.strip()
    candidates: list[str] = []

    # Non-greedy so we capture each fenced block independently rather than spanning
    # from the first ``` to the last; collect all fences as candidates.
    for fence in re.finditer(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL):
        candidates.append(fence.group(1).strip())
    candidates.append(text)  # fallback: scan the whole response

    for candidate in candidates:
        # Fast path: candidate is itself valid JSON.
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        # Robust path: find the first complete JSON object within the candidate.
        obj = _decode_first_object(candidate)
        if obj is not None:
            return obj

    logger.warning("_extract_json: no valid JSON object found in model response.")
    return None


def synthesize_diagnosis(
    *,
    client: OpenAI,
    settings: AnalysisSettings,
    window_findings: list[dict[str, str]],
    windows: list[AnalysisWindow],
) -> Diagnosis:
    """Fold per-window specialist findings into one global, structured diagnosis."""
    blocks = []
    budget = 42000
    for win, find in zip(windows, window_findings):
        seg = (
            f"=== 第 {win.index + 1} 段"
            f"{('（第' + str(win.chapter_index + 1) + '章）') if win.chapter_title or win.chapter_index else ''} ===\n"
            f"[連貫] {find.get('continuity','')}\n"
            f"[結構節奏] {find.get('structure','')}\n"
            f"[人物] {find.get('character','')}\n"
            f"[問題] {find.get('problems','')}"
        )
        blocks.append(seg[:5000])
    joined = "\n\n".join(blocks)
    if len(joined) > budget:
        joined = joined[:budget] + "\n\n[後續段落從略]"

    system = (
        "你是長篇小說的總編輯與診斷統合者。根據各段的專家分析，產出一份全書改寫診斷。"
        f"使用{settings.language}。嚴格只輸出一個 JSON 物件，不要任何解釋或 markdown 圍欄。"
    )
    user = (
        "各段專家分析如下：\n\n" + joined + "\n\n"
        "請統合為一個 JSON 物件，鍵如下：\n"
        '{\n'
        '  "overall_analysis": "全篇分析：核心前提與主線、主要人物與關係、文風、整體結構與優缺點（條列文字）",\n'
        '  "continuity_diagnosis": "脈絡診斷：時間線與人物弧線、最關鍵的連貫風險與矛盾、未解伏筆（條列文字）",\n'
        '  "spine_seed": "全書骨幹：給改寫引擎當主敘事模組的穩定背景（前提、人物檔案與稱謂、世界觀、中心衝突、主題），條列、精簡、無矛盾",\n'
        '  "continuity_notes": "改寫時必須遵守的連貫鐵則與稱謂/設定一致性要求（精簡條列）",\n'
        '  "problems": [ {"location": "第N段或概述", "severity": "high|medium|low", "problem": "具體問題", "fix": "建議改寫方向"} ]\n'
        "}\n"
        "problems 依嚴重度排序，聚焦真正影響閱讀的問題（連貫矛盾、人物失常、重複冗長、邏輯硬傷等），最多 25 條。只輸出 JSON。"
    )
    raw = grok_chat(
        client=client,
        model=settings.model,
        system=system,
        user=user,
        temperature=0.2,
        max_tokens=6000,
    )
    data = _extract_json(raw) or {}
    problems = data.get("problems") or []
    clean_problems = []
    for p in problems:
        if isinstance(p, dict) and p.get("problem"):
            clean_problems.append(
                {
                    "location": str(p.get("location", "")).strip(),
                    "severity": str(p.get("severity", "")).strip().lower() or "medium",
                    "problem": str(p.get("problem", "")).strip(),
                    "fix": str(p.get("fix", "")).strip(),
                }
            )

    # Per-window problem text (the 問題獵手 + 連貫 findings), for local injection in rewrite.
    window_problems = []
    for find in window_findings:
        local = "\n".join(
            part for part in [find.get("problems", ""), find.get("continuity", "")] if part and "無明顯問題" not in part
        ).strip()
        window_problems.append(local)

    return Diagnosis(
        overall_analysis=str(data.get("overall_analysis", "")).strip(),
        continuity_diagnosis=str(data.get("continuity_diagnosis", "")).strip(),
        problems=clean_problems,
        spine_seed=str(data.get("spine_seed", "")).strip(),
        continuity_notes=str(data.get("continuity_notes", "")).strip(),
        window_count=len(windows),
        window_problems=window_problems,
        model=settings.model,
    )


def diagnosis_to_markdown(diagnosis: Diagnosis, *, source_chars: int, started_at: datetime) -> str:
    lines = [
        "# 改寫診斷書",
        "",
        f"- 生成時間：{started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 分析模型：{diagnosis.model}",
        f"- 原文字數：{source_chars}　分析段數：{diagnosis.window_count}",
        f"- 偵測問題數：{len(diagnosis.problems)}",
        "",
        "## 一、全篇分析",
        "",
        diagnosis.overall_analysis or "（無）",
        "",
        "## 二、脈絡診斷（連貫風險）",
        "",
        diagnosis.continuity_diagnosis or "（無）",
        "",
        "## 三、問題清單",
        "",
    ]
    if diagnosis.problems:
        lines.append("| # | 位置 | 嚴重度 | 問題 | 建議改寫方向 |")
        lines.append("|---|------|--------|------|--------------|")
        for i, p in enumerate(diagnosis.problems, start=1):
            prob = p["problem"].replace("|", "／").replace("\n", " ")
            fix = p["fix"].replace("|", "／").replace("\n", " ")
            loc = p["location"].replace("|", "／")
            lines.append(f"| {i} | {loc} | {p['severity']} | {prob} | {fix} |")
    else:
        lines.append("（未偵測到明顯問題）")
    lines += [
        "",
        "## 附錄：改寫骨幹（將自動注入改寫的主敘事模組）",
        "",
        diagnosis.spine_seed or "（無）",
        "",
        "### 連貫鐵則（改寫時必守）",
        "",
        diagnosis.continuity_notes or "（無）",
        "",
        "---",
        "_可自行編輯本檔後再用於改寫；或直接在改寫時套用。_",
    ]
    return "\n".join(lines).strip() + "\n"


def analyze_manuscript(
    *,
    source_text: str,
    settings: AnalysisSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[Diagnosis, Path, Path]:
    """Run the full Phase-1 analysis. Writes 改寫診斷書_<ts>.md and .json."""
    if not settings.api_key.strip():
        raise ValueError("缺少 Grok（XAI）API 金鑰。請在 .env 或環境變數設定 XAI_API_KEY。")
    windows = split_analysis_windows(source_text, settings.analysis_chunk_chars)
    if not windows:
        raise ValueError("分析目標是空的。")
    client = build_grok_client(settings.api_key, settings.base_url)

    started_at = datetime.now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / f"改寫診斷書_{timestamp}.md"
    json_path = OUTPUT_DIR / f"改寫診斷書_{timestamp}.json"

    window_findings: list[dict[str, str]] = []
    prior_summary = ""
    for win in windows:
        if progress:
            progress(f"Grok 分析第 {win.index + 1}/{len(windows)} 段（{len(win.text)} 字，{len(SPECIALISTS)} 位專家）...")
        findings = analyze_window(
            client=client,
            settings=settings,
            window=win,
            total_windows=len(windows),
            prior_summary=prior_summary,
        )
        window_findings.append(findings)
        prior_summary = _rolling_summary(client, settings, prior_summary, win)

    if progress:
        progress("Grok 統合全書診斷...")
    diagnosis = synthesize_diagnosis(
        client=client, settings=settings, window_findings=window_findings, windows=windows
    )

    report_md = diagnosis_to_markdown(diagnosis, source_chars=len(source_text), started_at=started_at)
    md_path.write_text(report_md, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": started_at.isoformat(timespec="seconds"),
                "model": diagnosis.model,
                "source_chars": len(source_text),
                "settings": asdict(settings) | {"api_key": "***"},
                "diagnosis": asdict(diagnosis),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8-sig",
    )
    if progress:
        progress(f"診斷完成：{md_path.name}（{len(diagnosis.problems)} 個問題）")
    return diagnosis, md_path, json_path


def load_diagnosis(json_path: str | Path) -> Diagnosis:
    try:
        raw = read_text_file(json_path)
    except FileNotFoundError as exc:
        raise ValueError(f"找不到診斷書檔案：{json_path}") from exc
    except (PermissionError, OSError) as exc:
        logger.warning("load_diagnosis: failed to read %s: %s", json_path, exc)
        raise ValueError(f"診斷書檔案無法讀取：{json_path}") from exc
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("load_diagnosis: corrupted JSON in %s: %s", json_path, exc)
        raise ValueError(f"診斷書檔案損毀或格式錯誤：{json_path}。請重新分析。") from exc
    if not isinstance(data, dict):
        raise ValueError(f"診斷書檔案格式不符（非 JSON 物件）：{json_path}。請重新分析。")
    d = data.get("diagnosis", data)
    if not isinstance(d, dict):
        raise ValueError(f"診斷書內容格式錯誤：{json_path}。請重新分析。")
    return Diagnosis(
        overall_analysis=d.get("overall_analysis", ""),
        continuity_diagnosis=d.get("continuity_diagnosis", ""),
        problems=d.get("problems", []) or [],
        spine_seed=d.get("spine_seed", ""),
        continuity_notes=d.get("continuity_notes", ""),
        window_count=d.get("window_count", 0),
        window_problems=d.get("window_problems", []) or [],
        model=d.get("model", ""),
    )


def find_latest_diagnosis() -> Path | None:
    files = sorted(OUTPUT_DIR.glob("改寫診斷書_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _parse_args():
    import argparse

    key, base, model = grok_defaults()
    parser = argparse.ArgumentParser(description="Phase 1: Grok manuscript analysis & diagnosis.")
    parser.add_argument("target", help="Path to the source text file.")
    parser.add_argument("--model", default=model, help="Grok model (default from XAI_MODEL).")
    parser.add_argument("--base-url", default=base)
    parser.add_argument("--api-key", default=key)
    parser.add_argument("--window-chars", type=int, default=5000, help="分析視窗字數。")
    parser.add_argument("--language", default="繁體中文")
    parser.add_argument("--workers", type=int, default=4, help="每段並行專家數。")
    parser.add_argument("--instruction", default="", help="額外分析關注點。")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = AnalysisSettings(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        analysis_chunk_chars=args.window_chars,
        language=args.language,
        max_workers=args.workers,
        instruction=args.instruction,
    )
    source_text = read_text_file(args.target)
    diagnosis, md_path, json_path = analyze_manuscript(
        source_text=source_text, settings=settings, progress=print
    )
    print(f"診斷書：{md_path}")
    print(f"診斷資料：{json_path}")
    print(f"接著改寫可用： --diagnosis \"{json_path}\"")


if __name__ == "__main__":
    main()
