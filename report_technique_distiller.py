"""Distill chapter-craft reports into compact writing-agent technique libraries."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from skill_technique_review import (
    drop_unsafe_lines,
    extract_headings,
    extract_keyword_lines,
    read_text,
    sanitize_review_text,
    to_positive_int,
    trim_text,
)

logger = logging.getLogger(__name__)


DEFAULT_REPORT_DISTILL_GOAL = (
    "將 full_report.md 蒸餾成可給寫作 AGENT 使用的精簡技法庫。"
    "只保留可複製的敘事技巧、場景調度、節奏、人物推進、轉折、文風規則與 Director Instruction；"
    "不要抄原文，不要延續原作情節，不要輸出敏感或不適合進入寫作提示詞的內容。"
)

TECHNIQUE_LOAD_MODES = [
    "Replace Technique Library + Append Memory/Director",
    "Append Technique Library + Append Memory/Director",
    "Replace Technique Library Only",
    "Append Technique Library Only",
]

SAFE_TECHNIQUE_KEYWORDS = [
    "技巧",
    "技法",
    "節奏",
    "衝突",
    "人物",
    "場景",
    "視角",
    "轉折",
    "伏筆",
    "鋪陳",
    "公式",
    "模板",
    "感官",
    "對話",
    "動作",
    "氛圍",
    "張力",
    "鉤子",
    "可複製",
]

UNSAFE_LINE_MARKERS = [
    "[redacted:",
    "拒絕",
    "未成年",
    "未滿",
    "性內容",
    "拒絕分析",
    "不予處理",
    "不予協助",
    "自慰",
    "性交",
    "下體",
    "裸體",
    "性化",
    "露骨",
    "調教",
    "褲襠",
    "慾火",
    "欲望核心",
    "原則",
]


@dataclass(frozen=True)
class DistilledLibrary:
    source_label: str
    mode: str
    technique_library: str
    memory_insert: str
    director_instruction: str
    output_path: Path


def distill_full_report_to_agent_library(
    report_file: Any,
    report_path: str,
    report_text: str,
    distill_goal: str,
    output_language: str,
    max_report_chars: float | int | None,
    max_library_chars: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None, str, str, str]:
    """Gradio entry point: full_report.md -> compact technique library."""
    try:
        raw_text, source_label = load_report_source(report_file, report_path, report_text)
        if not raw_text.strip():
            return "[ERROR] 請上傳 full_report.md、貼上報告內容，或輸入報告路徑。", "", None, "", "", ""

        goal = (distill_goal or DEFAULT_REPORT_DISTILL_GOAL).strip()
        max_source = to_positive_int(max_report_chars) or 45000
        max_library = to_positive_int(max_library_chars) or 9000
        distillation_source = build_distillation_source(raw_text, max_source)

        if dry_run:
            library = build_local_compact_library(distillation_source, goal, output_language, max_library)
            mode = "Dry Run / Local Extract"
        else:
            if not analysis_base_url.strip() or not analysis_model_name.strip():
                return "[ERROR] 請先設定 Analysis / Grok 的 Base URL 與 Model Name。", "", None, "", "", ""
            client = OpenAI(
                api_key=(analysis_api_key or "not-needed").strip(),
                base_url=analysis_base_url.strip().rstrip("/"),
                timeout=900,
            )
            library = ask_grok_to_distill_library(
                client=client,
                model_name=analysis_model_name.strip(),
                source_text=distillation_source,
                goal=goal,
                output_language=output_language or "繁體中文",
                max_library_chars=max_library,
            )
            mode = f"Grok / {analysis_model_name.strip()}"

        sections = parse_distilled_sections(library)
        output_path, preview = write_distilled_library(
            source_label=source_label,
            mode=mode,
            goal=goal,
            raw_library=library,
            technique_library=sections["technique_library"],
            memory_insert=sections["memory_insert"],
            director_instruction=sections["director_instruction"],
        )
        status = (
            "[OK] full_report 已蒸餾成寫作 AGENT 精簡技法庫。\n"
            f"來源：{source_label}\n"
            f"模式：{mode}\n"
            f"技法庫字數：約 {len(sections['technique_library'])}\n"
            f"報告：{output_path}"
        )
        return (
            status,
            trim_text(preview, max_library + 6000),
            str(output_path),
            sections["technique_library"],
            sections["memory_insert"],
            sections["director_instruction"],
        )
    except ValueError as exc:
        # Bad/missing input paths surfaced by load_report_source(); user-facing.
        logger.warning("Distillation input error: %s", exc)
        return f"[ERROR] {exc}", "", None, "", "", ""
    except OpenAIError as exc:
        logger.exception("Distillation API error")
        return f"[ERROR] 分析 API 失敗：{exc}", "", None, "", "", ""
    except OSError as exc:
        # File I/O: missing dir, disk full, permission denied, etc.
        logger.exception("Distillation file I/O error")
        return f"[ERROR] 檔案讀寫失敗：{exc}", "", None, "", "", ""
    except Exception as exc:  # noqa: BLE001 - last-resort guard keeps the UI responsive
        logger.exception("Unexpected error during report distillation")
        return f"[ERROR] {exc}", "", None, "", "", ""


def load_distilled_library_to_agent_fields(
    technique_library: str,
    memory_insert: str,
    director_instruction: str,
    current_technique_library: str,
    current_memory: str,
    current_director_instruction: str,
    load_mode: str,
) -> tuple[str, str, str, str]:
    """Load a distilled library into writing-agent reference fields."""
    if not (technique_library or "").strip():
        return (
            current_technique_library or "",
            current_memory or "",
            current_director_instruction or "",
            "[ERROR] 目前沒有可載入的蒸餾技法庫，請先蒸餾 full_report.md。",
        )

    mode = load_mode or TECHNIQUE_LOAD_MODES[0]
    replace_library = mode.startswith("Replace")
    include_memory_director = "Memory/Director" in mode

    new_library = merge_text_field(
        current=current_technique_library,
        incoming=technique_library,
        header="Imported Technique Library",
        replace=replace_library,
    )
    new_memory = current_memory or ""
    new_instruction = current_director_instruction or ""
    if include_memory_director:
        new_memory = merge_text_field(
            current=current_memory,
            incoming=memory_insert,
            header="Technique Memory Insert",
            replace=False,
        )
        new_instruction = merge_text_field(
            current=current_director_instruction,
            incoming=director_instruction,
            header="Technique Director Instruction",
            replace=False,
        )

    return (
        new_library,
        new_memory,
        new_instruction,
        "[OK] 已載入寫作 AGENT 參考欄位：Technique Library"
        + ("、Story Memory、Director Instruction。" if include_memory_director else "。"),
    )


def load_latest_distilled_library_to_agent_fields(
    current_technique_library: str,
    current_memory: str,
    current_director_instruction: str,
    load_mode: str,
) -> tuple[str, str, str, str]:
    latest = find_latest_distilled_library()
    if latest is None:
        return (
            current_technique_library or "",
            current_memory or "",
            current_director_instruction or "",
            "[ERROR] 找不到已保存的精簡技法庫，請先蒸餾 full_report.md。",
        )
    return load_distilled_library_to_agent_fields(
        latest.technique_library,
        latest.memory_insert,
        latest.director_instruction,
        current_technique_library,
        current_memory,
        current_director_instruction,
        load_mode,
    )


def load_report_source(report_file: Any, report_path: str, report_text: str) -> tuple[str, str]:
    if report_file:
        path = get_file_path(report_file)
        return read_text(Path(path)), str(Path(path))
    clean_path = (report_path or "").strip().strip('"')
    if clean_path:
        path = Path(clean_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise ValueError(f"找不到報告檔案：{path}")
        return read_text(path), str(path)
    if (report_text or "").strip():
        return report_text, "pasted-report"
    latest = find_latest_full_report()
    if latest:
        # No explicit source given; fall back to the most recent report but log it
        # so an accidental empty-input run is traceable.
        logger.warning("No report source provided; falling back to latest full report: %s", latest)
        return read_text(latest), str(latest)
    return "", ""


def get_file_path(file_value: Any) -> str:
    if isinstance(file_value, (str, Path)):
        return str(file_value)
    name = getattr(file_value, "name", None)
    if name:
        return str(name)
    path = getattr(file_value, "path", None)
    if path:
        return str(path)
    raise ValueError("無法讀取上傳檔案路徑。")


def build_distillation_source(raw_text: str, max_chars: int) -> str:
    text = sanitize_review_text(raw_text)
    headings = extract_headings(text, limit=160)
    technique_lines = [
        line
        for line in extract_keyword_lines(text, limit=650)
        if is_safe_craft_line(line)
    ]
    source = [
        "# Report Headings",
        "\n".join(f"- {heading}" for heading in headings),
        "",
        "# Craft / Technique Lines",
        "\n".join(f"- {line}" for line in technique_lines),
    ]
    compact = "\n".join(part for part in source if part.strip()).strip()
    if len(compact) < 1500:
        compact = sanitize_review_text(raw_text)
    return trim_text(compact, max_chars)


def is_safe_craft_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    if not compact:
        return False
    lower = compact.lower()
    if any(marker.lower() in lower for marker in UNSAFE_LINE_MARKERS):
        return False
    return any(keyword in compact for keyword in SAFE_TECHNIQUE_KEYWORDS)


def build_local_compact_library(source_text: str, goal: str, output_language: str, max_chars: int) -> str:
    lines = [
        re.sub(r"^[-*]\s*", "", line).strip()
        for line in source_text.splitlines()
        if is_safe_craft_line(line)
    ]
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        line = trim_text(line, 240)
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
        if len(unique_lines) >= 90:
            break

    inferred = infer_safe_craft_templates(unique_lines)
    core_rules = inferred["core_rules"]
    scene_rules = inferred["scene_rules"]
    style_rules = inferred["style_rules"]
    if not core_rules:
        core_rules = [
            "以場景功能先行：每段先確定鉤子、主要推進、轉折與收束。",
            "用角色可見動作外化情緒，減少抽象心理說明。",
            "用資訊延遲、短句動作與旁觀者反應製造節奏變化。",
        ]

    director_lines = build_director_lines(core_rules + scene_rules)
    memory_insert = (
        "Technique memory: 寫作時優先套用精簡技法庫中的場景功能、節奏轉折、"
        "人物動作外化、感官配置與章尾鉤子；只借鑑技法，不複製原作情節或措辭。"
    )
    library = [
        "# Agent Compact Technique Library",
        "",
        f"`/goal`: {goal}",
        f"Output language: {output_language or '繁體中文'}",
        "",
        "## Technique Library",
        "",
        "### Core Craft Rules",
        *[f"- {line}" for line in core_rules],
        "",
        "### Scene / Action / Situation Methods",
        *[f"- {line}" for line in scene_rules[:30]],
        "",
        "### Style And Rhythm Rules",
        *[f"- {line}" for line in style_rules[:20]],
        "",
        "## Story Memory Insert",
        "",
        memory_insert,
        "",
        "## Director Instruction Insert",
        "",
        "\n".join(f"- {line}" for line in director_lines),
    ]
    return trim_text("\n".join(library).strip(), max_chars)


def infer_safe_craft_templates(lines: list[str]) -> dict[str, list[str]]:
    joined = "\n".join(lines)

    core_templates = [
        ("場景功能", "先決定本段功能：開場鉤子、主事件推進、人物變化、轉折、收束鉤子。"),
        ("開場鉤子", "開場先給具體動作、異常資訊或關係壓力，再補背景。"),
        ("衝突", "讓每場戲都有可見阻力：角色想做一件事，但被權力、秘密、情感或環境阻擋。"),
        ("人物", "用選擇、退讓、沉默、反擊等行為推進人物，而不是只用旁白說明性格。"),
        ("資訊", "採用「先露出線索，再延遲解釋」的方式釋放設定，讓讀者保持追問。"),
        ("轉折", "段尾安排狀態改變：資訊翻面、關係變冷、目標受阻、危機逼近或下一步行動出現。"),
        ("伏筆", "把伏筆放進物件、短句、旁觀者反應或未完成動作，不要額外解釋。"),
        ("章尾", "章尾保留未解問題、下一個行動或新壓力，形成連載鉤子。"),
    ]
    scene_templates = [
        ("動作", "用一個具體動作外化情緒，例如停頓、避開視線、收緊手指、改變站位。"),
        ("感官", "每個場景只選 2 到 3 種主感官，讓畫面集中，不要平均鋪滿。"),
        ("節奏", "快節奏段落用短句與連續動作，慢節奏段落用停頓、環境聲與內在反應。"),
        ("對話", "對話要同時完成兩件事：表面交換資訊，暗中推動關係或權力位置。"),
        ("旁觀者", "用旁觀者的細微反應放大主角色的壓迫感、魅力或危險度。"),
        ("場面", "先建立空間位置，再安排角色移動，讓讀者知道誰靠近、誰退後、誰掌控入口。"),
        ("視角", "限制視角資訊量，只讓讀者知道視角角色當下能感知或誤判的內容。"),
        ("沉默", "在高壓場景中用沉默、未回答、打斷與改口製造張力。"),
    ]
    style_templates = [
        ("文風", "文風以可觀察物象承載情緒，少用抽象形容詞直接判斷。"),
        ("句式", "句式長短交替：短句承接衝擊，長句承接氛圍與心理流動。"),
        ("意象", "重複意象要服務於情緒變化，每次回來都讓關係或局勢更進一步。"),
        ("描寫", "描寫不要停在外觀，讓描寫同時透露身分、壓力、意圖或風險。"),
        ("模板", "可複製模板：目標出現 -> 阻力介入 -> 角色反應 -> 資訊翻面 -> 留下一個行動鉤子。"),
        ("公式", "場景公式：環境壓力 + 角色目的 + 具體動作 + 關係反應 + 轉折收束。"),
    ]

    return {
        "core_rules": select_templates(joined, core_templates, fallback_count=5),
        "scene_rules": select_templates(joined, scene_templates, fallback_count=6),
        "style_rules": select_templates(joined, style_templates, fallback_count=4),
    }


def select_templates(joined: str, templates: list[tuple[str, str]], fallback_count: int) -> list[str]:
    selected = [template for keyword, template in templates if keyword in joined]
    if len(selected) < fallback_count:
        for _, template in templates:
            if template not in selected:
                selected.append(template)
            if len(selected) >= fallback_count:
                break
    return selected


def build_director_lines(candidates: list[str]) -> list[str]:
    defaults = [
        "先判斷本段場景功能，再安排角色目標、阻力、轉折與收束。",
        "用一個具體動作或感官細節外化角色情緒，不直接解釋情緒。",
        "每 3 到 5 個敘事節拍放入一次資訊釋放、反應或小轉折。",
        "章尾或段尾保留一個未解問題、關係張力或下一步行動鉤子。",
        "借鑑技法，不複製原作情節、句子、角色或敏感內容。",
    ]
    distilled = []
    for line in candidates:
        if any(keyword in line for keyword in ("公式", "模板", "技巧", "轉折", "節奏")):
            distilled.append(line)
        if len(distilled) >= 5:
            break
    # Strategy: prefer up to 5 distilled craft lines, then top up with defaults,
    # capped at 8 total. Defaults always backfill so the result is never empty.
    return (distilled + defaults)[:8]


def ask_grok_to_distill_library(
    *,
    client: OpenAI,
    model_name: str,
    source_text: str,
    goal: str,
    output_language: str,
    max_library_chars: int,
    model_max_output_tokens: int = 6000,
) -> str:
    # Cap requested completion tokens by the model's output budget (minus a small
    # safety margin) so we don't trip 'max_tokens exceeds limit' API errors.
    output_token_ceiling = max(min(model_max_output_tokens - 100, 6000), 256)
    max_tokens = min(max(max_library_chars // 2, 1600), output_token_ceiling)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是小說寫作技法蒸餾 Agent。你的任務是把分析報告壓縮成安全、精簡、"
                        "可直接放進寫作 AGENT prompt 的技法庫。不要引用原文句子，不要延續原作情節，"
                        "不要輸出未成年性內容或其他不適合進入寫作提示詞的內容。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"/goal: {goal}\n"
                        f"Output language: {output_language}\n"
                        f"Maximum output characters: about {max_library_chars}\n\n"
                        "請嚴格使用以下 Markdown 結構輸出：\n"
                        "# Agent Compact Technique Library\n"
                        "## Technique Library\n"
                        "### Core Craft Rules\n"
                        "### Scene / Action / Situation Methods\n"
                        "### Style And Rhythm Rules\n"
                        "## Story Memory Insert\n"
                        "## Director Instruction Insert\n\n"
                        "蒸餾規則：\n"
                        "- 只輸出高階寫作技巧、節奏規則、場景調度與可用指令。\n"
                        "- 不要引用原文，不要保留角色名，不要保留原作情節。\n"
                        "- 如果報告中出現拒絕分析或敏感題材，只轉成安全的避用提醒，不展開內容。\n\n"
                        f"來源報告摘錄：\n{source_text}"
                    ),
                },
            ],
            temperature=0.25,
            max_tokens=max_tokens,
        )
    except (OpenAIError, TimeoutError, ConnectionError) as exc:
        # Network / API failure (incl. the 900s timeout): degrade gracefully to a
        # locally-built compact library instead of crashing the distillation flow.
        logger.warning("Grok distillation API call failed (%s); using local fallback.", exc)
        return build_local_compact_library(source_text, goal, output_language, max_library_chars)

    try:
        content = response.choices[0].message.content
    except (IndexError, AttributeError) as exc:
        logger.warning("Grok distillation returned an unexpected response shape (%s); using local fallback.", exc)
        return build_local_compact_library(source_text, goal, output_language, max_library_chars)

    content = (content or "").strip()
    if not content:
        logger.warning("Grok distillation returned empty content; using local fallback.")
        return build_local_compact_library(source_text, goal, output_language, max_library_chars)
    return content


def parse_distilled_sections(library: str) -> dict[str, str]:
    text = clean_for_agent_prompt(library).strip()
    if not re.search(r"(?m)^##\s+\S", text):
        # Output does not follow the expected '## heading' structure; section
        # extraction will fall back to generic content below.
        logger.warning("Distilled library lacks expected '## ' headings; using fallback sections.")
    technique = extract_section(text, "Technique Library")
    memory = extract_section(text, "Story Memory Insert")
    director = extract_section(text, "Director Instruction Insert")
    if not technique:
        logger.warning("Technique Library section missing; falling back to full distilled text.")
        technique = text
    if not memory:
        logger.warning("Story Memory Insert section missing; using default memory insert.")
        memory = "Technique memory: Use the compact technique library as soft craft guidance; do not copy source plot or wording."
    if not director:
        logger.warning("Director Instruction Insert section missing; synthesizing director lines.")
        director = "\n".join(build_director_lines([line for line in text.splitlines() if is_safe_craft_line(line)]))
    return {
        "technique_library": technique.strip(),
        "memory_insert": memory.strip(),
        "director_instruction": director.strip(),
    }


def extract_section(text: str, title: str) -> str:
    if not text:
        return ""
    # Tolerant of malformed markdown: allow leading whitespace, optional bold
    # markers around the title, and trailing text on the heading line. Stops at
    # the next level-1/2 heading or end of text.
    pattern = re.compile(
        rf"^[ \t]*#{{1,2}}\s+\**{re.escape(title)}\**.*$([\s\S]*?)(?=^[ \t]*#{{1,2}}\s+|\Z)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def write_distilled_library(
    *,
    source_label: str,
    mode: str,
    goal: str,
    raw_library: str,
    technique_library: str,
    memory_insert: str,
    director_instruction: str,
) -> tuple[Path, str]:
    raw_library = clean_for_agent_prompt(raw_library)
    technique_library = clean_for_agent_prompt(technique_library)
    memory_insert = clean_for_agent_prompt(memory_insert)
    director_instruction = clean_for_agent_prompt(director_instruction)
    output_dir = Path.cwd() / "book_output" / "report_technique_libraries" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "agent_technique_library.md"
    markdown = "\n".join(
        [
            "# Agent Compact Technique Library",
            "",
            f"- Source: {source_label}",
            f"- Mode: {mode}",
            f"- Goal: {goal}",
            "",
            "## Technique Library",
            "",
            technique_library,
            "",
            "## Story Memory Insert",
            "",
            memory_insert,
            "",
            "## Director Instruction Insert",
            "",
            director_instruction,
            "",
            "## Raw Distillation",
            "",
            raw_library,
        ]
    ).strip()
    output_path.write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "agent_technique_library.json").write_text(
        json.dumps(
            {
                "source": source_label,
                "mode": mode,
                "goal": goal,
                "technique_library": technique_library,
                "memory_insert": memory_insert,
                "director_instruction": director_instruction,
                "raw_distillation": raw_library,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Return the markdown we just wrote so callers don't re-read it from disk
    # (avoids a redundant read that could fail/mask the real error).
    return output_path, markdown


def find_latest_full_report() -> Path | None:
    files = [path for path in Path.cwd().glob("book_output/chapter_craft_reports/*/full_report.md") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def find_latest_distilled_library() -> DistilledLibrary | None:
    files = [
        path
        for path in Path.cwd().glob("book_output/report_technique_libraries/*/agent_technique_library.json")
        if path.is_file()
    ]
    if not files:
        return None
    # Prefer the real file mtime as the primary key; directory name (timestamp)
    # is only a stable tiebreaker. Avoids loading an older library on collisions.
    path = max(files, key=lambda item: (item.stat().st_mtime, item.parent.name))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping corrupted distilled library %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Skipping distilled library with unexpected JSON shape: %s", path)
        return None
    return DistilledLibrary(
        source_label=payload.get("source", str(path)),
        mode=payload.get("mode", "saved"),
        technique_library=clean_for_agent_prompt(payload.get("technique_library", "")),
        memory_insert=clean_for_agent_prompt(payload.get("memory_insert", "")),
        director_instruction=clean_for_agent_prompt(payload.get("director_instruction", "")),
        output_path=path.with_suffix(".md"),
    )


def merge_text_field(current: str, incoming: str, header: str, replace: bool) -> str:
    incoming = (incoming or "").strip()
    if not incoming:
        return current or ""
    # Keep the header on a single line so the '[header]' block stays well-formed
    # even if a caller ever passes a multi-line header.
    header = " ".join(str(header).split())
    block = f"[{header}]\n{incoming}"
    if replace:
        return incoming
    current = (current or "").strip()
    return f"{current}\n\n{block}".strip() if current else incoming


def clean_for_agent_prompt(text: str) -> str:
    # Delegates to the shared sanitize-then-filter helper in skill_technique_review
    # so the line-dropping logic lives in exactly one place.
    return drop_unsafe_lines(text, UNSAFE_LINE_MARKERS)
