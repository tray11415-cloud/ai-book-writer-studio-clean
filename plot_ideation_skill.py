"""Plot arrangement and ideation workflow built on the chapter craft skill."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from chapter_craft_skill import (
    Chapter,
    load_chapters,
    normalize_text,
    slugify,
    to_positive_int,
    trim_chapter_text,
    trim_preview,
)
from lora_runtime import LORA_BASE_URL, LORA_MODEL_NAME, ensure_lora_server_running


DEFAULT_PLOT_GOAL = (
    "根據故事種子、既有設定與參考文本技法，發想可長篇連載的劇情編排："
    "包含故事引擎、章節節拍、衝突升級、角色弧線、場景清單、章尾鉤子與可直接交給寫作區使用的指令。"
)

PLOT_ROLES = {
    "engine": {
        "title": "故事引擎策劃",
        "focus": "核心賣點、主角欲望、長線問題、讀者期待、可連載的推進燃料。",
    },
    "structure": {
        "title": "章節編排師",
        "focus": "章節順序、三幕/多幕節拍、每章功能、轉折、章尾鉤子。",
    },
    "conflict": {
        "title": "衝突設計師",
        "focus": "外部阻力、內在矛盾、關係張力、秘密與代價、升級節點。",
    },
    "scene": {
        "title": "場景發想師",
        "focus": "具體可寫場景、場面調度、情緒落點、感官元素、對白火花。",
    },
}


@dataclass(frozen=True)
class PlotRoleResult:
    role: str
    title: str
    model_label: str
    output: str


@dataclass(frozen=True)
class ModelRoute:
    label: str
    client: OpenAI | None
    model_name: str


PLOT_ROLE_ROUTES = {
    "engine": "Grok",
    "structure": "NALANG",
    "conflict": "Grok",
    "scene": "LoRA",
}


def generate_plot_ideation(
    reference_txt_file: Any,
    reference_text: str,
    reference_url: str,
    premise: str,
    plot_goal: str,
    genre_tone: str,
    arc_mode: str,
    target_chapters: float | int | None,
    output_language: str,
    reference_limit: float | int | None,
    max_reference_chars: float | int | None,
    dry_run: bool,
    background: str,
    roles: Any,
    lore: Any,
    memory: str,
    style_dna: str,
    chronicle: str,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
    writing_api_key: str,
    writing_base_url: str,
    writing_model_name: str,
    lora_base_url: str,
    lora_model_name: str,
) -> tuple[str, str, str | None]:
    """Gradio entry point for plot arrangement and brainstorming."""
    try:
        premise = normalize_text(premise)
        context = build_project_context(background, roles, lore, memory, style_dna, chronicle)
        if not premise and not context and not (reference_text or reference_txt_file or reference_url):
            return "[ERROR] 請至少提供故事種子、既有設定、參考文本或小說目錄網址。", "", None

        goal = normalize_text(plot_goal) or DEFAULT_PLOT_GOAL
        chapter_count = to_positive_int(target_chapters) or 12
        ref_limit = to_positive_int(reference_limit) or 3
        max_chars = to_positive_int(max_reference_chars) or 9000
        source_label = "no-reference"
        reference_chapters: list[Chapter] = []

        if reference_txt_file or reference_text or reference_url:
            reference_chapters, source_label = load_chapters(
                txt_file=reference_txt_file,
                pasted_text=reference_text,
                directory_url=reference_url,
                limit=ref_limit,
                fallback_chunk_chars=max_chars,
            )

        routes = build_dry_run_routes()
        if not dry_run:
            routes = build_model_routes(
                analysis_api_key=analysis_api_key,
                analysis_base_url=analysis_base_url,
                analysis_model_name=analysis_model_name,
                writing_api_key=writing_api_key,
                writing_base_url=writing_base_url,
                writing_model_name=writing_model_name,
                lora_base_url=lora_base_url,
                lora_model_name=lora_model_name,
            )

        craft_dna = extract_reference_craft_dna(
            chapters=reference_chapters,
            max_chars=max_chars,
            dry_run=dry_run,
            route=routes["Grok"],
            goal=goal,
            output_language=output_language,
        )
        role_results = run_plot_cluster(
            routes=routes,
            dry_run=dry_run,
            premise=premise,
            context=context,
            craft_dna=craft_dna,
            goal=goal,
            genre_tone=genre_tone,
            arc_mode=arc_mode,
            chapter_count=chapter_count,
            output_language=output_language,
        )
        synthesis = synthesize_plot_plan(
            route=routes["NALANG"],
            dry_run=dry_run,
            premise=premise,
            context=context,
            craft_dna=craft_dna,
            role_results=role_results,
            goal=goal,
            genre_tone=genre_tone,
            arc_mode=arc_mode,
            chapter_count=chapter_count,
            output_language=output_language,
        )

        output_dir = write_plot_outputs(
            source_label=source_label,
            premise=premise,
            context=context,
            craft_dna=craft_dna,
            role_results=role_results,
            synthesis=synthesis,
            goal=goal,
        )
        report_path = output_dir / "plot_ideation.md"
        preview = report_path.read_text(encoding="utf-8")
        status = (
            f"[OK] 劇情編排完成。\n"
            f"目標章數：{chapter_count}\n"
            f"參考來源：{source_label}\n"
            f"模式：{'Dry Run' if dry_run else 'Grok + NALANG + LoRA'}\n"
            f"報告：{report_path}"
        )
        return status, trim_preview(preview), str(report_path)
    except Exception as exc:
        return f"[ERROR] {exc}", "", None


def extract_reference_craft_dna(
    *,
    chapters: list[Chapter],
    max_chars: int,
    dry_run: bool,
    route: ModelRoute,
    goal: str,
    output_language: str,
) -> str:
    if not chapters:
        return "未提供參考文本；請依故事種子與專案設定自行建立技法方向。"

    sample = "\n\n".join(
        f"## 第 {chapter.index} 章｜{chapter.title}\n"
        f"{trim_chapter_text(chapter.text, max_chars // max(len(chapters), 1))}"
        for chapter in chapters[:5]
    )
    if dry_run:
        return dry_run_craft_dna(chapters)
    if route.client is None:
        raise RuntimeError("缺少 LLM client。")

    response = route.client.chat.completions.create(
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是小說技法蒸餾 Agent。你會把參考文本濃縮成可移植的寫作技法，"
                    "不抄原文、不延續原作情節，只萃取結構、節奏、人物與文風方法。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n\n"
                    "請從參考章節萃取「技法 DNA」，固定輸出：\n"
                    "- 故事引擎\n"
                    "- 章節節拍\n"
                    "- 衝突升級方式\n"
                    "- 角色推進方式\n"
                    "- 章尾鉤子模式\n"
                    "- 可借用但不抄襲的 8 條寫作規則\n\n"
                    f"參考章節：\n{sample}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1800,
    )
    return read_message(response)


def run_plot_cluster(
    *,
    routes: dict[str, ModelRoute],
    dry_run: bool,
    premise: str,
    context: str,
    craft_dna: str,
    goal: str,
    genre_tone: str,
    arc_mode: str,
    chapter_count: int,
    output_language: str,
) -> list[PlotRoleResult]:
    if dry_run:
        return [
            PlotRoleResult(
                role=role,
                title=spec["title"],
                model_label=PLOT_ROLE_ROUTES.get(role, "Dry Run"),
                output=dry_run_role_output(
                    spec["title"],
                    spec["focus"],
                    chapter_count,
                    PLOT_ROLE_ROUTES.get(role, "Dry Run"),
                ),
            )
            for role, spec in PLOT_ROLES.items()
        ]

    results: list[PlotRoleResult] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                ask_plot_role,
                routes[PLOT_ROLE_ROUTES[role]],
                role,
                spec["title"],
                spec["focus"],
                premise,
                context,
                craft_dna,
                goal,
                genre_tone,
                arc_mode,
                chapter_count,
                output_language,
            ): role
            for role, spec in PLOT_ROLES.items()
        }
        for future in as_completed(futures):
            results.append(future.result())

    role_order = list(PLOT_ROLES)
    return sorted(results, key=lambda item: role_order.index(item.role))


def ask_plot_role(
    route: ModelRoute,
    role: str,
    title: str,
    focus: str,
    premise: str,
    context: str,
    craft_dna: str,
    goal: str,
    genre_tone: str,
    arc_mode: str,
    chapter_count: int,
    output_language: str,
) -> PlotRoleResult:
    if route.client is None:
        raise RuntimeError(f"缺少 {route.label} client。")
    response = route.client.chat.completions.create(
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    f"你是「{title}」，負責小說劇情編排與發想。"
                    "給出具體、可寫、可連載的方案，不要只講抽象原則。"
                    "不要抄襲參考文本，只借用技法。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n"
                    f"你的焦點：{focus}\n"
                    f"目標章數：{chapter_count}\n"
                    f"類型/基調：{genre_tone or '未指定'}\n"
                    f"編排模式：{arc_mode or '章節連載'}\n\n"
                    f"故事種子：\n{premise or '未提供'}\n\n"
                    f"專案既有設定：\n{context or '未提供'}\n\n"
                    f"參考技法 DNA：\n{craft_dna}\n\n"
                    "請輸出：\n"
                    "1. 你的核心判斷\n"
                    "2. 具體劇情發想 8-12 點\n"
                    "3. 章節或場景排序建議\n"
                    "4. 可加強讀者期待的鉤子\n"
                    "5. 最適合交給寫作區的 3 條 Director Instruction"
                ),
            },
        ],
        temperature=0.75,
        max_tokens=1800,
    )
    return PlotRoleResult(
        role=role,
        title=title,
        model_label=f"{route.label} / {route.model_name}",
        output=read_message(response),
    )


def synthesize_plot_plan(
    *,
    route: ModelRoute,
    dry_run: bool,
    premise: str,
    context: str,
    craft_dna: str,
    role_results: list[PlotRoleResult],
    goal: str,
    genre_tone: str,
    arc_mode: str,
    chapter_count: int,
    output_language: str,
) -> str:
    if dry_run:
        return dry_run_synthesis(chapter_count, premise, genre_tone, arc_mode)
    if route.client is None:
        raise RuntimeError("缺少 NALANG client。")

    notes = "\n\n".join(f"## {result.title}\n{result.output}" for result in role_results)
    response = route.client.chat.completions.create(
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是總編型劇情編排 Agent，負責把多名發想員的方案整合成可直接使用的長篇小說規劃。"
                    "要具體、有順序、有章節功能、有衝突升級。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n"
                    f"目標章數：{chapter_count}\n"
                    f"類型/基調：{genre_tone or '未指定'}\n"
                    f"編排模式：{arc_mode or '章節連載'}\n\n"
                    f"故事種子：\n{premise or '未提供'}\n\n"
                    f"專案既有設定：\n{context or '未提供'}\n\n"
                    f"參考技法 DNA：\n{craft_dna}\n\n"
                    f"AI 發想叢集筆記：\n{notes}\n\n"
                    "請整合成固定格式：\n"
                    "- 一句話賣點\n"
                    "- 核心劇情引擎\n"
                    "- 主角目標、阻力、代價、秘密\n"
                    "- 整體篇章節奏圖\n"
                    f"- {chapter_count} 章章節表：章名、章節功能、主要事件、人物變化、衝突升級、章尾鉤子\n"
                    "- 10 個可直接寫成場景的場面\n"
                    "- 5 條可複製的寫作技法\n"
                    "- 可貼進 Interactive Writing 的 Director Instruction 5 條"
                ),
            },
        ],
        temperature=0.55,
        max_tokens=3600,
    )
    return read_message(response)


def write_plot_outputs(
    *,
    source_label: str,
    premise: str,
    context: str,
    craft_dna: str,
    role_results: list[PlotRoleResult],
    synthesis: str,
    goal: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "book_output" / "plot_ideation" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_label": source_label,
        "premise": premise,
        "context": context,
        "craft_dna": craft_dna,
        "goal": goal,
        "role_results": [
            {
                "role": result.role,
                "title": result.title,
                "model_label": result.model_label,
                "output": result.output,
            }
            for result in role_results
        ],
        "synthesis": synthesis,
    }
    (output_dir / "plot_ideation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "craft_dna.md").write_text("# 技法 DNA\n\n" + craft_dna + "\n", encoding="utf-8")

    role_sections = "\n\n".join(
        f"## {result.title}\n\n模型：{result.model_label}\n\n{result.output}"
        for result in role_results
    )
    report = (
        "# 劇情編排 / 發想報告\n\n"
        f"`/goal`：{goal}\n\n"
        f"參考來源：{source_label}\n\n"
        "## 故事種子\n\n"
        f"{premise or '未提供'}\n\n"
        "## 專案既有設定\n\n"
        f"{context or '未提供'}\n\n"
        "## 參考技法 DNA\n\n"
        f"{craft_dna}\n\n"
        "## 總編整合方案\n\n"
        f"{synthesis}\n\n"
        "## AI 發想叢集筆記\n\n"
        f"{role_sections}\n"
    )
    (output_dir / "plot_ideation.md").write_text(report, encoding="utf-8")
    return output_dir


def build_project_context(
    background: str,
    roles: Any,
    lore: Any,
    memory: str,
    style_dna: str,
    chronicle: str,
) -> str:
    sections = []
    if normalize_text(background):
        sections.append("World / Story Background:\n" + normalize_text(background))
    role_block = rows_to_context("Characters", roles, ["Name", "Role", "Traits"])
    if role_block:
        sections.append(role_block)
    lore_block = rows_to_context("Lorebook", lore, ["Keyword", "Content"])
    if lore_block:
        sections.append(lore_block)
    if normalize_text(memory):
        sections.append("Story Memory:\n" + normalize_text(memory))
    if normalize_text(style_dna):
        sections.append("Style DNA:\n" + normalize_text(style_dna))
    if normalize_text(chronicle):
        sections.append("Story Chronicle:\n" + normalize_text(chronicle))
    return "\n\n".join(sections)


def rows_to_context(title: str, rows: Any, labels: list[str]) -> str:
    normalized = []
    for row in rows or []:
        cells = [str(cell or "").strip() for cell in list(row)[: len(labels)]]
        if any(cells):
            normalized.append(cells)
    if not normalized:
        return ""
    lines = [title + ":"]
    for row in normalized:
        parts = [f"{labels[idx]}: {value}" for idx, value in enumerate(row) if value]
        if parts:
            lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def dry_run_craft_dna(chapters: list[Chapter]) -> str:
    titles = "、".join(chapter.title for chapter in chapters[:5])
    return (
        f"Dry Run 技法 DNA：已讀取 {len(chapters)} 個參考章節（{titles}）。\n"
        "- 故事引擎：以明確問題推動章節前進。\n"
        "- 章節節拍：開場鉤子 -> 信息揭示 -> 阻力升級 -> 章尾懸念。\n"
        "- 衝突升級：讓每章多一個代價或秘密。\n"
        "- 角色推進：用選擇暴露欲望，而不是只靠旁白解釋。\n"
        "- 章尾鉤子：留下新危機、新線索或情緒落差。"
    )


def dry_run_role_output(title: str, focus: str, chapter_count: int, model_label: str) -> str:
    return (
        f"離線檢查模式：{title}\n\n"
        f"- 預定模型：{model_label}\n"
        f"- 分析焦點：{focus}\n"
        f"- 會產生 {chapter_count} 章級別的劇情編排。\n"
        "- 正式模式會依故事種子、專案設定與技法 DNA 產生具體章節表。\n"
        "- 可先檢查故事是否有主角目標、阻力、代價與長線懸念。"
    )


def dry_run_synthesis(chapter_count: int, premise: str, genre_tone: str, arc_mode: str) -> str:
    return (
        "## 離線總編方案\n\n"
        f"目標章數：{chapter_count}\n\n"
        f"類型/基調：{genre_tone or '未指定'}\n\n"
        f"編排模式：{arc_mode or '章節連載'}\n\n"
        f"故事種子：{premise or '未提供'}\n\n"
        "正式模式會輸出一句話賣點、核心劇情引擎、章節表、場景清單、寫作技法與 Director Instruction。"
    )


def language_instruction(output_language: str) -> str:
    mapping = {
        "繁体中文": "請使用繁體中文。",
        "繁體中文": "請使用繁體中文。",
        "简体中文": "请使用简体中文。",
        "English": "Use English.",
        "日本語": "日本語で出力してください。",
    }
    return mapping.get(output_language or "", "請使用繁體中文。")


def read_message(response: Any) -> str:
    return (response.choices[0].message.content or "").strip()


def build_dry_run_routes() -> dict[str, ModelRoute]:
    return {
        "Grok": ModelRoute("Grok", None, "dry-run-grok"),
        "NALANG": ModelRoute("NALANG", None, "dry-run-nalang"),
        "LoRA": ModelRoute("LoRA", None, LORA_MODEL_NAME),
    }


def build_model_routes(
    *,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
    writing_api_key: str,
    writing_base_url: str,
    writing_model_name: str,
    lora_base_url: str,
    lora_model_name: str,
) -> dict[str, ModelRoute]:
    if not analysis_base_url.strip() or not analysis_model_name.strip():
        raise ValueError("請先設定 Analysis / Grok 的 Base URL 與 Model Name。")
    if not writing_base_url.strip() or not writing_model_name.strip():
        raise ValueError("請先設定 Writing / NALANG 的 Base URL 與 Model Name。")
    if not lora_base_url.strip() or not lora_model_name.strip():
        raise ValueError("請先設定 LoRA 的 Base URL 與 Model Name。")

    lora_base_url = lora_base_url.strip().rstrip("/")
    if lora_base_url.lower() == LORA_BASE_URL.lower():
        ensure_lora_server_running()
    return {
        "Grok": ModelRoute(
            "Grok",
            OpenAI(
                api_key=(analysis_api_key or "not-needed").strip(),
                base_url=analysis_base_url.strip().rstrip("/"),
                timeout=900,
            ),
            analysis_model_name.strip(),
        ),
        "NALANG": ModelRoute(
            "NALANG",
            OpenAI(
                api_key=(writing_api_key or "not-needed").strip(),
                base_url=writing_base_url.strip().rstrip("/"),
                timeout=900,
            ),
            writing_model_name.strip(),
        ),
        "LoRA": ModelRoute(
            "LoRA",
            OpenAI(api_key="not-needed", base_url=lora_base_url, timeout=900),
            lora_model_name.strip(),
        ),
    }
