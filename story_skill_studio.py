"""Story-Skill Studio: distill a novel's craft into a reusable, plot-bound skill,
orchestrate an original technique-bound writing prompt from it, then write with it.

The pipeline has three stages, all content-agnostic (the skill carries the source's
*how*, never its *what* — no source characters, places, objects, or concrete events):

1. distill_story_skill   — input novel -> a structured "skill" JSON that binds
   plot arrangement (劇情安排) to description technique (描寫技法): a beat_template
   where every beat references the description_techniques it applies.
2. orchestrate_story_prompt — skill + a NEW story seed -> an original plot plan
   (brand-new people/things/events) and an excellent, beat-by-beat technique-bound
   writing prompt. This is the skill-driven Plot Ideation step.
3. load_orchestration_to_agent_fields — push the orchestrated prompt + techniques +
   beat plan into the writing agent's fields so Interactive Writing writes with it.

It reuses the proven helpers from chapter_craft_skill (LLM call, language mapping,
chapter loading, IO) and the model-route pattern from plot_ideation_skill.
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

import repetition_guard as rg
from chapter_craft_skill import (
    API_TIMEOUT,
    SPECIALIST_MAX_WORKERS,
    Chapter,
    chat_complete,
    language_instruction,
    load_chapters,
    normalize_text,
    to_positive_int,
    trim_chapter_text,
    trim_preview,
    write_text_with_backup,
)

logger = logging.getLogger(__name__)


DEFAULT_SKILL_DISTILL_GOAL = (
    "把參考小說蒸餾成『可移植的寫作技能』：深度綁定劇情節拍與描寫技法，"
    "讓每個敘事節拍都標註它使用了哪些描寫技巧。"
    "只萃取敘事方式、描寫技法與節拍結構模板；"
    "嚴禁保留原作的任何人名、地名、物件或具體情節事件——只要『怎麼寫』，不要『寫了什麼』。"
)

# Max output tokens for the continuation director's manual (env-overridable).
# Larger = a longer/more detailed plan, but remember the plan is fed into the
# WRITING model as the Story Instruction, so it competes with the story context;
# keep it well under the writing model's context window.
CONTINUATION_PLAN_MAX_TOKENS = int(os.getenv("BOOK_WRITER_CONTINUATION_PLAN_MAX_TOKENS", "16000"))
# Each beat is expanded in its OWN LLM call (so the model can never merge beats into
# ranges or truncate after the first), run in parallel. This caps how many beats a
# single plan may request, to bound cost / number of calls.
CONTINUATION_MAX_BEATS = max(1, int(os.getenv("BOOK_WRITER_CONTINUATION_MAX_BEATS", "60")))

DEFAULT_ORCHESTRATION_GOAL = (
    "根據蒸餾出的寫作技能與一個全新的故事種子，編排一個全新原創故事："
    "依技能的節拍模板排出章節，並為每個節拍標註要套用的描寫技法，"
    "最後產出一段可直接貼進寫作區的高品質技法綁定提示詞。"
    "嚴禁沿用任何來源故事的人事物或情節。"
)

SKILL_LOAD_MODES = [
    "Replace writing fields (system prompt + techniques + beat plan)",
    "Append to writing fields",
]

# The JSON shape the distiller LLM must return. Kept compact and explicit so the
# model fills every field and binds beats to techniques by id.
SKILL_JSON_SPEC = """{
  "schema_version": 1,
  "narrative_method": {
    "pov": "視角（第一/第三限知/全知…）",
    "tense": "時態與時間處理",
    "narration_distance": "敘事距離（貼近內心/冷眼旁觀…）",
    "voice_register": "語感/腔調（華麗、冷硬、口語…）",
    "dialogue_style": "對白風格與比例",
    "scene_vs_summary": "場景 vs 概述的拿捏"
  },
  "description_techniques": [
    {
      "id": "t1",
      "name": "技法名稱（例：眼神落點分鏡）",
      "category": "容顏描寫/身材描寫/動作描寫/環境氛圍/感官/情緒…",
      "when_to_use": "在什麼情境/節拍適合用",
      "how": "具體手法（可操作，不空泛）",
      "sentence_rhythm": "句長、停頓、標點節奏",
      "sensory_layering": "先帶哪個感官、層次與數量",
      "word_palette": "偏好的具體動詞/名詞/質地；避免的抽象詞",
      "weak_vs_strong": "弱寫法：… / 強寫法：…"
    }
  ],
  "plot_arrangement": {
    "story_engine_pattern": "推進故事的核心引擎模式",
    "pacing_curve": "整體節奏曲線",
    "escalation_pattern": "衝突/賭注如何逐步升級",
    "hook_pattern": "章尾/段尾鉤子模式",
    "arc_shape": "整體弧線形狀（抽象，不含原作情節）"
  },
  "beat_template": [
    {
      "beat_id": "b1",
      "function": "節拍功能（開場鉤子/失衡/第一道阻力/揭示/升級/最低點/高潮/餘韻…）",
      "purpose": "這個節拍要達成什麼",
      "pacing": "快/中/慢",
      "emotional_target": "讀者該有的情緒落點",
      "bound_technique_ids": ["t1", "t3"],
      "technique_application": "在這個節拍如何具體套用上述技法"
    }
  ],
  "transferable_rules": ["8-12 條可借用但不抄襲的寫作規則"],
  "abstraction_note": "本技能不含任何原作人名、地名、物件或具體事件。"
}"""


@dataclass(frozen=True)
class ModelRoute:
    label: str
    client: OpenAI | None
    model_name: str


def _make_route(label: str, api_key: str, base_url: str, model_name: str) -> ModelRoute:
    base = (base_url or "").strip().rstrip("/")
    model = (model_name or "").strip()
    if not base or not model:
        raise ValueError(f"請先設定 {label} 的 Base URL 與 Model Name。")
    client = OpenAI(api_key=(api_key or "not-needed").strip(), base_url=base, timeout=API_TIMEOUT)
    return ModelRoute(label, client, model)


# --------------------------------------------------------------------------- #
# Stage 1: distill an input novel into a bound, content-stripped skill JSON.
# --------------------------------------------------------------------------- #

def distill_story_skill(
    reference_txt_file: Any,
    reference_text: str,
    reference_url: str,
    distill_goal: str,
    output_language: str,
    reference_limit: float | int | None,
    max_reference_chars: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None, str]:
    """Gradio entry: reference novel -> bound skill JSON + markdown preview.

    Returns (status, preview_markdown, skill_json_path, skill_json_str).
    """
    try:
        goal = normalize_text(distill_goal) or DEFAULT_SKILL_DISTILL_GOAL
        ref_limit = to_positive_int(reference_limit) or 5
        max_chars = to_positive_int(max_reference_chars) or 12000

        if not (reference_txt_file or (reference_text or "").strip() or (reference_url or "").strip()):
            return "[ERROR] 請上傳參考小說 TXT、貼上正文，或輸入章節目錄網址。", "", None, ""

        chapters, source_label = load_chapters(
            txt_file=reference_txt_file,
            pasted_text=reference_text,
            directory_url=reference_url,
            limit=ref_limit,
            fallback_chunk_chars=max_chars,
        )
        if not chapters:
            return "[ERROR] 未能從參考來源讀到任何章節內容。", "", None, ""

        if dry_run:
            skill = _dry_run_skill(source_label, chapters)
        else:
            route = _make_route("Analysis / Grok", analysis_api_key, analysis_base_url, analysis_model_name)
            skill = _distill_skill_via_llm(
                route=route,
                chapters=chapters,
                goal=goal,
                output_language=output_language,
                max_chars=max_chars,
            )

        skill["source_label"] = source_label
        skill["distilled_at"] = datetime.now().isoformat(timespec="seconds")
        skill_json_str = json.dumps(skill, ensure_ascii=False, indent=2)

        output_dir = Path.cwd() / "book_output" / "story_skills" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        skill_path = output_dir / "story_skill.json"
        write_text_with_backup(skill_path, skill_json_str)
        preview_md = render_skill_markdown(skill)
        write_text_with_backup(output_dir / "story_skill.md", preview_md)

        n_tech = len(skill.get("description_techniques") or [])
        n_beats = len(skill.get("beat_template") or [])
        status = (
            f"[OK] 技能蒸餾完成。\n"
            f"參考來源：{source_label}（{len(chapters)} 章）\n"
            f"描寫技法：{n_tech} 條；劇情節拍：{n_beats} 拍（已綁定技法）\n"
            f"模式：{'Dry Run' if dry_run else 'Grok 結構化蒸餾'}\n"
            f"技能檔：{skill_path}"
        )
        return status, trim_preview(preview_md), str(skill_path), skill_json_str
    except Exception as exc:  # noqa: BLE001 — surface a clean message to the UI
        logger.exception("distill_story_skill failed")
        return f"[ERROR] {exc}", "", None, ""


def _distill_skill_via_llm(
    *,
    route: ModelRoute,
    chapters: list[Chapter],
    goal: str,
    output_language: str,
    max_chars: int,
) -> dict[str, Any]:
    if route.client is None:
        raise RuntimeError("缺少 Analysis LLM client。")
    per_chapter = max(max_chars // max(len(chapters), 1), 800)
    sample = "\n\n".join(
        f"## 參考章節 {chapter.index}｜{chapter.title}\n{trim_chapter_text(chapter.text, per_chapter)}"
        for chapter in chapters
    )
    raw = chat_complete(
        route.client,
        label="技能蒸餾",
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是頂尖的小說『寫作技能蒸餾 Agent』。你的工作是把參考文本萃取成一份"
                    "**可移植、可重用的寫作技能**，核心要求是把『劇情安排』與『描寫技法』深度綁定："
                    "先抽出敘事方式與描寫技法，再抽出抽象的節拍結構模板，"
                    "然後在每個節拍上用 bound_technique_ids 明確標出它使用了哪些描寫技法。\n"
                    "絕對禁止：輸出原作的任何人名、地名、門派、物件、招式名或具體情節事件。"
                    "只能保留『怎麼寫』（方法、結構、節奏、技巧），不能保留『寫了什麼』（內容）。\n"
                    "只輸出 JSON，不要任何解說或 markdown code fence。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n\n"
                    "請嚴格依下列 JSON schema 輸出（鍵名保持英文，值用指定語言；"
                    "description_techniques 給 10-18 條，beat_template 給 10-20 拍，"
                    "每一拍的 bound_technique_ids 必須對應到 description_techniques 的 id）：\n"
                    f"{SKILL_JSON_SPEC}\n\n"
                    f"參考文本：\n{sample}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=6000,
    )
    skill = _parse_skill_json(raw)
    return _normalize_skill(skill)


def _loads_json_dict(raw: str, what: str = "JSON") -> dict[str, Any]:
    """Parse an LLM JSON object, tolerating code fences and surrounding prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{what}的 LLM 回應不是有效 JSON。請改用結構較穩的 Analysis 模型，或重試。") from exc
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError(f"{what}的 LLM 回應未包含 JSON 物件（請改用結構較穩的 Analysis 模型，或重試）。")


def _parse_skill_json(raw: str) -> dict[str, Any]:
    return _loads_json_dict(raw, "技能蒸餾")


def _normalize_skill(skill: dict[str, Any]) -> dict[str, Any]:
    """Coerce the parsed skill into the expected shape so downstream code is safe."""
    skill = dict(skill or {})
    skill.setdefault("schema_version", 1)
    nm = skill.get("narrative_method")
    skill["narrative_method"] = nm if isinstance(nm, dict) else {}
    pa = skill.get("plot_arrangement")
    skill["plot_arrangement"] = pa if isinstance(pa, dict) else {}
    techs = skill.get("description_techniques")
    skill["description_techniques"] = [t for t in techs if isinstance(t, dict)] if isinstance(techs, list) else []
    beats = skill.get("beat_template")
    skill["beat_template"] = [b for b in beats if isinstance(b, dict)] if isinstance(beats, list) else []
    rules = skill.get("transferable_rules")
    skill["transferable_rules"] = [str(r) for r in rules] if isinstance(rules, list) else []
    skill.setdefault("abstraction_note", "本技能不含任何原作人名、地名、物件或具體事件。")
    return skill


# --------------------------------------------------------------------------- #
# Stage 2: orchestrate an original, technique-bound writing prompt (skill-driven
# Plot Ideation) from the skill + a NEW story seed.
# --------------------------------------------------------------------------- #

def orchestrate_story_prompt(
    skill_json_str: str,
    skill_file: Any,
    premise: str,
    genre_tone: str,
    target_chapters: float | int | None,
    orchestration_goal: str,
    output_language: str,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None, str, str, str]:
    """Gradio entry: skill + new premise -> original plan + technique-bound prompt.

    Returns (status, preview_markdown, orchestration_path,
             system_prompt_text, technique_library_text, beat_plan_text).
    """
    try:
        skill = _load_skill_payload(skill_json_str, skill_file)
        premise = normalize_text(premise)
        if not premise:
            return ("[ERROR] 請提供一個全新的故事種子 / 設定（全新人事物，不要沿用來源故事）。", "", None, "", "", "")
        goal = normalize_text(orchestration_goal) or DEFAULT_ORCHESTRATION_GOAL
        chapter_count = to_positive_int(target_chapters) or 12

        # The technique library + beat plan are derived deterministically from the
        # skill so the writing agent always carries the bound craft even offline.
        technique_library_text = render_technique_library(skill)
        beat_plan_text = render_beat_plan(skill)

        if dry_run:
            system_prompt_text = _dry_run_orchestration_prompt(skill, premise, genre_tone, chapter_count)
            plan_md = system_prompt_text
        else:
            route = _make_route("Analysis / Grok", analysis_api_key, analysis_base_url, analysis_model_name)
            system_prompt_text, plan_md = _orchestrate_via_llm(
                route=route,
                skill=skill,
                premise=premise,
                genre_tone=genre_tone,
                chapter_count=chapter_count,
                goal=goal,
                output_language=output_language,
            )

        output_dir = Path.cwd() / "book_output" / "story_skills" / "orchestrations" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_text_with_backup(output_dir / "writing_prompt.md", system_prompt_text)
        write_text_with_backup(output_dir / "technique_library.md", technique_library_text)
        write_text_with_backup(output_dir / "beat_plan.md", beat_plan_text)
        preview_md = (
            "# 技法綁定寫作提示詞（可貼進寫作區）\n\n"
            + system_prompt_text
            + "\n\n---\n\n# 節拍×技法綁定計畫\n\n"
            + beat_plan_text
        )
        write_text_with_backup(output_dir / "orchestration.md", preview_md)

        status = (
            f"[OK] 劇情編排 + 提示詞完成（skill-driven Plot Ideation）。\n"
            f"目標章數：{chapter_count}；技法：{len(skill.get('description_techniques') or [])} 條\n"
            f"模式：{'Dry Run' if dry_run else 'Grok 編排'}\n"
            f"輸出：{output_dir}\n"
            "→ 按「載入到寫作區」把提示詞/技法/節拍灌進 Interactive Writing。"
        )
        return status, trim_preview(preview_md), str(output_dir / "orchestration.md"), system_prompt_text, technique_library_text, beat_plan_text
    except Exception as exc:  # noqa: BLE001
        logger.exception("orchestrate_story_prompt failed")
        return f"[ERROR] {exc}", "", None, "", "", ""


def _orchestrate_via_llm(
    *,
    route: ModelRoute,
    skill: dict[str, Any],
    premise: str,
    genre_tone: str,
    chapter_count: int,
    goal: str,
    output_language: str,
) -> tuple[str, str]:
    if route.client is None:
        raise RuntimeError("缺少 Analysis LLM client。")
    skill_compact = json.dumps(_skill_for_prompt(skill), ensure_ascii=False)
    prompt = chat_complete(
        route.client,
        label="技能編排",
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是頂尖的小說『劇情編排總監 + 提示詞工程師』。你會拿到一份蒸餾出的『寫作技能』(JSON)，"
                    "裡面有敘事方式、描寫技法清單，以及把節拍綁定到技法的 beat_template。\n"
                    "你的任務：依使用者的『全新故事種子』，編排一個**全新原創**故事，"
                    "並產出一段**可直接交給寫作 AI 的高品質提示詞**。提示詞必須：\n"
                    "1) 鎖定技能的敘事方式；2) 依 beat_template 排出章節節拍；"
                    "3) 在每一拍明確標註要套用『哪些描寫技法』以及『怎麼套用』（引用技法名稱）。\n"
                    "鐵則：全新人事物。嚴禁沿用任何來源故事的人名、地名、物件或情節——只借技法與結構。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n"
                    f"目標章數：{chapter_count}\n"
                    f"類型 / 基調：{genre_tone or '未指定'}\n\n"
                    f"全新故事種子：\n{premise}\n\n"
                    f"寫作技能 JSON：\n{skill_compact}\n\n"
                    "請輸出一段完整的『技法綁定寫作提示詞』，固定包含：\n"
                    "## 敘事方式（鎖定）\n"
                    "## 一句話賣點 + 核心劇情引擎（原創）\n"
                    "## 主要人物（全新原創，列 3-5 位：身分、欲望、阻力、秘密）\n"
                    "## 世界 / 設定（全新原創）\n"
                    f"## {chapter_count} 章節拍表：每章＝節拍功能｜主要事件｜情緒落點｜本章要套用的描寫技法（引用技能裡的技法名稱）｜章尾鉤子\n"
                    "## 寫作守則（從技能的 transferable_rules 轉化，可直接遵循）\n"
                    "## 給寫作 AI 的最終 System / Director 指令（一段可貼上即用、明確要求逐拍套用對應技法）"
                ),
            },
        ],
        temperature=0.6,
        max_tokens=5000,
    )
    return prompt, prompt


def _skill_for_prompt(skill: dict[str, Any]) -> dict[str, Any]:
    """Trim the skill to the fields the orchestrator needs, keeping the binding."""
    return {
        "narrative_method": skill.get("narrative_method", {}),
        "plot_arrangement": skill.get("plot_arrangement", {}),
        "description_techniques": [
            {"id": t.get("id"), "name": t.get("name"), "category": t.get("category"), "how": t.get("how")}
            for t in (skill.get("description_techniques") or [])
        ],
        "beat_template": [
            {
                "function": b.get("function"),
                "pacing": b.get("pacing"),
                "emotional_target": b.get("emotional_target"),
                "bound_technique_ids": b.get("bound_technique_ids"),
                "technique_application": b.get("technique_application"),
            }
            for b in (skill.get("beat_template") or [])
        ],
        "transferable_rules": skill.get("transferable_rules", []),
    }


def _load_skill_payload(skill_json_str: str, skill_file: Any) -> dict[str, Any]:
    text = (skill_json_str or "").strip()
    if not text and skill_file:
        from chapter_craft_skill import get_file_path  # local import to avoid cycle at import time

        path = get_file_path(skill_file)
        text = Path(path).read_text(encoding="utf-8")
    if not text:
        raise ValueError("請先蒸餾一份技能，或上傳 story_skill.json。")
    try:
        return _normalize_skill(json.loads(text))
    except json.JSONDecodeError as exc:
        raise RuntimeError("技能 JSON 無法解析；請重新蒸餾或檢查檔案。") from exc


# --------------------------------------------------------------------------- #
# Continuation: attach a novel you want to CONTINUE and read the info needed to
# continue it (its real characters / world / plot-so-far / where it left off).
# Unlike distillation, this DOES keep the source's entities — it is your own
# story to extend, not an abstract craft reference.
# --------------------------------------------------------------------------- #

def read_continuation_source(
    novel_file: Any,
    novel_text: str,
    novel_url: str,
    max_story_chars: float | int | None,
    output_language: str,
    extract_brief: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
    current_background: str,
    current_roles: Any,
    current_memory: str,
) -> tuple[str, str, str, str, Any, str, str, str]:
    """Read a to-be-continued novel and load continuation context into the writing area.

    Returns (status, preview_md, full_story, background, roles, memory, instruction,
    brief_block): full_story = the prose to continue from; background/roles =
    extracted (or kept); memory = current memory + a continuation brief;
    instruction = a continue directive; brief_block = the continuation brief text
    (stashed for the continuation-prompt step).
    """
    empty_roles = current_roles if isinstance(current_roles, list) and current_roles else [["", "", ""]]
    try:
        if not (novel_file or (novel_text or "").strip() or (novel_url or "").strip()):
            return ("[ERROR] 請附上要續寫的小說（上傳 TXT / 貼上正文 / 章節目錄網址）。", "",
                    "", current_background or "", empty_roles, current_memory or "", "", "")
        chapters, source_label = load_chapters(
            txt_file=novel_file, pasted_text=novel_text, directory_url=novel_url,
            limit=None, fallback_chunk_chars=200000,
        )
        full_story = "\n\n".join(ch.text for ch in chapters).strip()
        if not full_story:
            return ("[ERROR] 未能從來源讀到正文。", "", "", current_background or "", empty_roles, current_memory or "", "", "")
        cap = to_positive_int(max_story_chars) or 0
        if cap and len(full_story) > cap:
            # Keep the most recent part — that is what continuation writes from.
            full_story = full_story[-cap:]

        background = current_background or ""
        roles = empty_roles
        brief_lines: list[str] = []
        if extract_brief:
            route = _make_route("Analysis / Grok", analysis_api_key, analysis_base_url, analysis_model_name)
            brief = _extract_continuation_brief(route, chapters, output_language)
            if (brief.get("background") or "").strip():
                background = brief["background"].strip()
            table = [
                [c.get("name", ""), c.get("role", ""), c.get("traits", "")]
                for c in (brief.get("characters") or [])
                if isinstance(c, dict) and (c.get("name") or c.get("role") or c.get("traits"))
            ]
            if table:
                roles = table
            brief_lines = _render_continuation_brief(brief)

        brief_body = "\n".join(brief_lines) if brief_lines else "已載入要續寫的小說正文；請從最後一個場景自然接續。"
        brief_block = f"【續寫資訊（來源：{source_label}）】\n{brief_body}"
        memory = _join_blocks(current_memory, brief_block)
        instruction = (
            "請從目前故事的最後一個場景自然接續往下寫，保持既有人物、世界設定與未解線索的連貫，"
            "延續既有語氣；不要重述已發生的內容，直接推進新情節。"
        )
        preview = (
            f"# 續寫資訊\n\n- 來源：{source_label}\n- 載入正文字數：{len(full_story)}"
            f"\n- 擷取摘要：{'是' if extract_brief else '否（僅載入正文）'}\n\n{brief_block}"
        )
        status = (
            "[OK] 已讀取續寫所需資訊並載入寫作區（Full Story ＋ 背景／角色 ＋ 續寫摘要 ＋ 接續指令）。\n"
            "→ 接著到 Step 3 產生續寫 PROMPT（會套上技能技法並避免重複），再切到『3. 寫作』續寫。"
        )
        return status, trim_preview(preview), full_story, background, roles, memory, instruction, brief_block
    except Exception as exc:  # noqa: BLE001
        logger.exception("read_continuation_source failed")
        return f"[ERROR] {exc}", "", "", current_background or "", empty_roles, current_memory or "", "", ""


def _extract_continuation_brief(route: ModelRoute, chapters: list[Chapter], output_language: str) -> dict[str, Any]:
    if route.client is None:
        raise RuntimeError("缺少 Analysis LLM client。")
    # Bias toward the later chapters — that is where continuation picks up.
    tail = chapters[-6:] if len(chapters) > 6 else chapters
    text = "\n\n".join(f"## {c.title}\n{trim_chapter_text(c.text, 4000)}" for c in tail)
    raw = chat_complete(
        route.client,
        label="續寫資訊擷取",
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是小說『續寫前置分析 Agent』。閱讀使用者要續寫的小說，擷取續寫所需的關鍵資訊，"
                    "以便之後自然接續。只輸出 JSON，不要任何解說或 code fence。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    '請輸出 JSON：{"background":"世界觀與設定摘要",'
                    '"characters":[{"name":"","role":"","traits":""}],'
                    '"plot_so_far":"至今劇情摘要","current_situation":"故事目前停在哪、最後一個場景的狀態與懸念",'
                    '"open_threads":["未解線索或伏筆"],"tone":"語氣基調"}\n\n'
                    f"小說內容（以後段為主）：\n{text}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=2500,
    )
    return _loads_json_dict(raw, "續寫資訊擷取")


def _render_continuation_brief(brief: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if (brief.get("plot_so_far") or "").strip():
        lines.append("劇情至今：" + brief["plot_so_far"].strip())
    if (brief.get("current_situation") or "").strip():
        lines.append("目前情境（接續點）：" + brief["current_situation"].strip())
    threads = brief.get("open_threads") or []
    if threads:
        lines.append("未解線索：" + "；".join(str(t).strip() for t in threads if str(t).strip()))
    if (brief.get("tone") or "").strip():
        lines.append("語氣基調：" + brief["tone"].strip())
    return lines


def generate_continuation_prompt(
    skill_json_str: str,
    skill_file: Any,
    full_story: str,
    brief_text: str,
    direction: str,
    next_chapters: float | int | None,
    output_language: str,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str, str, str, str]:
    """Produce a continuation PROMPT that reuses every prior mechanism:
    the distilled skill (narrative method + technique binding), continuation-aware
    plot orchestration (plan the NEXT beats of the EXISTING story), and the
    repetition guard (mine already-written phrasings into an avoid-list).

    Returns (status, preview, system_prompt, technique_library, instruction, avoid_words)
    — the last four are loaded straight into the writing area.
    """
    try:
        story = (full_story or "").strip()
        if not story:
            return ("[ERROR] 沒有可續寫的正文，請先在 Step 2 載入要續寫的小說。", "", "", "", "", "")

        # Skill is optional — continuation still works (just without locked craft).
        skill: dict[str, Any] = {}
        if (skill_json_str or "").strip() or skill_file:
            try:
                skill = _load_skill_payload(skill_json_str, skill_file)
            except Exception as exc:  # noqa: BLE001
                logger.warning("continuation: skill load failed, proceeding without it: %s", exc)
                skill = {}

        n = min(to_positive_int(next_chapters) or 5, CONTINUATION_MAX_BEATS)
        cjk = output_language != "English"

        # 防止重複：mine the phrasings already used in the existing prose.
        overused = rg.extract_overused_phrases(story) if rg.GUARD_ENABLED else []
        avoid_directive = rg.build_avoid_directive(overused, cjk=cjk) if overused else ""
        avoid_words = "、".join(overused[:12]) if cjk else ", ".join(overused[:12])

        if dry_run:
            plan = _dry_run_continuation_plan(skill, brief_text, direction, n)
        else:
            route = _make_route("Analysis / Grok", analysis_api_key, analysis_base_url, analysis_model_name)
            plan = _orchestrate_continuation(
                route=route, skill=skill, story=story, brief_text=brief_text,
                direction=direction, n=n, output_language=output_language, overused=overused,
            )

        has_skill = bool(skill.get("narrative_method") or skill.get("description_techniques"))
        system_prompt = (
            render_skill_system_prompt(skill, continuation=True)
            if has_skill
            else "（未載入技能；依續寫上下文、續寫規劃與『避免重複』指令續寫即可。）"
        )
        technique_library = render_technique_library(skill)
        instruction_parts = ["【續寫規劃（接續既有故事，勿重啟、勿重述已寫內容）】\n" + plan]
        if avoid_directive:
            instruction_parts.append(avoid_directive)
        instruction = "\n\n".join(instruction_parts)

        output_dir = Path.cwd() / "book_output" / "continuations" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_text_with_backup(output_dir / "continuation_prompt.md", instruction)
        if has_skill:
            write_text_with_backup(output_dir / "locked_narrative.md", system_prompt)

        preview = (
            "# 續寫 PROMPT（已載入 Story Instruction）\n\n" + instruction
            + ("\n\n---\n# 鎖定敘事方式（System Prompt）\n\n" + system_prompt if has_skill else "")
        )
        status = (
            "[OK] 已產生續寫 PROMPT 並載入寫作區："
            f"{'敘事方式＋技法庫＋' if has_skill else ''}續寫指令＋避免重複詞（{len(overused)} 條）。\n"
            f"規劃接下來 {n} 個節拍。{'（Dry Run 範例）' if dry_run else ''}\n"
            "→ 切到『3. 寫作』直接續寫；寫作時仍會自動走重複守衛。"
        )
        return status, trim_preview(preview), system_prompt, technique_library, instruction, avoid_words
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_continuation_prompt failed")
        return f"[ERROR] {exc}", "", "", "", "", ""


def _orchestrate_continuation(
    *,
    route: ModelRoute,
    skill: dict[str, Any],
    story: str,
    brief_text: str,
    direction: str,
    n: int,
    output_language: str,
    overused: list[str],
) -> str:
    if route.client is None:
        raise RuntimeError("缺少 Analysis LLM client。")
    skill_compact = json.dumps(_skill_for_prompt(skill), ensure_ascii=False) if skill else "（未提供技能）"
    avoid_block = "、".join(overused[:20]) if overused else "（無）"
    lang = language_instruction(output_language)

    # Pass 1 — one call: overall direction + a one-line skeleton for ALL beats, so the
    # parallel per-beat expansions stay coherent and escalate as a sequence.
    skeleton_raw = chat_complete(
        route.client,
        label="續寫骨架",
        model=route.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是小說『續寫劇情編排總監』。先規劃一個既有故事『接下來』的整體走向，與每一拍的一句話骨架，"
                    "確保各拍連貫、張力逐步升級、延續既有人物與未解線索，不可重啟故事、不可重述已寫內容。"
                    f"{lang}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"續寫共要 {n} 拍。\n"
                    f"【續寫方向（可選）】\n{(direction or '未指定，請依未解線索與情境自然推進').strip()}\n\n"
                    f"【故事續寫資訊（摘要）】\n{brief_text or '（無摘要，請依最近正文判讀）'}\n\n"
                    f"【最近正文（接續點）】\n{story[-4000:]}\n\n"
                    f"【寫作技能 JSON（敘事方式＋技法＋節拍模板）】\n{skill_compact}\n\n"
                    "請只輸出：\n"
                    "## 續寫總方向\n（2-4 句，承接目前情境、整體要往哪走、收束哪些張力）\n\n"
                    "## 拍子骨架（每一拍一行，務必剛好 "
                    f"{n} 行，格式『第N拍：[節拍功能] 一句話事件』，不可合併、不可寫成範圍）：\n"
                    "第1拍：...\n第2拍：...\n（依此到第 " + str(n) + " 拍）"
                ),
            },
        ],
        temperature=0.5,
        max_tokens=min(CONTINUATION_PLAN_MAX_TOKENS, max(1200, n * 80)),
    )
    direction_block, beat_oneliners = _split_continuation_skeleton(skeleton_raw, n)
    full_skeleton = "\n".join(f"第{i}拍：{beat_oneliners.get(i, '（待定）')}" for i in range(1, n + 1))

    beat_format = (
        "只輸出『這一拍』的一個區塊，格式如下，每個子標都要具體展開成數行：\n"
        "### 第 {i} 拍：[節拍功能]\n"
        "- 場景與調度：時間、地點、在場人物、空間與道具\n"
        "- 事件推進：具體發生什麼（2-3 個前後因果的動作/轉折，不是一句話）\n"
        "- 人物動機與內心：每個在場角色此刻要什麼、怕什麼、潛台詞\n"
        "- 對白方向：語氣、權力關係、要藏/要逼出什麼（給 1-2 句示範台詞）\n"
        "- 描寫技法套用：逐一點名技法 → 在此處具體怎麼用 + 一句示範句\n"
        "- 感官與句法節奏：主導感官、句長與停頓\n"
        "- 銜接與避免重複：如何接上一段；本拍要避免的既有寫法、改用什麼新表達\n"
        "- 推進/收束的線索與章尾鉤子"
    )
    beat_system = (
        "你是小說續寫導演。你只負責把『指定的單一節拍』展開成詳細到可直接照寫的導演手冊區塊。鐵則：\n"
        "1) 只寫被指定的那一拍，輸出剛好一個 `### 第 N 拍：` 區塊；嚴禁寫到別拍、嚴禁合併多拍、嚴禁寫成範圍標題。\n"
        "2) 續寫：延續既有人物、世界、語氣與未解線索，不重啟、不重述已寫內容。\n"
        "3) 鎖定技能的敘事方式；逐一點名要套用的描寫技法並給示範句，不只列名稱。\n"
        "4) 避免重複既有措辞（下方清單），並提示本拍換什麼新寫法。"
        f"{lang}"
    )

    def _expand(i: int) -> tuple[int, str]:
        user = (
            f"這是整個續寫 {n} 拍中的『第 {i} 拍』，請只展開這一拍。\n\n"
            f"【本拍骨架】\n第{i}拍：{beat_oneliners.get(i, '（請依前後脈絡判斷本拍功能）')}\n\n"
            f"【整體方向】\n{direction_block}\n\n"
            f"【全拍骨架（前後脈絡，幫助銜接，勿展開別拍）】\n{full_skeleton}\n\n"
            f"【故事續寫資訊（摘要）】\n{brief_text or '（無摘要）'}\n\n"
            f"【最近正文（僅第 1 拍需緊接；其餘依骨架銜接）】\n{story[-2500:]}\n\n"
            f"【寫作技能 JSON】\n{skill_compact}\n\n"
            f"【避免重複的既有措辞】\n{avoid_block}\n\n"
            + beat_format.replace("{i}", str(i))
        )
        try:
            text = chat_complete(
                route.client,
                label=f"續寫第{i}拍",
                model=route.model_name,
                messages=[
                    {"role": "system", "content": beat_system},
                    {"role": "user", "content": user},
                ],
                temperature=0.6,
                max_tokens=CONTINUATION_PLAN_MAX_TOKENS,
            ).strip()
        except Exception as exc:  # noqa: BLE001 — one failed beat shouldn't sink the rest
            logger.error("continuation beat %d failed: %s", i, exc)
            text = f"### 第 {i} 拍：（產生失敗：{exc}）"
        if not re.search(r"###\s*第", text):
            text = f"### 第 {i} 拍\n{text}"
        return i, text

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=SPECIALIST_MAX_WORKERS) as executor:
        futures = [executor.submit(_expand, i) for i in range(1, n + 1)]
        for future in as_completed(futures):
            i, text = future.result()
            results[i] = text

    beats = "\n\n".join(results[i] for i in range(1, n + 1) if i in results)
    footer = (
        "\n\n## 給寫作 AI 的最終 Director 指令\n"
        "延續既有故事與語氣，逐拍照上面手冊寫、逐拍套用指定技法，不重啟、不重述已寫內容，並避免重複既有措辞。"
    )
    return (direction_block.strip() + "\n\n" + beats).strip() + footer


def _split_continuation_skeleton(raw: str, n: int) -> tuple[str, dict[int, str]]:
    """Split the pass-1 output into the overall direction block and per-beat one-liners."""
    text = (raw or "").strip()
    oneliners: dict[int, str] = {}
    for num, body in re.findall(r"第\s*(\d+)\s*拍\s*[：:]\s*([^\n]+)", text):
        idx = int(num)
        if 1 <= idx <= n and idx not in oneliners:
            oneliners[idx] = body.strip()
    # Direction block = everything before the first per-beat line (or the 總方向 section).
    first = re.search(r"第\s*\d+\s*拍\s*[：:]", text)
    direction_block = text[: first.start()].strip() if first else text
    if not direction_block:
        direction_block = "## 續寫總方向\n（承接目前情境，依未解線索自然推進。）"
    return direction_block, oneliners


def _dry_run_continuation_plan(skill: dict[str, Any], brief_text: str, direction: str, n: int) -> str:
    techs = "、".join(t.get("name", "") for t in (skill.get("description_techniques") or [])[:3]) or "（未載入技能技法）"
    beats = []
    for i in range(1, n + 1):
        beats.append(
            f"### 第 {i} 拍：[節拍功能]\n"
            "- 場景與調度：（時間/地點/在場人物/空間道具）\n"
            "- 事件推進：（2-3 個前後因果的動作或轉折，延續既有人物與線索）\n"
            "- 人物動機與內心：（各角色此刻的欲望/恐懼/潛台詞）\n"
            "- 對白方向：（語氣、權力關係，含 1-2 句示範台詞）\n"
            f"- 描寫技法套用：{techs} → （在此處具體怎麼用 + 一句示範句）\n"
            "- 感官與句法節奏：（主導感官、句長停頓）\n"
            "- 銜接與避免重複：（如何接上段結尾；避免重複哪些既有寫法、換什麼新表達）\n"
            "- 推進/收束線索與章尾鉤子：（...）"
        )
    return (
        "## 續寫總方向\n"
        f"（離線範例）承接目前情境，依未解線索推進；方向：{(direction or '自然推進').strip()}。\n"
        "（正式模式會由 Grok 依最近正文與技能，把每一拍展開成可直接照寫的詳細指示。）\n\n"
        f"## 接下來 {n} 拍\n" + "\n\n".join(beats)
        + "\n\n## 整體要推進/收束的未解線索\n（...）\n\n"
        "## 給寫作 AI 的最終 Director 指令\n"
        "延續既有故事與語氣，逐拍照上面手冊寫、逐拍套用指定技法，不重啟、不重述已寫內容，並避免重複既有措辞。"
    )


# --------------------------------------------------------------------------- #
# Stage 3: load the orchestration into the writing-agent fields.
# --------------------------------------------------------------------------- #

def load_orchestration_to_agent_fields(
    system_prompt_text: str,
    technique_library_text: str,
    beat_plan_text: str,
    current_system_prompt: str,
    current_technique_library: str,
    current_memory: str,
    load_mode: str,
) -> tuple[str, str, str, str]:
    """Push the orchestrated prompt/techniques/beat-plan into the writing fields.

    Returns (system_prompt, technique_library, memory, status) for the
    System Prompt Override, Technique Library, and Story Memory components.
    """
    if not (system_prompt_text or "").strip():
        return (
            current_system_prompt or "",
            current_technique_library or "",
            current_memory or "",
            "[ERROR] 沒有可載入的提示詞，請先執行『劇情編排 + 提示詞』。",
        )
    append = "Append" in (load_mode or "")
    beat_block = "【節拍×技法綁定計畫】\n" + (beat_plan_text or "").strip()

    if append:
        system_prompt = _join_blocks(current_system_prompt, system_prompt_text)
        technique_library = _join_blocks(current_technique_library, technique_library_text)
        memory = _join_blocks(current_memory, beat_block)
    else:
        system_prompt = (system_prompt_text or "").strip()
        technique_library = (technique_library_text or "").strip()
        memory = beat_block

    status = (
        "[OK] 已載入到寫作區：System Prompt Override（技法綁定提示詞）、"
        "Technique Library（描寫技法）、Story Memory（節拍計畫）。"
        f"模式：{'Append' if append else 'Replace'}。請切到『3. 寫作』開始生成。"
    )
    return system_prompt, technique_library, memory, status


def load_skill_techniques_to_agent_fields(
    skill_json_str: str,
    skill_file: Any,
    current_system_prompt: str,
    current_technique_library: str,
    current_memory: str,
    load_mode: str,
) -> tuple[str, str, str, str]:
    """Load ONLY the craft (narrative method + techniques + beat binding) into the
    writing fields, with no invented plot — for "distill then write your own story".

    Returns (system_prompt, technique_library, memory, status).
    """
    try:
        skill = _load_skill_payload(skill_json_str, skill_file)
    except Exception as exc:  # noqa: BLE001
        return (
            current_system_prompt or "",
            current_technique_library or "",
            current_memory or "",
            f"[ERROR] {exc}（請先在 Step 1 蒸餾，或於 Step 2 上傳 story_skill.json）",
        )
    system_block = render_skill_system_prompt(skill)
    technique_block = render_technique_library(skill)
    binding_block = "【節拍×技法綁定參考】\n" + (render_beat_plan(skill) or "（無）")
    append = "Append" in (load_mode or "")
    if append:
        system_prompt = _join_blocks(current_system_prompt, system_block)
        technique_library = _join_blocks(current_technique_library, technique_block)
        memory = _join_blocks(current_memory, binding_block)
    else:
        system_prompt = system_block
        technique_library = technique_block
        memory = binding_block
    status = (
        "[OK] 已把『敘事方式 + 描寫技法 + 節拍×技法綁定』載入寫作區（不含劇情，"
        "劇情由你在 Story Instruction 主導）。請切到『3. 寫作』開始寫。"
        f"模式：{'Append' if append else 'Replace'}。"
    )
    return system_prompt, technique_library, memory, status


def load_latest_skill_to_agent_fields(
    current_system_prompt: str,
    current_technique_library: str,
    current_memory: str,
    load_mode: str,
) -> tuple[str, str, str, str]:
    """Convenience: load the most recent orchestration from disk into the fields."""
    base = Path.cwd() / "book_output" / "story_skills" / "orchestrations"
    if not base.exists():
        return (current_system_prompt or "", current_technique_library or "", current_memory or "",
                "[ERROR] 找不到已保存的編排；請先執行『劇情編排 + 提示詞』。")
    runs = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True)
    for run in runs:
        prompt_file = run / "writing_prompt.md"
        if prompt_file.exists():
            return load_orchestration_to_agent_fields(
                prompt_file.read_text(encoding="utf-8"),
                _read_if_exists(run / "technique_library.md"),
                _read_if_exists(run / "beat_plan.md"),
                current_system_prompt,
                current_technique_library,
                current_memory,
                load_mode,
            )
    return (current_system_prompt or "", current_technique_library or "", current_memory or "",
            "[ERROR] 最近的編排缺少 writing_prompt.md。")


def _read_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _join_blocks(existing: str, addition: str) -> str:
    existing = (existing or "").strip()
    addition = (addition or "").strip()
    if not existing:
        return addition
    if not addition:
        return existing
    return existing + "\n\n" + addition


# --------------------------------------------------------------------------- #
# Rendering helpers (skill -> human-readable + writing-agent-ready text).
# --------------------------------------------------------------------------- #

def render_skill_markdown(skill: dict[str, Any]) -> str:
    nm = skill.get("narrative_method", {})
    pa = skill.get("plot_arrangement", {})
    techs = skill.get("description_techniques") or []
    beats = skill.get("beat_template") or []
    tech_by_id = {t.get("id"): t for t in techs}

    lines = [
        f"# 寫作技能：{skill.get('source_label', '(未命名)')}",
        "",
        f"> {skill.get('abstraction_note', '')}",
        "",
        "## 敘事方式",
    ]
    lines += [f"- **{k}**：{v}" for k, v in nm.items()] or ["- （無）"]
    lines += ["", "## 劇情安排（抽象模式）"]
    lines += [f"- **{k}**：{v}" for k, v in pa.items()] or ["- （無）"]

    lines += ["", "## 描寫技法"]
    for t in techs:
        lines += [
            f"### [{t.get('id', '?')}] {t.get('name', '(未命名技法)')}  ｜ {t.get('category', '')}",
            f"- 時機：{t.get('when_to_use', '')}",
            f"- 手法：{t.get('how', '')}",
            f"- 句法節奏：{t.get('sentence_rhythm', '')}",
            f"- 感官層次：{t.get('sensory_layering', '')}",
            f"- 用詞調色盤：{t.get('word_palette', '')}",
            f"- 弱 vs 強：{t.get('weak_vs_strong', '')}",
            "",
        ]

    lines += ["## 節拍 × 技法綁定模板"]
    for b in beats:
        bound = b.get("bound_technique_ids") or []
        names = "、".join(
            f"{tech_by_id.get(tid, {}).get('name', tid)}" for tid in bound
        ) or "（未綁定）"
        lines += [
            f"### {b.get('beat_id', '?')}｜{b.get('function', '')}（節奏：{b.get('pacing', '')}）",
            f"- 目的：{b.get('purpose', '')}",
            f"- 情緒落點：{b.get('emotional_target', '')}",
            f"- 綁定技法：{names}",
            f"- 套用方式：{b.get('technique_application', '')}",
            "",
        ]

    rules = skill.get("transferable_rules") or []
    if rules:
        lines += ["## 可移植寫作守則"]
        lines += [f"{i}. {r}" for i, r in enumerate(rules, 1)]
    return "\n".join(lines).strip() + "\n"


def render_skill_system_prompt(skill: dict[str, Any], *, continuation: bool = False) -> str:
    """A craft-only System Prompt: lock the narrative method + rules, leave plot to
    the user. The concrete beat->technique mapping rides in Story Memory.

    continuation=False (original-creation): demand entirely new entities.
    continuation=True: keep the existing story's entities (we are extending it),
    only the narrative method + techniques are locked.
    """
    nm = skill.get("narrative_method", {})
    rules = skill.get("transferable_rules") or []
    parts = ["【寫作技能·鎖定敘事方式與描寫技法（劇情由你在 Story Instruction 自行主導）】"]
    if nm:
        parts.append("敘事方式（鎖定）：")
        parts += [f"- {k}：{v}" for k, v in nm.items()]
    if rules:
        parts.append("寫作守則：")
        parts += [f"- {r}" for r in rules]
    if continuation:
        parts.append(
            "本次為『續寫』：延續既有故事的人物、世界與設定（見 World/Background、Characters、Story Memory），"
            "自然接續、不要重啟故事或重述已寫內容；只鎖定上述敘事方式，並在對應節拍套用 Technique Library 的描寫技法。"
        )
    else:
        parts.append(
            "請依使用者給的情節推進；當前情節走到哪一種節拍，就套用 Technique Library 與 "
            "Story Memory 裡對應的描寫技法。全程使用全新原創人事物，不得出現任何來源故事的人名、地名、物件或情節。"
        )
    return "\n".join(parts).strip()


def render_technique_library(skill: dict[str, Any]) -> str:
    """Compact technique cards for the writing-agent Technique Library field."""
    techs = skill.get("description_techniques") or []
    if not techs:
        return ""
    lines = ["【描寫技法庫（來自蒸餾技能）】"]
    for t in techs:
        lines += [
            f"◆ {t.get('name', '')}（{t.get('category', '')}）",
            f"  時機：{t.get('when_to_use', '')}",
            f"  手法：{t.get('how', '')}",
            f"  句法：{t.get('sentence_rhythm', '')}｜感官：{t.get('sensory_layering', '')}",
            f"  用詞：{t.get('word_palette', '')}",
        ]
    rules = skill.get("transferable_rules") or []
    if rules:
        lines += ["", "【寫作守則】"] + [f"- {r}" for r in rules]
    return "\n".join(lines).strip()


def render_beat_plan(skill: dict[str, Any]) -> str:
    """The beat x technique binding as a continuity plan for Story Memory."""
    beats = skill.get("beat_template") or []
    techs = {t.get("id"): t.get("name") for t in (skill.get("description_techniques") or [])}
    if not beats:
        return ""
    lines = []
    for b in beats:
        bound = "、".join(techs.get(tid, tid) for tid in (b.get("bound_technique_ids") or [])) or "（未綁定）"
        lines.append(
            f"- {b.get('function', '')}（{b.get('pacing', '')}）→ 技法：{bound}｜{b.get('technique_application', '')}"
        )
    return "節拍推進時，逐拍套用對應描寫技法：\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dry-run fixtures (offline UI check, no API calls).
# --------------------------------------------------------------------------- #

def _dry_run_skill(source_label: str, chapters: list[Chapter]) -> dict[str, Any]:
    return _normalize_skill({
        "schema_version": 1,
        "narrative_method": {
            "pov": "第三人稱限知，貼近主角內心",
            "tense": "過去式為主，關鍵時刻切現在式特寫",
            "narration_distance": "近—在情緒高點拉到極近",
            "voice_register": "冷硬中帶抒情",
            "dialogue_style": "短句交鋒，潛台詞多",
            "scene_vs_summary": "重場景，轉場用一句概述帶過",
        },
        "plot_arrangement": {
            "story_engine_pattern": "一個未解的問題驅動每章前進",
            "pacing_curve": "慢熱開場→中段加速→高潮前急停→爆發",
            "escalation_pattern": "每章多一個代價或一個秘密",
            "hook_pattern": "章尾留新危機或情緒落差",
            "arc_shape": "下沉再反彈的 V 型",
        },
        "description_techniques": [
            {"id": "t1", "name": "眼神落點分鏡", "category": "容顏描寫",
             "when_to_use": "兩人初次對峙或情緒轉折", "how": "先寫視線落點，再寫眼球微動，最後一個濕潤/光的細節",
             "sentence_rhythm": "短—短—長", "sensory_layering": "視覺先行，收在一個觸覺", "word_palette": "具體動詞，避免『美麗』類抽象詞",
             "weak_vs_strong": "弱：她很漂亮 / 強：她睫毛壓下，眼底那點光像被風掐熄"},
            {"id": "t2", "name": "動作切碎升溫", "category": "動作描寫",
             "when_to_use": "衝突或親密升級", "how": "把一個動作拆成 3-4 個可觀察微動作，逐步加壓",
             "sentence_rhythm": "短句連擊", "sensory_layering": "觸覺主導", "word_palette": "碾、扣、抵、掐等力度動詞",
             "weak_vs_strong": "弱：他抱住她 / 強：他先扣住她手腕，再把人抵向牆，呼吸落在她耳側"},
        ],
        "beat_template": [
            {"beat_id": "b1", "function": "開場鉤子", "purpose": "用失衡瞬間抓住讀者", "pacing": "快",
             "emotional_target": "不安與好奇", "bound_technique_ids": ["t1"], "technique_application": "用眼神落點分鏡帶出對手的威脅感"},
            {"beat_id": "b2", "function": "第一道阻力", "purpose": "暴露主角欲望與代價", "pacing": "中",
             "emotional_target": "緊繃", "bound_technique_ids": ["t2"], "technique_application": "用動作切碎升溫表現對抗"},
            {"beat_id": "b3", "function": "章尾鉤子", "purpose": "拋出新危機", "pacing": "快",
             "emotional_target": "懸念", "bound_technique_ids": ["t1", "t2"], "technique_application": "眼神＋動作雙線收尾，留下情緒落差"},
        ],
        "transferable_rules": [
            "描寫用具體動詞與感官細節，禁止抽象形容詞堆砌。",
            "每章至少一個代價或秘密推進長線問題。",
            "情緒高點才拉近敘事距離，平時保持克制。",
        ],
        "abstraction_note": f"Dry Run 範例技能（讀取了 {len(chapters)} 章；不含任何原作人事物）。",
    })


def _dry_run_orchestration_prompt(skill: dict[str, Any], premise: str, genre_tone: str, chapter_count: int) -> str:
    nm = skill.get("narrative_method", {})
    return (
        "## 敘事方式（鎖定）\n"
        + "\n".join(f"- {k}：{v}" for k, v in nm.items())
        + "\n\n## 一句話賣點 + 核心劇情引擎（原創）\n"
        + f"（離線範例）依故事種子「{premise[:40]}…」與技能引擎模式編排的全新原創故事。\n\n"
        + "## 主要人物（全新原創）\n- （正式模式會產生 3-5 位全新角色：身分/欲望/阻力/秘密）\n\n"
        + f"## {chapter_count} 章節拍表\n"
        + render_beat_plan(skill)
        + "\n\n## 給寫作 AI 的最終 System / Director 指令\n"
        + "鎖定上述敘事方式；逐章依節拍功能推進；每一拍套用其綁定的描寫技法；全程使用全新原創人事物，"
        + "不得出現任何來源故事的人名、地名、物件或情節。"
    )
