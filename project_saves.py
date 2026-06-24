"""Named multi-slot project saves.

The original Save/Load tab wrote a single timestamped JSON and reloaded it by file
upload. This adds **named slots**: save the whole working state — story bible,
prompts, AND the currently distilled Story Skill — under a name, keep many of them,
see an info table, and load/delete any by picking it from a dropdown.

Each slot is one JSON file under book_output/saves/<slug>.json. The payload bundles
the writing-context fields the studio cares about plus light metadata so the slot
list can show what each save contains (story length, whether it carries a skill, the
skill's source / technique & beat counts, and a free-text note).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# The fields a slot restores, in the exact order load_project_slot returns them
# (the trailing status string is added on top of these).
SLOT_FIELDS = [
    "background",
    "roles",
    "lore",
    "full_story",
    "memory",
    "style_dna",
    "style_samples",
    "chronicle",
    "technique_library",
    "system_prompt",
    "custom_director",
    "instruction",
    "skill_json",
]

_EMPTY_ROLES = [["", "", ""]]
_EMPTY_LORE = [["", ""]]


def _saves_dir() -> Path:
    path = Path.cwd() / "book_output" / "saves"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(name: str) -> str:
    """Filesystem-safe slug; keeps CJK, replaces unsafe/space runs with '_'."""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", (name or "").strip())
    cleaned = cleaned.strip("_.")[:80]
    return cleaned or datetime.now().strftime("save_%Y%m%d_%H%M%S")


def _read_slot(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Skipping unreadable save slot: %s", path)
        return None


def _all_slots() -> list[dict[str, Any]]:
    """Every slot payload (with its file path), newest first."""
    slots: list[dict[str, Any]] = []
    for path in _saves_dir().glob("*.json"):
        data = _read_slot(path)
        if data is None:
            continue
        data["_path"] = str(path)
        data.setdefault("name", path.stem)
        data.setdefault("saved_at", "")
        slots.append(data)
    slots.sort(key=lambda d: d.get("saved_at", ""), reverse=True)
    return slots


def list_slot_names() -> list[str]:
    return [s["name"] for s in _all_slots()]


def _find_slot(name: str) -> dict[str, Any] | None:
    name = (name or "").strip()
    if not name:
        return None
    for slot in _all_slots():
        if slot.get("name") == name:
            return slot
    # Fall back to slug match so a hand-typed name still resolves.
    target = _slug(name)
    for slot in _all_slots():
        if _slug(slot.get("name", "")) == target:
            return slot
    return None


def _skill_summary(skill_json: str) -> dict[str, Any]:
    """Lightweight facts about an embedded skill for the info table."""
    text = (skill_json or "").strip()
    if not text:
        return {"has_skill": False}
    try:
        skill = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"has_skill": True, "skill_source": "(unparsed)", "techniques": 0, "beats": 0}
    if not isinstance(skill, dict):
        return {"has_skill": True, "skill_source": "(unparsed)", "techniques": 0, "beats": 0}
    return {
        "has_skill": True,
        "skill_source": skill.get("source_label", "(unknown)"),
        "techniques": len(skill.get("description_techniques") or []),
        "beats": len(skill.get("beat_template") or []),
    }


def describe_slots() -> str:
    slots = _all_slots()
    if not slots:
        return "（目前沒有任何命名存檔。輸入名稱後按「Save To Slot」建立第一個。）"
    lines = [
        f"### 已保存的存檔（{len(slots)} 個）",
        "",
        "| 名稱 | 存檔時間 | 故事字數 | 技能 | 備註 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in slots:
        info = _skill_summary(s.get("skill_json", ""))
        if info.get("has_skill"):
            skill_cell = f"✓ {info.get('skill_source', '')}（技法 {info.get('techniques', 0)}／節拍 {info.get('beats', 0)}）"
        else:
            skill_cell = "—"
        story_chars = len((s.get("full_story") or ""))
        note = (s.get("note") or "").replace("|", "/").replace("\n", " ")[:60]
        name = (s.get("name") or "").replace("|", "/")
        lines.append(
            f"| {name} | {s.get('saved_at', '')} | {story_chars} | {skill_cell} | {note} |"
        )
    return "\n".join(lines)


def _dropdown_update(selected: str | None = None) -> Any:
    names = list_slot_names()
    value = selected if (selected in names) else (names[0] if names else None)
    return gr.update(choices=names, value=value)


def refresh_slots() -> tuple[Any, str]:
    return _dropdown_update(), describe_slots()


def save_project_slot(
    slot_name: str,
    note: str,
    background: Any,
    roles: Any,
    lore: Any,
    full_story: str,
    memory: str,
    style_dna: str,
    style_samples: str,
    chronicle: str,
    technique_library: str,
    system_prompt: str,
    custom_director: str,
    instruction: str,
    skill_json: str,
) -> tuple[str, Any, str]:
    """Save the full working state (incl. the distilled skill) under a named slot.

    Returns (status, dropdown_update, info_markdown).
    """
    name = (slot_name or "").strip()
    if not name:
        return "[ERROR] 請先輸入存檔名稱。", _dropdown_update(), describe_slots()

    payload = {
        "name": name,
        "note": (note or "").strip(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "background": background or "",
        "roles": roles if isinstance(roles, list) else _EMPTY_ROLES,
        "lore": lore if isinstance(lore, list) else _EMPTY_LORE,
        "full_story": full_story or "",
        "memory": memory or "",
        "style_dna": style_dna or "",
        "style_samples": style_samples or "",
        "chronicle": chronicle or "",
        "technique_library": technique_library or "",
        "system_prompt": system_prompt or "",
        "custom_director": custom_director or "",
        "instruction": instruction or "",
        "skill_json": skill_json or "",
    }
    path = _saves_dir() / f"{_slug(name)}.json"
    existed = path.exists()
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.exception("Failed to write save slot")
        return f"[ERROR] 寫入存檔失敗：{exc}", _dropdown_update(), describe_slots()

    info = _skill_summary(skill_json)
    skill_note = (
        f"含技能（技法 {info.get('techniques', 0)}／節拍 {info.get('beats', 0)}）"
        if info.get("has_skill")
        else "未含技能"
    )
    status = (
        f"[OK] {'已更新' if existed else '已建立'}存檔「{name}」（{skill_note}）。\n檔案：{path}"
    )
    return status, _dropdown_update(name), describe_slots()


def load_project_slot(slot_name: str) -> tuple[Any, ...]:
    """Load a named slot. Returns the SLOT_FIELDS values (in order) + status."""
    slot = _find_slot(slot_name)
    if slot is None:
        empty = ("", _EMPTY_ROLES, _EMPTY_LORE, "", "", "", "", "", "", "", "", "", "")
        return (*empty, f"[ERROR] 找不到存檔「{slot_name}」。")
    roles = slot.get("roles") or _EMPTY_ROLES
    lore = slot.get("lore") or _EMPTY_LORE
    info = _skill_summary(slot.get("skill_json", ""))
    skill_note = (
        f"，並還原技能（技法 {info.get('techniques', 0)}／節拍 {info.get('beats', 0)}，可到『11. 故事技能』編排或直接載入）"
        if info.get("has_skill")
        else "（此存檔未含技能）"
    )
    status = f"[OK] 已載入存檔「{slot.get('name', slot_name)}」{skill_note}。存檔時間：{slot.get('saved_at', '')}"
    return (
        slot.get("background", ""),
        roles if isinstance(roles, list) else _EMPTY_ROLES,
        lore if isinstance(lore, list) else _EMPTY_LORE,
        slot.get("full_story", ""),
        slot.get("memory", ""),
        slot.get("style_dna", ""),
        slot.get("style_samples", ""),
        slot.get("chronicle", ""),
        slot.get("technique_library", ""),
        slot.get("system_prompt", ""),
        slot.get("custom_director", ""),
        slot.get("instruction", ""),
        slot.get("skill_json", ""),
        status,
    )


def delete_project_slot(slot_name: str) -> tuple[str, Any, str]:
    """Delete a named slot. Returns (status, dropdown_update, info_markdown)."""
    slot = _find_slot(slot_name)
    if slot is None:
        return f"[ERROR] 找不到存檔「{slot_name}」。", _dropdown_update(), describe_slots()
    try:
        Path(slot["_path"]).unlink()
    except OSError as exc:
        logger.exception("Failed to delete save slot")
        return f"[ERROR] 刪除失敗：{exc}", _dropdown_update(), describe_slots()
    return f"[OK] 已刪除存檔「{slot.get('name', slot_name)}」。", _dropdown_update(), describe_slots()
