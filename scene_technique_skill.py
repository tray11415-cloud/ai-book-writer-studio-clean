"""Technique aggregation for specific scenes, actions, and situations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from chapter_craft_skill import (
    API_TIMEOUT,
    Chapter,
    chat_complete,
    language_instruction,
    load_chapters,
    normalize_text,
    read_message,
    to_positive_int,
    trim_chapter_text,
    trim_preview,
    write_text_with_backup,
)


logger = logging.getLogger(__name__)


DEFAULT_TECHNIQUE_GOAL = (
    "彙整描寫特定場景、特定動作、特定情境時可使用的小說寫作手法，"
    "輸出可套用的感官配置、動作拆拍、心理壓力、視角距離、句式節奏、場面調度與可直接貼進寫作區的指令。"
)

DEFAULT_LIBRARY_GOAL = (
    "將整本小說蒸餾成 Technique Finder 手法庫：逐章找出可複製的場景、動作、情境描寫方法，"
    "整理成可搜尋、可套用、可貼進寫作區的 Technique Finder 卡片。"
)


@dataclass(frozen=True)
class TechniqueRequest:
    target_scene: str
    target_action: str
    target_situation: str
    desired_effect: str
    goal: str
    output_language: str


@dataclass(frozen=True)
class TechniqueCard:
    title: str
    source_chapter: str
    scene: str
    action: str
    situation: str
    reader_effect: str
    technique_summary: str
    scene_texture: str
    action_beats: str
    sensory_focus: str
    pov_camera: str
    sentence_rhythm: str
    formulas: list[str]
    director_instruction: str


def aggregate_scene_techniques(
    reference_txt_file: Any,
    reference_text: str,
    reference_url: str,
    target_scene: str,
    target_action: str,
    target_situation: str,
    desired_effect: str,
    technique_goal: str,
    output_language: str,
    reference_limit: float | int | None,
    max_reference_chars: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None]:
    """Gradio entry point for scene/action/situation technique aggregation."""
    try:
        request = TechniqueRequest(
            target_scene=normalize_text(target_scene),
            target_action=normalize_text(target_action),
            target_situation=normalize_text(target_situation),
            desired_effect=normalize_text(desired_effect),
            goal=normalize_text(technique_goal) or DEFAULT_TECHNIQUE_GOAL,
            output_language=output_language or "繁體中文",
        )
        if not any([request.target_scene, request.target_action, request.target_situation]):
            return "[ERROR] 請至少填入特定場景、特定動作或特定情境。", "", None

        ref_limit = to_positive_int(reference_limit) or 3
        max_chars = to_positive_int(max_reference_chars) or 9000
        source_label = "no-reference"
        chapters: list[Chapter] = []
        if reference_txt_file or reference_text or reference_url:
            chapters, source_label = load_chapters(
                txt_file=reference_txt_file,
                pasted_text=reference_text,
                directory_url=reference_url,
                limit=ref_limit,
                fallback_chunk_chars=max_chars,
            )

        if dry_run:
            report = dry_run_techniques(request, chapters)
            mode = "Dry Run"
        else:
            if not analysis_base_url.strip() or not analysis_model_name.strip():
                return "[ERROR] 請先設定 Analysis / Grok 的 Base URL 與 Model Name。", "", None
            client = OpenAI(
                api_key=(analysis_api_key or "not-needed").strip(),
                base_url=analysis_base_url.strip().rstrip("/"),
                timeout=API_TIMEOUT,
            )
            report = ask_scene_technique_agent(
                client=client,
                model_name=analysis_model_name.strip(),
                request=request,
                chapters=chapters,
                max_chars=max_chars,
            )
            mode = f"Grok / {analysis_model_name.strip()}"

        output_dir = write_technique_report(
            request=request,
            report=report,
            source_label=source_label,
            mode=mode,
        )
        report_path = output_dir / "scene_techniques.md"
        status = (
            "[OK] 描寫手法彙整完成。\n"
            f"場景：{request.target_scene or '未指定'}\n"
            f"動作：{request.target_action or '未指定'}\n"
            f"情境：{request.target_situation or '未指定'}\n"
            f"參考來源：{source_label}\n"
            f"模式：{mode}\n"
            f"報告：{report_path}"
        )
        return status, trim_preview(report_path.read_text(encoding="utf-8")), str(report_path)
    except Exception as exc:
        return f"[ERROR] {exc}", "", None


def distill_novel_to_technique_finder(
    novel_txt_file: Any,
    novel_text: str,
    novel_url: str,
    library_goal: str,
    output_language: str,
    chapter_limit: float | int | None,
    cards_per_chapter: float | int | None,
    max_chapter_chars: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None]:
    """Distill a whole novel into a searchable Technique Finder library."""
    try:
        limit = to_positive_int(chapter_limit)
        cards_each = to_positive_int(cards_per_chapter) or 3
        max_chars = to_positive_int(max_chapter_chars) or 9000
        goal = normalize_text(library_goal) or DEFAULT_LIBRARY_GOAL
        chapters, source_label = load_chapters(
            txt_file=novel_txt_file,
            pasted_text=novel_text,
            directory_url=novel_url,
            limit=limit,
            fallback_chunk_chars=max_chars,
        )
        if not chapters:
            return "[ERROR] 沒有可蒸餾的小說章節。", "", None

        if dry_run:
            cards = dry_run_library_cards(chapters, cards_each)
            mode = "Dry Run"
        else:
            if not analysis_base_url.strip() or not analysis_model_name.strip():
                return "[ERROR] 請先設定 Analysis / Grok 的 Base URL 與 Model Name。", "", None
            client = OpenAI(
                api_key=(analysis_api_key or "not-needed").strip(),
                base_url=analysis_base_url.strip().rstrip("/"),
                timeout=API_TIMEOUT,
            )
            cards = []
            for chapter in chapters:
                cards.extend(
                    ask_chapter_technique_cards(
                        client=client,
                        model_name=analysis_model_name.strip(),
                        chapter=chapter,
                        goal=goal,
                        output_language=output_language or "繁體中文",
                        cards_per_chapter=cards_each,
                        max_chars=max_chars,
                    )
                )
            mode = f"Grok / {analysis_model_name.strip()}"

        output_dir = write_technique_library(
            source_label=source_label,
            goal=goal,
            mode=mode,
            cards=cards,
        )
        report_path = output_dir / "technique_finder_library.md"
        status = (
            "[OK] 小說已蒸餾成 Technique Finder 手法庫。\n"
            f"來源：{source_label}\n"
            f"章節數：{len(chapters)}\n"
            f"卡片數：{len(cards)}\n"
            f"模式：{mode}\n"
            f"報告：{report_path}"
        )
        return status, trim_preview(report_path.read_text(encoding="utf-8")), str(report_path)
    except Exception as exc:
        return f"[ERROR] {exc}", "", None


def ask_scene_technique_agent(
    *,
    client: OpenAI,
    model_name: str,
    request: TechniqueRequest,
    chapters: list[Chapter],
    max_chars: int,
) -> str:
    reference = build_reference_excerpt(chapters, max_chars)
    return chat_complete(
        client,
        label="場景手法彙整",
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是小說描寫手法彙整 Agent。你專門把參考文本與指定條件轉成可操作的描寫技法。"
                    "不要抄襲原文，不要續寫故事，不要輸出完整小說段落；只輸出手法、公式、注意事項與短句級示例。"
                    f"{language_instruction(request.output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {request.goal}\n\n"
                    f"特定場景：{request.target_scene or '未指定'}\n"
                    f"特定動作：{request.target_action or '未指定'}\n"
                    f"特定情境：{request.target_situation or '未指定'}\n"
                    f"想達成的讀者感受：{request.desired_effect or '未指定'}\n\n"
                    "你不是寫讀後感，而是要解剖『這個場景／動作，到底是一句一句怎麼寫出來的』。"
                    "不要用一兩句空泛帶過，每一段都要深到能照著寫。請彙整成固定格式：\n"
                    "- 核心描寫策略\n"
                    "- 場景質地：空間、光線、聲音、溫度、物件、背景動勢（各舉一個可觀察細節）\n"
                    "- 動作解剖（最重要）：把這個動作拆成 5-7 個可觀察微單元並排出書寫順序，"
                    "每個單元寫明『身體哪裡先動、帶多少力、碰到什麼、改變了什麼』。"
                    "例如喝酒＝端杯目的→器物聲→入口停頓→喉結餘味→放杯→話語轉向。\n"
                    "- 情境壓力：祕密、危險、權力差、時間限制、心理遮掩\n"
                    "- 感官分層：指明哪個感官領頭、其餘感官的先後與份量（強調節制，通常一主一輔，不要堆五感）\n"
                    "- 視角距離與鏡頭調度\n"
                    "- 句法節奏：具體講句子長短、停頓與標點怎麼配合動作節拍（如『動作用短句加速，停頓用句號硬斷』）\n"
                    "- 用詞調色盤：列出適合的動詞／名詞／質感詞，以及該避免的抽象判斷詞（迅速、用力、美麗、憤怒等）\n"
                    "- 對白/無聲互動處理\n"
                    "- 10 條可複製寫法公式\n"
                    "- 5 條可直接貼進 Interactive Writing 的 Director Instruction\n"
                    "- 避免事項：哪些寫法會變俗、變慢或失焦\n"
                    "- 弱寫→強寫對照：給 3 組，每組第一句是常見平庸寫法，第二句是運用本手法後的短句級改寫"
                    "（都要你自己寫，短、具體、不抄原文）\n\n"
                    f"參考文本：\n{reference}"
                ),
            },
        ],
        temperature=0.35,
        max_tokens=2600,
    )


def ask_chapter_technique_cards(
    *,
    client: OpenAI,
    model_name: str,
    chapter: Chapter,
    goal: str,
    output_language: str,
    cards_per_chapter: int,
    max_chars: int,
) -> list[TechniqueCard]:
    chapter_text = trim_chapter_text(chapter.text, max_chars)
    create_kwargs: dict[str, Any] = dict(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是小說 Technique Finder 蒸餾 Agent。你會從一章小說中抽取可複製的描寫手法卡片。"
                    "每張卡片必須對應一種「場景 + 動作 + 情境」組合。不要抄原文，不要續寫正文。"
                    "只輸出 JSON array，不要 Markdown，不要解釋。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n\n"
                    f"章節：第 {chapter.index} 章｜{chapter.title}\n"
                    f"請抽取 {cards_per_chapter} 張 Technique Finder 卡片。\n\n"
                    "每張卡片必須包含這些鍵：\n"
                    "title, source_chapter, scene, action, situation, reader_effect, "
                    "technique_summary, scene_texture, action_beats, sensory_focus, "
                    "pov_camera, sentence_rhythm, formulas, director_instruction。\n\n"
                    "深度要求（不要一句空泛帶過，要拆到能照著寫）：\n"
                    "- action_beats：把動作拆成 5-7 個可觀察微單元並排序，每拍寫明『哪裡先動、帶多少力、"
                    "碰到什麼、改變了什麼距離或關係』，而不是只列名詞。\n"
                    "- scene_texture：空間/光線/聲音/溫度/物件各給一個可觀察細節，並指出哪個細節在推動情緒。\n"
                    "- sensory_focus：指明哪個感官領頭、其餘的先後與份量（通常一主一輔，不要堆五感）。\n"
                    "- sentence_rhythm：具體講句子長短、停頓與標點如何配合動作節拍（如短句加速、句號硬斷）。\n"
                    "- technique_summary：點出這個寫法為何有效、避免哪個常見平庸寫法。\n"
                    "- formulas 必須是字串陣列，至少 3 條，要可直接套用。\n\n"
                    f"章節正文：\n{chapter_text}"
                ),
            },
        ],
        temperature=0.25,
        max_tokens=4000,
    )
    try:
        response = client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        logger.error("第 %s 章卡片抽取呼叫失敗：%s", chapter.index, exc)
        raise RuntimeError(f"卡片抽取呼叫失敗：{exc}") from exc
    # Warn if the model may have hit the token ceiling and truncated the JSON.
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if isinstance(completion_tokens, int) and completion_tokens >= 3900:
        logger.warning(
            "第 %s 章卡片抽取接近 token 上限（%s/4000），JSON 可能被截斷。",
            chapter.index,
            completion_tokens,
        )
    raw = read_message(response)
    if raw and not raw.rstrip().endswith("]"):
        logger.warning(
            "第 %s 章卡片 JSON 可能被截斷（結尾：%r）。",
            chapter.index,
            raw[-10:],
        )
    parsed = parse_card_json(raw)
    if parsed:
        return [card_from_mapping(item, chapter) for item in parsed[:cards_per_chapter]]
    return [
        TechniqueCard(
            title=f"{chapter.title}｜手法彙整",
            source_chapter=f"第 {chapter.index} 章｜{chapter.title}",
            scene="見原始回應",
            action="見原始回應",
            situation="見原始回應",
            reader_effect="見原始回應",
            technique_summary=raw,
            scene_texture="",
            action_beats="",
            sensory_focus="",
            pov_camera="",
            sentence_rhythm="",
            formulas=[],
            director_instruction="",
        )
    ]


def build_reference_excerpt(chapters: list[Chapter], max_chars: int) -> str:
    if not chapters:
        return "未提供參考文本；請依指定場景、動作、情境自行歸納通用技法。"
    per_chapter = max(1200, max_chars // max(len(chapters), 1))
    return "\n\n".join(
        f"## 第 {chapter.index} 章｜{chapter.title}\n{trim_chapter_text(chapter.text, per_chapter)}"
        for chapter in chapters[:6]
    )


def write_technique_report(
    *,
    request: TechniqueRequest,
    report: str,
    source_label: str,
    mode: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "book_output" / "scene_techniques" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_label": source_label,
        "mode": mode,
        "request": {
            "target_scene": request.target_scene,
            "target_action": request.target_action,
            "target_situation": request.target_situation,
            "desired_effect": request.desired_effect,
            "goal": request.goal,
            "output_language": request.output_language,
        },
        "report": report,
    }
    write_text_with_backup(
        output_dir / "scene_techniques.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    markdown = (
        "# 特定場景 / 動作 / 情境描寫手法\n\n"
        f"`/goal`：{request.goal}\n\n"
        f"模式：{mode}\n\n"
        f"參考來源：{source_label}\n\n"
        "## 條件\n\n"
        f"- 場景：{request.target_scene or '未指定'}\n"
        f"- 動作：{request.target_action or '未指定'}\n"
        f"- 情境：{request.target_situation or '未指定'}\n"
        f"- 讀者感受：{request.desired_effect or '未指定'}\n\n"
        "## 手法彙整\n\n"
        f"{report}\n"
    )
    write_text_with_backup(output_dir / "scene_techniques.md", markdown)
    return output_dir


def write_technique_library(
    *,
    source_label: str,
    goal: str,
    mode: str,
    cards: list[TechniqueCard],
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "book_output" / "scene_techniques" / "library" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_label": source_label,
        "goal": goal,
        "mode": mode,
        "cards": [card.__dict__ for card in cards],
    }
    write_text_with_backup(
        output_dir / "technique_finder_library.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    markdown = render_technique_library(source_label, goal, mode, cards)
    write_text_with_backup(output_dir / "technique_finder_library.md", markdown)
    return output_dir


def render_technique_library(
    source_label: str,
    goal: str,
    mode: str,
    cards: list[TechniqueCard],
) -> str:
    lines = [
        "# Technique Finder 手法庫",
        "",
        f"`/goal`：{goal}",
        "",
        f"來源：{source_label}",
        "",
        f"模式：{mode}",
        "",
        f"卡片數：{len(cards)}",
        "",
        "## 索引",
        "",
    ]
    for index, card in enumerate(cards, start=1):
        lines.append(
            f"{index}. {card.title}｜場景：{card.scene}｜動作：{card.action}｜情境：{card.situation}"
        )
    lines.append("")
    lines.append("## 卡片")
    lines.append("")
    for index, card in enumerate(cards, start=1):
        lines.extend(
            [
                f"### {index}. {card.title}",
                "",
                f"- 來源章節：{card.source_chapter}",
                f"- 場景：{card.scene}",
                f"- 動作：{card.action}",
                f"- 情境：{card.situation}",
                f"- 讀者感受：{card.reader_effect}",
                "",
                f"**手法摘要**：{card.technique_summary}",
                "",
                f"**場景質地**：{card.scene_texture}",
                "",
                f"**動作拆拍**：{card.action_beats}",
                "",
                f"**感官焦點**：{card.sensory_focus}",
                "",
                f"**視角/鏡頭**：{card.pov_camera}",
                "",
                f"**句式節奏**：{card.sentence_rhythm}",
                "",
                "**可複製公式**：",
            ]
        )
        for formula in card.formulas:
            lines.append(f"- {formula}")
        lines.extend(
            [
                "",
                f"**Director Instruction**：{card.director_instruction}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def dry_run_techniques(request: TechniqueRequest, chapters: list[Chapter]) -> str:
    reference_note = (
        f"已讀取 {len(chapters)} 個參考章節。"
        if chapters
        else "未提供參考文本；此為通用技法檢查。"
    )
    return (
        "## 離線手法彙整\n\n"
        f"{reference_note}\n\n"
        "### 核心描寫策略\n"
        "- 先確定視角距離：遠景建立場域，中景交代動作，近景落在身體反應或物件細節。\n"
        "- 動作不要一次寫完，拆成起勢、停頓、接近、接觸、反應、後果。\n"
        "- 情境壓力要外化到環境：聲音變少、光線偏移、物件位置、呼吸和皮膚感。\n\n"
        "### 可複製公式\n"
        "- 場景壓迫 = 空間限制 + 聲音稀薄 + 角色不敢明說的目的。\n"
        "- 動作張力 = 動作前停頓 + 微小身體反應 + 旁觀者或環境回聲。\n"
        "- 情境懸念 = 表面動作 + 隱藏意圖 + 即將被發現的風險。\n\n"
        "### Director Instruction\n"
        f"- 描寫「{request.target_scene or '指定場景'}」中「{request.target_action or '指定動作'}」發生於「{request.target_situation or '指定情境'}」時，先寫環境壓力，再拆動作節拍，最後落在角色身體反應。\n"
        "- 避免直接解釋情緒；用光線、聲音、手部停頓、呼吸與物件位置承載情緒。"
    )


def dry_run_library_cards(chapters: list[Chapter], cards_per_chapter: int) -> list[TechniqueCard]:
    cards: list[TechniqueCard] = []
    for chapter in chapters:
        for offset in range(cards_per_chapter):
            cards.append(
                TechniqueCard(
                    title=f"{chapter.title}｜Technique Card {offset + 1}",
                    source_chapter=f"第 {chapter.index} 章｜{chapter.title}",
                    scene="章節核心場域",
                    action="角色推進關鍵動作",
                    situation="衝突或秘密即將升級",
                    reader_effect="懸念、壓迫、期待",
                    technique_summary="Dry Run 卡片：正式模式會由 Grok 從章節中抽取具體場景/動作/情境手法。",
                    scene_texture="空間限制、光線、聲音、物件位置。",
                    action_beats="起勢 -> 停頓 -> 接近 -> 接觸 -> 反應 -> 後果。",
                    sensory_focus="視覺先定位，聽覺降噪，觸覺落到身體反應。",
                    pov_camera="遠景建立場域，中景跟動作，近景落在微反應。",
                    sentence_rhythm="短句壓迫，長句鋪陳，章尾收束在未解決問題。",
                    formulas=[
                        "場景壓迫 = 空間限制 + 聲音稀薄 + 不可明說的目的",
                        "動作張力 = 動作前停頓 + 微反應 + 後果延遲",
                        "情境懸念 = 表面行為 + 隱藏意圖 + 暴露風險",
                    ],
                    director_instruction=(
                        "將本章手法套用到新場景：先建立場域壓力，再拆角色動作，"
                        "最後用身體反應或物件變化留下鉤子。"
                    ),
                )
            )
    return cards


def parse_card_json(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("無法解析卡片 JSON：%s（原始長度：%d）", exc, len(raw))
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def card_from_mapping(item: dict[str, Any], chapter: Chapter) -> TechniqueCard:
    formulas = item.get("formulas", [])
    if isinstance(formulas, str):
        formulas = [formulas]
    if not isinstance(formulas, list):
        formulas = []
    return TechniqueCard(
        title=str(item.get("title") or f"{chapter.title}｜Technique Card").strip(),
        source_chapter=str(
            item.get("source_chapter") or f"第 {chapter.index} 章｜{chapter.title}"
        ).strip(),
        scene=str(item.get("scene") or "").strip(),
        action=str(item.get("action") or "").strip(),
        situation=str(item.get("situation") or "").strip(),
        reader_effect=str(item.get("reader_effect") or "").strip(),
        technique_summary=str(item.get("technique_summary") or "").strip(),
        scene_texture=str(item.get("scene_texture") or "").strip(),
        action_beats=str(item.get("action_beats") or "").strip(),
        sensory_focus=str(item.get("sensory_focus") or "").strip(),
        pov_camera=str(item.get("pov_camera") or "").strip(),
        sentence_rhythm=str(item.get("sentence_rhythm") or "").strip(),
        formulas=[str(formula).strip() for formula in formulas if str(formula).strip()],
        director_instruction=str(item.get("director_instruction") or "").strip(),
    )
