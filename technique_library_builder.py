"""Build and query a categorized writing-technique book library."""
from __future__ import annotations

import hashlib
import json
import re
from glob import glob
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from chapter_craft_skill import to_positive_int, trim_preview
from report_technique_distiller import TECHNIQUE_LOAD_MODES, merge_text_field
from skill_technique_review import read_text, sanitize_review_text, trim_text


DEFAULT_BOOK_LIBRARY_GOAL = (
    "把多本小說、full_report.md、Technique Finder 報告自動蒸餾成可查詢的寫作技法書庫。"
    "請依照大分類與小分類整理，例如容顏描寫、身材描寫、動作描寫、情緒描寫、場景氛圍、對話關係、節奏章法；"
    "每個小分類底下輸出可複製的寫法技巧、適用場景、公式與可載入寫作 AGENT 的 Director Instruction。"
    "只蒸餾方法，不抄原文，不延續原作劇情。"
)

BOOK_LIBRARY_LOAD_MODES = TECHNIQUE_LOAD_MODES

BLOCKED_CONTENT_MARKERS = [
    "[redacted:",
    "未成年",
    "未滿",
    "性內容",
    "自慰",
    "性交",
    "下體",
    "裸體",
    "褲襠",
    "露骨",
    "調教",
]

TAXONOMY: list[dict[str, Any]] = [
    {
        "category": "容顏描寫",
        "subcategories": [
            {
                "name": "眼睛描寫",
                "aliases": ["眼神", "目光", "視線", "眸", "瞳"],
                "keywords": ["眼", "眸", "瞳", "目光", "視線", "凝視", "垂眸", "抬眼", "眼尾"],
                "summary": "用視線方向、停頓、光影與眨眼頻率外化情緒，讓讀者先看見心緒，再理解人物想法。",
                "formula": ["視線落點", "細微變化", "情緒壓力", "對方反應"],
                "scenes": ["初見", "試探", "壓迫", "心虛", "決意"],
            },
            {
                "name": "眉眼描寫",
                "aliases": ["眉", "眉梢", "眉心"],
                "keywords": ["眉", "眉心", "眉梢", "蹙眉", "揚眉", "眉眼"],
                "summary": "用眉心、眉梢與眼部連動表現判斷、挑釁、忍耐或不悅，適合短促地推動對話張力。",
                "formula": ["眉部動作", "眼神變化", "未說出口的判斷", "語氣或沉默"],
                "scenes": ["對話交鋒", "質問", "壓抑怒意", "不動聲色"],
            },
            {
                "name": "嘴巴描寫",
                "aliases": ["嘴唇", "唇", "唇角", "口吻"],
                "keywords": ["嘴", "唇", "唇角", "抿唇", "薄唇", "咬唇", "口吻"],
                "summary": "用唇角、抿唇、咬字和吞吐來暗示人物的忍耐、誘惑、嘲諷、退讓或心虛。",
                "formula": ["唇部細節", "語氣變化", "情緒遮掩", "關係距離改變"],
                "scenes": ["欲言又止", "冷笑", "安撫", "談判", "告白前後"],
            },
            {
                "name": "臉部輪廓描寫",
                "aliases": ["輪廓", "鼻梁", "下頜", "面容"],
                "keywords": ["臉", "面容", "輪廓", "鼻梁", "下頜", "面頰", "臉色"],
                "summary": "用輪廓、光線與臉色改變建立人物第一印象，並把外貌轉成氣勢或狀態。",
                "formula": ["輪廓特徵", "光影角度", "當下狀態", "旁人感受"],
                "scenes": ["人物登場", "受傷", "疲憊", "權威壓場"],
            },
            {
                "name": "神情氣質描寫",
                "aliases": ["神情", "氣質", "美感"],
                "keywords": ["神情", "神色", "氣質", "清冷", "凌厲", "溫潤", "端莊", "妖冶", "冷淡"],
                "summary": "把抽象氣質落在姿態、語氣、空氣感與他人反應上，避免只用形容詞堆疊。",
                "formula": ["抽象氣質", "具體外化", "場景襯托", "他人反應"],
                "scenes": ["角色亮相", "權力對峙", "宴席", "私密談話"],
            },
        ],
    },
    {
        "category": "身材描寫",
        "subcategories": [
            {
                "name": "身形輪廓描寫",
                "aliases": ["身形", "身段", "體態"],
                "keywords": ["身形", "身段", "身影", "體態", "修長", "纖細", "高挑", "魁梧"],
                "summary": "用遠近鏡頭、衣料線條與動作造成的輪廓變化，讓身形成為角色性格的一部分。",
                "formula": ["遠景輪廓", "衣料或光影", "動作牽動", "性格暗示"],
                "scenes": ["登場", "轉身", "行走", "戰鬥前"],
            },
            {
                "name": "腰身姿態描寫",
                "aliases": ["腰", "姿態", "站姿"],
                "keywords": ["腰", "腰身", "腰肢", "站姿", "姿態", "挺直", "俯身", "倚"],
                "summary": "用重心、腰背角度與停頓表現自持、疲憊、誘惑、戒備或屈服。",
                "formula": ["重心位置", "腰背角度", "動作停頓", "情緒或權力變化"],
                "scenes": ["行禮", "靠近", "受壓", "疲憊支撐"],
            },
            {
                "name": "手與指節描寫",
                "aliases": ["手", "手指", "指尖"],
                "keywords": ["手", "指", "指尖", "指節", "掌心", "握住", "攥", "按住", "撫"],
                "summary": "用手部微動作承接心理波動，讓人物的控制、緊張、猶豫或保護欲可見。",
                "formula": ["手的位置", "力度變化", "接觸物", "心理暗流"],
                "scenes": ["沉默", "試探", "護住", "壓抑怒意", "遞物"],
            },
            {
                "name": "肩背與脊線描寫",
                "aliases": ["肩", "背", "脊背"],
                "keywords": ["肩", "背", "脊", "脊背", "肩線", "肩頭", "挺背", "塌下"],
                "summary": "用肩背線條呈現承擔、戒備、失落或威嚴，適合不用心理獨白時表現角色狀態。",
                "formula": ["肩背線條", "環境壓力", "細微變形", "讀者推知的情緒"],
                "scenes": ["背影", "離別", "承壓", "統御場面"],
            },
            {
                "name": "服飾襯托描寫",
                "aliases": ["衣", "袍", "裙", "甲"],
                "keywords": ["衣", "袍", "裙", "衫", "袖", "衣襬", "衣角", "甲", "披風"],
                "summary": "讓服飾跟動作、身份、場景互相牽動，而不是孤立描述衣服本身。",
                "formula": ["服飾材質", "動作牽引", "身份暗示", "場景反差"],
                "scenes": ["宴席", "出征", "潛行", "儀式", "雨雪場景"],
            },
        ],
    },
    {
        "category": "動作描寫",
        "subcategories": [
            {
                "name": "走路步伐描寫",
                "aliases": ["步伐", "行走", "走近"],
                "keywords": ["走", "步", "腳步", "步伐", "走近", "踏入", "踱", "退後"],
                "summary": "用步速、距離變化與地面聲音控制場面壓力，讓人物靠近本身形成劇情推進。",
                "formula": ["起步", "距離縮短或拉開", "聲音/地面", "對方反應"],
                "scenes": ["逼近", "退讓", "巡視", "追逐", "登場"],
            },
            {
                "name": "轉身回眸描寫",
                "aliases": ["轉身", "回頭", "回眸"],
                "keywords": ["轉身", "回頭", "回眸", "側身", "偏頭", "轉過"],
                "summary": "把轉身拆成方向、停頓、視線與未完成的話，讓一個小動作帶出關係變化。",
                "formula": ["身體轉向", "停頓", "視線落點", "未說出口的變化"],
                "scenes": ["離開前", "被叫住", "忽然察覺", "情緒轉折"],
            },
            {
                "name": "喝酒飲茶描寫",
                "aliases": ["喝酒", "飲酒", "飲茶", "舉杯"],
                "keywords": ["酒", "杯", "飲", "喝", "茶", "盞", "壺", "舉杯", "抿了一口"],
                "summary": "用倒酒、舉杯、入口、吞嚥、放杯的節拍，把對話中的試探、權力或情緒壓住。",
                "formula": ["器物聲", "入口節拍", "停頓或吞嚥", "話語轉向"],
                "scenes": ["宴席", "談判", "失意", "試探", "故作鎮定"],
            },
            {
                "name": "拔劍出招描寫",
                "aliases": ["拔劍", "出劍", "出招"],
                "keywords": ["劍", "刀", "拔", "出鞘", "劍光", "招式", "斬", "刺", "劈"],
                "summary": "先寫意圖與距離，再寫武器出現和結果，避免只堆招式名稱而缺少壓力。",
                "formula": ["殺意或決心", "距離/角度", "出招瞬間", "造成的後果"],
                "scenes": ["決鬥", "偷襲", "護人", "威懾", "破局"],
            },
            {
                "name": "戰鬥閃避描寫",
                "aliases": ["戰鬥", "閃避", "格擋"],
                "keywords": ["閃", "避", "擋", "格", "退", "攻", "掌風", "拳", "血", "傷"],
                "summary": "用攻防節拍、受力方向和代價讓戰鬥有因果，不讓動作變成流水帳。",
                "formula": ["攻擊來源", "防禦選擇", "受力後果", "下一個破綻"],
                "scenes": ["近身戰", "追殺", "以弱抗強", "救援"],
            },
            {
                "name": "觸碰拉近描寫",
                "aliases": ["觸碰", "靠近", "抓住"],
                "keywords": ["碰", "觸", "抓", "握", "拉住", "靠近", "扶", "按"],
                "summary": "用接觸點、力度、停留時間與被碰者反應，讓距離變化承載關係張力。",
                "formula": ["接觸點", "力度", "停留時間", "反應與後續選擇"],
                "scenes": ["救人", "阻止", "安撫", "威脅", "曖昧拉扯"],
            },
            {
                "name": "坐臥起身描寫",
                "aliases": ["坐下", "起身", "躺下"],
                "keywords": ["坐", "起身", "站起", "躺", "倚", "靠", "跪", "俯身"],
                "summary": "用身體高度改變表現權力、疲憊、屈服或反擊前的蓄力。",
                "formula": ["高度變化", "支撐點", "場面權力", "下一步動作"],
                "scenes": ["審問", "病榻", "行禮", "談判結束"],
            },
        ],
    },
    {
        "category": "情緒描寫",
        "subcategories": [
            {
                "name": "壓抑情緒描寫",
                "aliases": ["忍耐", "克制"],
                "keywords": ["忍", "壓下", "克制", "沉默", "垂下", "指節發白", "喉間"],
                "summary": "用身體壓住情緒的痕跡替代直白說明，讓讀者自己讀出未爆發的部分。",
                "formula": ["情緒衝擊", "壓住的動作", "外表平靜", "微小裂縫"],
                "scenes": ["忍怒", "忍痛", "隱瞞真相", "長輩或上位者面前"],
            },
            {
                "name": "憤怒描寫",
                "aliases": ["怒意", "殺意"],
                "keywords": ["怒", "冷聲", "殺意", "咬牙", "握緊", "震怒", "厲聲"],
                "summary": "讓怒意先改變聲音、手部、呼吸與空氣，再讓人物做出選擇。",
                "formula": ["外界刺激", "身體反應", "語氣降溫或爆發", "行動代價"],
                "scenes": ["質問", "背叛", "護短", "逼供"],
            },
            {
                "name": "恐懼危機描寫",
                "aliases": ["害怕", "危險"],
                "keywords": ["怕", "恐", "寒意", "發冷", "顫", "驚", "危險", "心口一緊"],
                "summary": "把恐懼放進感官錯位、呼吸和判斷變窄，讓危機感貼近身體。",
                "formula": ["危險信號", "感官收縮", "身體失控", "求生判斷"],
                "scenes": ["被追", "暗殺前", "揭露真相", "密室"],
            },
            {
                "name": "悲傷失落描寫",
                "aliases": ["哀傷", "失落"],
                "keywords": ["悲", "痛", "失落", "空", "怔", "眼眶", "苦笑", "沉下"],
                "summary": "用空白、遲滯、物件停留和聲音變輕表現悲傷，不急著讓角色解釋。",
                "formula": ["失去的觸發", "動作停住", "感官變遠", "一句收束"],
                "scenes": ["離別", "喪失", "真相揭開", "夢醒"],
            },
            {
                "name": "決意覺悟描寫",
                "aliases": ["決心", "覺悟"],
                "keywords": ["決", "定了", "不退", "抬頭", "站直", "深吸", "目光沉定"],
                "summary": "用姿態變穩、視線明確和語句變短，讓決意看起來像一個可見的轉折。",
                "formula": ["猶豫殘留", "身體穩住", "目標明確", "短句行動"],
                "scenes": ["反擊前", "立誓", "接受代價", "保護他人"],
            },
        ],
    },
    {
        "category": "場景氛圍",
        "subcategories": [
            {
                "name": "雨雪風描寫",
                "aliases": ["雨", "雪", "風"],
                "keywords": ["雨", "雪", "風", "霜", "寒", "濕", "雷", "霧"],
                "summary": "讓天氣與人物情緒、行動阻力互相作用，而不是只作背景板。",
                "formula": ["天氣觸感", "動作阻力", "情緒映照", "場面轉折"],
                "scenes": ["夜行", "追逐", "離別", "壓抑場景"],
            },
            {
                "name": "室內宮殿描寫",
                "aliases": ["房間", "大殿", "宮殿"],
                "keywords": ["殿", "房", "室", "廳", "門", "窗", "榻", "簾", "燭"],
                "summary": "用空間層次、門窗位置和器物聲音安排人物權力與動線。",
                "formula": ["空間格局", "人物站位", "器物/光源", "權力距離"],
                "scenes": ["審問", "密談", "宮廷對峙", "病榻"],
            },
            {
                "name": "夜色燭火描寫",
                "aliases": ["夜", "燭", "月"],
                "keywords": ["夜", "月", "燭", "燈", "暗", "影", "昏", "火光"],
                "summary": "用明暗邊界、影子和看不清的部分製造隱祕、危險或曖昧。",
                "formula": ["光源位置", "陰影遮蔽", "可見細節", "未知壓力"],
                "scenes": ["潛入", "密會", "刺殺前", "夢境"],
            },
            {
                "name": "宴席人群描寫",
                "aliases": ["宴席", "人群", "眾人"],
                "keywords": ["宴", "席", "眾人", "人群", "賓客", "杯盞", "喧", "靜了一瞬"],
                "summary": "用群體聲浪的升降凸顯某個人的動作或一句話，放大社交壓力。",
                "formula": ["群體背景聲", "異常停頓", "焦點人物", "局勢改變"],
                "scenes": ["公開羞辱", "身份揭露", "談判", "暗流交鋒"],
            },
        ],
    },
    {
        "category": "對話與關係",
        "subcategories": [
            {
                "name": "對話交鋒描寫",
                "aliases": ["交鋒", "質問"],
                "keywords": ["問", "答", "冷聲", "低聲", "笑了", "沉默", "反問", "質問"],
                "summary": "讓每句話都帶有目的、遮掩或反擊，並用停頓和動作補足沒說出的部分。",
                "formula": ["表面話題", "真正目的", "反應動作", "局勢推進"],
                "scenes": ["審問", "談判", "試探", "爭吵"],
            },
            {
                "name": "潛台詞描寫",
                "aliases": ["未說出口", "試探"],
                "keywords": ["沒有說", "未說", "沉默", "只道", "淡淡", "似笑非笑", "試探"],
                "summary": "用避開的詞、轉移的視線和不完整回應，讓讀者讀到話外之意。",
                "formula": ["避而不答", "轉移焦點", "小動作洩露", "對方追問"],
                "scenes": ["互相試探", "隱瞞秘密", "權力不對等", "感情拉扯"],
            },
            {
                "name": "權力壓迫描寫",
                "aliases": ["壓迫", "威壓"],
                "keywords": ["威壓", "壓迫", "跪", "命令", "不敢", "退下", "俯首", "上位"],
                "summary": "用高度、距離、稱謂和是否能開口呈現權力差，而不是直接說誰比較強。",
                "formula": ["位置高低", "命令或沉默", "受壓反應", "代價"],
                "scenes": ["上位者登場", "審判", "師徒/君臣", "逼迫選擇"],
            },
            {
                "name": "曖昧拉扯描寫",
                "aliases": ["拉扯", "親近"],
                "keywords": ["近", "退", "避開", "靠近", "呼吸", "停了一瞬", "耳邊", "指尖"],
                "summary": "用距離進退、呼吸停頓和不確定反應寫關係張力，避免直接宣告情緒。",
                "formula": ["距離變化", "身體細節", "克制或退避", "關係未定"],
                "scenes": ["相互試探", "救護後", "密談", "離別前"],
            },
        ],
    },
    {
        "category": "節奏章法",
        "subcategories": [
            {
                "name": "開場鉤子",
                "aliases": ["開場", "鉤子"],
                "keywords": ["開場", "開頭", "鉤子", "第一句", "異常", "問題"],
                "summary": "用異常資訊、動作壓力或未解問題開場，先讓讀者想知道發生了什麼。",
                "formula": ["異常信號", "當下動作", "缺口資訊", "短暫延遲解答"],
                "scenes": ["章節開頭", "轉場", "新事件引入"],
            },
            {
                "name": "信息釋放",
                "aliases": ["資訊", "伏筆"],
                "keywords": ["伏筆", "線索", "秘密", "揭露", "真相", "隱瞞", "信息", "資訊"],
                "summary": "一次只釋放足以改變判斷的資訊，保留下一個問題讓讀者繼續追。",
                "formula": ["先給現象", "局部解釋", "新疑問", "角色選擇"],
                "scenes": ["查案", "身世揭露", "陰謀推進", "對話轉折"],
            },
            {
                "name": "章尾轉折",
                "aliases": ["章尾", "結尾", "轉折"],
                "keywords": ["章尾", "結尾", "轉折", "忽然", "卻", "下一刻", "沒想到"],
                "summary": "用選擇、揭露、來人、代價或危機收尾，讓章末成為下一章的推力。",
                "formula": ["當前目標", "意外變量", "代價或新問題", "未完成動作"],
                "scenes": ["章節結尾", "戰鬥前", "秘密揭露後"],
            },
            {
                "name": "視角鏡頭調度",
                "aliases": ["視角", "鏡頭"],
                "keywords": ["視角", "鏡頭", "近景", "遠景", "看見", "聽見", "轉向"],
                "summary": "控制遠近、可見與不可見資訊，讓讀者的注意力跟著場面焦點移動。",
                "formula": ["遠景定位", "近景細節", "焦點切換", "情緒落點"],
                "scenes": ["群像", "戰鬥", "密室", "人物登場"],
            },
        ],
    },
]

ALL_CATEGORY_CHOICES = ["All"]
for _category in TAXONOMY:
    ALL_CATEGORY_CHOICES.append(_category["category"])
    for _sub in _category["subcategories"]:
        ALL_CATEGORY_CHOICES.append(f"{_category['category']} / {_sub['name']}")


@dataclass(frozen=True)
class SourceDocument:
    label: str
    text: str


@dataclass
class ReferenceShelfItem:
    id: str
    label: str
    kind: str
    text: str
    chars: int
    digest: str
    added_at: str


@dataclass
class TechniqueBookEntry:
    id: str
    category: str
    subcategory: str
    title: str
    aliases: list[str]
    trigger_keywords: list[str]
    summary: str
    formula_steps: list[str]
    applicable_scenes: list[str]
    director_instruction: str
    source_labels: list[str]
    evidence_count: int
    evidence_signals: list[str]
    deep_breakdown: list[str]
    detail_lenses: list[str]
    micro_techniques: list[str]
    common_mistakes: list[str]
    practice_prompts: list[str]
    # --- Deeper craft dissection (added for "how each body part / action is written") ---
    anatomy_breakdown: list[str]  # 把這個部位/動作拆成「可觀察的微單元」並排出書寫順序
    sentence_rhythm: list[str]  # 句長、停頓、標點、段落節奏的具體規格
    word_palette: list[str]  # 適合的動詞/名詞/質感詞，與該避免的抽象詞
    sensory_layering: list[str]  # 哪個感官領頭、順序、各放幾個細節
    weak_vs_strong: list[str]  # 弱寫 → 強寫 的短句級對照示例


def add_references_to_shelf(
    source_files: Any,
    source_paths_text: str,
    pasted_source_text: str,
    pasted_label: str,
    shelf_state_json: str,
    max_chars_per_source: float | int | None,
) -> tuple[str, str, str, list[str]]:
    """Add uploaded/path/pasted sources to the persistent reference shelf."""
    try:
        max_chars = to_positive_int(max_chars_per_source) or 65000
        shelf = load_shelf_payload(shelf_state_json)
        existing = {item.get("digest") for item in shelf.get("items", [])}
        docs = load_source_documents(source_files, source_paths_text, pasted_source_text, max_chars)
        added = 0
        skipped = 0
        for doc in docs:
            label = normalize_reference_label(doc.label, pasted_label)
            digest = text_digest(doc.text)
            if digest in existing:
                skipped += 1
                continue
            existing.add(digest)
            shelf.setdefault("items", []).append(
                asdict(
                    ReferenceShelfItem(
                        id=f"REF-{digest[:10]}",
                        label=label,
                        kind=guess_source_kind(label),
                        text=doc.text,
                        chars=len(doc.text),
                        digest=digest,
                        added_at=datetime.now().isoformat(timespec="seconds"),
                    )
                )
            )
            added += 1
        shelf["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_reference_shelf_payload(shelf)
        status = (
            "[OK] Reference Shelf 已更新。\n"
            f"新增：{added}\n"
            f"略過重複：{skipped}\n"
            f"目前來源：{len(shelf.get('items', []))}\n"
            f"保存：{reference_shelf_path()}"
        )
        return status, render_reference_shelf(shelf), json.dumps(shelf, ensure_ascii=False), shelf_choices(shelf)
    except Exception as exc:
        shelf = load_shelf_payload(shelf_state_json)
        return f"[ERROR] {exc}", render_reference_shelf(shelf), json.dumps(shelf, ensure_ascii=False), shelf_choices(shelf)


def remove_references_from_shelf(
    selected_reference_labels: list[str] | str | None,
    shelf_state_json: str,
) -> tuple[str, str, str, list[str]]:
    """Remove selected references from the shelf."""
    try:
        shelf = load_shelf_payload(shelf_state_json)
        selected = selected_reference_labels or []
        if isinstance(selected, str):
            selected = [selected]
        selected_ids = {choice_to_reference_id(item) for item in selected}
        before = len(shelf.get("items", []))
        shelf["items"] = [
            item for item in shelf.get("items", []) if str(item.get("id", "")) not in selected_ids
        ]
        removed = before - len(shelf.get("items", []))
        shelf["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_reference_shelf_payload(shelf)
        status = (
            "[OK] Reference Shelf 已刪除選取來源。\n"
            f"刪除：{removed}\n"
            f"目前來源：{len(shelf.get('items', []))}"
        )
        return status, render_reference_shelf(shelf), json.dumps(shelf, ensure_ascii=False), shelf_choices(shelf)
    except Exception as exc:
        shelf = load_shelf_payload(shelf_state_json)
        return f"[ERROR] {exc}", render_reference_shelf(shelf), json.dumps(shelf, ensure_ascii=False), shelf_choices(shelf)


def clear_reference_shelf() -> tuple[str, str, str, list[str]]:
    """Clear the persistent reference shelf."""
    shelf = empty_shelf_payload()
    shelf["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_reference_shelf_payload(shelf)
    return (
        f"[OK] Reference Shelf 已清空。\n保存：{reference_shelf_path()}",
        render_reference_shelf(shelf),
        json.dumps(shelf, ensure_ascii=False),
        [],
    )


def load_saved_reference_shelf() -> tuple[str, str, str, list[str]]:
    """Load the saved reference shelf from disk."""
    shelf = load_shelf_payload("")
    return (
        f"[OK] Reference Shelf 已載入。\n目前來源：{len(shelf.get('items', []))}\n來源檔：{reference_shelf_path()}",
        render_reference_shelf(shelf),
        json.dumps(shelf, ensure_ascii=False),
        shelf_choices(shelf),
    )


def build_integrated_technique_book_library_from_shelf(
    shelf_state_json: str,
    library_goal: str,
    output_language: str,
    max_entries_per_subcategory: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None, str | None, str]:
    """Build the technique book library from the current reference shelf."""
    try:
        shelf = load_shelf_payload(shelf_state_json)
        docs = [
            SourceDocument(label=str(item.get("label") or item.get("id")), text=str(item.get("text") or ""))
            for item in shelf.get("items", [])
            if str(item.get("text") or "").strip()
        ]
        if not docs:
            return "[ERROR] Reference Shelf 是空的。請先新增小說、報告或 Technique JSON。", "", None, None, ""
        return build_library_from_documents(
            docs=docs,
            library_goal=library_goal,
            output_language=output_language,
            max_entries_per_subcategory=max_entries_per_subcategory,
            dry_run=dry_run,
            analysis_api_key=analysis_api_key,
            analysis_base_url=analysis_base_url,
            analysis_model_name=analysis_model_name,
            source_mode="Reference Shelf",
        )
    except Exception as exc:
        return f"[ERROR] {exc}", "", None, None, ""


def build_integrated_technique_book_library(
    source_files: Any,
    source_paths_text: str,
    pasted_source_text: str,
    library_goal: str,
    output_language: str,
    max_chars_per_source: float | int | None,
    max_entries_per_subcategory: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None, str | None, str]:
    """Gradio entry point for the integrated searchable technique book library."""
    try:
        max_chars = to_positive_int(max_chars_per_source) or 65000
        docs = load_source_documents(source_files, source_paths_text, pasted_source_text, max_chars)
        if not docs:
            return "[ERROR] 請上傳多個 MD/TXT/JSON、輸入檔案/資料夾路徑，或貼上來源內容。", "", None, None, ""
        return build_library_from_documents(
            docs=docs,
            library_goal=library_goal,
            output_language=output_language,
            max_entries_per_subcategory=max_entries_per_subcategory,
            dry_run=dry_run,
            analysis_api_key=analysis_api_key,
            analysis_base_url=analysis_base_url,
            analysis_model_name=analysis_model_name,
            source_mode="Direct Batch Input",
        )
    except Exception as exc:
        return f"[ERROR] {exc}", "", None, None, ""


def build_library_from_documents(
    *,
    docs: list[SourceDocument],
    library_goal: str,
    output_language: str,
    max_entries_per_subcategory: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
    source_mode: str,
) -> tuple[str, str, str | None, str | None, str]:
    """Shared implementation for direct input and reference-shelf builds."""
    try:
        max_entries = to_positive_int(max_entries_per_subcategory) or 3
        goal = (library_goal or DEFAULT_BOOK_LIBRARY_GOAL).strip()
        local_entries = build_local_entries(docs)
        if dry_run:
            entries = local_entries
            mode = f"{source_mode} / Dry Run / Local Categorizer"
        else:
            if not analysis_base_url.strip() or not analysis_model_name.strip():
                return "[ERROR] 請先設定 Analysis / Grok 的 Base URL 與 Model Name。", "", None, None, ""
            client = OpenAI(
                api_key=(analysis_api_key or "not-needed").strip(),
                base_url=analysis_base_url.strip().rstrip("/"),
                timeout=900,
            )
            entries = ask_grok_for_book_entries(
                client=client,
                model_name=analysis_model_name.strip(),
                docs=docs,
                goal=goal,
                output_language=output_language or "繁體中文",
                max_entries_per_subcategory=max_entries,
            )
            if not entries:
                entries = local_entries
                mode = f"{source_mode} / Grok / {analysis_model_name.strip()} (fallback local categorizer)"
            else:
                entries = merge_entries(entries, max_entries)
                mode = f"{source_mode} / Grok / {analysis_model_name.strip()}"

        if not entries:
            return "[ERROR] 沒有抽取到可歸類的技法。請提高每檔字數上限，或改用 Grok 蒸餾。", "", None, None, ""

        output_dir = write_book_library(
            docs=docs,
            entries=entries,
            goal=goal,
            mode=mode,
            output_language=output_language or "繁體中文",
        )
        md_path = output_dir / "integrated_technique_book_library.md"
        json_path = output_dir / "integrated_technique_book_library.json"
        state_json = json_path.read_text(encoding="utf-8")
        preview = md_path.read_text(encoding="utf-8")
        status = (
            "[OK] Integrated Technique Book Library 已建立。\n"
            f"來源檔數：{len(docs)}\n"
            f"技法卡數：{len(entries)}\n"
            f"模式：{mode}\n"
            f"Markdown：{md_path}\n"
            f"JSON：{json_path}"
        )
        return status, trim_preview(preview, 26000), str(md_path), str(json_path), state_json
    except Exception as exc:
        return f"[ERROR] {exc}", "", None, None, ""


def search_integrated_technique_book_library(
    query: str,
    category_filter: str,
    library_json_path: str,
    library_state_json: str,
    result_limit: float | int | None,
) -> tuple[str, str]:
    """Search current or latest integrated technique book library."""
    try:
        payload = load_library_payload(library_state_json, library_json_path)
        entries = entries_from_payload(payload)
        limit = to_positive_int(result_limit) or 12
        results = rank_entries(entries, query, category_filter)[:limit]
        markdown = render_search_results(results, query, category_filter, payload)
        status = (
            "[OK] 查詢完成。\n"
            f"查詢：{(query or '').strip() or 'All'}\n"
            f"分類：{category_filter or 'All'}\n"
            f"結果：{len(results)} / {len(entries)}"
        )
        return status, markdown
    except Exception as exc:
        return f"[ERROR] {exc}", ""


def load_technique_book_to_agent_fields(
    query: str,
    category_filter: str,
    library_json_path: str,
    library_state_json: str,
    current_technique_library: str,
    current_memory: str,
    current_director_instruction: str,
    load_mode: str,
    result_limit: float | int | None,
) -> tuple[str, str, str, str]:
    """Load selected searchable book-library entries into writing-agent fields."""
    try:
        payload = load_library_payload(library_state_json, library_json_path)
        entries = entries_from_payload(payload)
        limit = to_positive_int(result_limit) or 16
        selected = rank_entries(entries, query, category_filter)[:limit]
        if not selected:
            return (
                current_technique_library or "",
                current_memory or "",
                current_director_instruction or "",
                "[ERROR] 沒有符合條件的技法卡，請調整查詢或分類。",
            )
        incoming_library = render_agent_reference(selected, query, category_filter)
        memory_insert = (
            "Technique Book memory: 使用已載入的分類技法作為寫作方法參考；"
            "只借用描寫策略、節奏與動作公式，不複製來源小說情節或句子。"
        )
        director_insert = "\n".join(
            f"- {entry.director_instruction}"
            for entry in selected[:8]
            if entry.director_instruction.strip()
        )
        mode = load_mode or BOOK_LIBRARY_LOAD_MODES[0]
        replace_library = mode.startswith("Replace")
        include_memory_director = "Memory/Director" in mode
        new_library = merge_text_field(
            current=current_technique_library,
            incoming=incoming_library,
            header="Integrated Technique Book Selection",
            replace=replace_library,
        )
        new_memory = current_memory or ""
        new_director = current_director_instruction or ""
        if include_memory_director:
            new_memory = merge_text_field(
                current=current_memory,
                incoming=memory_insert,
                header="Technique Book Memory Insert",
                replace=False,
            )
            new_director = merge_text_field(
                current=current_director_instruction,
                incoming=director_insert,
                header="Technique Book Director Instruction",
                replace=False,
            )
        status = (
            "[OK] 已將書庫查詢結果載入寫作 AGENT。\n"
            f"載入技法卡：{len(selected)}\n"
            f"模式：{mode}"
        )
        return new_library, new_memory, new_director, status
    except Exception as exc:
        return (
            current_technique_library or "",
            current_memory or "",
            current_director_instruction or "",
            f"[ERROR] {exc}",
        )


def load_latest_technique_book_to_agent_fields(
    current_technique_library: str,
    current_memory: str,
    current_director_instruction: str,
    load_mode: str,
    result_limit: float | int | None,
) -> tuple[str, str, str, str]:
    latest = find_latest_book_library_json()
    if latest is None:
        return (
            current_technique_library or "",
            current_memory or "",
            current_director_instruction or "",
            "[ERROR] 找不到已建立的 Integrated Technique Book Library。",
        )
    return load_technique_book_to_agent_fields(
        "",
        "All",
        str(latest),
        "",
        current_technique_library,
        current_memory,
        current_director_instruction,
        load_mode,
        result_limit,
    )


def load_source_documents(
    source_files: Any,
    source_paths_text: str,
    pasted_source_text: str,
    max_chars_per_source: int,
) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    seen: set[str] = set()
    for path in iter_uploaded_paths(source_files):
        add_source_doc(docs, seen, Path(path), max_chars_per_source)
    for path in iter_path_inputs(source_paths_text):
        add_source_doc(docs, seen, path, max_chars_per_source)
    if (pasted_source_text or "").strip():
        text = clean_source_text(pasted_source_text)
        docs.append(SourceDocument(label="pasted-source", text=trim_text(text, max_chars_per_source)))
    return docs[:200]


def reference_shelf_path() -> Path:
    output_dir = Path.cwd() / "book_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir / "technique_reference_shelf.json"


def empty_shelf_payload() -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "items": [],
    }


def load_shelf_payload(shelf_state_json: str) -> dict[str, Any]:
    if (shelf_state_json or "").strip():
        try:
            payload = json.loads(shelf_state_json)
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                return payload
        except json.JSONDecodeError:
            pass
    path = reference_shelf_path()
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("items"), list):
                return payload
        except json.JSONDecodeError:
            pass
    return empty_shelf_payload()


def save_reference_shelf_payload(shelf: dict[str, Any]) -> None:
    path = reference_shelf_path()
    path.write_text(json.dumps(shelf, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_reference_label(label: str, pasted_label: str) -> str:
    if label == "pasted-source" and (pasted_label or "").strip():
        return (pasted_label or "").strip()
    return (label or "unnamed-reference").strip()


def guess_source_kind(label: str) -> str:
    suffix = Path(label).suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".md":
        return "markdown"
    if suffix == ".txt":
        return "novel-txt"
    return "pasted"


def text_digest(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def shelf_choices(shelf: dict[str, Any]) -> list[str]:
    choices = []
    for item in shelf.get("items", []):
        choices.append(
            f"{item.get('id', '')} | {item.get('label', 'unnamed')} | {item.get('chars', 0)} chars"
        )
    return choices


def choice_to_reference_id(choice: str) -> str:
    return (choice or "").split("|", 1)[0].strip()


def render_reference_shelf(shelf: dict[str, Any]) -> str:
    items = shelf.get("items", [])
    lines = [
        "# Reference Shelf",
        "",
        f"- Updated: {shelf.get('updated_at', '')}",
        f"- Sources: {len(items)}",
        "",
    ]
    if not items:
        lines.append("目前沒有小說參考。請新增 TXT / MD / JSON / pasted source。")
        return "\n".join(lines)
    lines.append("## Current References")
    lines.append("")
    for index, item in enumerate(items, start=1):
        label = str(item.get("label") or "unnamed")
        lines.extend(
            [
                f"### {index}. {label}",
                "",
                f"- ID: {item.get('id', '')}",
                f"- Kind: {item.get('kind', '')}",
                f"- Chars: {item.get('chars', 0)}",
                f"- Added: {item.get('added_at', '')}",
                f"- Digest: {str(item.get('digest', ''))[:12]}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def iter_uploaded_paths(source_files: Any) -> list[str]:
    if not source_files:
        return []
    files = source_files if isinstance(source_files, list) else [source_files]
    paths = []
    for item in files:
        name = getattr(item, "name", None) or getattr(item, "path", None) or item
        if name:
            paths.append(str(name))
    return paths


def iter_path_inputs(source_paths_text: str) -> list[Path]:
    paths: list[Path] = []
    for raw in (source_paths_text or "").splitlines():
        item = raw.strip().strip('"').strip("'")
        if not item:
            continue
        expanded = Path(item).expanduser()
        if not expanded.is_absolute():
            expanded = Path.cwd() / expanded
        if expanded.is_dir():
            for suffix in ("*.md", "*.txt", "*.json"):
                paths.extend(sorted(expanded.rglob(suffix)))
        else:
            matched = [Path(item) for item in sorted(glob(str(expanded)))] if any(ch in str(expanded) for ch in "*?") else [expanded]
            paths.extend(matched)
    return paths


def add_source_doc(docs: list[SourceDocument], seen: set[str], path: Path, max_chars_per_source: int) -> None:
    path = path.expanduser()
    if not path.is_file():
        return
    resolved = str(path.resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    text = read_any_source(path)
    text = clean_source_text(text)
    if not text.strip():
        return
    docs.append(SourceDocument(label=str(path), text=trim_text(text, max_chars_per_source)))


def read_any_source(path: Path) -> str:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(read_text(path))
            return flatten_json_source(payload)
        except Exception:
            return read_text(path)
    return read_text(path)


def flatten_json_source(payload: Any) -> str:
    if isinstance(payload, dict):
        parts: list[str] = []
        for key in ("technique_library", "raw_distillation", "report"):
            if isinstance(payload.get(key), str):
                parts.append(payload[key])
        cards = payload.get("cards")
        if isinstance(cards, list):
            for card in cards:
                if isinstance(card, dict):
                    parts.append(" / ".join(str(card.get(k, "")) for k in (
                        "title",
                        "scene",
                        "action",
                        "situation",
                        "technique_summary",
                        "formulas",
                        "director_instruction",
                    )))
        if parts:
            return "\n\n".join(parts)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def clean_source_text(text: str) -> str:
    text = sanitize_review_text(text or "")
    clean_lines = []
    for line in text.splitlines():
        lower = line.lower()
        if any(marker.lower() in lower for marker in BLOCKED_CONTENT_MARKERS):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def build_local_entries(docs: list[SourceDocument]) -> list[TechniqueBookEntry]:
    entries: list[TechniqueBookEntry] = []
    for category in TAXONOMY:
        for sub in category["subcategories"]:
            evidence_count = 0
            source_labels: list[str] = []
            evidence_signals: list[str] = []
            for doc in docs:
                count, signals = match_subcategory(doc.text, sub["keywords"])
                if count:
                    evidence_count += count
                    source_labels.append(doc.label)
                    evidence_signals.extend(signals)
            if evidence_count:
                entries.append(
                    make_taxonomy_entry(
                        category=category["category"],
                        sub=sub,
                        source_labels=source_labels,
                        evidence_count=evidence_count,
                        evidence_signals=dedupe(evidence_signals)[:8],
                    )
                )
    return entries


def match_subcategory(text: str, keywords: list[str]) -> tuple[int, list[str]]:
    count = 0
    signals: list[str] = []
    compact = re.sub(r"\s+", " ", text)
    for keyword in keywords:
        occurrences = compact.count(keyword)
        if occurrences:
            count += occurrences
            signals.append(f"偵測到「{keyword}」相關描寫 {occurrences} 次")
    context_signals = extract_context_signals(compact, keywords)
    signals.extend(context_signals)
    return count, signals


def extract_context_signals(text: str, keywords: list[str]) -> list[str]:
    signals: list[str] = []
    for keyword in keywords[:4]:
        index = text.find(keyword)
        if index < 0:
            continue
        start = max(0, index - 24)
        end = min(len(text), index + len(keyword) + 24)
        window = text[start:end]
        nearby = [
            token
            for token in keywords
            if token != keyword and token in window
        ]
        if nearby:
            signals.append(f"「{keyword}」常與「{'、'.join(nearby[:4])}」同場出現")
        else:
            signals.append(f"「{keyword}」出現在可抽取的描寫段落")
    return signals


def make_taxonomy_entry(
    *,
    category: str,
    sub: dict[str, Any],
    source_labels: list[str],
    evidence_count: int,
    evidence_signals: list[str],
) -> TechniqueBookEntry:
    title = f"{sub['name']}：{sub['summary'][:26]}"
    entry_id = stable_entry_id(category, sub["name"], title)
    formula = [str(step) for step in sub.get("formula", [])]
    deep = build_deep_profile(category, sub["name"], formula)
    director_instruction = (
        f"需要{category}/{sub['name']}時，按「"
        + " -> ".join(formula)
        + f"」寫；讓細節推動情緒、關係或局勢，不要只堆形容詞。"
    )
    return TechniqueBookEntry(
        id=entry_id,
        category=category,
        subcategory=sub["name"],
        title=title,
        aliases=[str(item) for item in sub.get("aliases", [])],
        trigger_keywords=[str(item) for item in sub.get("keywords", [])],
        summary=str(sub.get("summary", "")).strip(),
        formula_steps=formula,
        applicable_scenes=[str(item) for item in sub.get("scenes", [])],
        director_instruction=director_instruction,
        source_labels=dedupe(source_labels),
        evidence_count=evidence_count,
        evidence_signals=evidence_signals,
        deep_breakdown=deep["deep_breakdown"],
        detail_lenses=deep["detail_lenses"],
        micro_techniques=deep["micro_techniques"],
        common_mistakes=deep["common_mistakes"],
        practice_prompts=deep["practice_prompts"],
        anatomy_breakdown=deep["anatomy_breakdown"],
        sentence_rhythm=deep["sentence_rhythm"],
        word_palette=deep["word_palette"],
        sensory_layering=deep["sensory_layering"],
        weak_vs_strong=deep["weak_vs_strong"],
    )


def build_deep_profile(category: str, subcategory: str, formula_steps: list[str]) -> dict[str, list[str]]:
    """Return reusable, sentence-level deep-analysis fields for a category/subcategory.

    Every block carries the full deep schema, including the added craft-dissection
    fields (anatomy_breakdown / sentence_rhythm / word_palette / sensory_layering /
    weak_vs_strong) so even the offline / dry-run path teaches "how a body part or an
    action is actually written", not a generic checklist.
    """
    body_common = {
        "deep_breakdown": [
            "先決定讀者此刻要感到的是吸引、壓迫、疏離、心疼還是警戒，再選擇要放大的部位。",
            "不要把身體部位孤立成外貌清單；每個細節都要連到人物狀態、權力位置或關係距離。",
            "用一個可見變化承接情緒：視線停住、唇角收緊、指節發白、肩背塌下，比直接說情緒更有力。",
            "描寫順序通常是整體印象 -> 一個部位特寫 -> 動作造成的變化 -> 旁人的反應或局勢改變。",
        ],
        "detail_lenses": [
            "光影落在哪裡",
            "哪個部位正在動或被迫不動",
            "動作力度是放鬆、克制、顫抖還是壓住",
            "對方看見後產生什麼誤判或情緒變化",
        ],
        "micro_techniques": [
            "把形容詞改寫成動作痕跡，例如把『冷』寫成眼神越過對方、聲音沒有起伏。",
            "用停頓寫心理：看見、停住、移開、再開口。",
            "讓部位描寫和場景物件互動，例如杯沿映出目光、袖口遮住指尖。",
        ],
        "common_mistakes": [
            "只堆美貌詞，沒有情緒目的。",
            "連續描寫多個部位，卻沒有焦點。",
            "每次角色登場都用同一套五官模板。",
        ],
        "practice_prompts": [
            "用 120 字寫一個人壓住怒意，只能寫眼睛、嘴角與手。",
            "把『她很美但很危險』改寫成三個可見細節。",
        ],
        "anatomy_breakdown": [
            "整體印象（一句定調：氣質、距離感、此刻的權力位置）。",
            "鎖定單一部位特寫（只放大一個，不要五官全寫）。",
            "寫這個部位的『此刻變化』而非靜態形狀（收緊、移開、發白、垂下）。",
            "讓這個變化和場景物件或對方互動（杯沿、袖口、門縫、對方的視線）。",
            "收在旁人或對方的反應、誤讀或局勢位移上。",
        ],
        "sentence_rhythm": [
            "定調句用中長句鋪一個整體印象，再用一個短句切到特寫。",
            "部位特寫那一句要短、要具體，逗號不超過兩個。",
            "變化的瞬間可用句號硬斷，製造『被看見』的停頓感。",
        ],
        "word_palette": [
            "多用可觀察的具體名詞與動詞：垂、抬、抿、攥、扣、頓、移。",
            "質感詞挑一個就好：濕、冷、白、緊、虛、沉。",
            "避免抽象判斷詞：美麗、優雅、迷人、危險——這些要由細節讓讀者自己得出。",
        ],
        "sensory_layering": [
            "視覺領頭（落點、光影），但只給一個主細節。",
            "第二層補一個非視覺感官（溫度、氣息、布料摩擦聲）即可，不要堆五感。",
            "最後一層交給對方的反應，讓描寫變成互動而非靜物。",
        ],
        "weak_vs_strong": [
            "弱：她有一雙美麗動人的眼睛。",
            "強：她抬眼，目光在他手腕上停了半息，又若無其事地移開。",
        ],
    }
    action_common = {
        "deep_breakdown": [
            "先確定動作的戲劇功能：靠近、試探、威脅、遮掩、逃避、反擊或示弱。",
            "把動作拆成起手、接觸/轉折、停頓、後果四段，讓讀者看見因果，而不是看見流水帳。",
            "每個動作都要改變至少一件事：距離、權力、資訊、情緒、節奏或危險程度。",
            "用器物聲、呼吸、衣料、地面、光影補足動作質感，讓動作有重量。",
        ],
        "detail_lenses": [
            "動作開始前的人物目的",
            "速度、力度、方向與距離",
            "動作中斷或停頓的位置",
            "動作完成後誰被迫回應",
        ],
        "micro_techniques": [
            "用三拍節奏：準備 -> 動作 -> 後果。",
            "在關鍵動作前插入一個極短停頓，製造壓力。",
            "讓對方反應收束動作，避免動作自己結束。",
        ],
        "common_mistakes": [
            "只寫動詞，沒有目的與阻力。",
            "招式或動作太多，讀者不知道哪一拍重要。",
            "每個動作都同速同力，缺少節奏起伏。",
        ],
        "practice_prompts": [
            "用 150 字寫『倒酒』，讓它變成一次試探，而不是普通喝酒。",
            "把『他拔劍』拆成殺意、距離、出鞘、後果四拍。",
        ],
        "anatomy_breakdown": [
            "起手前先給一個目的或殺意（讀者要知道這個動作為何發生）。",
            "起勢：身體哪個部位先動，往哪個方向，帶多少力。",
            "接觸或轉折：碰到了什麼、改變了什麼距離或關係。",
            "停頓：在最關鍵的一拍插一個極短的靜止，讓讀者屏息。",
            "後果：對方被迫反應，或局勢/危險/資訊發生位移。",
        ],
        "sentence_rhythm": [
            "準備段用一個中句交代目的與距離。",
            "動作本身用短句或斷句加速：主詞＋動詞，少形容。",
            "停頓那一拍單獨成句；後果用一句稍長的句子讓重量落地。",
        ],
        "word_palette": [
            "動詞要準：扣、抵、挑、壓、撤、收、頓，而非『做了一個動作』。",
            "用器物名詞承載聲音與質感：杯底、刀鞘、衣料、地磚。",
            "避免『迅速地』『用力地』這類副詞，改用具體結果（指節發白、酒液濺出）。",
        ],
        "sensory_layering": [
            "聽覺常是動作的節拍器：器物聲、呼吸、衣料摩擦。",
            "觸覺/體感放在接觸與停頓那一拍（重量、阻力、溫度）。",
            "視覺收尾，定格在後果上（裂痕、濺痕、對方的臉）。",
        ],
        "weak_vs_strong": [
            "弱：他快速地拔出劍，狠狠地砍了過去。",
            "強：他拇指先抵上劍格。半寸寒光出鞘，又停住——他在等對方先動。",
        ],
    }
    emotion_common = {
        "deep_breakdown": [
            "先寫情緒如何被壓住，再寫壓不住的裂縫。",
            "情緒不一定要由角色說出，可由呼吸、手、語速、視線與沉默洩露。",
            "讓外在場景或對方一句話成為情緒變形的觸發點。",
        ],
        "detail_lenses": ["呼吸", "手部", "語速", "沉默", "視線", "身體重心"],
        "micro_techniques": [
            "把內心句縮短，讓讀者從外部細節推回情緒。",
            "用反向描寫：越平靜越危險、越禮貌越疏離。",
        ],
        "common_mistakes": ["直接命名情緒太多。", "角色反應和事件重量不匹配。", "情緒沒有推動下一個選擇。"],
        "practice_prompts": ["寫一段角色聽見噩耗後不哭，只寫他整理袖口。"],
        "anatomy_breakdown": [
            "觸發點：哪一句話、哪一個動作引爆情緒。",
            "壓抑層：身體先做出克制動作（屏息、攥手、放緩語速）。",
            "裂縫層：某個壓不住的小細節洩露真相（聲音一抖、指節發白）。",
            "選擇層：情緒最後推動一個決定或一句話，不讓它空轉。",
        ],
        "sentence_rhythm": [
            "壓抑時用短而克制的句子，標點偏多、語氣偏平。",
            "裂縫出現時讓一個句子忽然斷掉，或留一個未完成的破折號——",
            "避免長段內心獨白；用外部細節替代心理形容。",
        ],
        "word_palette": [
            "用身體動詞外化情緒：攥、抿、頓、撇、垂、繃。",
            "避免直接命名：憤怒、悲傷、絕望、心痛。",
            "用『沒有』『仍』『只是』等克制副詞暗示壓抑（聲音沒有起伏）。",
        ],
        "sensory_layering": [
            "體感領頭：呼吸、心跳、喉間、指尖的緊與冷。",
            "聽覺次之：自己聲音的變化、周遭忽然安靜。",
            "視覺收束在一個被攥皺或被放下的物件上。",
        ],
        "weak_vs_strong": [
            "弱：她感到無比憤怒，幾乎要崩潰了。",
            "強：她把那封信折好，又折了一次，指腹把摺痕壓得死白。",
        ],
    }
    scene_common = {
        "deep_breakdown": [
            "場景不是背景板，要給人物行動造成阻力或誘惑。",
            "先定空間權力：誰站高、誰靠近門、誰被光照見、誰留在陰影裡。",
            "讓聲音、溫度、氣味、光線服務於當下情緒。",
        ],
        "detail_lenses": ["光源", "門窗位置", "聲音遠近", "溫度觸感", "人物站位"],
        "micro_techniques": [
            "遠景定位後立刻切一個能推動情緒的近景。",
            "用場景變化標記局勢變化，例如聲音忽然停下。",
        ],
        "common_mistakes": ["環境描寫太長但不影響劇情。", "所有場景都只剩氣氛詞。", "人物和空間沒有互動。"],
        "practice_prompts": ["用 180 字寫一間大殿，讓讀者感到有人即將被審判。"],
        "anatomy_breakdown": [
            "定位：一句話給空間規模與權力結構（誰高、誰近門、誰在光裡）。",
            "主感官：選一個能定調的感官細節（燭火、寒氣、迴響）放大。",
            "近景切換：立刻切到一個能推動情緒的近物（搖曳的影、半掩的門）。",
            "互動：讓人物與空間發生關係（被光照見、被陰影吞沒、被門擋住）。",
        ],
        "sentence_rhythm": [
            "定位句可長，用來鋪空間；轉到情緒時改用短句。",
            "用一個單句景物切換標記局勢轉變（『燭火忽然矮下去。』）。",
            "段落密度隨緊張度遞增：越接近衝突，句子越短。",
        ],
        "word_palette": [
            "用能投射情緒的景物名詞：燭、瓦、簾、影、階、風。",
            "光與溫度的動詞化：壓、漫、吞、滲、晃。",
            "避免成串氣氛形容詞（陰森森、靜悄悄）疊用，挑一個最準的。",
        ],
        "sensory_layering": [
            "視覺定空間（光源、站位），但別寫成佈景清單。",
            "聽覺管節奏（遠近、忽然的靜），溫度管壓迫感。",
            "讓最後一個感官細節直接連到人物的處境。",
        ],
        "weak_vs_strong": [
            "弱：大殿非常莊嚴肅穆，氣氛十分緊張。",
            "強：燭火順著穿堂風一起矮下去，殿上眾人的影子同時往御座方向倒。",
        ],
    }
    dialogue_common = {
        "deep_breakdown": [
            "每句對話都要有表面話題與真正目的。",
            "把沒說出口的內容交給停頓、稱謂、視線和手部動作。",
            "對話結束時局勢必須改變：資訊、關係、威脅或選擇至少變一項。",
        ],
        "detail_lenses": ["稱謂", "反問", "沉默", "打斷", "語速", "看向哪裡"],
        "micro_techniques": [
            "先讓角色避開核心問題，再用一個動作洩露真正反應。",
            "用短句壓迫，用長句遮掩。",
        ],
        "common_mistakes": ["對話只交代資訊，沒有攻防。", "角色都用同一種語氣。", "缺少停頓和未回答。"],
        "practice_prompts": ["寫一段兩人談茶，實際上在談背叛。"],
        "anatomy_breakdown": [
            "表面話題：兩人嘴上在談什麼（茶、天氣、舊事）。",
            "真正目的：底下在爭什麼（試探、施壓、求饒、宣戰）。",
            "潛台詞載體：用稱謂變化、反問、停頓、看向哪裡來傳遞沒說的。",
            "局勢位移：一段對話結束，至少有一項（資訊/關係/威脅）改變。",
        ],
        "sentence_rhythm": [
            "攻擊用短句，閃避用長句或岔開話題的句子。",
            "在關鍵問句後留一個動作節拍（他放下杯）再回答，製造壓力。",
            "用未說完的半句和省略號承載最重的潛台詞。",
        ],
        "word_palette": [
            "靠稱謂和敬語的細微變化標記關係溫度（『你』與『大人』之間）。",
            "動作標籤取代情緒形容（他笑了笑 → 他指尖在杯沿頓了一下）。",
            "避免用『冷冷地』『譏諷地』直接標註語氣，讓台詞自己冷。",
        ],
        "sensory_layering": [
            "聽覺管語速與停頓；視覺管視線落點與手部小動作。",
            "讓一個共用物件（茶、信、棋）成為兩人之間的攻防焦點。",
            "沉默本身是一層——寫周遭的聲音填補空白。",
        ],
        "weak_vs_strong": [
            "弱：『你到底是不是想背叛我？』她憤怒地質問。",
            "強：『再添些茶吧。』她沒抬頭，『冷了，就不好喝了。』",
        ],
    }
    rhythm_common = {
        "deep_breakdown": [
            "先用異常或缺口抓住讀者，再延遲解答。",
            "資訊釋放要一口給判斷、一口留疑問。",
            "章尾要讓當前目標被打斷，或讓新代價突然出現。",
        ],
        "detail_lenses": ["異常", "缺口", "延遲", "反轉", "新代價", "未完成動作"],
        "micro_techniques": [
            "開頭三句內給出異常動作或不該出現的人事物。",
            "章尾用一句短句或一個新動作切斷安全感。",
        ],
        "common_mistakes": ["開頭先解釋設定。", "章尾只是自然停下，沒有下一章推力。", "伏筆給得太直白。"],
        "practice_prompts": ["把平淡章尾改成有人敲門、信物碎裂或一句反常稱呼。"],
        "anatomy_breakdown": [
            "鉤子：開頭三句內放一個異常（不該出現的人、反常的動作、被打斷的日常）。",
            "延遲：先給局部判斷，把完整解答往後押。",
            "升溫：每段釋放一點資訊，同時製造一個新疑問。",
            "切斷：章尾在最不安全的一拍硬停，把安全感留給下一章。",
        ],
        "sentence_rhythm": [
            "鉤子句要短、要具體，避免先鋪設定。",
            "章尾最後一句用最短的句子收，製造懸停。",
            "資訊段用中句，懸念點用斷句。",
        ],
        "word_palette": [
            "用具體動作收尾而非情緒總結（門開了 / 信碎了 / 她改了稱呼）。",
            "避免『欲知後事如何』式的廉價懸念。",
            "用反常的小詞標記不對勁（『又是』『不該』『竟然』）。",
        ],
        "sensory_layering": [
            "章尾常靠單一聲音或單一動作切斷（敲門聲、瓷裂聲）。",
            "把多餘的環境描寫砍掉，只留製造懸念的那一個細節。",
            "讓未完成的動作懸在半空，視覺定格。",
        ],
        "weak_vs_strong": [
            "弱：這一夜註定不平靜，一場風暴即將來臨。",
            "強：她正要吹熄燭火，門外傳來第二聲叩響——而她今夜誰也沒約。",
        ],
    }
    base = {
        "容顏描寫": body_common,
        "身材描寫": body_common,
        "動作描寫": action_common,
        "情緒描寫": emotion_common,
        "場景氛圍": scene_common,
        "對話與關係": dialogue_common,
        "節奏章法": rhythm_common,
    }.get(category, action_common)
    profile = {key: list(value) for key, value in base.items()}

    for tokens, override in DEEP_SUBCATEGORY_PROFILES:
        if any(token in subcategory for token in tokens):
            for key, value in override.items():
                profile[key] = list(value)
            break

    if formula_steps:
        profile["deep_breakdown"].append("本卡公式：" + " -> ".join(formula_steps))
    return profile


# Per-subcategory deep overrides. Each entry is (match_tokens, partial_profile).
# These teach, at sentence level, "how this exact body part / action is written" —
# the granularity the writing AGENT needs, not a generic checklist. The first
# matching token group wins, so order from most specific to least.
DEEP_SUBCATEGORY_PROFILES: list[tuple[tuple[str, ...], dict[str, list[str]]]] = [
    (
        ("眼睛", "眉眼", "眼神", "目光"),
        {
            "anatomy_breakdown": [
                "落點：眼睛先看向哪裡——杯沿、門縫、對方手腕、地面。落點本身要洩露情緒或意圖。",
                "動作：抬眼／垂眸／斜睨／避開／停住，挑一個，不要連用三個。",
                "停留時長：用『半息』『一瞬』『久久』量化注視，時長即態度。",
                "細節：只給一個眼部濕度或光的細節（眼尾泛紅、瞳孔縮了一下），絕不堆兩個。",
                "回應：讓被看的人誤讀或被迫反應，眼神就從描寫變成了行動。",
            ],
            "detail_lenses": ["視線落點", "眼神停留時長", "眨眼/瞳孔變化", "眼周的光與濕度", "被看者的反應"],
            "micro_techniques": [
                "視線先找落點，落點本身要暗示情緒，再寫眼神怎麼變。",
                "用注視的『時長』寫態度：停半息是試探，久久不移是宣告。",
                "讓眼神成為動作鏈的起點：他看過去 → 對方被迫先開口。",
            ],
            "sentence_rhythm": [
                "落點句短，動作句更短；注視的停留用一個句號懸住。",
                "避免在同一句塞進『抬眼、瞇眼、又垂下』三個連續動作。",
            ],
            "word_palette": [
                "動詞：抬、垂、斜、掃、定、瞥、瞇、移。",
                "光濕細節挑一個：泛紅、發亮、一縮、蒙了層水汽。",
                "避免：美麗動人、水汪汪、深邃迷人。",
            ],
            "sensory_layering": ["視覺主導（落點＋一個光濕細節），其餘交給對方反應，不必動用其他感官。"],
            "weak_vs_strong": [
                "弱：他有一雙深邃迷人的眼睛，正深情地望著她。",
                "強：他的目光落在她攥緊的手上，停了一瞬，才慢慢抬到她臉上。",
            ],
        },
    ),
    (
        ("嘴巴", "唇", "嘴角"),
        {
            "anatomy_breakdown": [
                "嘴形先壓住話：抿住／微啟／咬字變慢／唇角收回，動作即潛台詞。",
                "矛盾：讓嘴角與語氣相反——笑意在唇角，聲音卻沒有溫度。",
                "未說：用吞回去的半句或嚥下的稱呼製造關係張力。",
                "破綻：最後讓一個唇部小動作洩露真實情緒（咬了下唇、唇線一緊）。",
            ],
            "detail_lenses": ["唇角方向", "咬字節奏", "話前的停頓", "吞回去的話", "笑意真假"],
            "micro_techniques": [
                "先寫唇部如何壓住話語，再讓它和語氣製造矛盾。",
                "把最重的情緒放在『沒說出口的半句』上。",
            ],
            "sentence_rhythm": ["話前插一個唇部動作節拍再開口；半句用破折號或省略號收。"],
            "word_palette": [
                "動詞：抿、咬、扯、勾、抽、嚥。",
                "避免：嫣然一笑、櫻桃小嘴、冷笑一聲（太套路）。",
            ],
            "sensory_layering": ["視覺看唇形，聽覺聽咬字與停頓，兩者夠了。"],
            "weak_vs_strong": [
                "弱：她冷笑一聲，譏諷地說道。",
                "強：她唇角往上勾了勾，那點笑卻沒到聲音裡：『原來如此。』",
            ],
        },
    ),
    (
        ("臉部輪廓", "臉部", "輪廓", "下頜", "顴"),
        {
            "anatomy_breakdown": [
                "用光勾輪廓：讓光從某個角度打來，只描被照亮的那條線（顴、下頜、鼻樑）。",
                "用陰影藏情緒：另一半臉留在暗處，未明的情緒交給陰影。",
                "用一個肌肉動作破靜態：下頜一緊、太陽穴跳了一下。",
            ],
            "detail_lenses": ["光從哪來", "被照亮的那條線", "陰影遮住什麼", "一個細微的肌肉收緊"],
            "micro_techniques": ["不要逐項描述五官，用光影選一條輪廓線，把其餘留白。"],
            "sentence_rhythm": ["輪廓句可稍長以鋪光影，破靜態的肌肉動作用一個短句切入。"],
            "word_palette": ["動詞：勾、削、繃、跳、沉。避免：稜角分明、英俊／美麗。"],
            "sensory_layering": ["純視覺，靠光影調度即可，別加多餘感官。"],
            "weak_vs_strong": [
                "弱：他有一張稜角分明、十分英俊的臉。",
                "強：燭光只照亮他半邊下頜，那條線繃得很緊，另外半張臉沉在暗裡。",
            ],
        },
    ),
    (
        ("身形", "身材", "腰身", "姿態", "肩背", "脊"),
        {
            "anatomy_breakdown": [
                "用衣物與動作勾身形，不直接量化三圍：束腰的勒痕、垂落的衣袂、脊背的線條。",
                "姿態即心理：挺直是抗拒，微傾是試探，塌下是認輸。",
                "讓身體與空間互動：靠著柱、抵著案、立在階前，位置即權力。",
                "用一個重心轉移寫變化（重心後撤＝戒備，前傾＝進逼）。",
            ],
            "detail_lenses": ["衣料如何貼或垂", "脊背/肩線的角度", "重心在前還是後", "身體靠著什麼"],
            "micro_techniques": ["用服飾的貼合與垂墜暗示身形，用姿態角度寫態度。"],
            "sentence_rhythm": ["姿態定調句中長，重心轉移用一個短句標記。"],
            "word_palette": ["動詞：挺、傾、塌、撐、抵、繃。避免：前凸後翹、玲瓏有致、魔鬼身材。"],
            "sensory_layering": ["視覺為主（線條＋姿態），觸覺只在衣料貼身或倚靠時補一筆。"],
            "weak_vs_strong": [
                "弱：她身材玲瓏，腰肢纖細，十分動人。",
                "強：束腰的襦裙勒出一道緊繃的線，她沒坐下，只把指尖抵在案沿，脊背挺得筆直。",
            ],
        },
    ),
    (
        ("手", "指", "腕", "掌"),
        {
            "anatomy_breakdown": [
                "定位：手此刻在哪——袖中、杯邊、劍柄、門框、對方腕上。位置即意圖。",
                "力度：輕扣／攥緊／鬆開／按住／指節發白，力度即情緒強度。",
                "接觸物：讓手與一個物件互動，物件承載觸覺與聲音。",
                "時機：握緊或鬆開的『瞬間』要和對話或事件的轉折對齊。",
            ],
            "detail_lenses": ["手的位置", "力度大小", "接觸的物件", "指節/掌心狀態", "握緊或鬆開的時機"],
            "micro_techniques": [
                "先定位手在哪，再寫力度，最後讓它洩露人物真正的選擇。",
                "用指節發白、青筋、顫抖把壓抑的情緒落到手上。",
            ],
            "sentence_rhythm": ["手部動作多用短句；力度到頂點時用一個句號定格。"],
            "word_palette": ["動詞：扣、攥、按、扣、捏、鬆、抵、抽。避免：纖纖玉手、修長好看。"],
            "sensory_layering": ["觸覺與視覺並重：看見位置與發白，感到力度與溫度。"],
            "weak_vs_strong": [
                "弱：她緊張地握緊了拳頭。",
                "強：她垂在袖中的手攥住了帕子，指節一點點發白，臉上卻還掛著笑。",
            ],
        },
    ),
    (
        ("走路", "步伐", "腳步"),
        {
            "anatomy_breakdown": [
                "速度與聲音：腳步的快慢、輕重、是否有聲，本身就是情緒與身份。",
                "方向與目的：朝誰走、繞開誰、在誰面前停。",
                "停頓：在關鍵的一步停住，比走完更有戲。",
                "他人感知：讓周遭根據腳步聲先一步反應（眾人噤聲、回頭）。",
            ],
            "detail_lenses": ["步速", "腳步聲輕重", "行進方向", "在哪裡停住", "旁人因腳步的反應"],
            "micro_techniques": ["用腳步聲當入場的節拍器，讓聲音先到、人後到。"],
            "sentence_rhythm": ["行進用帶節奏的中句，停步用一個短句硬斷。"],
            "word_palette": ["動詞：踏、頓、踱、繞、停、退。用『一步』『半步』量化距離。"],
            "sensory_layering": ["聽覺領頭（腳步聲），視覺補方向與停點。"],
            "weak_vs_strong": [
                "弱：他緩緩地走了進來，氣勢逼人。",
                "強：靴聲在空殿裡一下一下地響，眾人說話的聲音隨之低了下去——他走到一半，停住了。",
            ],
        },
    ),
    (
        ("轉身", "回眸", "回頭"),
        {
            "anatomy_breakdown": [
                "轉身的『起點』：先動的是肩、是腰、還是只是頭。",
                "速度差：猛回頭是失態，緩緩轉身是從容，半轉是欲走還留。",
                "回眸定格：轉到一半時視線與對方相接的那一瞬最關鍵。",
                "未完成：讓動作停在半轉，把後續交給懸念。",
            ],
            "detail_lenses": ["先動的部位", "轉身速度", "視線相接的瞬間", "衣袂/髮的餘勢"],
            "micro_techniques": ["把戲放在『轉到一半』的那一拍，而非轉完之後。"],
            "sentence_rhythm": ["用一個逗號切出轉身的兩段，回眸那一瞬單獨成句。"],
            "word_palette": ["動詞：轉、側、回、頓、掠。避免：嫣然回首、風情萬種。"],
            "sensory_layering": ["視覺為主，補一筆衣料或髮絲的餘勢（動勢）。"],
            "weak_vs_strong": [
                "弱：她嫣然回首，對他笑了一笑。",
                "強：她已經轉過身去，腳步卻在門檻前頓住，肩微微側了側，終究沒有回頭。",
            ],
        },
    ),
    (
        ("喝酒", "飲酒", "飲茶", "舉杯", "斟"),
        {
            "anatomy_breakdown": [
                "端杯前先給目的：拖延、試探、遮掩、壓住怒意——這杯為何在此刻被舉起。",
                "器物聲做節拍：杯底落案、酒液入盞、指尖碰瓷，每一聲是一拍。",
                "入口不只是『喝』：寫停頓、喉結滾動、餘味、舌尖的反應。",
                "放杯：杯子放回的位置與輕重，是這一回合的句點。",
                "話語轉向：讓對方根據這一杯做出判斷，喝酒就成了攻防。",
            ],
            "detail_lenses": ["端杯前的目的", "杯盞的聲音", "入口的停頓", "喉結/餘味", "放杯的位置與輕重", "之後的話語轉向"],
            "micro_techniques": [
                "用器物聲當節奏點，把對話的試探壓在喝酒的節拍裡。",
                "入口處插一個停頓，讓讀者以為他要說話，結果他只是嚥下。",
                "放杯的輕重決定這一回合誰贏。",
            ],
            "sentence_rhythm": [
                "倒、舉、抿、放，用短句一拍一拍推進，像慢動作。",
                "入口的停頓用一個句號懸住，再接話語轉向。",
            ],
            "word_palette": [
                "動詞：斟、舉、抿、啜、嚥、頓、擱、推。",
                "器物與聲音名詞：盞、壺、杯底、瓷、案。",
                "避免：一飲而盡、開懷暢飲（除非要寫的就是失態）。",
            ],
            "sensory_layering": [
                "聽覺領頭（瓷與酒的聲音），體感補喉間與溫度，視覺收在放杯位置。",
                "味覺只點一筆餘味即可，不要寫成品酒筆記。",
            ],
            "weak_vs_strong": [
                "弱：他端起酒杯一飲而盡，然後說道。",
                "強：他指尖在杯沿頓了頓，才抿了一口，慢慢嚥下。酒盞擱回案上，發出極輕的一聲。『所以呢？』",
            ],
        },
    ),
    (
        ("拔劍", "出招", "出鞘", "劍"),
        {
            "anatomy_breakdown": [
                "拔劍前先寫距離與殺意：讀者要明白為何此刻非拔不可。",
                "出鞘瞬間只抓一兩個最鋒利的細節：聲音、冷光、衣袖、地面。",
                "起手的停頓：半寸出鞘又停住，比一氣呵成更有壓力。",
                "招式後立刻寫代價或局勢：見血、退步、破綻、對方的臉，不讓動作懸空。",
            ],
            "detail_lenses": ["雙方距離", "殺意/動機", "出鞘聲與冷光", "起手的停頓", "受力後的後果", "下一個破綻"],
            "micro_techniques": [
                "把張力放在『要拔未拔』的那一拍。",
                "出招只抓最鋒利的一兩個細節，不要逐招描述。",
            ],
            "sentence_rhythm": [
                "蓄勢用一個中句交代距離殺意，出招用斷句加速。",
                "停頓那一拍單獨成句，後果用一句讓重量落地。",
            ],
            "word_palette": [
                "動詞：抵、扣、撤、挑、劈、刺、收、格。",
                "用冷光、寒、鳴、裂等通感詞寫鋒利。",
                "避免：眼花繚亂、勢如破竹這類空泛總結。",
            ],
            "sensory_layering": ["聽覺（劍鳴）與視覺（冷光）領頭，觸覺補受力，收在後果上。"],
            "weak_vs_strong": [
                "弱：他迅速拔劍，使出凌厲的一招，將對方擊退。",
                "強：他拇指抵上劍格。半寸寒光出鞘，又生生停住——對面那人喉結一動，先退了半步。",
            ],
        },
    ),
    (
        ("戰鬥", "閃避", "纏鬥", "交手"),
        {
            "anatomy_breakdown": [
                "選一條主線：不寫每一招，只追一條最關鍵的攻防線。",
                "用身體承重寫真實：呼吸、汗、舊傷、發顫的腿。",
                "節奏起伏：快—慢—快，在最危險處放一個慢鏡停頓。",
                "代價遞增：每一回合都讓某人更接近輸，危險可量化。",
            ],
            "detail_lenses": ["主攻防線", "身體的負荷", "快慢節奏點", "每回合的代價"],
            "micro_techniques": ["用一個慢鏡停頓切開連續打鬥，讓讀者喘口氣再加速。"],
            "sentence_rhythm": ["連擊用短句疊加，危機點用一個長句的慢鏡拉開。"],
            "word_palette": ["動詞要狠且準：撞、絞、卸、欺、墊、崩。避免招式名堆砌。"],
            "sensory_layering": ["體感領頭（重量、痛、喘），聽覺補撞擊，視覺定關鍵一擊。"],
            "weak_vs_strong": [
                "弱：兩人你來我往，打得難分難解，十分激烈。",
                "強：他卸開第三劍時，左膝那道舊傷終於不聽使喚。就這一瞬——對方的刃已經貼上了他的頸側。",
            ],
        },
    ),
    (
        ("觸碰", "拉近", "接觸", "牽", "扶"),
        {
            "anatomy_breakdown": [
                "接觸前的距離：先把兩人之間的空氣寫緊，碰才有重量。",
                "接觸點要具體：指尖、手腕、衣袖、髮梢——碰哪裡決定了曖昧的層級。",
                "停留時長：碰一下就收是試探，停住不放是宣告。",
                "餘溫：分開後留一個體感餘韻（腕上的溫度、未散的氣息）。",
            ],
            "detail_lenses": ["接觸前的距離", "接觸的部位", "力度與停留", "分開後的餘溫"],
            "micro_techniques": ["把曖昧放在『接觸前』與『分開後』，而非接觸本身。"],
            "sentence_rhythm": ["接近用放慢的長句，接觸的瞬間用一個極短句定格。"],
            "word_palette": ["動詞：擦、觸、扣、攥、扶、收。避免：電流般、火熱、悸動。"],
            "sensory_layering": ["觸覺與體感領頭（溫度、力度），視覺只給接觸點一個特寫。"],
            "weak_vs_strong": [
                "弱：他握住她的手，兩人都感到一陣悸動。",
                "強：他的指尖先碰到她的腕，又像被燙到似的收回去半寸，終究還是覆了上來。",
            ],
        },
    ),
    (
        ("坐", "臥", "起身", "倚"),
        {
            "anatomy_breakdown": [
                "坐臥起身是權力與心理的姿態：先坐／站著不坐／起身離席，各有意味。",
                "用過程寫態度：怎麼坐下（緩、沉、跌）、怎麼起身（驟、撐、踉蹌）。",
                "與家具/空間互動：靠著、撐著、佔據主位，位置即地位。",
                "讓一個起身或落座的動作標記對話的轉折。",
            ],
            "detail_lenses": ["坐還是不坐", "落座/起身的速度與重量", "佔據的位置", "與家具的互動"],
            "micro_techniques": ["用『不坐』或『起身』這種姿態變化標記局勢翻轉。"],
            "sentence_rhythm": ["姿態過程用中句，轉折的起身用一個短句切。"],
            "word_palette": ["動詞：落、沉、撐、倚、起、踱。避免：優雅地坐下這類空評。"],
            "sensory_layering": ["視覺為主（姿態與位置），體感補重量（沉沉坐下／猛地起身）。"],
            "weak_vs_strong": [
                "弱：他優雅地在椅子上坐了下來。",
                "強：他沒有坐主位旁的客席，反而徑直在主位上落了座，沉沉地，像本就該屬於那裡。",
            ],
        },
    ),
    (
        ("壓抑", "隱忍", "克制"),
        {
            "anatomy_breakdown": [
                "先寫『壓』：身體做出克制動作（屏息、攥手、放緩語速、別開臉）。",
                "再寫『漏』：一個壓不住的小破綻洩露真相（聲音一抖、眼尾紅、指節白）。",
                "對比張力：表面越平靜，破綻越要小而精準。",
                "推進：讓壓抑最終逼出一個選擇或一句話，不讓它空轉。",
            ],
            "detail_lenses": ["克制的身體動作", "壓不住的那個破綻", "表面與內裡的落差", "最後被逼出的選擇"],
            "micro_techniques": ["九分克制配一分破綻，破綻要小到只有讀者看見。"],
            "sentence_rhythm": ["克制段句子短而平，破綻處讓一個句子忽然斷掉或用破折號——"],
            "word_palette": ["用『沒有』『仍』『只是』寫克制；避免直接說憤怒、悲傷、心痛。"],
            "sensory_layering": ["體感領頭（喉、胸、指尖的緊與冷），聽覺補自己聲音的變化。"],
            "weak_vs_strong": [
                "弱：她強忍著悲傷，內心痛苦極了。",
                "強：她應了聲『好』，聲音很穩。只是去接茶盞的手，懸在半空停了一下，才穩穩地接過去。",
            ],
        },
    ),
    (
        ("憤怒", "暴怒", "盛怒"),
        {
            "anatomy_breakdown": [
                "怒的層級先定：是冷怒、壓怒，還是爆發。層級決定寫法。",
                "冷怒寫『靜』：聲音變低、動作變慢、用詞變客氣，反而更可怕。",
                "爆發寫『一個破壞性動作』：摔、拍、攥碎，但只給一個。",
                "餘波：怒後的安靜與他人的噤聲，比怒本身更有壓力。",
            ],
            "detail_lenses": ["怒的層級", "聲音與語速的變化", "一個外化的動作", "周遭的反應"],
            "micro_techniques": ["高位者的怒往往越壓越低；用客氣與緩慢寫殺機。"],
            "sentence_rhythm": ["冷怒用短而慢的句子；爆發用一個動作短句，之後接一片安靜。"],
            "word_palette": ["動詞：擲、拍、捏、碾、頓。避免：怒火中燒、暴跳如雷。"],
            "sensory_layering": ["聽覺管聲音的低與靜，視覺定那一個破壞性動作。"],
            "weak_vs_strong": [
                "弱：皇帝勃然大怒，怒火中燒。",
                "強：皇帝沒有抬高聲音。他只是把那份奏摺輕輕擱下，指腹在『謀逆』兩個字上慢慢按了按。殿內，沒有人敢喘氣。",
            ],
        },
    ),
    (
        ("恐懼", "危機", "驚恐"),
        {
            "anatomy_breakdown": [
                "先給威脅的『線索』而非威脅本身：不對的聲音、不該開的門、過分的安靜。",
                "身體先於意識反應：汗、屏息、發冷、後退半步。",
                "用感知收窄寫恐懼：聽覺放大、視野只剩一點、時間變慢。",
                "讓恐懼逼出一個倉促的選擇或失誤。",
            ],
            "detail_lenses": ["威脅的線索", "身體的本能反應", "被放大的單一感官", "恐懼逼出的行動"],
            "micro_techniques": ["寫恐懼靠『感知收窄』：把世界縮小到一個聲音、一道縫。"],
            "sentence_rhythm": ["危機逼近時句子越來越短，最後用一個極短句引爆。"],
            "word_palette": ["動詞：縮、僵、退、攥、屏。避免：嚇得魂飛魄散這類成語化。"],
            "sensory_layering": ["聽覺領頭（放大的某個聲音），體感補冷與汗，視覺收窄成一點。"],
            "weak_vs_strong": [
                "弱：她嚇得魂飛魄散，渾身發抖。",
                "強：殿裡太靜了。靜到她能聽見自己的心跳，和身後——那一聲本不該存在的、極輕的腳步。",
            ],
        },
    ),
    (
        ("夜色", "燭火", "月色", "燈"),
        {
            "anatomy_breakdown": [
                "選一個主光源（燭、月、燈）當情緒的調光器。",
                "寫光的『動』而非『有』：搖、矮、漫、明滅，光動則氣氛動。",
                "用光影分配權力與隱秘：誰在光裡受審，誰在暗處旁觀。",
                "讓光的一次變化標記局勢或心理的轉折。",
            ],
            "detail_lenses": ["主光源", "光的動態", "光照亮誰/陰影藏誰", "光變化標記的轉折"],
            "micro_techniques": ["用一次『燭火忽然矮下去』式的光變，標記不安或來人。"],
            "sentence_rhythm": ["鋪光影可用中長句，光的突變用一個短句切。"],
            "word_palette": ["動詞：搖、矮、漫、晃、滅、跳。避免：燈火通明、月色皎潔的靜態套語。"],
            "sensory_layering": ["視覺主導（光與影），補一筆溫度（燭火的暖／夜的寒）。"],
            "weak_vs_strong": [
                "弱：夜色很美，燭火通明，氣氛安詳。",
                "強：穿堂風過，滿殿燭火一起矮了矮，影子在牆上晃了一下，又重新站直。",
            ],
        },
    ),
    (
        ("章尾", "轉折", "鉤子", "開場"),
        {
            "anatomy_breakdown": [
                "找到最不安全的一拍：在角色以為要結束、要安全的那一刻。",
                "用一個具體動作或聲音切斷：門響、信碎、一句反常的稱呼。",
                "不解釋：把完整含義留給下一章，只給讀者一個無法忽視的異常。",
                "最後一句要短：句子越短，懸停越強。",
            ],
            "detail_lenses": ["切斷的時機", "切斷用的具體動作/聲音", "留下的疑問", "最後一句的長度"],
            "micro_techniques": ["章尾不要總結，要『打斷』；用動作收尾而非情緒。"],
            "sentence_rhythm": ["最後一句用最短的句子收，製造戛然而止。"],
            "word_palette": ["用『又』『竟』『不該』標記不對勁；避免『欲知後事如何』。"],
            "sensory_layering": ["靠單一聲音或單一動作切斷，砍掉其餘環境描寫。"],
            "weak_vs_strong": [
                "弱：這一夜，註定是個不眠之夜，一切才剛剛開始。",
                "強：她正要吹熄那盞燈，門外傳來第二聲叩響。今夜，她誰也沒有約。",
            ],
        },
    ),
]


def ask_grok_for_book_entries(
    *,
    client: OpenAI,
    model_name: str,
    docs: list[SourceDocument],
    goal: str,
    output_language: str,
    max_entries_per_subcategory: int,
) -> list[TechniqueBookEntry]:
    source_pack = build_source_pack(docs)
    taxonomy_text = render_taxonomy_for_prompt()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是頂尖的小說寫作技法解剖師（craft anatomist），不是泛泛的讀後感寫手。"
                    "你的任務是把多份小說、full_report、Technique Finder 報告，蒸餾成階層式、可搜尋、"
                    "可直接拿去寫作的『深度技法卡』。\n"
                    "核心要求：每張卡要回答『這個部位／這個動作，到底是一句一句怎麼寫出來的』——"
                    "要拆到可觀察的微單元、句子的長短與標點、用哪些動詞與意象、感官的先後順序、"
                    "以及弱寫與強寫的差別。\n"
                    "嚴禁：抄錄原文句子、續寫原作情節、保留角色姓名、輸出敏感或未成年不當內容。"
                    "示例句必須是你自己寫的、通用的短句級示範（弱／強各一句），不得直接複製來源。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n"
                    f"Output language: {output_language}\n"
                    f"Max entries per subcategory: {max_entries_per_subcategory}\n\n"
                    "請只輸出 JSON array（不要 Markdown、不要解說）。每個 item 必須包含這些鍵：\n"
                    "category, subcategory, title, aliases, trigger_keywords, summary, "
                    "formula_steps, applicable_scenes, director_instruction, source_labels, evidence_count, evidence_signals, "
                    "deep_breakdown, detail_lenses, micro_techniques, common_mistakes, practice_prompts, "
                    "anatomy_breakdown, sentence_rhythm, word_palette, sensory_layering, weak_vs_strong。\n\n"
                    "深度規格（每張卡都必須達到，不可用一兩句空泛帶過）：\n"
                    "- anatomy_breakdown：4-6 條。把這個部位或動作拆成『可觀察的微單元』並排出書寫順序。"
                    "例如寫眼睛＝視線落點→眼神變化→停留時長→一個光濕細節→對方反應；"
                    "例如寫喝酒＝端杯目的→器物聲→入口停頓→喉結餘味→放杯→話語轉向。要具體到能照著寫。\n"
                    "- sentence_rhythm：3-5 條。講句子的長短安排、停頓與標點、段落密度。"
                    "例如『特寫句要短、逗號不超過兩個』『關鍵變化用句號硬斷製造停頓』。\n"
                    "- word_palette：3-5 條。列出該技法適合的具體動詞／名詞／質感詞，以及『該避免的抽象詞』"
                    "（如：避免美麗、優雅、迅速、憤怒等直接判斷詞，改用可觀察細節）。\n"
                    "- sensory_layering：2-4 條。指明哪個感官領頭、其餘感官的先後與份量、各放幾個細節"
                    "（強調節制：通常一個主感官＋一個輔助，不要堆五感）。\n"
                    "- weak_vs_strong：剛好 2 條，第一條以『弱：』開頭（常見的平庸寫法），"
                    "第二條以『強：』開頭（運用本技法後的短句級改寫）。兩句都要你自己寫，短、具體、不抄原文。\n"
                    "- deep_breakdown：4-6 條，說明技法為何有效、如何驅動情緒/關係/局勢。\n"
                    "- detail_lenses：列出觀察鏡頭（要盯著看的點）。\n"
                    "- micro_techniques：像教學步驟一樣拆出可操作的微動作。\n"
                    "- common_mistakes：指出 3 條以上常見失誤。\n"
                    "- practice_prompts：能直接拿去練習的題目。\n"
                    "- director_instruction：一句話，可直接貼進寫作 AGENT 當本技法的導演指令。\n\n"
                    "一張合格卡片的範例（節錄，供你對齊深度，不要照抄內容）：\n"
                    "{\n"
                    '  "category": "動作描寫", "subcategory": "喝酒飲茶描寫", "title": "把一杯酒寫成攻防",\n'
                    '  "anatomy_breakdown": ["端杯前先給目的：拖延或試探", "器物聲做節拍：杯底落案、酒液入盞", '
                    '"入口寫停頓與喉結，而非『喝下去』", "放杯的輕重是這一回合的句點", "讓對方據此判斷，喝酒成攻防"],\n'
                    '  "sentence_rhythm": ["倒、舉、抿、放用短句一拍一拍推進", "入口停頓用一個句號懸住再接話"],\n'
                    '  "word_palette": ["動詞：斟、舉、抿、嚥、頓、擱", "避免：一飲而盡、開懷暢飲"],\n'
                    '  "sensory_layering": ["聽覺領頭（瓷與酒聲），體感補喉間，味覺只點一筆餘味"],\n'
                    '  "weak_vs_strong": ["弱：他端起酒杯一飲而盡。", '
                    '"強：他指尖在杯沿頓了頓，才抿一口，慢慢嚥下。酒盞擱回案上，極輕一聲。"]\n'
                    "}\n\n"
                    "分類優先使用下列 taxonomy（已附每個小分類的定位、關鍵詞與公式骨架，請據此精準歸類並深化）；"
                    "若來源出現明確的新技法，可新增小分類：\n"
                    f"{taxonomy_text}\n\n"
                    "來源材料（請從中觀察真實寫法，再抽象成通用技法，不要回貼原句）：\n"
                    f"{source_pack}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=8000,
    )
    raw = (response.choices[0].message.content or "").strip()
    mappings = parse_json_array(raw)
    entries: list[TechniqueBookEntry] = []
    for item in mappings:
        entries.append(entry_from_mapping(item))
    return entries


def build_source_pack(docs: list[SourceDocument]) -> str:
    parts = []
    per_doc = max(8000, min(24000, 90000 // max(len(docs), 1)))
    for index, doc in enumerate(docs[:20], start=1):
        parts.append(f"## Source {index}: {doc.label}\n{trim_text(doc.text, per_doc)}")
    return "\n\n".join(parts)


def render_taxonomy_for_prompt() -> str:
    """Send the model the rich taxonomy (定位 + 關鍵詞 + 公式骨架), not just the names,
    so it has concrete anchors to categorize against and deepen."""
    lines = []
    for category in TAXONOMY:
        lines.append(f"# {category['category']}")
        for sub in category["subcategories"]:
            keywords = "、".join(str(k) for k in sub.get("keywords", [])[:8])
            formula = " -> ".join(str(s) for s in sub.get("formula", []))
            summary = str(sub.get("summary", "")).strip()
            detail = f"  - {sub['name']}：{summary}"
            if formula:
                detail += f"｜骨架：{formula}"
            if keywords:
                detail += f"｜觸發詞：{keywords}"
            lines.append(detail)
    return "\n".join(lines)


def parse_json_array(raw: str) -> list[dict[str, Any]]:
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
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def entry_from_mapping(item: dict[str, Any]) -> TechniqueBookEntry:
    category = clean_short(item.get("category") or "未分類技法")
    subcategory = clean_short(item.get("subcategory") or "一般描寫")
    title = clean_short(item.get("title") or f"{subcategory}技法")
    formula_steps = clean_list(item.get("formula_steps"))
    fallback_deep = build_deep_profile(category, subcategory, formula_steps)
    return TechniqueBookEntry(
        id=stable_entry_id(category, subcategory, title),
        category=category,
        subcategory=subcategory,
        title=title,
        aliases=clean_list(item.get("aliases")),
        trigger_keywords=clean_list(item.get("trigger_keywords")),
        summary=clean_paragraph(item.get("summary")),
        formula_steps=formula_steps,
        applicable_scenes=clean_list(item.get("applicable_scenes")),
        director_instruction=clean_paragraph(item.get("director_instruction")),
        source_labels=clean_list(item.get("source_labels")),
        evidence_count=to_positive_int(item.get("evidence_count")) or 1,
        evidence_signals=clean_list(item.get("evidence_signals"))[:8],
        deep_breakdown=clean_list(item.get("deep_breakdown")) or fallback_deep["deep_breakdown"],
        detail_lenses=clean_list(item.get("detail_lenses")) or fallback_deep["detail_lenses"],
        micro_techniques=clean_list(item.get("micro_techniques")) or fallback_deep["micro_techniques"],
        common_mistakes=clean_list(item.get("common_mistakes")) or fallback_deep["common_mistakes"],
        practice_prompts=clean_list(item.get("practice_prompts")) or fallback_deep["practice_prompts"],
        anatomy_breakdown=clean_list(item.get("anatomy_breakdown")) or fallback_deep["anatomy_breakdown"],
        sentence_rhythm=clean_list(item.get("sentence_rhythm")) or fallback_deep["sentence_rhythm"],
        word_palette=clean_list(item.get("word_palette")) or fallback_deep["word_palette"],
        sensory_layering=clean_list(item.get("sensory_layering")) or fallback_deep["sensory_layering"],
        weak_vs_strong=clean_list(item.get("weak_vs_strong")) or fallback_deep["weak_vs_strong"],
    )


def merge_entries(entries: list[TechniqueBookEntry], max_entries_per_subcategory: int) -> list[TechniqueBookEntry]:
    grouped: dict[tuple[str, str], list[TechniqueBookEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.category, entry.subcategory), []).append(entry)
    merged: list[TechniqueBookEntry] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        seen_titles: set[str] = set()
        for entry in bucket:
            if entry.title in seen_titles:
                continue
            seen_titles.add(entry.title)
            merged.append(entry)
            if len([item for item in merged if (item.category, item.subcategory) == key]) >= max_entries_per_subcategory:
                break
    return merged


def write_book_library(
    *,
    docs: list[SourceDocument],
    entries: list[TechniqueBookEntry],
    goal: str,
    mode: str,
    output_language: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path.cwd() / "book_output" / "integrated_technique_libraries" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "goal": goal,
        "mode": mode,
        "output_language": output_language,
        "sources": [{"label": doc.label, "chars": len(doc.text)} for doc in docs],
        "entries": [asdict(entry) for entry in entries],
    }
    json_path = output_dir / "integrated_technique_book_library.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output_dir / "integrated_technique_book_library.md"
    md_path.write_text(render_book_library_markdown(payload), encoding="utf-8")
    return output_dir


def render_book_library_markdown(payload: dict[str, Any]) -> str:
    entries = entries_from_payload(payload)
    lines = [
        "# Integrated Technique Book Library",
        "",
        f"- Created: {payload.get('created_at', '')}",
        f"- Mode: {payload.get('mode', '')}",
        f"- Goal: {payload.get('goal', '')}",
        f"- Sources: {len(payload.get('sources', []))}",
        f"- Technique Cards: {len(entries)}",
        "",
        "## Search Index",
        "",
    ]
    grouped = group_entries(entries)
    for category, subs in grouped.items():
        lines.append(f"### {category}")
        for subcategory, items in subs.items():
            aliases = "、".join(dedupe(sum((item.aliases for item in items), []))[:8])
            lines.append(f"- {subcategory}: {len(items)} cards; aliases: {aliases}")
        lines.append("")
    lines.append("## Technique Cards")
    lines.append("")
    for category, subs in grouped.items():
        lines.append(f"### {category}")
        lines.append("")
        for subcategory, items in subs.items():
            lines.append(f"#### {subcategory}")
            lines.append("")
            for entry in items:
                lines.extend(render_entry_block(entry))
                lines.append("")
    lines.extend(
        [
            "## Writing AGENT Compact Reference",
            "",
            render_agent_reference(entries[:24], "", "All"),
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_entry_block(entry: TechniqueBookEntry) -> list[str]:
    return [
        f"##### {entry.title}",
        "",
        f"- ID: {entry.id}",
        f"- Trigger Keywords: {'、'.join(entry.trigger_keywords[:12])}",
        f"- Source Coverage: {len(entry.source_labels)} sources / {entry.evidence_count} signals",
        f"- Summary: {entry.summary}",
        f"- Applicable Scenes: {'、'.join(entry.applicable_scenes[:8])}",
        f"- Formula: {' -> '.join(entry.formula_steps)}",
        f"- Director Instruction: {entry.director_instruction}",
        f"- Evidence Signals: {'；'.join(entry.evidence_signals[:5])}",
        "",
        "**Anatomy Breakdown（部位／動作如何一步步寫）**",
        *[f"- {line}" for line in entry.anatomy_breakdown[:6]],
        "",
        "**Sentence Rhythm（句法節奏與標點）**",
        *[f"- {line}" for line in entry.sentence_rhythm[:5]],
        "",
        "**Word Palette（用詞與意象調色盤）**",
        *[f"- {line}" for line in entry.word_palette[:5]],
        "",
        "**Sensory Layering（感官分層順序）**",
        *[f"- {line}" for line in entry.sensory_layering[:4]],
        "",
        "**Weak vs Strong（弱寫 → 強寫對照）**",
        *[f"- {line}" for line in entry.weak_vs_strong[:4]],
        "",
        "**Deep Breakdown**",
        *[f"- {line}" for line in entry.deep_breakdown[:6]],
        "",
        "**Detail Lenses**",
        *[f"- {line}" for line in entry.detail_lenses[:8]],
        "",
        "**Micro Techniques**",
        *[f"- {line}" for line in entry.micro_techniques[:8]],
        "",
        "**Common Mistakes**",
        *[f"- {line}" for line in entry.common_mistakes[:6]],
        "",
        "**Practice Prompts**",
        *[f"- {line}" for line in entry.practice_prompts[:4]],
    ]


def render_search_results(
    entries: list[TechniqueBookEntry],
    query: str,
    category_filter: str,
    payload: dict[str, Any],
) -> str:
    lines = [
        "# Technique Book Search Results",
        "",
        f"- Library: {payload.get('created_at', '')} / {payload.get('mode', '')}",
        f"- Query: {(query or '').strip() or 'All'}",
        f"- Category Filter: {category_filter or 'All'}",
        f"- Results: {len(entries)}",
        "",
    ]
    if not entries:
        lines.append("沒有符合條件的技法卡。")
        return "\n".join(lines)
    current_header = ""
    for entry in entries:
        header = f"{entry.category} / {entry.subcategory}"
        if header != current_header:
            lines.append(f"## {header}")
            lines.append("")
            current_header = header
        lines.extend(render_entry_block(entry))
        lines.append("")
    lines.extend(["## Agent Reference Block", "", render_agent_reference(entries, query, category_filter)])
    return "\n".join(lines).strip()


def render_agent_reference(entries: list[TechniqueBookEntry], query: str, category_filter: str) -> str:
    lines = [
        "# Technique Book Selection",
        f"Query: {(query or '').strip() or 'All'}",
        f"Category: {category_filter or 'All'}",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## {entry.category} / {entry.subcategory}",
                f"- Technique: {entry.summary}",
                f"- Formula: {' -> '.join(entry.formula_steps)}",
                f"- Use when: {'、'.join(entry.applicable_scenes[:6])}",
                f"- Anatomy（如何一步步寫）: {' → '.join(entry.anatomy_breakdown[:5])}",
                f"- Sentence rhythm: {'；'.join(entry.sentence_rhythm[:3])}",
                f"- Word palette: {'；'.join(entry.word_palette[:3])}",
                f"- Sensory order: {'；'.join(entry.sensory_layering[:2])}",
                f"- Example weak→strong: {' || '.join(entry.weak_vs_strong[:2])}",
                f"- Deep breakdown: {'；'.join(entry.deep_breakdown[:3])}",
                f"- Detail lenses: {'、'.join(entry.detail_lenses[:6])}",
                f"- Micro techniques: {'；'.join(entry.micro_techniques[:4])}",
                f"- Avoid: {'；'.join(entry.common_mistakes[:3])}",
                f"- Director Instruction: {entry.director_instruction}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def load_library_payload(library_state_json: str, library_json_path: str) -> dict[str, Any]:
    if (library_state_json or "").strip():
        return json.loads(library_state_json)
    clean_path = (library_json_path or "").strip().strip('"')
    if clean_path:
        path = Path(clean_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise ValueError(f"找不到書庫 JSON：{path}")
        return json.loads(path.read_text(encoding="utf-8"))
    latest = find_latest_book_library_json()
    if latest is None:
        raise ValueError("找不到已建立的 Integrated Technique Book Library。")
    return json.loads(latest.read_text(encoding="utf-8"))


def find_latest_book_library_json() -> Path | None:
    files = [
        path
        for path in Path.cwd().glob("book_output/integrated_technique_libraries/*/integrated_technique_book_library.json")
        if path.is_file()
    ]
    if not files:
        return None
    return max(files, key=lambda item: (item.parent.name, item.stat().st_mtime))


def entries_from_payload(payload: dict[str, Any]) -> list[TechniqueBookEntry]:
    entries = []
    for item in payload.get("entries", []):
        if isinstance(item, dict):
            entries.append(entry_from_mapping(item))
    return entries


def group_entries(entries: list[TechniqueBookEntry]) -> dict[str, dict[str, list[TechniqueBookEntry]]]:
    grouped: dict[str, dict[str, list[TechniqueBookEntry]]] = {}
    for entry in entries:
        grouped.setdefault(entry.category, {}).setdefault(entry.subcategory, []).append(entry)
    return grouped


def rank_entries(entries: list[TechniqueBookEntry], query: str, category_filter: str) -> list[TechniqueBookEntry]:
    category_filter = (category_filter or "All").strip()
    filtered = [entry for entry in entries if category_matches(entry, category_filter)]
    terms = [term for term in re.split(r"[\s,，、/]+", (query or "").strip()) if term]
    if not terms:
        return sorted(filtered, key=lambda entry: (-entry.evidence_count, entry.category, entry.subcategory, entry.title))

    scored: list[tuple[int, TechniqueBookEntry]] = []
    for entry in filtered:
        haystack = " ".join(
            [
                entry.category,
                entry.subcategory,
                entry.title,
                " ".join(entry.aliases),
                " ".join(entry.trigger_keywords),
                entry.summary,
                " ".join(entry.formula_steps),
                " ".join(entry.applicable_scenes),
                entry.director_instruction,
                " ".join(entry.evidence_signals),
                " ".join(entry.deep_breakdown),
                " ".join(entry.detail_lenses),
                " ".join(entry.micro_techniques),
                " ".join(entry.common_mistakes),
                " ".join(entry.practice_prompts),
                " ".join(entry.anatomy_breakdown),
                " ".join(entry.sentence_rhythm),
                " ".join(entry.word_palette),
                " ".join(entry.sensory_layering),
                " ".join(entry.weak_vs_strong),
            ]
        ).lower()
        score = 0
        for term in terms:
            lower = term.lower()
            if lower in entry.subcategory.lower():
                score += 80
            if lower in entry.category.lower():
                score += 55
            if lower in entry.title.lower():
                score += 40
            if lower in haystack:
                score += 20
        if score:
            scored.append((score + min(entry.evidence_count, 50), entry))
    return [entry for _, entry in sorted(scored, key=lambda item: (-item[0], item[1].category, item[1].subcategory))]


def category_matches(entry: TechniqueBookEntry, category_filter: str) -> bool:
    if not category_filter or category_filter == "All":
        return True
    if " / " in category_filter:
        category, subcategory = category_filter.split(" / ", 1)
        return entry.category == category and entry.subcategory == subcategory
    return entry.category == category_filter


def clean_short(value: Any) -> str:
    return clip_text(clean_paragraph(value), 80) or "未命名"


def clean_paragraph(value: Any) -> str:
    text = str(value or "")
    text = clean_source_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = re.split(r"[\n,，、;；]+", value)
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    cleaned = []
    for item in raw:
        text = clean_paragraph(item)
        if text:
            cleaned.append(clip_text(text, 80))
    return dedupe(cleaned)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def stable_entry_id(category: str, subcategory: str, title: str) -> str:
    digest = hashlib.sha1(f"{category}|{subcategory}|{title}".encode("utf-8")).hexdigest()[:8]
    return f"TB-{digest}"


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Auto template watcher: detect which depiction topics the in-progress text is
# touching (eyes / drinking / drawing a sword / suppressed emotion ...) and
# surface the matching "how to write it" technique template for review.
# ---------------------------------------------------------------------------

def detect_active_subcategories(text: str, max_topics: int = 4) -> list[tuple[int, str, dict[str, Any], list[str]]]:
    """Score TAXONOMY subcategories by how strongly their trigger words appear in `text`.

    Returns up to `max_topics` tuples of (hit_count, category_name, subcategory_dict, matched_terms),
    strongest first.
    """
    compact = re.sub(r"\s+", " ", text or "")
    if not compact.strip():
        return []
    scored: list[tuple[int, str, dict[str, Any], list[str]]] = []
    for category in TAXONOMY:
        for sub in category["subcategories"]:
            count = 0
            matched: list[str] = []
            for term in list(sub.get("keywords", [])) + list(sub.get("aliases", [])):
                term = str(term)
                if not term:
                    continue
                occ = compact.count(term)
                if occ:
                    count += occ
                    if term not in matched:
                        matched.append(term)
            if count > 0:
                scored.append((count, category["category"], sub, matched))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]["name"]))
    return scored[: max(int(max_topics or 4), 1)]


def cards_for_subcategory(
    category: str,
    sub: dict[str, Any],
    hit_count: int,
    matched_terms: list[str],
    library_entries: list[TechniqueBookEntry],
    cards_per_topic: int = 1,
) -> list[TechniqueBookEntry]:
    """Prefer real cards from a built library for this subcategory; otherwise synthesize
    a deep card straight from the taxonomy so the watcher always has something to show."""
    matches = [e for e in library_entries if e.category == category and e.subcategory == sub["name"]]
    if not matches:
        matches = [e for e in library_entries if e.subcategory == sub["name"]]
    if matches:
        return matches[: max(int(cards_per_topic or 1), 1)]
    synthesized = make_taxonomy_entry(
        category=category,
        sub=sub,
        source_labels=["taxonomy"],
        evidence_count=hit_count,
        evidence_signals=[f"文中偵測到「{term}」" for term in matched_terms[:5]],
    )
    return [synthesized]


def render_template_watch_block(entry: TechniqueBookEntry, matched_terms: list[str]) -> list[str]:
    """Compact, write-focused view: just the parts a writer needs at the moment of writing."""
    return [
        f"#### 🎯 {entry.category} / {entry.subcategory}",
        f"- 觸發詞：{('、'.join(matched_terms[:8])) or '—'}",
        f"- 一句話：{entry.summary}",
        f"- 怎麼寫（解剖）：{' → '.join(entry.anatomy_breakdown[:5])}",
        f"- 句法節奏：{'；'.join(entry.sentence_rhythm[:2])}",
        f"- 用詞：{'；'.join(entry.word_palette[:2])}",
        f"- 感官：{'；'.join(entry.sensory_layering[:1])}",
        f"- 弱→強：{' ｜ '.join(entry.weak_vs_strong[:2])}",
        f"- 避免：{'；'.join(entry.common_mistakes[:2])}",
        f"- 導演指令：{entry.director_instruction}",
        "",
    ]


def suggest_technique_templates(
    text: str,
    library_state_json: str = "",
    library_json_path: str = "",
    max_topics: int = 4,
    cards_per_topic: int = 1,
) -> tuple[str, list[TechniqueBookEntry]]:
    """Given the in-progress text, return (markdown_review, matched_entries).

    Works even with no built library (synthesizes from the taxonomy), so the watcher
    is useful out of the box and gets richer once a Technique Book is built.
    """
    detected = detect_active_subcategories(text, max_topics)
    if not detected:
        return (
            "尚未偵測到特定描寫主題。\n\n"
            "繼續寫，或在 **Story Instruction** 點名主題（例如：眼睛、喝酒、拔劍、壓抑、章尾鉤子），"
            "系統就會自動跳出「該怎麼寫」的技法模板。",
            [],
        )
    library_entries: list[TechniqueBookEntry] = []
    try:
        payload = load_library_payload(library_state_json, library_json_path)
        library_entries = entries_from_payload(payload)
    except Exception:
        library_entries = []
    source_note = "目前技法書庫" if library_entries else "內建技法 taxonomy（建一次深度技法書庫後會更貼合你的參考小說）"
    topics_line = "、".join(f"{cat}/{sub['name']}({count})" for count, cat, sub, _ in detected)
    lines = [
        "### ✍️ 偵測到的描寫主題 → 該怎麼寫",
        f"- 來源：{source_note}",
        f"- 命中主題：{topics_line}",
        "",
    ]
    chosen: list[TechniqueBookEntry] = []
    for count, category, sub, matched in detected:
        cards = cards_for_subcategory(category, sub, count, matched, library_entries, cards_per_topic)
        for card in cards:
            chosen.append(card)
            lines.extend(render_template_watch_block(card, matched))
    return "\n".join(lines).strip(), chosen
