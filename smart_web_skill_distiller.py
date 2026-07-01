"""Web discovery and gap-aware skill distillation for the Gradio studio.

This module is deliberately compliance-first:
- it only fetches public HTTP(S) pages,
- checks robots.txt before fetching target pages,
- rate-limits per host,
- never bypasses login, paywalls, CAPTCHA, or platform protections,
- stores short evidence snippets and craft abstractions instead of full works.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chapter_craft_skill import (
    API_TIMEOUT,
    chat_complete,
    extract_main_text,
    language_instruction,
    normalize_text,
    to_positive_int,
    trim_preview,
    write_text_with_backup,
)
from story_skill_studio import (
    SKILL_JSON_SPEC,
    _loads_json_dict,
    _normalize_skill,
    render_skill_markdown,
)


logger = logging.getLogger(__name__)

WEB_SOURCE_MODE_CHOICES = [
    "Seed URLs only",
    "Search public web",
    "Search Pixiv via public search",
]

DEFAULT_WEB_SKILL_GOAL = (
    "根據目標小說類型描述或我貼上的片段，從公開網頁/短摘錄中找出我目前缺少的寫作技巧，"
    "抽象成可移植的敘事方式、故事推進原則、劇情發想思路與描寫技法庫。"
    "只保留『怎麼寫』，不要保留來源人物、地名、事件、設定或可辨識原文。"
)

USER_AGENT = "AI-Book-Writer-WebSkillDistiller/0.1 (+local; respects robots.txt)"
MAX_QUERY_CHARS = 180
MAX_TARGET_EXCERPT_CHARS = 3500
MAX_SOURCE_SNIPPET_CHARS = 1200
REQUEST_DELAY_SECONDS = 1.2


@dataclass(frozen=True)
class CandidatePage:
    title: str
    url: str
    snippet: str = ""
    source: str = "seed"


@dataclass(frozen=True)
class SourceEvidence:
    title: str
    url: str
    source: str
    status: str
    fetched: bool
    chars: int
    digest: str
    snippets: list[str]


class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[bool, RobotFileParser | None, str]] = {}

    def can_fetch(self, session: requests.Session, url: str, user_agent: str = USER_AGENT) -> tuple[bool, str]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False, "unsupported URL scheme"
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._cache:
            robots_url = urljoin(root, "/robots.txt")
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = session.get(robots_url, timeout=10)
                if response.status_code == 404:
                    self._cache[root] = (True, None, "robots.txt not found")
                elif response.status_code >= 400:
                    self._cache[root] = (False, None, f"robots.txt returned {response.status_code}")
                else:
                    parser.parse(response.text.splitlines())
                    self._cache[root] = (True, parser, "robots.txt parsed")
            except requests.RequestException as exc:
                self._cache[root] = (False, None, f"robots.txt unavailable: {exc}")
        ok, parser, note = self._cache[root]
        if not ok:
            return False, note
        if parser is None:
            return True, note
        allowed = parser.can_fetch(user_agent, url)
        return allowed, "robots.txt allows" if allowed else "robots.txt disallows"


def discover_and_distill_web_skill(
    target_description: str,
    target_excerpt: str,
    seed_urls_text: str,
    source_modes: list[str] | str | None,
    allowed_domains_text: str,
    distill_goal: str,
    output_language: str,
    max_search_results: float | int | None,
    max_pages_to_fetch: float | int | None,
    max_snippet_chars: float | int | None,
    dry_run: bool,
    analysis_api_key: str,
    analysis_base_url: str,
    analysis_model_name: str,
) -> tuple[str, str, str | None, str | None, str]:
    """Gradio entry: discover public novel references and distill a skill.

    Returns (status, preview_markdown, story_skill_json_path,
             gap_report_markdown_path, story_skill_json_str).
    """
    try:
        description = normalize_text(target_description)
        excerpt = trim_text(normalize_text(target_excerpt), MAX_TARGET_EXCERPT_CHARS)
        goal = normalize_text(distill_goal) or DEFAULT_WEB_SKILL_GOAL
        if not description and not excerpt and not (seed_urls_text or "").strip():
            return "[ERROR] 請輸入目標小說類型描述、小說片段，或至少提供一個 URL。", "", None, None, ""

        modes = normalize_modes(source_modes)
        max_results = min(to_positive_int(max_search_results) or 8, 30)
        max_pages = min(to_positive_int(max_pages_to_fetch) or 5, 20)
        snippet_chars = min(to_positive_int(max_snippet_chars) or MAX_SOURCE_SNIPPET_CHARS, 5000)
        allowed_domains = parse_allowed_domains(allowed_domains_text)

        session = build_session()
        candidates = discover_candidates(
            session=session,
            target_description=description,
            target_excerpt=excerpt,
            seed_urls_text=seed_urls_text,
            modes=modes,
            allowed_domains=allowed_domains,
            max_results=max_results,
        )
        if candidates:
            evidence, compliance_notes = collect_public_evidence(
                session=session,
                candidates=candidates,
                target_description=description,
                target_excerpt=excerpt,
                max_pages=max_pages,
                max_snippet_chars=snippet_chars,
            )
        elif description or excerpt:
            evidence = []
            compliance_notes = [
                "No external candidate pages were found; generated a local gap skill from the target description/excerpt only.",
            ]
        else:
            return "[ERROR] 找不到候選來源。可貼幾個公開章節 URL，或放寬 Allowed Domains。", "", None, None, ""

        if dry_run:
            skill = build_local_gap_skill(description, excerpt, evidence)
            mode = "Dry Run / local gap distiller"
        else:
            if not analysis_base_url.strip() or not analysis_model_name.strip():
                return "[ERROR] 請先在 Core Settings 設定 Analysis Base URL 與 Model Name。", "", None, None, ""
            client = OpenAI(
                api_key=(analysis_api_key or "not-needed").strip(),
                base_url=analysis_base_url.strip().rstrip("/"),
                timeout=API_TIMEOUT,
            )
            skill = distill_web_skill_via_llm(
                client=client,
                model_name=analysis_model_name.strip(),
                target_description=description,
                target_excerpt=excerpt,
                evidence=evidence,
                goal=goal,
                output_language=output_language,
            )
            mode = f"Analysis LLM / {analysis_model_name.strip()}"

        skill = _normalize_skill(skill)
        skill["source_label"] = "smart-web-skill-distillation"
        skill["distilled_at"] = datetime.now().isoformat(timespec="seconds")
        skill["distillation_mode"] = mode
        skill["source_policy"] = {
            "robots_checked": True,
            "stored_full_text": False,
            "stored_short_snippets_only": True,
            "login_or_paywall_bypass": False,
        }

        run_dir = Path.cwd() / "book_output" / "web_skill_distillations" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        skill_json_str = json.dumps(skill, ensure_ascii=False, indent=2)
        skill_path = run_dir / "story_skill.json"
        skill_md_path = run_dir / "story_skill.md"
        sources_path = run_dir / "web_skill_sources.json"
        report_path = run_dir / "web_skill_gap_report.md"

        write_text_with_backup(skill_path, skill_json_str)
        skill_md = render_skill_markdown(skill)
        write_text_with_backup(skill_md_path, skill_md)
        write_text_with_backup(
            sources_path,
            json.dumps([asdict(item) for item in evidence], ensure_ascii=False, indent=2),
        )
        report_md = render_gap_report(
            skill=skill,
            target_description=description,
            target_excerpt=excerpt,
            evidence=evidence,
            compliance_notes=compliance_notes,
            mode=mode,
            skill_path=skill_path,
            sources_path=sources_path,
        )
        write_text_with_backup(report_path, report_md)

        fetched = sum(1 for item in evidence if item.fetched)
        blocked = sum(1 for note in compliance_notes if "disallows" in note or "unavailable" in note)
        status = (
            "[OK] 智慧網路技巧探索完成。\n"
            f"候選來源：{len(candidates)}；短摘錄來源：{len(evidence)}；實際抓取：{fetched}；robots/連線略過：{blocked}\n"
            f"模式：{mode}\n"
            f"Skill JSON：{skill_path}\n"
            f"缺口報告：{report_path}\n"
            "已更新 Story Skill 狀態，可直接用 ①b 載入技法，或接著做 Step 2 編排。"
        )
        preview = report_md + "\n\n---\n\n" + skill_md
        return status, trim_preview(preview, max_chars=18000), str(skill_path), str(report_path), skill_json_str
    except Exception as exc:  # noqa: BLE001
        logger.exception("discover_and_distill_web_skill failed")
        return f"[ERROR] {exc}", "", None, None, ""


def normalize_modes(source_modes: list[str] | str | None) -> set[str]:
    if source_modes is None:
        return {"Seed URLs only", "Search public web", "Search Pixiv via public search"}
    if isinstance(source_modes, str):
        return {source_modes}
    return {str(item) for item in source_modes if str(item).strip()} or {"Seed URLs only"}


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def discover_candidates(
    *,
    session: requests.Session,
    target_description: str,
    target_excerpt: str,
    seed_urls_text: str,
    modes: set[str],
    allowed_domains: set[str],
    max_results: int,
) -> list[CandidatePage]:
    seen: set[str] = set()
    candidates: list[CandidatePage] = []

    for url in parse_urls(seed_urls_text + "\n" + target_description):
        add_candidate(candidates, seen, CandidatePage(title=url, url=url, source="seed"), allowed_domains)

    query = build_search_query(target_description, target_excerpt)
    queries: list[tuple[str, str]] = []
    if query and "Search public web" in modes:
        queries.append((f"{query} 小說 描寫 技巧 章節", "web-search"))
    if query and "Search Pixiv via public search" in modes:
        queries.append((f"site:pixiv.net/novel {query}", "pixiv-search"))
    for domain in sorted(allowed_domains):
        if query:
            queries.append((f"site:{domain} {query} 小說", f"domain-search:{domain}"))

    remaining = max(0, max_results - len(candidates))
    for search_query, source in queries:
        if remaining <= 0:
            break
        try:
            for item in search_duckduckgo(session, search_query, source=source, limit=remaining):
                add_candidate(candidates, seen, item, allowed_domains)
                remaining = max_results - len(candidates)
                if remaining <= 0:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("search failed for %r: %s", search_query, exc)
    return candidates[:max_results]


def add_candidate(
    candidates: list[CandidatePage],
    seen: set[str],
    candidate: CandidatePage,
    allowed_domains: set[str],
) -> None:
    normalized = normalize_url(candidate.url)
    if not normalized or normalized in seen:
        return
    if allowed_domains and not host_allowed(normalized, allowed_domains):
        return
    seen.add(normalized)
    candidates.append(CandidatePage(candidate.title or normalized, normalized, candidate.snippet, candidate.source))


def search_duckduckgo(
    session: requests.Session,
    query: str,
    *,
    source: str,
    limit: int,
) -> list[CandidatePage]:
    url = "https://duckduckgo.com/html/?q=" + quote_plus(query[:MAX_QUERY_CHARS])
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return parse_duckduckgo_results(response.text, source=source, limit=limit)


def parse_duckduckgo_results(html: str, *, source: str, limit: int) -> list[CandidatePage]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[CandidatePage] = []
    for anchor in soup.select("a.result__a, a[data-testid='result-title-a']"):
        href = extract_search_href(anchor.get("href", ""))
        if not href:
            continue
        container = anchor.find_parent(class_="result") or anchor.parent
        snippet = ""
        if container is not None:
            snippet_node = container.select_one(".result__snippet")
            if snippet_node is not None:
                snippet = snippet_node.get_text(" ", strip=True)
        title = anchor.get_text(" ", strip=True) or href
        results.append(CandidatePage(title=title, url=href, snippet=snippet, source=source))
        if len(results) >= limit:
            break
    return results


def extract_search_href(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//duckduckgo.com/l/"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
    if parsed.scheme in {"http", "https"}:
        return href
    return ""


def collect_public_evidence(
    *,
    session: requests.Session,
    candidates: list[CandidatePage],
    target_description: str,
    target_excerpt: str,
    max_pages: int,
    max_snippet_chars: int,
) -> tuple[list[SourceEvidence], list[str]]:
    robots = RobotsCache()
    evidence: list[SourceEvidence] = []
    notes: list[str] = []
    last_fetch_by_host: dict[str, float] = {}
    terms = extract_terms(target_description + "\n" + target_excerpt)

    for candidate in candidates:
        if len([item for item in evidence if item.fetched]) >= max_pages:
            if candidate.snippet:
                evidence.append(evidence_from_snippet(candidate, "not fetched: page limit reached"))
            continue

        allowed, note = robots.can_fetch(session, candidate.url)
        if not allowed:
            notes.append(f"{candidate.url}: {note}")
            if candidate.snippet:
                evidence.append(evidence_from_snippet(candidate, f"not fetched: {note}"))
            continue

        parsed = urlparse(candidate.url)
        host = parsed.netloc.lower()
        now = time.monotonic()
        elapsed = now - last_fetch_by_host.get(host, 0.0)
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

        try:
            response = session.get(candidate.url, timeout=25)
            last_fetch_by_host[host] = time.monotonic()
            response.raise_for_status()
            ctype = response.headers.get("content-type", "")
            if ctype and "text/html" not in ctype and "text/plain" not in ctype:
                notes.append(f"{candidate.url}: skipped non-text content ({ctype})")
                continue
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding
            text = extract_main_text(response.text)
            snippets = select_snippets(text, terms, max_snippet_chars)
            if not snippets and candidate.snippet:
                snippets = [trim_text(candidate.snippet, min(max_snippet_chars, 500))]
            if snippets:
                joined = "\n".join(snippets)
                evidence.append(
                    SourceEvidence(
                        title=candidate.title,
                        url=candidate.url,
                        source=candidate.source,
                        status=note,
                        fetched=True,
                        chars=len(joined),
                        digest=hash_text(joined),
                        snippets=snippets,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{candidate.url}: fetch failed: {exc}")
            if candidate.snippet:
                evidence.append(evidence_from_snippet(candidate, f"not fetched: {exc}"))
    return evidence, notes


def evidence_from_snippet(candidate: CandidatePage, status: str) -> SourceEvidence:
    snippet = trim_text(normalize_text(candidate.snippet), 500)
    return SourceEvidence(
        title=candidate.title,
        url=candidate.url,
        source=candidate.source,
        status=status,
        fetched=False,
        chars=len(snippet),
        digest=hash_text(snippet),
        snippets=[snippet] if snippet else [],
    )


def select_snippets(text: str, terms: list[str], max_chars: int) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?<=[。！？.!?])\s+", cleaned) if len(p.strip()) >= 40]
    if not paragraphs:
        paragraphs = [cleaned]
    scored: list[tuple[int, int, str]] = []
    for index, para in enumerate(paragraphs[:400]):
        score = sum(1 for term in terms if term and term.lower() in para.lower())
        if score or index < 8:
            scored.append((score, -index, para))
    scored.sort(reverse=True)

    snippets: list[str] = []
    used = 0
    seen: set[str] = set()
    for _, _, para in scored:
        snippet = trim_text(para, min(700, max_chars))
        digest = hash_text(snippet)
        if digest in seen:
            continue
        if used + len(snippet) > max_chars and snippets:
            continue
        snippets.append(snippet)
        seen.add(digest)
        used += len(snippet)
        if used >= max_chars:
            break
    return snippets


def distill_web_skill_via_llm(
    *,
    client: OpenAI,
    model_name: str,
    target_description: str,
    target_excerpt: str,
    evidence: list[SourceEvidence],
    goal: str,
    output_language: str,
) -> dict[str, Any]:
    source_pack = []
    for index, item in enumerate(evidence[:12], start=1):
        source_pack.append(
            {
                "id": f"S{index}",
                "source": item.source,
                "title": item.title,
                "url": item.url,
                "status": item.status,
                "snippets": item.snippets[:3],
            }
        )
    raw = chat_complete(
        client,
        label="smart web skill distillation",
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是小說寫作技巧偵測與 skill 蒸餾 Agent。你只抽取可移植的『怎麼寫』，"
                    "不可保留來源作品的人名、地名、具體事件、特殊設定、長句原文或可辨識橋段。"
                    "你要比較使用者的目標描述/片段與來源短摘錄，找出缺失的寫作技巧與更好的表現方法。"
                    "輸出必須是 JSON object，不能有 markdown code fence。"
                    "JSON 要符合 story skill schema，並額外加入 gap_analysis 欄位："
                    "{target_signal, current_strengths, missing_techniques, upgrade_path, source_limits}。"
                    "description_techniques 請給 10-16 張具體、可操作的技法卡。"
                    "嚴禁複製來源句子；必要時只用抽象描述。"
                    f"{language_instruction(output_language)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n\n"
                    f"目標小說類型描述：\n{target_description or '（未提供）'}\n\n"
                    f"使用者片段（用來診斷缺口，不要重寫）：\n{target_excerpt or '（未提供）'}\n\n"
                    "可參考的 story skill JSON schema：\n"
                    f"{SKILL_JSON_SPEC}\n\n"
                    "公開來源短摘錄包（只做技巧觀察，不得複製內容）：\n"
                    f"{json.dumps(source_pack, ensure_ascii=False, indent=2)}"
                ),
            },
        ],
        temperature=0.25,
        max_tokens=7000,
    )
    return _loads_json_dict(raw, "web skill JSON")


def build_local_gap_skill(
    target_description: str,
    target_excerpt: str,
    evidence: list[SourceEvidence],
) -> dict[str, Any]:
    excerpt = target_excerpt or ""
    strengths, gaps = diagnose_excerpt(excerpt)
    target_terms = extract_terms(target_description)[:8]
    source_count = len(evidence)
    source_hint = "、".join(target_terms[:4]) if target_terms else "目標類型"

    techniques = [
        technique("t1", "目標讀者效果回推", "定位", "先寫清讀者應感到的壓力/期待/情緒，再反推場景信息、動作與語氣。", "讓技巧選擇不散，所有描寫都服務同一閱讀效果。", "先定效果，再放細節；每段只服務一個主要效果。", "只堆漂亮詞，卻不知道要讓讀者感到什麼。"),
        technique("t2", "晚進場景與早出鉤子", "場景節奏", "從衝突已經發生或即將爆發處切入，在答案剛出現或代價剛升高處收束。", "減少鋪墊疲乏，讓段落天然帶推進力。", "開場用短動作，段尾留新問題或情緒落差。", "從起床、走路、背景介紹一路寫到正題。"),
        technique("t3", "壓力源三層化", "衝突", "同一場景同時放入外部阻力、人物欲望、不可說的秘密。", "讀者會感到情境在多方向擠壓人物，而不是單線任務。", "每 600-1000 字至少讓其中一層壓力有變化。", "只有事件，沒有代價；只有情緒，沒有局勢。"),
        technique("t4", "動作鏈取代概括情緒", "描寫", "用手、眼、步伐、呼吸、停頓等連續微動作承載心理變化。", "情緒被看見而非被告知，畫面更有可信度。", "動作先行，心理後落；一句心理對應兩三個外部信號。", "大量使用『很悲傷』『很緊張』等直接標籤。"),
        technique("t5", "感官主次排序", "感官", "每段指定一個主感官，再用第二感官補強，不同段落輪換主感官。", "避免感官雜訊，讓畫面有焦點和層次。", "視覺定位置，聲音推壓力，觸覺落到人物身上。", "五感一起堆，讀者抓不到重心。"),
        technique("t6", "內外反差句", "人物", "外在行為保持克制，內心或身體反應泄露真實波動。", "人物更立體，也能製造含蓄張力。", "短外部句接一個細小失控信號。", "把人物想法全部說白，失去潛台詞。"),
        technique("t7", "資訊分期揭露", "懸念", "先給徵兆，再給錯誤解釋，最後給真正原因或更大的問題。", "讀者會主動補全，黏著度更高。", "每次只揭一層，並同步增加代價。", "一次把背景、原因、真相全說完。"),
        technique("t8", "句長變速", "文風", "平穩段用中句鋪流動，危險/轉折處改用短句切斷。", "節奏本身會形成緊迫感。", "長句鋪氛圍，短句落刀；不要整段同一速度。", "所有句子長短相近，讀起來像說明文。"),
        technique("t9", "具體名詞與動詞替換抽象形容", "語言", "把抽象情緒改成物件、力道、方向、材質、動詞。", "文字更可見、可觸、可表演。", "每段檢查三個抽象詞，替換成可拍攝細節。", "依靠『絕美、震撼、複雜』等泛形容。"),
        technique("t10", "來源技巧去情節化", "蒸餾", "只抽取節奏、視角、壓力配置、描寫秩序，不搬運來源人物與事件。", "讓學到的技巧能安全套用到原創故事。", "每張技法卡都寫成行為規則，不寫來源橋段。", "把參考作品改名重演。"),
    ]

    if "對話太少" in gaps:
        techniques.append(technique("t11", "對話潛台詞壓縮", "對話", "讓人物用短句迴避、試探、轉移焦點，真正意圖藏在停頓與反問裡。", "對話會帶衝突，不只是交換資訊。", "一問一避一反擊，三句內完成權力變化。", "角色把想法完整解釋給對方聽。"))
    if "感官密度偏低" in gaps:
        techniques.append(technique("t12", "場景材質錨點", "氛圍", "選一個可觸摸的材質作為場景錨點，反覆變形回收。", "場景記憶點更強，也能承載情緒變化。", "同一物件在開頭、中段、結尾各變一次意義。", "背景只是裝飾，和人物情緒沒有互動。"))

    return {
        "schema_version": 2,
        "skill_name": f"智慧網路蒸餾·{source_hint}技巧補完",
        "narrative_method": {
            "pov": "以目標類型需求選擇貼近角色的限知視角，讓讀者跟著誤判、期待與承受代價。",
            "tense": "穩定敘事時保持順流，轉折與感官高點用短句製造即時感。",
            "narration_distance": "資訊交代時略遠，情緒與危險時貼近身體反應。",
            "voice_register": "具體、可拍攝、少抽象判斷；讓畫面先說話。",
            "dialogue_style": "對話承載試探與權力變化，避免純說明。",
            "scene_vs_summary": "重點場景深寫；過場壓縮成推進句。",
            "interiority": "內心不直接倒出，透過選擇、遲疑、微動作泄露。",
        },
        "story_progression": {
            "story_engine": "用未解問題、人物欲望和外部阻力形成持續牽引。",
            "escalation_logic": "每段不是重複情緒，而是新增代價、秘密、錯判或關係位移。",
            "tension_rhythm": "鋪陳、逼近、停頓、落刀交替。",
            "reveal_and_withhold": "徵兆先於解釋；答案後面接更大的問題。",
            "hook_principles": "段尾留下新線索、情緒反差、決定或危機。",
            "arc_heuristics": "人物在壓力下暴露不願承認的欲望或弱點。",
            "scene_entry_exit": "晚進早出，從變化點切入，在不可逆落點離開。",
        },
        "plot_ideation": {
            "conflict_sources": "身分差、秘密、交易、禁令、誤會、旁觀者目光。",
            "character_engine": "人物的外在目標和內在缺口互相拉扯。",
            "power_dynamics": "每場至少有一個權力籌碼變動。",
            "parallel_threads": "明線推事件，暗線推情緒和真相。",
            "seed_planting": "提前埋物件、口頭禪、禁忌或錯誤印象，後續回收。",
            "stakes_axes": "同時照顧生存/關係/名聲/信念至少兩軸代價。",
        },
        "description_techniques": techniques,
        "transferable_rules": [
            f"本次蒐集到 {source_count} 個短摘錄來源；只抽象技巧，不保存或仿寫來源情節。",
            "每次寫作前先問：這段缺的是壓力、畫面、節奏、潛台詞，還是鉤子。",
            "優先補足目前片段的弱項：" + ("、".join(gaps[:5]) if gaps else "依目標類型建立更強的場景推進。"),
        ],
        "usage_contract": "這份 skill 是技巧工具箱，不是模板；不得搬運來源角色、事件、設定或可辨識句子。",
        "abstraction_note": "由公開網頁短摘錄與使用者目標/片段蒸餾，只保留可移植寫法。",
        "gap_analysis": {
            "target_signal": target_description or "使用者片段推斷",
            "current_strengths": strengths,
            "missing_techniques": gaps,
            "upgrade_path": ["先定讀者效果", "補壓力源", "補動作與感官", "用句長變速", "段尾留鉤"],
            "source_limits": "Dry Run 未調用 LLM，診斷為本地啟發式；可取消 Dry Run 取得更細分析。",
        },
    }


def technique(
    tid: str,
    name: str,
    category: str,
    craft_move: str,
    why: str,
    rhythm: str,
    pitfalls: str,
) -> dict[str, str]:
    return {
        "id": tid,
        "name": name,
        "category": category,
        "craft_move": craft_move,
        "why_it_works": why,
        "rhythm_and_sensory": rhythm,
        "pitfalls": pitfalls,
    }


def diagnose_excerpt(excerpt: str) -> tuple[list[str], list[str]]:
    text = excerpt or ""
    strengths: list[str] = []
    gaps: list[str] = []
    if len(text) >= 800:
        strengths.append("已有足夠篇幅可觀察節奏與段落推進")
    else:
        gaps.append("樣本偏短，難以判斷長段節奏")
    dialogue_marks = text.count("「") + text.count("\"") + text.count("“")
    if dialogue_marks >= 4:
        strengths.append("已有對話或聲音交鋒")
    else:
        gaps.append("對話太少")
    sensory_hits = len(re.findall(r"光|影|聲|香|冷|熱|痛|汗|風|雨|血|手|眼|唇|指|呼吸", text))
    if sensory_hits >= 8:
        strengths.append("已有可用感官與身體細節")
    else:
        gaps.append("感官密度偏低")
    if re.search(r"然而|卻|但|只是|不能|不得|秘密|忽然|突然", text):
        strengths.append("已有轉折或限制信號")
    else:
        gaps.append("衝突/限制信號不明顯")
    if len(re.findall(r"[。！？.!?]", text)) >= 12:
        strengths.append("已有基本句群節奏")
    else:
        gaps.append("句群節奏樣本不足")
    if not gaps:
        gaps.append("可進一步強化段尾鉤子與多層壓力")
    return strengths or ["已有可作為診斷起點的目標描述"], gaps


def render_gap_report(
    *,
    skill: dict[str, Any],
    target_description: str,
    target_excerpt: str,
    evidence: list[SourceEvidence],
    compliance_notes: list[str],
    mode: str,
    skill_path: Path,
    sources_path: Path,
) -> str:
    gap = skill.get("gap_analysis") if isinstance(skill.get("gap_analysis"), dict) else {}
    lines = [
        "# 智慧網路技巧缺口報告",
        "",
        f"- 模式：{mode}",
        f"- Skill JSON：`{skill_path}`",
        f"- 來源短摘錄索引：`{sources_path}`",
        "- 合規策略：尊重 robots.txt；不登入、不繞付費牆、不保存全文；只保存短摘錄與技巧摘要。",
        "",
        "## 目標信號",
        target_description or "（未提供目標描述）",
        "",
    ]
    if target_excerpt:
        lines += ["## 使用者片段診斷樣本", trim_text(target_excerpt, 900), ""]
    if gap:
        lines += ["## 缺失與補強路線"]
        for key in ("current_strengths", "missing_techniques", "upgrade_path"):
            value = gap.get(key)
            if isinstance(value, list):
                lines.append(f"### {key}")
                lines += [f"- {item}" for item in value]
        if gap.get("source_limits"):
            lines += ["", "### source_limits", f"- {gap['source_limits']}"]
        lines.append("")
    lines += ["## 來源摘要"]
    if evidence:
        for index, item in enumerate(evidence, start=1):
            lines += [
                f"### S{index}. {item.title}",
                f"- URL：{item.url}",
                f"- Source：{item.source}",
                f"- Status：{item.status}",
                f"- Snippets：{len(item.snippets)}（短摘錄，非全文）",
                "",
            ]
    else:
        lines.append("- 沒有可用來源；skill 只根據目標/片段做本地診斷。")
    if compliance_notes:
        lines += ["", "## 略過/限制記錄"]
        lines += [f"- {trim_text(note, 260)}" for note in compliance_notes[:30]]
    return "\n".join(lines).strip() + "\n"


def build_search_query(target_description: str, target_excerpt: str) -> str:
    text = target_description or target_excerpt
    terms = extract_terms(text)
    query = " ".join(terms[:8])
    if not query and text:
        query = text[:MAX_QUERY_CHARS]
    return query[:MAX_QUERY_CHARS].strip()


def extract_terms(text: str) -> list[str]:
    text = normalize_text(text)
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,8}", text):
        if token not in terms and not is_low_value_term(token):
            terms.append(token)
    return terms


def is_low_value_term(token: str) -> bool:
    low_value = {
        "小說",
        "片段",
        "描述",
        "目標",
        "寫作",
        "技巧",
        "更好",
        "自動",
        "輸入",
        "找出",
    }
    return token in low_value


def parse_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s<>\"]+", text or "")


def parse_allowed_domains(text: str) -> set[str]:
    domains: set[str] = set()
    for raw in re.split(r"[\s,;，；]+", text or ""):
        item = raw.strip().lower()
        if not item:
            continue
        parsed = urlparse(item if "://" in item else "https://" + item)
        host = parsed.hostname or parsed.netloc or parsed.path
        host = host.strip().lstrip(".").lower()
        if host:
            domains.add(host)
    return domains


def host_allowed(url: str, allowed_domains: set[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or parsed.netloc).lower()
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


def normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip(").,，。")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return parsed.geturl()


def hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def trim_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[trimmed]"


__all__ = [
    "DEFAULT_WEB_SKILL_GOAL",
    "WEB_SOURCE_MODE_CHOICES",
    "discover_and_distill_web_skill",
    "parse_duckduckgo_results",
    "extract_search_href",
    "build_local_gap_skill",
]
