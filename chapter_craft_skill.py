"""Chapter-level fiction craft analysis skill for the Gradio studio."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


DEFAULT_GOAL = (
    "逐章分析小說寫作技巧，抓出章節功能、敘事節奏、人物推進、衝突設計、"
    "語言風格、可複製技法與可改善處。"
)

SPECIALISTS = {
    "structure": {
        "title": "結構編劇",
        "focus": "章節功能、開場鉤子、事件因果、場景節拍、轉折與收束。",
    },
    "pacing": {
        "title": "節奏讀者",
        "focus": "懸念密度、資訊釋放、快慢切換、段落推進、讀者翻頁動機。",
    },
    "character": {
        "title": "人物弧線分析師",
        "focus": "角色欲望、阻力、選擇、關係張力、台詞與行動是否推動人物。",
    },
    "style": {
        "title": "文風技術編輯",
        "focus": "敘述視角、句式節奏、意象、感官細節、對白、場面調度與語氣。",
    },
}

CHAPTER_HEADING_RE = re.compile(
    r"^\s*(第[零一二三四五六七八九十百千萬万\d]+[章回節节卷部][^\n\r]{0,60}|"
    r"Chapter\s+\d+[^\n\r]{0,60}|"
    r"\d{1,4}[\.、]\s*[^\n\r]{1,60})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

DEFAULT_AUTO_CHUNK_CHARS = 12000
MIN_AUTO_CHUNK_CHARS = 3000


@dataclass(frozen=True)
class Chapter:
    index: int
    title: str
    text: str
    source_url: str | None = None


@dataclass(frozen=True)
class SpecialistResult:
    role: str
    title: str
    analysis: str


@dataclass(frozen=True)
class ChapterReport:
    chapter: Chapter
    goal: str
    specialist_results: list[SpecialistResult]
    synthesis: str


def analyze_chapter_craft(
    txt_file: Any,
    pasted_text: str,
    directory_url: str,
    goal: str,
    chapter_limit: float | int | None,
    max_chapter_chars: float | int | None,
    dry_run: bool,
    api_key: str,
    base_url: str,
    model_name: str,
) -> tuple[str, str, str | None]:
    """Gradio entry point for TXT, pasted text, or chapter-directory URL analysis."""
    try:
        goal = (goal or DEFAULT_GOAL).strip()
        limit = to_positive_int(chapter_limit)
        max_chars = to_positive_int(max_chapter_chars) or 12000

        chapters, source_label = load_chapters(
            txt_file=txt_file,
            pasted_text=pasted_text,
            directory_url=directory_url,
            limit=limit,
            fallback_chunk_chars=max_chars,
        )
        if not chapters:
            return "[ERROR] 沒有可分析的章節。", "", None

        client = None
        if not dry_run:
            if not base_url.strip() or not model_name.strip():
                return "[ERROR] 請先在 Core Settings 設定 Base URL 與 Model Name。", "", None
            client = OpenAI(
                api_key=(api_key or "not-needed").strip(),
                base_url=base_url.strip().rstrip("/"),
                timeout=900,
            )

        reports: list[ChapterReport] = []
        for chapter in chapters:
            reports.append(
                analyze_one_chapter(
                    chapter=chapter,
                    goal=goal,
                    max_chapter_chars=max_chars,
                    dry_run=dry_run,
                    client=client,
                    model_name=model_name.strip(),
                )
            )

        output_dir = write_reports(reports, source_label)
        combined_path = output_dir / "full_report.md"
        preview = combined_path.read_text(encoding="utf-8")
        status = (
            f"[OK] 已分析 {len(reports)} 章。\n"
            f"來源：{source_label}\n"
            f"模式：{'Dry Run' if dry_run else model_name.strip()}\n"
            f"報告：{combined_path}"
        )
        return status, trim_preview(preview), str(combined_path)
    except Exception as exc:
        return f"[ERROR] {exc}", "", None


def load_chapters(
    *,
    txt_file: Any,
    pasted_text: str,
    directory_url: str,
    limit: int | None,
    fallback_chunk_chars: int | None = None,
) -> tuple[list[Chapter], str]:
    url = (directory_url or "").strip()
    if txt_file:
        path = get_file_path(txt_file)
        chapters = split_chapters(read_text_file(path), fallback_chunk_chars=fallback_chunk_chars)
        return apply_limit(chapters, limit), Path(path).stem
    if (pasted_text or "").strip():
        chapters = split_chapters(pasted_text, fallback_chunk_chars=fallback_chunk_chars)
        return apply_limit(chapters, limit), "pasted-text"
    if url:
        chapters = load_chapters_from_directory_url(url, limit=limit)
        return chapters, url
    raise ValueError("請上傳 TXT、貼上小說正文，或輸入小說章節目錄網址。")


def read_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    file_path = Path(path)
    encodings = [encoding, "utf-8-sig", "cp950", "big5", "gb18030"]
    tried: set[str] = set()
    for candidate in encodings:
        if candidate in tried:
            continue
        tried.add(candidate)
        try:
            return file_path.read_text(encoding=candidate)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding=encoding)


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


def split_chapters(raw_text: str, *, fallback_chunk_chars: int | None = None) -> list[Chapter]:
    text = normalize_text(raw_text)
    matches = list(CHAPTER_HEADING_RE.finditer(text))
    if not matches:
        return split_text_into_auto_chapters(text, fallback_chunk_chars)

    chapters: list[Chapter] = []
    preface = text[: matches[0].start()].strip()
    if preface:
        chapters.append(Chapter(index=1, title="序章/前言", text=preface))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chapters.append(
                Chapter(index=len(chapters) + 1, title=clean_title(match.group(1)), text=body)
            )
    return chapters


def split_text_into_auto_chapters(text: str, chunk_chars: int | None) -> list[Chapter]:
    text = text.strip()
    if not text:
        return []

    target = chunk_chars or DEFAULT_AUTO_CHUNK_CHARS
    target = max(MIN_AUTO_CHUNK_CHARS, int(target))
    if len(text) <= target:
        return [Chapter(index=1, title="Auto Chunk 001", text=text)]

    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if len(blocks) <= 1:
        blocks = [block.strip() for block in text.split("\n") if block.strip()]
    if len(blocks) <= 1:
        blocks = split_long_block(text, target)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for block in blocks:
        if len(block) > target:
            flush_current()
            chunks.extend(split_long_block(block, target))
            continue

        extra_len = len(block) + (2 if current else 0)
        if current and current_len + extra_len > target:
            flush_current()
        current.append(block)
        current_len += extra_len

    flush_current()
    return [
        Chapter(index=index, title=f"Auto Chunk {index:03d}", text=chunk)
        for index, chunk in enumerate(chunks, start=1)
        if chunk
    ]


def split_long_block(block: str, chunk_chars: int) -> list[str]:
    return [
        block[start : start + chunk_chars].strip()
        for start in range(0, len(block), chunk_chars)
        if block[start : start + chunk_chars].strip()
    ]


def load_chapters_from_directory_url(url: str, limit: int | None) -> list[Chapter]:
    session = requests.Session()
    session.headers.update({"User-Agent": "AI-Book-Writer-ChapterCraft/0.1"})
    html = fetch_text(session, url)
    links = extract_chapter_links(html, url)
    if limit:
        links = links[:limit]
    if not links:
        raise ValueError("找不到章節連結；請確認網址是靜態 HTML 目錄頁，或改用 TXT。")

    chapters: list[Chapter] = []
    for index, (title, chapter_url) in enumerate(links, start=1):
        chapter_html = fetch_text(session, chapter_url)
        text = extract_main_text(chapter_html)
        if text:
            chapters.append(Chapter(index=index, title=title, text=text, source_url=chapter_url))
        if index < len(links):
            time.sleep(0.8)
    return chapters


def fetch_text(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding
    return response.text


def extract_chapter_links(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        title = anchor.get_text(" ", strip=True)
        if not title or not looks_like_chapter_title(title):
            continue
        absolute = urljoin(base_url, anchor["href"].strip())
        if absolute in seen or not is_same_site_or_relative(base_url, absolute):
            continue
        seen.add(absolute)
        links.append((clean_title(title), absolute))
    return links


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    selectors = [
        "#content",
        ".content",
        ".chapter-content",
        ".chapter",
        ".read-content",
        ".book-content",
        "article",
        "main",
    ]
    best = ""
    for selector in selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        text = normalize_text(node.get_text("\n", strip=True))
        if len(text) > len(best):
            best = text
    return best or normalize_text(soup.get_text("\n", strip=True))


def analyze_one_chapter(
    *,
    chapter: Chapter,
    goal: str,
    max_chapter_chars: int,
    dry_run: bool,
    client: OpenAI | None,
    model_name: str,
) -> ChapterReport:
    source_text = trim_chapter_text(chapter.text, max_chapter_chars)
    if dry_run:
        specialist_results = [
            SpecialistResult(
                role=role,
                title=spec["title"],
                analysis=dry_run_role_analysis(chapter, source_text, spec["focus"]),
            )
            for role, spec in SPECIALISTS.items()
        ]
        synthesis = dry_run_synthesis(chapter, source_text)
    else:
        if client is None:
            raise RuntimeError("缺少 LLM client。")
        specialist_results = run_specialists(client, model_name, chapter, source_text, goal)
        synthesis = synthesize_report(client, model_name, chapter, specialist_results, goal)
    return ChapterReport(chapter, goal, specialist_results, synthesis)


def run_specialists(
    client: OpenAI,
    model_name: str,
    chapter: Chapter,
    source_text: str,
    goal: str,
) -> list[SpecialistResult]:
    results: list[SpecialistResult] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                ask_specialist,
                client,
                model_name,
                role,
                spec["title"],
                spec["focus"],
                chapter,
                source_text,
                goal,
            ): role
            for role, spec in SPECIALISTS.items()
        }
        for future in as_completed(futures):
            results.append(future.result())

    role_order = list(SPECIALISTS)
    return sorted(results, key=lambda item: role_order.index(item.role))


def ask_specialist(
    client: OpenAI,
    model_name: str,
    role: str,
    title: str,
    focus: str,
    chapter: Chapter,
    source_text: str,
    goal: str,
) -> SpecialistResult:
    messages = [
        {
            "role": "system",
            "content": (
                f"你是「{title}」，專門分析小說寫作技巧。"
                "請使用繁體中文，直接、具體、可操作。"
                "不要改寫原文，不要劇透未提供章節，不要空泛稱讚。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"/goal: {goal}\n\n"
                f"章節：第 {chapter.index} 章｜{chapter.title}\n"
                f"分析焦點：{focus}\n\n"
                "請輸出：\n"
                "1. 本章在小說中的功能\n"
                "2. 可觀察到的寫作技巧，至少 5 點，每點要引用短語或描述原文現象\n"
                "3. 技巧如何影響讀者\n"
                "4. 可學習的寫法模板\n"
                "5. 若要強化，本章最值得調整的 2 點\n\n"
                f"章節正文：\n{source_text}"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.25,
        max_tokens=1800,
    )
    return SpecialistResult(role=role, title=title, analysis=read_message(response))


def synthesize_report(
    client: OpenAI,
    model_name: str,
    chapter: Chapter,
    specialist_results: list[SpecialistResult],
    goal: str,
) -> str:
    cluster_notes = "\n\n".join(
        f"## {result.title}\n{result.analysis}" for result in specialist_results
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是總編型 AI Agent，負責把多名分析員的觀察整合成一份可學習、"
                    "可執行的章節寫作技巧報告。使用繁體中文。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"/goal: {goal}\n\n"
                    f"章節：第 {chapter.index} 章｜{chapter.title}\n\n"
                    "請整合下列 AI 叢集筆記，輸出固定格式：\n"
                    "- 章節核心作用\n"
                    "- 本章最重要的 8 個寫作技巧\n"
                    "- 技巧拆解：鋪陳、衝突、人物、節奏、文風\n"
                    "- 逐句寫法解剖（最重要，不要空泛）：挑本章 2-3 個關鍵描寫片段（如某個身體部位、"
                    "某個動作、某段對話），各自拆解『它是一句一句怎麼寫成的』——可觀察微單元的順序、"
                    "句子長短與標點節奏、用了哪些具體動詞與意象、哪個感官領頭。每段並附一組弱寫→強寫對照"
                    "（你自己寫的短句級示例，不要抄原文）。\n"
                    "- 可複製寫作公式\n"
                    "- 仿寫練習題 3 題\n"
                    "- 修稿建議 3 點\n\n"
                    f"AI 叢集筆記：\n{cluster_notes}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=3200,
    )
    return read_message(response)


def read_message(response: Any) -> str:
    return (response.choices[0].message.content or "").strip()


def write_reports(reports: list[ChapterReport], source_label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.cwd() / "book_output" / "chapter_craft_reports" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        f"# {source_label}｜逐章寫作技巧分析",
        "",
        f"共分析 {len(reports)} 個章節。",
        "",
    ]
    combined = [index_lines[0], "", index_lines[2], ""]
    for report in reports:
        file_name = f"{report.chapter.index:03d}-{slugify(report.chapter.title)}.md"
        (output_dir / file_name).write_text(render_chapter_report(report), encoding="utf-8")
        index_lines.append(f"- [第 {report.chapter.index} 章｜{report.chapter.title}]({file_name})")
        combined.append(render_chapter_report(report))
        combined.append("\n---\n")

    (output_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    (output_dir / "full_report.md").write_text("\n".join(combined).strip() + "\n", encoding="utf-8")
    return output_dir


def render_chapter_report(report: ChapterReport) -> str:
    source = f"\n\n來源：{report.chapter.source_url}" if report.chapter.source_url else ""
    specialist_sections = "\n\n".join(
        f"## {item.title}\n\n{item.analysis}" for item in report.specialist_results
    )
    return (
        f"# 第 {report.chapter.index} 章｜{report.chapter.title}\n\n"
        f"`/goal`：{report.goal}{source}\n\n"
        "## AI 叢集總結\n\n"
        f"{report.synthesis}\n\n"
        "## 專家分工筆記\n\n"
        f"{specialist_sections}\n"
    )


def dry_run_role_analysis(chapter: Chapter, source_text: str, focus: str) -> str:
    paragraphs = [p.strip() for p in source_text.split("\n") if p.strip()]
    avg_len = int(sum(len(p) for p in paragraphs) / max(len(paragraphs), 1))
    return (
        f"離線檢查模式：{chapter.title}\n\n"
        f"- 分析焦點：{focus}\n"
        f"- 字數：約 {len(source_text)} 字；段落數：{len(paragraphs)}；平均段長：約 {avg_len} 字。\n"
        "- 檢查章首是否快速建立人物目標、阻力與場景問題。\n"
        "- 檢查每 3 到 5 段是否有新的資訊、行動或情緒變化。\n"
        "- 標記對白前後的敘事動作，判斷台詞是否推動關係或只在解釋設定。\n"
        "- 正式模式會由目前設定的 API 模型產生完整技巧拆解。"
    )


def dry_run_synthesis(chapter: Chapter, source_text: str) -> str:
    return (
        "## 離線總結\n\n"
        f"第 {chapter.index} 章〈{chapter.title}〉已完成 dry-run。"
        "章節擷取與 AI 叢集流程可正常運作；關閉 Dry Run 後會輸出完整分析。\n\n"
        "### 建議正式分析時檢查\n"
        "- 章首鉤子是否在一頁內建立問題。\n"
        "- 本章中段是否有明確轉折，而不只是資訊補充。\n"
        "- 章尾是否留下新的選擇、危機、秘密或情緒落差。\n"
        "- 文風是否透過動作、感官與對白承載資訊。"
    )


def trim_chapter_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n[...中段略，因章節過長已自動截取...]\n\n{tail}"


def trim_preview(text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[預覽截斷；請下載 full_report.md 查看完整內容。]"


def apply_limit(chapters: list[Chapter], limit: int | None) -> list[Chapter]:
    return chapters[:limit] if limit else chapters


def to_positive_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()[:120]


def slugify(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "-", value.strip())
    value = re.sub(r"\s+", "-", value)
    return value[:80] or "chapter"


def looks_like_chapter_title(title: str) -> bool:
    stripped = title.strip()
    if len(stripped) > 120:
        return False
    patterns: Iterable[str] = (
        r"第[零一二三四五六七八九十百千萬万\d]+[章回節节卷部]",
        r"Chapter\s+\d+",
        r"^\d{1,4}[\.、]\s*",
    )
    return any(re.search(pattern, stripped, re.IGNORECASE) for pattern in patterns)


def is_same_site_or_relative(base_url: str, candidate_url: str) -> bool:
    base = urlparse(base_url)
    candidate = urlparse(candidate_url)
    return not candidate.netloc or candidate.netloc == base.netloc
