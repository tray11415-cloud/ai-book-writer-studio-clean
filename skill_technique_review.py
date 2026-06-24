"""Local Skill and Technique review helpers for the Gradio studio."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


REVIEW_CHOICES = [
    "Skill + Latest Technique",
    "Skill Only",
    "Latest Integrated Technique Book Library",
    "Latest Technique Finder Library",
    "Latest Focused Technique Sheet",
    "Latest Chapter Craft Report",
    "Latest Plot Ideation Report",
    "Custom Markdown / JSON Path",
]

SKILL_RELATIVE_FILES = [
    Path("skills/novel-chapter-craft-analysis/SKILL.md"),
    Path("skills/novel-chapter-craft-analysis/references/prompt-contract.md"),
    Path("skills/novel-chapter-craft-analysis/agents/openai.yaml"),
]

REPORT_PATTERNS = {
    "Latest Integrated Technique Book Library": "book_output/integrated_technique_libraries/*/integrated_technique_book_library.md",
    "Latest Technique Finder Library": "book_output/scene_techniques/library/*/technique_finder_library.md",
    "Latest Focused Technique Sheet": "book_output/scene_techniques/*/scene_techniques.md",
    "Latest Chapter Craft Report": "book_output/chapter_craft_reports/*/full_report.md",
    "Latest Plot Ideation Report": "book_output/plot_ideation/*/plot_ideation.md",
}

KEYWORDS = [
    "Technique",
    "Director Instruction",
    "STYLE_DNA",
    "手法",
    "技法",
    "公式",
    "場景",
    "動作",
    "情境",
    "節奏",
    "視角",
    "感官",
    "可複製",
    "寫作",
    "Deep Breakdown",
    "Detail Lenses",
    "Micro Techniques",
    "Common Mistakes",
    "Practice Prompts",
    "Anatomy Breakdown",
    "Sentence Rhythm",
    "Word Palette",
    "Sensory Layering",
    "Weak vs Strong",
    "Reference Shelf",
]

AGE_MARKERS = [
    "未成年",
    "未滿",
    "十四",
    "十五",
    "十六",
    "14歲",
    "15歲",
    "16歲",
    "小女孩",
    "小丫頭",
]

SEXUAL_MARKERS = [
    "性",
    "自慰",
    "性交",
    "裸體",
    "下體",
    "乳",
    "調教",
    "寵物",
]


@dataclass(frozen=True)
class ReviewArtifact:
    label: str
    path: Path
    text: str


def review_skill_and_technique(
    review_target: str,
    custom_path: str,
    max_preview_chars: float | int | None,
) -> tuple[str, str, str | None]:
    """Build a local review report for Skill definitions and technique outputs."""
    try:
        target = review_target or REVIEW_CHOICES[0]
        max_chars = to_positive_int(max_preview_chars) or 22000
        sections: list[str] = []
        artifacts: list[ReviewArtifact] = []

        if target in {"Skill + Latest Technique", "Skill Only"}:
            skill_section, skill_artifacts = build_skill_review(max_chars=max_chars)
            sections.append(skill_section)
            artifacts.extend(skill_artifacts)

        if target != "Skill Only":
            if target == "Skill + Latest Technique":
                artifact = find_latest_any_report()
            elif target == "Custom Markdown / JSON Path":
                artifact = read_custom_artifact(custom_path)
            else:
                artifact = find_latest_report(target)

            if artifact:
                sections.append(build_artifact_review(artifact, max_chars=max_chars))
                artifacts.append(artifact)
            else:
                sections.append("## Technique Review\n\nNo matching technique/report artifact was found.")

        markdown = "\n\n---\n\n".join(section for section in sections if section.strip()).strip()
        output_path = write_review_report(markdown)
        status = build_status(target, artifacts, output_path)
        return status, markdown, str(output_path)
    except Exception as exc:
        return f"[ERROR] {exc}", "", None


def build_skill_review(max_chars: int) -> tuple[str, list[ReviewArtifact]]:
    root = Path.cwd()
    artifacts: list[ReviewArtifact] = []
    lines = [
        "# Skill Review",
        "",
        "## Capability Checklist",
        "",
        f"- `chapter_craft_skill.py`: {mark_exists(root / 'chapter_craft_skill.py')}",
        f"- `plot_ideation_skill.py`: {mark_exists(root / 'plot_ideation_skill.py')}",
        f"- `scene_technique_skill.py`: {mark_exists(root / 'scene_technique_skill.py')}",
        f"- `technique_library_builder.py`: {mark_exists(root / 'technique_library_builder.py')}",
        f"- `skill_technique_review.py`: {mark_exists(root / 'skill_technique_review.py')}",
        f"- Project Skill folder: {mark_exists(root / 'skills/novel-chapter-craft-analysis')}",
        "",
        "## Skill Files",
        "",
    ]

    for relative in SKILL_RELATIVE_FILES:
        path = root / relative
        text = read_text(path)
        if text:
            artifacts.append(ReviewArtifact(relative.as_posix(), path, text))
            lines.extend(
                [
                    f"### `{relative.as_posix()}`",
                    "",
                    f"- Path: `{path}`",
                    f"- Size: {path.stat().st_size:,} bytes",
                    f"- Modified: {path.stat().st_mtime:.0f}",
                    "",
                    "```markdown",
                    trim_text(text, max_chars // 3),
                    "```",
                    "",
                ]
            )
        else:
            lines.extend([f"### `{relative.as_posix()}`", "", "- Missing or unreadable.", ""])

    return "\n".join(lines), artifacts


def build_artifact_review(artifact: ReviewArtifact, max_chars: int) -> str:
    text = sanitize_review_text(artifact.text)
    headings = extract_headings(text, limit=80)
    keyword_lines = extract_keyword_lines(text, limit=80)
    director_blocks = extract_director_blocks(text, limit=12)
    deep_blocks = extract_deep_technique_blocks(text, limit=16)
    json_summary = summarize_json_if_possible(artifact.path, text)

    lines = [
        "# Technique Review",
        "",
        f"- Source: {artifact.label}",
        f"- Path: `{artifact.path}`",
        f"- Size: {artifact.path.stat().st_size:,} bytes",
        f"- Modified: {datetime.fromtimestamp(artifact.path.stat().st_mtime).isoformat(timespec='seconds')}",
        "",
    ]
    if json_summary:
        lines.extend(["## JSON Summary", "", json_summary, ""])
    if headings:
        lines.extend(["## Headings", "", "\n".join(f"- {heading}" for heading in headings), ""])
    if director_blocks:
        lines.extend(["## Director Instructions", "", "\n\n".join(director_blocks), ""])
    if deep_blocks:
        lines.extend(["## Deep Technique Blocks", "", "\n\n".join(deep_blocks), ""])
    if keyword_lines:
        lines.extend(["## Technique Lines", "", "\n".join(f"- {line}" for line in keyword_lines), ""])

    lines.extend(
        [
            "## Preview",
            "",
            trim_text(text, max_chars),
        ]
    )
    return "\n".join(lines)


def find_latest_any_report() -> ReviewArtifact | None:
    candidates = [artifact for label in REPORT_PATTERNS for artifact in [find_latest_report(label)] if artifact]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.path.stat().st_mtime)


def find_latest_report(label: str) -> ReviewArtifact | None:
    pattern = REPORT_PATTERNS.get(label)
    if not pattern:
        return None
    files = [path for path in Path.cwd().glob(pattern) if path.is_file()]
    if not files:
        return None
    path = max(files, key=lambda item: item.stat().st_mtime)
    return ReviewArtifact(label, path, read_text(path))


def read_custom_artifact(custom_path: str) -> ReviewArtifact | None:
    clean = (custom_path or "").strip().strip('"')
    if not clean:
        return None
    raw = Path(clean).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    # Resolve symlinks / '..' segments so a relative input cannot traverse out of
    # the project (e.g. '../../etc/passwd') while still allowing intentional
    # absolute paths the operator chooses.
    path = raw.resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    if not (is_within(path, cwd) or is_within(path, home)):
        logger.warning("Rejected custom artifact path outside allowed roots: %s", path)
        raise ValueError(f"Custom path is outside the allowed directories: {path}")
    if not path.is_file():
        raise ValueError(f"Custom path does not exist: {path}")
    return ReviewArtifact("Custom Markdown / JSON Path", path, read_text(path))


def is_within(path: Path, root: Path) -> bool:
    """Return True if resolved ``path`` is ``root`` or lives beneath it."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "cp950", "big5", "gb18030"):
        try:
            return path.read_text(encoding=encoding, errors="strict")
        except UnicodeDecodeError:
            continue
    # No known encoding matched: decode lossily so the UI still works, but warn
    # because multi-byte (e.g. Chinese) text may be corrupted by replacements.
    logger.warning("No clean encoding for %s; falling back to utf-8 with replacement characters.", path)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_headings(text: str, limit: int) -> list[str]:
    headings = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,4}\s+\S+", stripped):
            headings.append(stripped)
        if len(headings) >= limit:
            break
    return headings


def extract_keyword_lines(text: str, limit: int) -> list[str]:
    matches: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if len(stripped) < 8:
            continue
        if any(keyword in stripped for keyword in KEYWORDS) and stripped not in seen:
            matches.append(trim_text(stripped, 260))
            seen.add(stripped)
        if len(matches) >= limit:
            break
    return matches


def extract_director_blocks(text: str, limit: int) -> list[str]:
    blocks: list[str] = []
    pattern = re.compile(r"(?im)^.*Director Instruction.*(?:\n(?:[-*].*|.{1,260})){0,5}")
    for match in pattern.finditer(text):
        block = match.group(0).strip()
        if block and block not in blocks:
            blocks.append(trim_text(block, 900))
        if len(blocks) >= limit:
            break
    return blocks


def extract_deep_technique_blocks(text: str, limit: int, follow_lines: int = 8) -> list[str]:
    blocks: list[str] = []
    headings = (
        "Anatomy Breakdown",
        "Sentence Rhythm",
        "Word Palette",
        "Sensory Layering",
        "Weak vs Strong",
        "Deep Breakdown",
        "Detail Lenses",
        "Micro Techniques",
        "Common Mistakes",
        "Practice Prompts",
    )
    # Build the alternation from the headings tuple so the list lives in one place.
    heading_alt = "|".join(re.escape(heading) for heading in headings)
    pattern = re.compile(
        rf"(?im)^(?:\*\*)?({heading_alt})(?:\*\*)?.*$"
        rf"(?:\n(?:[-*].*|.{{1,260}})){{0,{follow_lines}}}"
    )
    for match in pattern.finditer(text):
        title = match.group(1)
        if title not in headings:
            continue
        block = match.group(0).strip()
        if block and block not in blocks:
            blocks.append(trim_text(block, 1200))
        if len(blocks) >= limit:
            break
    return blocks


def sanitize_review_text(text: str) -> str:
    return "\n".join(sanitize_line(line) for line in text.splitlines())


def drop_unsafe_lines(text: str, markers: list[str]) -> str:
    """Sanitize then drop any line containing one of the given unsafe markers.

    Shared helper so report_technique_distiller and this module use a single
    line-filtering implementation instead of duplicating the loop.
    """
    lowered = [marker.lower() for marker in markers]
    kept = []
    for line in sanitize_review_text(text).splitlines():
        low = line.lower()
        if any(marker in low for marker in lowered):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def sanitize_line(line: str) -> str:
    compact = line.lower()
    has_age_marker = any(marker.lower() in compact for marker in AGE_MARKERS)
    has_sexual_marker = any(marker.lower() in compact for marker in SEXUAL_MARKERS)
    if has_age_marker and has_sexual_marker:
        return "[redacted: source report references underage sexual content]"
    return line


def summarize_json_if_possible(path: Path, text: str) -> str:
    if path.suffix.lower() != ".json":
        sibling = path.with_suffix(".json")
        if sibling.exists():
            path = sibling
            text = read_text(path)
        else:
            return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, list):
        return f"- JSON list items: {len(payload)}"
    if isinstance(payload, dict):
        keys = ", ".join(str(key) for key in list(payload.keys())[:20])
        cards = payload.get("cards")
        card_line = f"\n- Cards: {len(cards)}" if isinstance(cards, list) else ""
        entries = payload.get("entries")
        entry_line = ""
        if isinstance(entries, list):
            deep_count = 0
            for item in entries:
                if isinstance(item, dict) and any(
                    item.get(key)
                    for key in (
                        "deep_breakdown",
                        "detail_lenses",
                        "micro_techniques",
                        "common_mistakes",
                        "practice_prompts",
                        "anatomy_breakdown",
                        "sentence_rhythm",
                        "word_palette",
                        "sensory_layering",
                        "weak_vs_strong",
                    )
                ):
                    deep_count += 1
            entry_line = f"\n- Integrated technique entries: {len(entries)}\n- Entries with deep fields: {deep_count}"
        return f"- JSON object keys: {keys}{card_line}{entry_line}"
    return f"- JSON type: {type(payload).__name__}"


def write_review_report(markdown: str) -> Path:
    output_dir = Path.cwd() / "book_output" / "skill_technique_reviews" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "skill_technique_review.md"
    output_path.write_text(markdown + "\n", encoding="utf-8")
    return output_path


def build_status(target: str, artifacts: list[ReviewArtifact], output_path: Path) -> str:
    artifact_lines = "\n".join(f"- {item.label}: {item.path}" for item in artifacts) or "- No artifacts loaded."
    return (
        "[OK] Skill / Technique review built.\n"
        f"Target: {target}\n"
        f"Artifacts:\n{artifact_lines}\n"
        f"Review: {output_path}"
    )


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n[...review preview truncated...]\n\n{tail}"


def mark_exists(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def to_positive_int(value: float | int | None, maximum: int = 2_000_000) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    # Clamp to a generous ceiling so an accidental/huge char limit cannot trigger
    # multi-megabyte string allocations downstream. Well above any realistic UI input.
    return min(number, maximum)
