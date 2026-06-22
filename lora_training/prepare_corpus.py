from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".yaml",
    ".yml",
    ".epub",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "book_output",
    "dist",
    "build",
    "lora_output",
    "runs",
}

EXCLUDED_RELATIVE_PREFIXES = {
    ("lora_training", "data"),
    ("lora_training", "lora_output"),
    ("lora_training", "runs"),
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.example",
    ".gitignore",
    "LORA_TRAINING.md",
    "README.md",
    "requirements-lora.txt",
    "requirements.txt",
}

SYSTEM_PROMPT = (
    "你是一位專業的繁體中文小說寫手。請維持自然、連貫、具有畫面感的敘事，"
    "避免條列式說明，專注於人物、場景、情緒與劇情推進。"
)

USER_PROMPT = "請依照訓練文本的文風，創作或續寫一段繁體中文小說正文。"


def get_opencc_converter():
    try:
        from opencc import OpenCC  # type: ignore

        return OpenCC("s2twp")
    except Exception:
        return None


def score_decoded_text(text: str) -> int:
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    bad_markers = ["�", "锟斤拷", "Ã", "Â", "", "", "", "", "蝣", "隤", "銝"]
    penalty = sum(text.count(marker) * 30 for marker in bad_markers)
    control_penalty = sum(1 for char in text if ord(char) < 32 and char not in "\r\n\t")
    return cjk * 3 + ascii_letters - penalty - control_penalty * 10


def decode_bytes(raw: bytes) -> tuple[str, str]:
    candidates: list[tuple[int, str, str]] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "cp950"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        candidates.append((score_decoded_text(text), encoding, text))

    if not candidates:
        text = raw.decode("utf-8", errors="replace")
        return text, "utf-8-replace"

    _, encoding, text = max(candidates, key=lambda item: item[0])
    return text, encoding


def normalize_text(text: str, converter, convert_to_traditional: bool) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()

    if convert_to_traditional and converter is not None:
        text = converter.convert(text)

    return text


def strip_html(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "\n", markup)
    markup = re.sub(r"(?i)<br\s*/?>", "\n", markup)
    markup = re.sub(r"(?i)</(p|div|section|article|h[1-6]|li|tr)>", "\n", markup)
    markup = re.sub(r"(?s)<[^>]+>", "", markup)
    text = html.unescape(markup)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def epub_spine_members(zf: zipfile.ZipFile) -> list[str]:
    names = set(zf.namelist())
    try:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        opf_path = next(
            element.attrib["full-path"]
            for element in container.iter()
            if _tag_name(element) == "rootfile" and "full-path" in element.attrib
        )
        opf_root = ET.fromstring(zf.read(opf_path))
        opf_dir = posixpath.dirname(opf_path)

        manifest: dict[str, str] = {}
        for element in opf_root.iter():
            if _tag_name(element) == "item" and "id" in element.attrib and "href" in element.attrib:
                manifest[element.attrib["id"]] = element.attrib["href"]

        ordered: list[str] = []
        for element in opf_root.iter():
            if _tag_name(element) == "itemref" and "idref" in element.attrib:
                href = manifest.get(element.attrib["idref"])
                if not href:
                    continue
                member = posixpath.normpath(posixpath.join(opf_dir, href))
                if member in names:
                    ordered.append(member)
        if ordered:
            return ordered
    except Exception:
        pass

    return sorted(
        name
        for name in names
        if name.lower().endswith((".xhtml", ".html", ".htm"))
        and "nav" not in posixpath.basename(name).lower()
        and "toc" not in posixpath.basename(name).lower()
    )


def extract_epub_text(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for member in epub_spine_members(zf):
            try:
                raw = zf.read(member)
            except KeyError:
                continue
            text, _ = decode_bytes(raw)
            stripped = strip_html(text)
            if stripped:
                parts.append(stripped)
    return "\n\n".join(parts)


def read_source_text(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".epub":
        return extract_epub_text(path), "epub"
    raw = path.read_bytes()
    return decode_bytes(raw)


def should_include(path: Path, source_root: Path, include_code: bool) -> bool:
    relative_parts = path.relative_to(source_root).parts
    relative_parts_lower = tuple(part.lower() for part in relative_parts)
    for prefix in EXCLUDED_RELATIVE_PREFIXES:
        if relative_parts_lower[: len(prefix)] == prefix:
            return False
    if any(part in EXCLUDED_DIR_NAMES for part in relative_parts[:-1]):
        return False
    if path.name in EXCLUDED_FILE_NAMES:
        return False
    if path.name.startswith("."):
        return False
    if include_code and path.suffix.lower() in TEXT_EXTENSIONS | {".py"}:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def iter_source_files(source_root: Path, include_code: bool) -> Iterable[Path]:
    for path in source_root.rglob("*"):
        if path.is_file() and should_include(path, source_root, include_code):
            yield path


def chunk_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current and current_len >= min_chars:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if len(piece) >= min_chars:
                    chunks.append(piece)
            continue

        if current_len + len(paragraph) + 2 > max_chars and current:
            joined = "\n\n".join(current).strip()
            if len(joined) >= min_chars:
                chunks.append(joined)
            current = []
            current_len = 0

        current.append(paragraph)
        current_len += len(paragraph) + 2

    joined = "\n\n".join(current).strip()
    if len(joined) >= min_chars:
        chunks.append(joined)

    return chunks


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_output_name(index: int, relative: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", relative).strip("._ ")
    cleaned = cleaned or f"source_{index:04d}"
    if len(cleaned) > 140:
        cleaned = cleaned[-140:]
    return f"{index:04d}_{cleaned}.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean Chinese novel text and build LoRA SFT data.")
    parser.add_argument("--source", type=Path, default=Path.cwd(), help="Folder containing source text files.")
    parser.add_argument("--out", type=Path, default=Path("lora_training/data"), help="Output data folder.")
    parser.add_argument("--max-chars", type=int, default=2200, help="Maximum characters per training sample.")
    parser.add_argument("--min-chars", type=int, default=300, help="Minimum characters per training sample.")
    parser.add_argument("--include-code", action="store_true", help="Also include .py files. Usually not recommended.")
    parser.add_argument("--no-traditional", action="store_true", help="Do not convert Simplified Chinese to Traditional Chinese.")
    parser.add_argument(
        "--no-cleaned-copy",
        action="store_true",
        help="Do not write per-source cleaned copies. Useful when the source folder is already cleaned_files.",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = get_opencc_converter()
    convert_to_traditional = not args.no_traditional
    if convert_to_traditional and converter is None:
        print("[WARN] opencc is not installed. Text will be decoded but not converted to Traditional Chinese.")

    records: list[dict] = []
    clean_sections: list[str] = []
    included_sources: list[dict] = []
    files = list(iter_source_files(source_root, args.include_code))
    clean_files_dir = output_dir / "cleaned_files"
    if not args.no_cleaned_copy:
        clean_files_dir.mkdir(parents=True, exist_ok=True)

    for file_index, path in enumerate(files, 1):
        text, encoding = read_source_text(path)
        text = normalize_text(text, converter, convert_to_traditional)
        if not text or len(text) < args.min_chars:
            continue

        relative = str(path.relative_to(source_root))
        cleaned_file = None
        if not args.no_cleaned_copy:
            cleaned_file = clean_files_dir / safe_output_name(file_index, relative)
            cleaned_file.write_text(text + "\n", encoding="utf-8-sig")
        included_sources.append(
            {
                "source": relative,
                "decoded_as": encoding,
                "chars": len(text),
                "cleaned_file": str(cleaned_file) if cleaned_file is not None else str(path),
            }
        )
        clean_sections.append(f"\n\n===== {relative} | decoded_as={encoding} =====\n\n{text}")
        for chunk in chunk_text(text, args.max_chars, args.min_chars):
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        {"role": "assistant", "content": chunk},
                    ],
                    "source": relative,
                    "decoded_as": encoding,
                    "chars": len(chunk),
                }
            )

    clean_corpus_path = output_dir / "clean_corpus.txt"
    train_path = output_dir / "train.jsonl"
    report_path = output_dir / "dataset_report.json"

    clean_corpus_path.write_text("".join(clean_sections).strip() + "\n", encoding="utf-8-sig")
    write_jsonl(train_path, records)
    report = {
        "source": str(source_root),
        "files_scanned": len(files),
        "samples": len(records),
        "convert_to_traditional": convert_to_traditional and converter is not None,
        "include_code": args.include_code,
        "clean_corpus": str(clean_corpus_path),
        "train_jsonl": str(train_path),
        "cleaned_files_dir": None if args.no_cleaned_copy else str(clean_files_dir),
        "cleaned_copy_skipped": args.no_cleaned_copy,
        "included_sources": included_sources,
        "note": (
            "If samples is very small, add real novel .txt/.md/.epub files under this project "
            "and rerun prepare_lora_data.bat before training."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    print(f"Scanned files: {len(files)}")
    print(f"Training samples: {len(records)}")
    print(f"Clean corpus: {clean_corpus_path}")
    print(f"Train JSONL: {train_path}")
    print(f"Report: {report_path}")
    if len(records) < 20:
        print("[WARN] Very few samples. This is not enough for a useful novel-writing LoRA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
