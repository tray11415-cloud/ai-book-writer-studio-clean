"""Gradio UI for the full-text rewrite agent."""
from __future__ import annotations

import logging
import queue
import socket
import threading
from pathlib import Path
from typing import Callable, Iterator

import gradio as gr

logger = logging.getLogger(__name__)

# How long (seconds) the UI waits for a single progress message before deciding the
# worker has hung. Generous because a single long manuscript window can take minutes.
_QUEUE_TIMEOUT_SECONDS = 1800
# Defensive upper bound on joining the finished worker thread.
_JOIN_TIMEOUT_SECONDS = 30

from rewrite_agent import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DZMM_MODELS,
    INPUT_DIR,
    LORA_BASE_URL,
    LORA_MODEL_NAME,
    MODE_HYBRID,
    MODES,
    OP_REWRITE,
    OPERATIONS,
    RewriteSettings,
    STRENGTHS,
    full_rewrite,
    load_reference_text,
    read_text_file,
)
from analysis_agent import (
    AnalysisSettings,
    analyze_manuscript,
    find_latest_diagnosis,
    grok_defaults,
    load_diagnosis,
)


def can_bind_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def get_gradio_port(host: str, default_port: int = 7870) -> int:
    for port in range(default_port, 7900):
        if can_bind_port(host, port):
            return port
    raise OSError("No empty Gradio port found in range 7870-7899.")


def _file_path(file_obj) -> str | None:
    if file_obj is None:
        return None
    path = str(getattr(file_obj, "name", file_obj)).strip()
    if not path:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(f"找不到檔案或無法讀取：{path}")
    return path


def _safe_error(exc: BaseException) -> str:
    """User-facing one-liner for an exception; full detail is logged server-side."""
    logger.exception("Worker failed", exc_info=exc)
    detail = str(exc).strip().splitlines()
    brief = detail[0] if detail else exc.__class__.__name__
    # Keep it short so we never leak stack traces / API payloads into the UI.
    return brief[:200]


# Sentinel pushed onto the progress queue when the worker thread finishes.
_DONE = object()


def _stream_worker(work: Callable[[Callable[[str], None]], object], state: dict) -> Iterator[list[str]]:
    """Run `work` in a daemon thread, yielding accumulated status lines as they arrive.

    `work` receives a `report(msg)` callback and returns its result; on success the
    result is stored in state["result"], on failure the exception in state["error"].
    Yields the growing list of status lines. Uses bounded waits so the UI cannot
    freeze forever if the worker hangs or dies without signalling completion.
    """
    progress_queue: queue.Queue = queue.Queue()

    def report(message: str) -> None:
        progress_queue.put(message)

    def worker() -> None:
        try:
            state["result"] = work(report)
        except Exception as exc:  # noqa: BLE001
            state["error"] = exc
        finally:
            progress_queue.put(_DONE)

    # daemon=True: if the interpreter is shutting down we don't want a lingering
    # network call to block process exit. The bounded waits below handle the UX.
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    status_lines: list[str] = []
    while True:
        try:
            message = progress_queue.get(timeout=_QUEUE_TIMEOUT_SECONDS)
        except queue.Empty:
            if not worker_thread.is_alive():
                # Worker died without enqueuing _DONE; surface whatever we have.
                break
            logger.error("Worker produced no progress for %ss; aborting wait.", _QUEUE_TIMEOUT_SECONDS)
            state.setdefault("error", TimeoutError("作業逾時，未在預期時間內回應。"))
            break
        if message is _DONE:
            break
        status_lines.append(str(message))
        yield status_lines
    worker_thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
    state["status_lines"] = status_lines


def run_rewrite(
    target_file,
    target_text,
    instruction,
    continuity_notes,
    style_files,
    style_text,
    mode,
    rewrite_strength,
    operation,
    output_language,
    chunk_chars,
    max_tokens,
    temperature,
    top_p,
    repetition_penalty,
    use_story_context,
    chapter_continuity,
    apply_diagnosis,
    dedup_similarity,
    api_key,
    api_base_url,
    api_model,
):
    source_text = ""
    try:
        target_path = _file_path(target_file)
    except FileNotFoundError as exc:
        yield _safe_error(exc), "", None
        return
    if target_path:
        source_text = read_text_file(target_path)
    elif target_text and target_text.strip():
        source_text = target_text

    if not source_text.strip():
        yield "請先上傳 txt/md 檔，或貼上要改寫的全文。", "", None
        return

    # Optionally apply the most recent Grok diagnosis (Phase 1 output).
    diag_spine_seed = ""
    diag_brief = ""
    diag_window_problems: list[str] = []
    if apply_diagnosis:
        latest = find_latest_diagnosis()
        if latest is None:
            yield "找不到診斷書。請先在上方「① 全篇分析診斷（Grok）」按開始分析，再勾選套用。", "", None
            return
        try:
            diagnosis = load_diagnosis(latest)
        except Exception as exc:  # noqa: BLE001
            yield _safe_error(exc), "", None
            return
        diag_spine_seed = diagnosis.spine_seed
        diag_brief = diagnosis.rewrite_brief()
        diag_window_problems = diagnosis.window_problems

    try:
        style_paths = [_file_path(file_obj) for file_obj in (style_files or [])]
    except FileNotFoundError as exc:
        yield _safe_error(exc), "", None
        return
    style_paths = [path for path in style_paths if path]
    style_reference = "\n\n".join(part for part in [style_text.strip(), load_reference_text(style_paths)] if part)

    settings = RewriteSettings(
        mode=mode,
        instruction=instruction,
        rewrite_strength=rewrite_strength,
        operation=operation,
        output_language=output_language,
        api_key=api_key or "not-needed",
        api_base_url=api_base_url,
        api_model=api_model,
        chunk_chars=int(chunk_chars),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
        style_reference=style_reference,
        continuity_notes=continuity_notes,
        use_story_context=bool(use_story_context),
        chapter_continuity=bool(chapter_continuity),
        dedup_similarity=float(dedup_similarity),
        diagnosis_spine_seed=diag_spine_seed,
        diagnosis_brief=diag_brief,
        diagnosis_window_problems=diag_window_problems,
    )

    # Run the (blocking) rewrite in a worker thread and stream progress lines to
    # the UI through a queue, so the user sees per-chunk progress live instead of
    # a frozen screen until the whole manuscript is done.
    yield "準備開始。第一次本地 LoRA 可能會先載入模型，畫面會等比較久。", "", None

    state: dict = {}
    for status_lines in _stream_worker(
        lambda report: full_rewrite(source_text=source_text, settings=settings, progress=report),
        state,
    ):
        yield "\n".join(status_lines), "", None

    status_lines = state.get("status_lines", [])
    if state.get("error") is not None:
        yield "\n".join(status_lines + [f"失敗：{_safe_error(state['error'])}"]), "", None
        return

    result = state["result"]
    status = "\n".join(
        status_lines
        + [
            f"完成。共 {result.chunks_total} 段。",
            f"輸出：{result.output_path}",
            f"紀錄：{result.log_path}",
            f"QA：{result.report_path}",
        ]
    )
    yield status, result.output_text, str(result.output_path)


def run_analysis(
    target_file,
    target_text,
    analysis_api_key,
    analysis_base_url,
    analysis_model,
    window_chars,
    analysis_instruction,
):
    source_text = ""
    try:
        target_path = _file_path(target_file)
    except FileNotFoundError as exc:
        yield _safe_error(exc), ""
        return
    if target_path:
        source_text = read_text_file(target_path)
    elif target_text and target_text.strip():
        source_text = target_text

    if not source_text.strip():
        yield "請先上傳 txt/md 檔，或貼上要分析的全文。", ""
        return
    if not (analysis_api_key or "").strip():
        yield "缺少 Grok（XAI）API 金鑰。請設定環境變數 XAI_API_KEY，或填入上方欄位。", ""
        return

    # Validate window_chars against the same bounds the UI slider enforces, so a
    # direct/out-of-range call gets a clear message instead of silent clamping.
    try:
        window_chars_int = int(window_chars)
    except (TypeError, ValueError):
        yield "分析視窗字數必須是數字（建議 2000–12000）。", ""
        return
    if not (2000 <= window_chars_int <= 12000):
        yield "分析視窗字數需介於 2000 至 12000 之間。", ""
        return

    settings = AnalysisSettings(
        api_key=analysis_api_key,
        base_url=analysis_base_url,
        model=analysis_model,
        analysis_chunk_chars=window_chars_int,
        instruction=analysis_instruction or "",
    )

    yield "Grok 深度分析開始（逐段多專家，長文會花較久）...", ""

    state: dict = {}
    for status_lines in _stream_worker(
        lambda report: analyze_manuscript(source_text=source_text, settings=settings, progress=report),
        state,
    ):
        yield "\n".join(status_lines), ""

    status_lines = state.get("status_lines", [])
    if state.get("error") is not None:
        yield "\n".join(status_lines + [f"分析失敗：{_safe_error(state['error'])}"]), ""
        return

    diagnosis, md_path, _json_path = state["result"]
    report_md = Path(md_path).read_text(encoding="utf-8-sig")
    status = "\n".join(
        status_lines
        + [
            f"分析完成，共 {diagnosis.window_count} 段、{len(diagnosis.problems)} 個問題。",
            f"診斷書：{md_path}",
            "→ 接著在「② 全篇改寫」勾選「套用最新診斷書」再按開始改寫。",
        ]
    )
    yield status, report_md


with gr.Blocks(title="改寫AGENT") as demo:
    gr.Markdown("# 改寫AGENT")

    with gr.Row():
        with gr.Column(scale=3):
            target_file = gr.File(label="改寫目標檔案", file_count="single", file_types=[".txt", ".md"])
            target_text = gr.Textbox(label="或直接貼上全文", lines=16)
            instruction = gr.Textbox(
                label="改寫要求",
                lines=5,
                placeholder="例如：保留劇情，改成更成熟的仙俠小說文風；加強場景描寫與人物心理，減少流水帳。",
            )
            continuity_notes = gr.Textbox(
                label="全篇設定 / 連貫要求",
                lines=4,
                placeholder="角色關係、世界觀、不能改動的設定、稱謂、伏筆等。",
            )
        with gr.Column(scale=2):
            mode = gr.Dropdown(label="使用模式", choices=MODES, value=MODE_HYBRID)
            rewrite_strength = gr.Dropdown(label="改寫強度", choices=STRENGTHS, value="強改寫")
            operation = gr.Dropdown(
                label="操作模式（增補 / 改寫 / 刪減）",
                choices=OPERATIONS,
                value=OP_REWRITE,
            )
            output_language = gr.Textbox(label="輸出語言", value="繁體中文")
            chunk_chars = gr.Slider(label="每段原文字數", minimum=500, maximum=2400, value=900, step=100)
            max_tokens = gr.Slider(label="每段輸出上限 tokens", minimum=300, maximum=2000, value=900, step=100)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.2, value=0.75, step=0.05)
            top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
            repetition_penalty = gr.Slider(
                label="Repetition Penalty（外部 API，DZMM 建議 1.05）",
                minimum=1.0,
                maximum=1.3,
                value=1.05,
                step=0.01,
            )
            use_story_context = gr.Checkbox(
                label="維護跨段落故事脈絡（建議開啟，可降低重複、提升連貫）",
                value=True,
            )
            chapter_continuity = gr.Checkbox(
                label="主敘事模組 + 章節連貫系統（建議開啟，強化章節之間的連貫）",
                value=True,
            )
            dedup_similarity = gr.Slider(
                label="近似重複判定門檻（越低刪越多）",
                minimum=0.80,
                maximum=1.0,
                value=0.9,
                step=0.01,
            )
            api_key = gr.Textbox(label="外部 API Key", value=DEFAULT_API_KEY, type="password")
            api_base_url = gr.Textbox(label="外部 API Base URL（NALANG 走本地代理 127.0.0.1:8000）", value=DEFAULT_BASE_URL)
            api_model = gr.Dropdown(
                label="改寫模型（NALANG／x-apex，可自選）",
                choices=DZMM_MODELS,
                value=DEFAULT_MODEL if DEFAULT_MODEL in DZMM_MODELS else (DZMM_MODELS[0] if DZMM_MODELS else DEFAULT_MODEL),
                allow_custom_value=True,
            )
            gr.Textbox(label="本地 LoRA", value=f"{LORA_MODEL_NAME} @ {LORA_BASE_URL}", interactive=False)

    with gr.Accordion("參考風格", open=False):
        style_files = gr.File(label="參考風格檔案", file_count="multiple", file_types=[".txt", ".md"])
        style_text = gr.Textbox(label="或貼上參考風格片段", lines=8)

    _gk, _gb, _gm = grok_defaults()
    with gr.Accordion("① 全篇分析診斷（Grok）— 先分析、再改寫", open=True):
        gr.Markdown(
            "用 Grok 對上面的全文做深度分析：**全篇分析 / 脈絡診斷（連貫風險）/ 問題發現**，"
            "產出可審閱的「改寫診斷書」。之後在②勾選「套用最新診斷書」，改寫就會自動修正診斷出的問題。"
        )
        with gr.Row():
            analysis_api_key = gr.Textbox(label="Grok API Key (XAI)", value=_gk, type="password")
            analysis_base_url = gr.Textbox(label="Grok Base URL", value=_gb)
            analysis_model = gr.Textbox(label="Grok Model", value=_gm)
        with gr.Row():
            window_chars = gr.Slider(label="分析視窗字數（每段給 Grok 的量）", minimum=2000, maximum=12000, value=5000, step=500)
            analysis_instruction = gr.Textbox(
                label="額外分析關注（選填）",
                placeholder="例如：特別檢查人物稱謂、時間線與設定前後矛盾。",
            )
        analyze_btn = gr.Button("開始分析診斷（Grok）", variant="secondary")
        analysis_status = gr.Textbox(label="分析進度", lines=6)
        analysis_report = gr.Markdown()

    apply_diagnosis = gr.Checkbox(
        label="套用最新診斷書（先按①分析；改寫會 seed 全書骨幹並修正診斷問題）",
        value=False,
    )
    run_btn = gr.Button("② 開始全篇改寫", variant="primary")
    status_output = gr.Textbox(label="進度", lines=12)
    output_text = gr.Textbox(label="改寫結果", lines=24)
    output_file = gr.File(label="下載改寫結果")

    analyze_btn.click(
        run_analysis,
        inputs=[
            target_file,
            target_text,
            analysis_api_key,
            analysis_base_url,
            analysis_model,
            window_chars,
            analysis_instruction,
        ],
        outputs=[analysis_status, analysis_report],
        api_name=False,
    )

    run_btn.click(
        run_rewrite,
        inputs=[
            target_file,
            target_text,
            instruction,
            continuity_notes,
            style_files,
            style_text,
            mode,
            rewrite_strength,
            operation,
            output_language,
            chunk_chars,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            use_story_context,
            chapter_continuity,
            apply_diagnosis,
            dedup_similarity,
            api_key,
            api_base_url,
            api_model,
        ],
        outputs=[status_output, output_text, output_file],
        api_name=False,
    )


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    server_name = "127.0.0.1"
    server_port = get_gradio_port(server_name)
    demo.queue(default_concurrency_limit=1)
    print(f"Starting rewrite agent on http://{server_name}:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=True,
        show_api=False,
        quiet=True,
    )


if __name__ == "__main__":
    main()
