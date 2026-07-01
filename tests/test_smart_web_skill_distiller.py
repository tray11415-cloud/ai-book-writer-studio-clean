from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from smart_web_skill_distiller import (
    build_local_gap_skill,
    discover_and_distill_web_skill,
    parse_duckduckgo_results,
)


def test_parse_duckduckgo_results_unwraps_redirect() -> None:
    html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnovel%2F1">Example Novel</a>
      <a class="result__snippet">short craft snippet</a>
    </div>
    """
    results = parse_duckduckgo_results(html, source="web-search", limit=3)
    assert len(results) == 1
    assert results[0].url == "https://example.com/novel/1"
    assert results[0].source == "web-search"


def test_local_gap_skill_keeps_story_skill_shape() -> None:
    skill = build_local_gap_skill("仙俠慢熱、壓迫感、每段有鉤子", "他停在門外。風很冷。", [])
    assert skill["schema_version"] == 2
    assert skill["narrative_method"]
    assert skill["story_progression"]
    assert skill["plot_ideation"]
    assert len(skill["description_techniques"]) >= 10
    assert "gap_analysis" in skill


def test_discover_dry_run_from_seed_url(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/robots.txt":
                body = b"User-agent: *\nAllow: /\n"
            else:
                body = (
                    "<html><main><p>夜雨落在石階上，主角沒有立刻回答，只用手指按住劍柄。"
                    "遠處鐘聲斷續，讓沉默像一層冷霧壓下來。</p>"
                    "<p>她先退半步，再抬眼看他，話裡留了一個沒有說完的問題。</p></main></html>"
                ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        url = f"http://127.0.0.1:{server.server_port}/chapter"
        status, preview, skill_path, report_path, skill_json = discover_and_distill_web_skill(
            target_description="仙俠壓迫感與含蓄對話",
            target_excerpt="他站在門口，很緊張。",
            seed_urls_text=url,
            source_modes=["Seed URLs only"],
            allowed_domains_text="127.0.0.1",
            distill_goal="找缺失技法並蒸餾成 skill",
            output_language="繁體中文",
            max_search_results=1,
            max_pages_to_fetch=1,
            max_snippet_chars=800,
            dry_run=True,
            analysis_api_key="",
            analysis_base_url="",
            analysis_model_name="",
        )
    finally:
        os.chdir(old_cwd)
        server.shutdown()
        thread.join(timeout=2)

    assert status.startswith("[OK]")
    assert "智慧網路技巧缺口報告" in preview
    assert skill_path is not None and Path(skill_path).is_file()
    assert report_path is not None and Path(report_path).is_file()
    parsed = json.loads(skill_json)
    assert parsed["source_policy"]["stored_full_text"] is False
