"""Local web UI for conversational story generation."""
import logging
import os
import socket
import threading
import time
import uuid
from typing import Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request, session
from openai import OpenAI

from compat_proxy import HOST as PROXY_HOST
from compat_proxy import PORT as PROXY_PORT
from compat_proxy import main as run_compat_proxy
from config import get_config
from env_utils import get_dotenv_path
from lora_runtime import ensure_lora_server_running, is_lora_base_url
import repetition_guard as rg


load_dotenv(get_dotenv_path())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("web_app")

app = Flask(__name__)
# Never hardcode a secret: read from env, fall back to a random per-process key.
# The fallback is safe here because session state (CHAT_SESSIONS) is in-memory and
# ephemeral; production deployments should set FLASK_SECRET_KEY to a persistent value.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
# Harden session cookies. SESSION_COOKIE_SECURE is opt-in (env) so local HTTP dev keeps
# working; HttpOnly + SameSite=Lax are always safe and mitigate XSS/CSRF cookie theft.
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_SESSION_COOKIE_SECURE", "0").strip().lower() in {
    "1",
    "true",
    "on",
    "yes",
}
# Cap request body size to avoid memory-exhaustion DoS via huge JSON payloads.
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("BOOK_WRITER_WEB_MAX_BODY_BYTES", str(5 * 1024 * 1024)))

CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}
# Last-access timestamp per session, used to prune stale in-memory histories.
SESSION_LAST_ACCESS: Dict[str, float] = {}
_SESSIONS_LOCK = threading.Lock()

WEB_CHAT_MAX_TOKENS = int(os.getenv("BOOK_WRITER_CHAT_MAX_TOKENS", "4000"))
# Max characters accepted in a single user message (DoS / token-explosion guard).
WEB_CHAT_MAX_MESSAGE_CHARS = int(os.getenv("BOOK_WRITER_WEB_MAX_MESSAGE_CHARS", "8000"))
# Sliding context window (characters) for the raw chat history sent to the model, so
# context cannot grow unbounded as the story lengthens.
WEB_CHAT_CONTEXT_WINDOW = int(os.getenv("BOOK_WRITER_CHAT_CONTEXT_WINDOW", "8000"))
# Hard cap on retained messages per session (after which oldest turns are dropped).
WEB_CHAT_MAX_HISTORY_MESSAGES = int(os.getenv("BOOK_WRITER_WEB_MAX_HISTORY_MESSAGES", "60"))
# Idle sessions older than this (seconds) are pruned from memory.
WEB_SESSION_TTL_SECONDS = int(os.getenv("BOOK_WRITER_WEB_SESSION_TTL_SECONDS", str(24 * 3600)))
# Token-level repetition penalties (env-configurable), forwarded to the completion.
WEB_CHAT_FREQUENCY_PENALTY = float(os.getenv("BOOK_WRITER_WEB_FREQUENCY_PENALTY", "0.5"))
WEB_CHAT_PRESENCE_PENALTY = float(os.getenv("BOOK_WRITER_WEB_PRESENCE_PENALTY", "0.6"))

SYSTEM_PROMPT = """You are a professional long-form fiction co-writer.
Work with the user through chat to create a story scene by scene.

Rules:
1. Preserve continuity across plot, worldbuilding, and character relationships.
2. If the user adds setting notes, absorb them and continue the story naturally.
3. Default to Simplified Chinese when the user writes in Chinese.
4. Write with concrete imagery, emotional texture, and scene detail.
5. If the user says "continue", continue seamlessly without repeating summaries.
6. Unless the user asks for an outline, default to writing story prose.
7. If key details are missing, make reasonable creative assumptions without stopping the flow."""

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Book Writer Web</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --panel: #fffaf2;
      --ink: #2e241c;
      --muted: #7a6756;
      --accent: #9a3412;
      --accent-2: #c2410c;
      --border: #e7d8c7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(194,65,12,.12), transparent 28%),
        radial-gradient(circle at bottom right, rgba(154,52,18,.10), transparent 24%),
        var(--bg);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 40px;
    }
    .hero {
      padding: 20px 0 14px;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(30px, 5vw, 52px);
      line-height: 1.02;
      letter-spacing: .02em;
    }
    .hero p {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 15px;
    }
    .layout {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 18px;
      align-items: start;
    }
    .card {
      background: rgba(255,250,242,.92);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: 0 10px 30px rgba(46,36,28,.06);
      backdrop-filter: blur(8px);
    }
    .side, .chat {
      overflow: hidden;
    }
    .side-inner, .chat-inner {
      padding: 18px;
    }
    .label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--border);
      background: #fffdf9;
      color: var(--ink);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      outline: none;
    }
    textarea {
      resize: vertical;
      min-height: 120px;
    }
    .actions, .quick-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      font: inherit;
      cursor: pointer;
      transition: transform .15s ease, opacity .15s ease, background .15s ease;
    }
    button:hover { transform: translateY(-1px); }
    button.primary {
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
    }
    button.secondary {
      background: #efe2d2;
      color: var(--ink);
    }
    .chat-window {
      min-height: 560px;
      max-height: 70vh;
      overflow: auto;
      padding-right: 6px;
    }
    .msg {
      padding: 14px 16px;
      border-radius: 18px;
      margin-bottom: 12px;
      white-space: pre-wrap;
      line-height: 1.7;
      animation: fadeIn .25s ease;
    }
    .msg.user {
      background: #efe2d2;
      margin-left: 12%;
    }
    .msg.assistant {
      background: #fff;
      border: 1px solid var(--border);
      margin-right: 12%;
    }
    .msg.system {
      background: #fbf3e7;
      color: var(--muted);
      border: 1px dashed var(--border);
    }
    .composer {
      border-top: 1px solid var(--border);
      padding: 16px 18px 18px;
      background: rgba(255,250,242,.96);
    }
    .composer textarea {
      min-height: 110px;
    }
    .status {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }
    .hint {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.6;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      .msg.user, .msg.assistant { margin-left: 0; margin-right: 0; }
      .chat-window { min-height: 420px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>AI Book Writer</h1>
      <p>Build your novel like a conversation. Add setting, characters, world rules, or just ask it to continue the story.</p>
    </section>

    <section class="layout">
      <aside class="card side">
        <div class="side-inner">
          <label class="label" for="starter">Quick Start</label>
          <textarea id="starter" placeholder="Example: Write a modern urban mystery. The heroine is a forensic doctor, and chapter one begins in a morgue at midnight."></textarea>
          <div class="actions">
            <button class="primary" id="startBtn">Start Story</button>
            <button class="secondary" id="resetBtn">Reset Chat</button>
          </div>
          <div class="quick-actions">
            <button class="secondary quick" data-text="Continue writing and keep the current tone, style, and plot momentum.">Continue</button>
            <button class="secondary quick" data-text="Rewrite this with stronger atmosphere, richer setting detail, and more inner emotion.">More Atmosphere</button>
            <button class="secondary quick" data-text="Push the plot forward and increase tension and conflict.">Push Plot</button>
            <button class="secondary quick" data-text="Summarize the current character dynamics and major clues, then continue the story.">Summarize Clues</button>
          </div>
          <p class="hint">A good quick start includes genre, protagonist, world rules, desired tone, and opening situation. After that, keep steering the story in the chat panel.</p>
        </div>
      </aside>

      <main class="card chat">
        <div class="chat-inner">
          <div id="chatWindow" class="chat-window">
            <div class="msg system">Your story chat will appear here. Start with a setup on the left, then continue the story through conversation.</div>
          </div>
        </div>
        <div class="composer">
          <label class="label" for="message">Chat Input</label>
          <textarea id="message" placeholder="Example: Make the leads confront each other for the first time in this chapter."></textarea>
          <div class="actions">
            <button class="primary" id="sendBtn">Send</button>
          </div>
          <div class="status" id="status"></div>
        </div>
      </main>
    </section>
  </div>

  <script>
    const chatWindow = document.getElementById('chatWindow');
    const statusEl = document.getElementById('status');
    const messageInput = document.getElementById('message');
    const starterInput = document.getElementById('starter');

    function addMessage(role, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      chatWindow.appendChild(div);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function setStatus(text) {
      statusEl.textContent = text || '';
    }

    async function sendMessage(text) {
      if (!text || !text.trim()) return;
      addMessage('user', text.trim());
      setStatus('Generating...');
      messageInput.value = '';

      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text.trim() })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        addMessage('assistant', data.reply);
        setStatus('Done');
      } catch (error) {
        addMessage('system', 'Error: ' + error.message);
        setStatus('Failed');
      }
    }

    async function resetChat() {
      await fetch('/api/reset', { method: 'POST' });
      chatWindow.innerHTML = '';
      addMessage('system', 'Chat reset. You can start a new story now.');
      setStatus('');
    }

    document.getElementById('sendBtn').addEventListener('click', () => sendMessage(messageInput.value));
    document.getElementById('startBtn').addEventListener('click', () => sendMessage(starterInput.value));
    document.getElementById('resetBtn').addEventListener('click', resetChat);

    document.querySelectorAll('.quick').forEach(btn => {
      btn.addEventListener('click', () => sendMessage(btn.dataset.text));
    });

    messageInput.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        sendMessage(messageInput.value);
      }
    });
  </script>
</body>
</html>"""


def _prune_stale_sessions(now: float) -> None:
    """Drop in-memory histories for sessions idle longer than the TTL."""
    if WEB_SESSION_TTL_SECONDS <= 0:
        return
    cutoff = now - WEB_SESSION_TTL_SECONDS
    stale = [sid for sid, ts in SESSION_LAST_ACCESS.items() if ts < cutoff]
    for sid in stale:
        CHAT_SESSIONS.pop(sid, None)
        SESSION_LAST_ACCESS.pop(sid, None)


def _touch_session(session_id: str) -> None:
    """Record last access and opportunistically clean up expired sessions."""
    now = time.time()
    with _SESSIONS_LOCK:
        SESSION_LAST_ACCESS[session_id] = now
        _prune_stale_sessions(now)


def _get_session_id() -> str:
    session_id = session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["chat_session_id"] = session_id
    return session_id


def _trim_text(text: str, max_chars: int) -> str:
    """Keep the most recent ``max_chars`` characters (mirrors app_gradio.trim_text)."""
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _window_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Bound the chat history sent to the model so context cannot grow unbounded.

    Keeps the most recent messages within both a message-count cap and a character
    budget, preserving the latest user turn.
    """
    windowed = history[-WEB_CHAT_MAX_HISTORY_MESSAGES:] if WEB_CHAT_MAX_HISTORY_MESSAGES > 0 else list(history)
    if WEB_CHAT_CONTEXT_WINDOW <= 0:
        return windowed
    kept: List[Dict[str, str]] = []
    used = 0
    # Walk newest-first, accumulating until the character budget is hit; always keep
    # at least the most recent message.
    for msg in reversed(windowed):
        size = len(msg.get("content") or "")
        if kept and used + size > WEB_CHAT_CONTEXT_WINDOW:
            break
        kept.append(msg)
        used += size
    kept.reverse()
    return kept


def _build_story_so_far(history: List[Dict[str, str]]) -> str:
    """Assemble the prior story prose from assistant messages for the repetition guard."""
    parts = [
        (m.get("content") or "").strip()
        for m in history
        if m.get("role") == "assistant" and (m.get("content") or "").strip()
    ]
    return "\n\n".join(parts)


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _ensure_proxy_running() -> None:
    agent_config = get_config()
    base_url = agent_config["config_list"][0]["base_url"]
    if is_lora_base_url(base_url):
        ensure_lora_server_running()
        return

    if _is_port_open(PROXY_HOST, PROXY_PORT):
        return

    thread = threading.Thread(target=run_compat_proxy, daemon=True)
    thread.start()


def _build_client() -> OpenAI:
    agent_config = get_config()
    llm = agent_config["config_list"][0]
    return OpenAI(
        api_key=llm["api_key"] or "not-needed",
        base_url=llm["base_url"],
        timeout=900,
    )


def _generate_story_reply(history: List[Dict[str, str]], extra_avoid_block: str = "") -> str:
    agent_config = get_config()
    model = agent_config["config_list"][0]["model"]
    client = _build_client()

    system_content = SYSTEM_PROMPT
    if extra_avoid_block.strip():
        system_content = SYSTEM_PROMPT + "\n\n" + extra_avoid_block.strip()

    # Window the raw chat history so the context sent to the model stays bounded
    # regardless of how long the conversation grows.
    windowed = _window_history(history)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_content}, *windowed],
        temperature=0.85,
        max_tokens=WEB_CHAT_MAX_TOKENS,
        frequency_penalty=WEB_CHAT_FREQUENCY_PENALTY,
        presence_penalty=WEB_CHAT_PRESENCE_PENALTY,
    )
    return response.choices[0].message.content or ""


@app.before_request
def before_request() -> None:
    _ensure_proxy_running()


@app.get("/")
def index():
    return HTML_PAGE


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    raw_message = payload.get("message")
    if not isinstance(raw_message, str):
        return jsonify({"error": "Message must be text."}), 400
    message = raw_message.strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400
    if len(message) > WEB_CHAT_MAX_MESSAGE_CHARS:
        return (
            jsonify({"error": f"Message too long (max {WEB_CHAT_MAX_MESSAGE_CHARS} characters)."}),
            413,
        )

    session_id = _get_session_id()
    _touch_session(session_id)
    history = CHAT_SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": message})
    # Keep the retained history bounded so memory per session cannot grow without limit.
    if WEB_CHAT_MAX_HISTORY_MESSAGES > 0 and len(history) > WEB_CHAT_MAX_HISTORY_MESSAGES:
        del history[: len(history) - WEB_CHAT_MAX_HISTORY_MESSAGES]

    try:
        # --- cross-response repetition guard + long-form memory ---------------
        # Mine overused phrasings from the assembled story-so-far, add an
        # anti-repetition directive, and regenerate if the continuation recycles
        # too much of the prior prose (mirrors app_gradio.py).
        current_story = _build_story_so_far(history)
        guard_on = rg.GUARD_ENABLED and bool(current_story.strip())
        overused = rg.extract_overused_phrases(current_story) if guard_on else []
        # Trim the story-so-far for ratio/longform comparisons so the guard work also
        # stays bounded on very long conversations.
        story_for_guard = _trim_text(current_story, max(WEB_CHAT_CONTEXT_WINDOW * 4, 1)) if guard_on else ""
        directive_cjk = not any(
            "english" in str(h.get("content", "")).lower() for h in history if h.get("role") == "user"
        )

        avoid_block = rg.build_avoid_directive(overused, cjk=directive_cjk) if guard_on else ""
        reply = _generate_story_reply(history, avoid_block)

        if guard_on:
            best_reply = reply
            best_ratio = rg.repetition_ratio(reply, story_for_guard)
            attempts = 0
            while best_ratio > rg.RATIO_THRESHOLD and attempts < rg.MAX_RETRIES:
                attempts += 1
                recycled = rg.repeated_spans(reply, story_for_guard)
                retry_block = rg.build_avoid_directive(overused, recycled, cjk=directive_cjk)
                reply = _generate_story_reply(history, retry_block)
                ratio = rg.repetition_ratio(reply, story_for_guard)
                if ratio < best_ratio:
                    best_reply, best_ratio = reply, ratio
            reply = best_reply

        history.append({"role": "assistant", "content": reply})
        if WEB_CHAT_MAX_HISTORY_MESSAGES > 0 and len(history) > WEB_CHAT_MAX_HISTORY_MESSAGES:
            del history[: len(history) - WEB_CHAT_MAX_HISTORY_MESSAGES]
        return jsonify({"reply": reply})
    except Exception:  # noqa: BLE001
        # Log full detail server-side; never leak raw exception text to the client.
        request_id = uuid.uuid4().hex[:12]
        logger.exception("Error in /api/chat (request_id=%s)", request_id)
        return (
            jsonify(
                {
                    "error": "An error occurred while generating the story. Please try again.",
                    "request_id": request_id,
                }
            ),
            500,
        )


@app.post("/api/reset")
def reset():
    session_id = _get_session_id()
    # Actually free the in-memory history on reset rather than leaving an empty list.
    with _SESSIONS_LOCK:
        CHAT_SESSIONS.pop(session_id, None)
        SESSION_LAST_ACCESS.pop(session_id, None)
    return jsonify({"ok": True})


def main() -> None:
    _ensure_proxy_running()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
