"""Local web UI for conversational story generation."""
import os
import socket
import threading
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


load_dotenv(get_dotenv_path())

app = Flask(__name__)
app.secret_key = "ai-book-writer-web-secret"

CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}
WEB_CHAT_MAX_TOKENS = int(os.getenv("BOOK_WRITER_CHAT_MAX_TOKENS", "4000"))

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


def _get_session_id() -> str:
    session_id = session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["chat_session_id"] = session_id
    return session_id


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


def _generate_story_reply(history: List[Dict[str, str]]) -> str:
    agent_config = get_config()
    model = agent_config["config_list"][0]["model"]
    client = _build_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history],
        temperature=0.85,
        max_tokens=WEB_CHAT_MAX_TOKENS,
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
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    session_id = _get_session_id()
    history = CHAT_SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": message})

    try:
        reply = _generate_story_reply(history)
        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/reset")
def reset():
    session_id = _get_session_id()
    CHAT_SESSIONS[session_id] = []
    return jsonify({"ok": True})


def main() -> None:
    _ensure_proxy_running()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
