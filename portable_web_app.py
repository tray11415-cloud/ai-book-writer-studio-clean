"""Portable entry point that opens the local browser for the bundled app."""
import threading
import time
import webbrowser

from web_app import _ensure_proxy_running, _is_port_open, app

HOST = "127.0.0.1"
PORT = 5000


def _open_browser_when_ready() -> None:
    """Wait for the Flask server to accept connections, then open the browser.

    Polling avoids the previous race where a fixed 1.5s delay could fire before a
    slow machine had the server listening, leaving the user on a connection error.
    """
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _is_port_open(HOST, PORT):
            break
        time.sleep(0.2)
    webbrowser.open(f"http://{HOST}:{PORT}/")


def main() -> None:
    _ensure_proxy_running()
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()
