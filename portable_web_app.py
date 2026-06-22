"""Portable entry point that opens the local browser for the bundled app."""
import threading
import webbrowser

from web_app import app, _ensure_proxy_running


def main() -> None:
    _ensure_proxy_running()
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
