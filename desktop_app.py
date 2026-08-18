#!/usr/bin/env python3
"""
Segriotate desktop launcher.

Starts the local Flask server in the background, opens a native window,
and shuts everything down when the window is closed.

    python desktop_app.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402

PORT = getattr(config, "SERVER_PORT", 8765)
BASE = f"http://127.0.0.1:{PORT}"

SPLASH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Segriotate</title>
<style>
  html, body {
    margin: 0; height: 100%;
    background: #14161a; color: #e8e6e1;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; align-items: center; justify-content: center;
  }
  .box { text-align: center; max-width: 420px; padding: 32px; }
  h1 { font-size: 22px; font-weight: 600; letter-spacing: 0.04em; margin: 0 0 8px; }
  p { color: #8b909c; font-size: 14px; line-height: 1.5; margin: 0; }
  .dot {
    width: 10px; height: 10px; border-radius: 50%; background: #5eead4;
    margin: 0 auto 18px; animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.35; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.15); }
  }
</style>
</head>
<body>
  <div class="box">
    <div class="dot"></div>
    <h1>Segriotate</h1>
    <p>Loading segmentation models… this can take a minute the first time.</p>
  </div>
</body>
</html>
"""


def probe_health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=2) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def health_ok() -> bool:
    data = probe_health()
    return bool(data) and data.get("app") == "segriotate"


def run_flask():
    import segment_server

    segment_server.app.run(
        host="127.0.0.1",
        port=PORT,
        threaded=True,
        use_reloader=False,
    )


class Bridge:
    """Called from the HTML window (native folder picker)."""

    def __init__(self):
        self.window = None

    def pick_images_folder(self):
        import webview
        import segment_server

        if self.window is None:
            return {"ok": False, "error": "window not ready"}
        result = self.window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return {"ok": False}
        folder = result[0]
        try:
            segment_server.set_images_dir(folder)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {
            "ok": True,
            "dir": folder,
            "files": segment_server.list_image_files(),
        }


def wait_then_load(window):
    deadline = time.time() + 180
    while time.time() < deadline:
        if health_ok():
            window.load_url(f"{BASE}/")
            return
        time.sleep(0.4)
    window.load_html(
        "<html><body style='background:#14161a;color:#f97362;font-family:sans-serif;"
        "padding:40px'>"
        "<h1>Segriotate failed to start</h1>"
        "<p>The local server did not become ready. Check the terminal for errors, "
        "and make sure port 8765 is free.</p>"
        "</body></html>"
    )


def main():
    try:
        import webview
    except ImportError:
        sys.exit("pywebview is not installed. Run: pip install pywebview")

    already = probe_health()
    if already and already.get("app") != "segriotate":
        sys.exit(
            f"Port {PORT} is already in use (probably an old python scripts/segment_server.py).\n"
            "Stop that process with Ctrl+C, then start Segriotate again."
        )
    already_running = bool(already) and already.get("app") == "segriotate"
    if not already_running:
        threading.Thread(target=run_flask, daemon=True).start()

    bridge = Bridge()
    if already_running:
        window = webview.create_window(
            "Segriotate",
            url=f"{BASE}/",
            js_api=bridge,
            width=1440,
            height=900,
            min_size=(960, 640),
        )
    else:
        window = webview.create_window(
            "Segriotate",
            html=SPLASH_HTML,
            js_api=bridge,
            width=1440,
            height=900,
            min_size=(960, 640),
        )
    bridge.window = window
    if not already_running:
        threading.Thread(target=wait_then_load, args=(window,), daemon=True).start()
    webview.start()


if __name__ == "__main__":
    main()
