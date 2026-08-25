#!/usr/bin/env python3
"""
Segriotate desktop launcher (PyQt).

Starts the local Flask server in the background, opens a native Qt window
with the HTML editor, and shuts everything down when the window is closed.

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

try:
    # WebEngine must be imported before QApplication is created.
    from PyQt6.QtCore import QObject, QStandardPaths, Qt, QTimer, QUrl, pyqtSlot
    from PyQt6.QtGui import QAction, QIcon
    from PyQt6.QtWebChannel import QWebChannel
    from PyQt6.QtWebEngineCore import (
        QWebEnginePage,
        QWebEngineProfile,
        QWebEngineScript,
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QMainWindow,
        QMessageBox,
    )
except ImportError:
    sys.exit("PyQt6 WebEngine is not installed. Run: pip install PyQt6 PyQt6-WebEngine")

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
  h1 { font-size: 22px; font-weight: 600; letter-spacing: 0.02em; margin: 0 0 8px; }
  p { color: #8b909c; font-size: 14px; line-height: 1.5; margin: 0; }
  #status { margin-top: 14px; color: #5eead4; font-size: 13px; min-height: 1.4em; }
  .spinner {
    width: 42px; height: 42px;
    border: 3px solid #2a2e36;
    border-top-color: #5eead4;
    border-radius: 50%;
    margin: 0 auto 22px;
    animation: spin 0.85s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="box">
    <div class="spinner"></div>
    <h1>Launching app.</h1>
    <p>Please wait</p>
    <p id="status">Checking for models…</p>
  </div>
</body>
</html>
"""

FAIL_HTML = """<!DOCTYPE html>
<html lang="en">
<body style="background:#14161a;color:#f97362;font-family:sans-serif;padding:40px">
  <h1>Segriotate failed to start</h1>
  <p>The local server did not become ready. Check the terminal or <code>segriotate.log</code>.
  First launch downloads FastSAM / MobileSAM into <code>models/dot-pt/</code>
  and can build TensorRT engines from those <code>.pt</code> files. That can take several minutes.</p>
</body>
</html>
"""

DESKTOP_FLAG_JS = "window.IS_DESKTOP_APP = true;"

BRIDGE_JS = """
(function connectBridge() {
  if (typeof QWebChannel === "undefined" || typeof qt === "undefined" || !qt.webChannelTransport) {
    setTimeout(connectBridge, 50);
    return;
  }
  new QWebChannel(qt.webChannelTransport, function (channel) {
    window.qtBridge = channel.objects.bridge;
    window.dispatchEvent(new Event("qtbridgeready"));
  });
})();
"""


def web_storage_dir() -> Path:
    base = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation
    )
    root = Path(base) if base else ROOT / ".segriotate"
    path = root / "web"
    path.mkdir(parents=True, exist_ok=True)
    return path


def probe_health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=2) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def run_flask():
    import segment_server

    segment_server.app.run(
        host="127.0.0.1",
        port=PORT,
        threaded=True,
        use_reloader=False,
    )


class Bridge(QObject):
    """Called from the HTML editor (native folder picker)."""

    def __init__(self, window: MainWindow):
        super().__init__(window)
        self._window = window

    @pyqtSlot(result=str)
    def pick_images_folder(self) -> str:
        return self._window.pick_images_folder_json()

    @pyqtSlot(result=str)
    def pick_labels_folder(self) -> str:
        return self._window.pick_labels_folder_json()


class MainWindow(QMainWindow):
    def __init__(self, already_running: bool):
        super().__init__()
        self.setWindowTitle("Segriotate")
        self.resize(1440, 900)
        self.setMinimumSize(960, 640)
        self.setStyleSheet("background: #14161a;")

        self.view = QWebEngineView(self)
        self.view.setStyleSheet("background: #14161a;")
        self.setCentralWidget(self.view)

        # Qt's default profile is off-the-record, so localStorage (where the
        # editor caches its class profiles) is discarded on exit. A named
        # profile with a storage path keeps it on disk. The profile is parented
        # to the application so that it outlives the page it backs.
        storage = web_storage_dir()
        self.profile = QWebEngineProfile("segriotate", QApplication.instance())
        self.profile.setPersistentStoragePath(str(storage))
        self.profile.setCachePath(str(storage / "cache"))
        self.profile.setPersistentCookiesPolicy(
            QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies
        )
        self.view.setPage(QWebEnginePage(self.profile, self))

        self.bridge = Bridge(self)
        self.channel = QWebChannel(self.view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)
        self._inject_bridge_scripts()

        self._make_menu()
        self.view.loadFinished.connect(self._on_load_finished)
        self._labels_chosen = False
        self._last_labels_dir = str(ROOT / "labels")
        self._last_images_dir = str(ROOT)

        self._deadline = time.time() + 3600
        if already_running:
            self.view.setUrl(QUrl(BASE + "/"))
        else:
            self.view.setHtml(SPLASH_HTML)
            self._poll = QTimer(self)
            self._poll.timeout.connect(self._check_server)
            self._poll.start(400)

    def _inject_script(self, name, source_code=None, source_url=None, when=None, subframes=False):
        script = QWebEngineScript()
        script.setName(name)
        if source_url is not None:
            script.setSourceUrl(source_url)
        if source_code is not None:
            script.setSourceCode(source_code)
        script.setInjectionPoint(when)
        script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
        script.setRunsOnSubFrames(subframes)
        self.view.page().scripts().insert(script)

    def _inject_bridge_scripts(self):
        created = QWebEngineScript.InjectionPoint.DocumentCreation
        ready = QWebEngineScript.InjectionPoint.DocumentReady
        self._inject_script(
            "qwebchannel",
            source_url=QUrl("qrc:///qtwebchannel/qwebchannel.js"),
            when=created,
            subframes=True,
        )
        self._inject_script("segriotate-desktop-flag", source_code=DESKTOP_FLAG_JS, when=created)
        self._inject_script("segriotate-bridge", source_code=BRIDGE_JS, when=ready)

    def _make_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        self._open_act = QAction("Open Images…", self)
        self._open_act.setShortcut("Ctrl+O")
        self._open_act.setEnabled(False)
        self._open_act.triggered.connect(self.open_images_from_menu)
        file_menu.addAction(self._open_act)
        self._labels_act = QAction("Choose Labels Folder…", self)
        self._labels_act.setEnabled(False)
        self._labels_act.triggered.connect(self.choose_labels_folder_from_menu)
        file_menu.addAction(self._labels_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(quit_act)

    def _on_load_finished(self, ok):
        url = self.view.url().toString()
        editor_ready = bool(ok) and url.startswith(BASE)
        self._open_act.setEnabled(editor_ready)
        self._labels_act.setEnabled(editor_ready)

    def _choose_directory(self, title: str, start: str | None = None) -> str:
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        # Native dialogs include a New Folder control on macOS/Windows.
        dialog.setDirectory(start or str(ROOT))
        try:
            dialog.setLabelText(QFileDialog.DialogLabel.Accept, "Select")
        except Exception:
            pass
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _check_server(self):
        data = probe_health()
        if data and data.get("app") == "segriotate":
            msg = data.get("message") or "Launching app. Please wait"
            self.view.page().runJavaScript(
                "var e=document.getElementById('status');"
                f"if(e) e.textContent={json.dumps(msg)};"
            )
            if data.get("error"):
                err = data.get("error")
                self._poll.stop()
                self.view.setHtml(
                    FAIL_HTML.replace(
                        "The local server did not become ready.",
                        str(err),
                    )
                )
                return
            if data.get("ready") is True:
                self._poll.stop()
                self.view.setUrl(QUrl(BASE + "/"))
                return
        if time.time() > self._deadline:
            self._poll.stop()
            self.view.setHtml(FAIL_HTML)

    def pick_images_folder_json(self) -> str:
        import segment_server

        folder = self._choose_directory("Open Images", str(ROOT))
        if not folder:
            return json.dumps({"ok": False})
        try:
            segment_server.set_images_dir(folder)
        except (ValueError, OSError) as e:
            return json.dumps({"ok": False, "error": str(e)})
        self._last_images_dir = str(ROOT)
        labels = segment_server.get_labels_dir()
        if labels:
            self._labels_chosen = True
            self._last_labels_dir = str(labels)
        return json.dumps({
            "ok": True,
            "dir": folder,
            "files": segment_server.list_image_files(),
            "labels_dir": str(labels) if labels else None,
        })

    def open_images_from_menu(self):
        data = json.loads(self.pick_images_folder_json())
        if not data.get("ok"):
            if data.get("error"):
                QMessageBox.warning(self, "Segriotate", data["error"])
            return
        files_js = json.dumps(data["files"])
        dir_js = json.dumps(data["dir"])
        self.view.page().runJavaScript(
            "if (window.applyServerFileList) {"
            f" window.applyServerFileList({files_js}, {dir_js});"
            "}"
        )
        if data.get("labels_dir"):
            self._notify_labels_dir(data["labels_dir"])

    def pick_labels_folder_json(self) -> str:
        import segment_server

        folder = self._choose_directory("Choose folder for labels", str(ROOT))
        if not folder:
            return json.dumps({"ok": False})
        try:
            saved = segment_server.set_labels_dir(folder)
        except (ValueError, OSError) as e:
            return json.dumps({"ok": False, "error": str(e)})
        self._labels_chosen = True
        self._last_labels_dir = str(saved)
        return json.dumps({"ok": True, "dir": str(saved)})

    def _notify_labels_dir(self, folder: str):
        dir_js = json.dumps(folder)
        self.view.page().runJavaScript(
            "if (window.setLabelsDir) {"
            f" window.setLabelsDir({dir_js});"
            "}"
        )

    def choose_labels_folder_from_menu(self):
        data = json.loads(self.pick_labels_folder_json())
        if not data.get("ok"):
            if data.get("error"):
                QMessageBox.warning(self, "Segriotate", data["error"])
            return
        self._notify_labels_dir(data["dir"])


def _windows_app_id() -> None:
    """So the taskbar uses Segriotate.ico instead of the Python icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.segritech.segriotate"
        )
    except Exception:
        pass


def _app_icon() -> QIcon:
    ico = ROOT / "Segriotate.ico"
    png = ROOT / "Segriotate.app" / "Contents" / "Resources" / "segriotate_icon_1024.png"
    if ico.is_file():
        return QIcon(str(ico))
    if png.is_file():
        return QIcon(str(png))
    return QIcon()


def main():
    _windows_app_id()
    already = probe_health()
    if already and already.get("app") != "segriotate":
        sys.exit(
            f"Port {PORT} is already in use (probably an old python scripts/segment_server.py).\n"
            "Stop that process with Ctrl+C, then start Segriotate again."
        )
    flask_up = bool(already) and already.get("app") == "segriotate"
    if not flask_up:
        threading.Thread(target=run_flask, daemon=True).start()
    editor_ready = flask_up and already.get("ready") is True

    try:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    except AttributeError:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("Segriotate")
    icon = _app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    window = MainWindow(editor_ready)
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
