"""Local HTTP bridge server: translates browser/terminal link clicks into IDE navigation.

Start once per session::

    python -m krrood.entity_query_language.verbalization.rendering.bridge_server
    python -m krrood.entity_query_language.verbalization.rendering.bridge_server --port 8765

The server listens on ``127.0.0.1`` only (not exposed to the network).  On each
request it locates the ``charm`` launcher (the JetBrains IDE command-line tool)
and calls ``charm --line N /path/file.py``.

Launcher auto-detection order
------------------------------
1. ``charm`` on PATH
2. JetBrains IDE scripts in ``~/.local/share/JetBrains/Toolbox/scripts/``
   (``pycharm``, ``idea``, ``charm``)
3. Common standalone install paths under ``~/``, ``/opt/``, ``/snap/``

If none is found, a warning is printed and the click is silently ignored.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

_log = logging.getLogger(__name__)

# Sent back to the browser so it closes the tab that opened for the request.
_CLOSE_TAB_HTML = b"""\
<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<script>window.close();setTimeout(function(){history.back();},200);</script>
</body></html>"""


def _find_charm() -> Optional[str]:
    """Return the path to the JetBrains IDE launcher, or ``None`` if not found."""
    # 1. Standard PATH lookup (works when the user added PyCharm bin to PATH)
    for name in ("charm", "pycharm", "idea"):
        found = shutil.which(name)
        if found:
            return found

    # 2. JetBrains Toolbox scripts directory
    toolbox = Path.home() / ".local/share/JetBrains/Toolbox/scripts"
    for name in ("pycharm", "idea", "charm"):
        candidate = toolbox / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    # 3. Common standalone install patterns
    search_roots = [Path.home() / "Applications", Path("/opt"), Path("/snap")]
    patterns = ["*/bin/pycharm.sh", "*/bin/idea.sh", "*/bin/charm"]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                if os.access(candidate, os.X_OK):
                    return str(candidate)

    return None


class _BridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        qs = parse_qs(urlparse(self.path).query)
        file_path = qs.get("file", [None])[0]
        line = qs.get("line", ["1"])[0]

        if file_path:
            launcher = _find_charm()
            if launcher:
                subprocess.Popen([launcher, "--line", line, file_path])
                _log.info("bridge: opened %s:%s via %s", file_path, line, launcher)
            else:
                _log.warning(
                    "bridge: no JetBrains launcher found (charm/pycharm/idea). "
                    "Add PyCharm's bin/ directory to PATH and restart the server."
                )

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_CLOSE_TAB_HTML)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(_CLOSE_TAB_HTML)

    def log_message(self, *args) -> None:
        pass  # suppress per-request HTTP log noise


def run(port: int = 8765) -> None:
    """Start the bridge server on *port*.  Blocks until Ctrl+C."""
    server = HTTPServer(("127.0.0.1", port), _BridgeHandler)
    launcher = _find_charm()
    if launcher:
        print(f"IDE launcher: {launcher}")
    else:
        print(
            "WARNING: no JetBrains launcher found on PATH or in Toolbox scripts.\n"
            "Add PyCharm's bin/ directory to PATH and restart."
        )
    print(f"Bridge server listening on http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()
    run(port=args.port)
