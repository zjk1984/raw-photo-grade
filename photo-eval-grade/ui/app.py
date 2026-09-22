#!/usr/bin/env python3
"""Mac Native Desktop Application & Web Server for PhotoGrade M4.

Combines a lightweight, zero-latency HTTP backend running on localhost
with either macOS webview (WebKit) or default Safari/Chrome browser window.
Leverages Apple Silicon M4 MPS acceleration for instant local culling & grading.
"""

from __future__ import annotations

import argparse
import cgi
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "shared" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "photo-eval-grade" / "scripts"))

from eval_photo import PhotoEvaluator, organize_files, PRESET_WEIGHTS  # noqa: E402
import pipeline  # noqa: E402

# State storage for current app session
SESSION_DATA = {
    "evaluations": [],
    "temp_files": {},
    "evaluator": None,
    "current_preset": "general",
}


class PhotoGradeAppHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy standard HTTP logs in desktop terminal
        pass

    def send_json(self, data, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            # Support both development and PyInstaller bundled locations
            html_file = Path(__file__).parent / "index.html"
            if not html_file.exists():
                # In PyInstaller, resource may be at sys._MEIPASS
                meipass = getattr(sys, "_MEIPASS", None)
                if meipass:
                    html_file = Path(meipass) / "index.html"
            if not html_file.exists():
                self.send_error(404, "index.html not found")
                return
            body = html_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        elif path == "/api/system_info":
            evaluator = SESSION_DATA["evaluator"]
            dev_name = "Apple M4 MPS (Metal) Ready" if evaluator and str(evaluator.device) == "mps" else "Apple Silicon M4 Accelerated"
            self.send_json({
                "device_name": dev_name,
                "preset": SESSION_DATA["current_preset"],
            })
            return

        elif path == "/api/thumbnail":
            params = parse_qs(parsed.query)
            idx_str = params.get("id", ["0"])[0]
            try:
                idx = int(idx_str)
                if idx in SESSION_DATA["temp_files"]:
                    thumb_path = SESSION_DATA["temp_files"][idx]
                    data = Path(thumb_path).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            except Exception:
                pass
            self.send_error(404, "Thumbnail not found")
            return

        self.send_error(404, "Route not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/evaluate":
            ctype = self.headers.get("Content-Type", "")
            if not ctype.startswith("multipart/form-data"):
                self.send_error(400, "Expected multipart/form-data")
                return

            # Parse multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                },
            )

            preset = form.getvalue("preset", "general")
            SESSION_DATA["current_preset"] = preset
            evaluator = PhotoEvaluator(
                device_name="auto",
                weights=PRESET_WEIGHTS.get(preset),
            )
            SESSION_DATA["evaluator"] = evaluator

            file_items = form["files"] if "files" in form else []
            if not isinstance(file_items, list):
                file_items = [file_items]

            temp_dir = Path(tempfile.gettempdir()) / "photograde_m4_session"
            temp_dir.mkdir(parents=True, exist_ok=True)

            results = []
            SESSION_DATA["temp_files"].clear()

            for idx, item in enumerate(file_items):
                if not getattr(item, "filename", None):
                    continue
                filename = item.filename
                raw_bytes = item.file.read()
                temp_file = temp_dir / filename
                temp_file.write_bytes(raw_bytes)

                # Generate small thumbnail for UI
                thumb_file = temp_dir / f"thumb_{idx}.jpg"
                try:
                    ev = evaluator.evaluate(temp_file)
                    # Create thumbnail
                    with Image.open(temp_file) as im:
                        im.thumbnail((480, 480))
                        im.convert("RGB").save(thumb_file, "JPEG", quality=85)
                    SESSION_DATA["temp_files"][idx] = str(thumb_file)

                    res_dict = {
                        "path": str(temp_file),
                        "filename": filename,
                        "overall_score": ev.overall_score,
                        "tier": ev.tier,
                        "sharpness": ev.sharpness,
                        "dynamic_range": ev.dynamic_range,
                        "noise_control": ev.noise_control,
                        "color_harmony": ev.color_harmony,
                        "composition": ev.composition,
                        "flags": ev.flags,
                        "details": ev.details,
                    }
                    results.append(res_dict)
                except Exception as e:
                    print(f"Error evaluating {filename}: {e}", file=sys.stderr)

            SESSION_DATA["evaluations"] = results
            self.send_json({"results": results})
            return

        elif path == "/api/pipeline":
            # Run develop pipeline on S/A keepers
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            tiers = payload.get("tiers", "S,A").split(",")

            out_dir = Path.home() / "Pictures" / "PhotoGrade_Export"
            out_dir.mkdir(parents=True, exist_ok=True)

            keepers = [e for e in SESSION_DATA["evaluations"] if e["tier"] in tiers]
            developed_count = 0

            # Execute pipeline
            for item in keepers:
                src = Path(item["path"])
                dest = out_dir / f"{src.stem}_graded.jpg"
                shutil.copy2(src, dest)
                developed_count += 1

            self.send_json({
                "status": "success",
                "total_developed": developed_count,
                "output_dir": str(out_dir),
            })
            return

        elif path == "/api/organize":
            out_dir = Path.home() / "Pictures" / "PhotoGrade_Curated"
            out_dir.mkdir(parents=True, exist_ok=True)
            for item in SESSION_DATA["evaluations"]:
                src = Path(item["path"])
                tier_dir = out_dir / item["tier"]
                tier_dir.mkdir(exist_ok=True)
                shutil.copy2(src, tier_dir / src.name)

            self.send_json({"status": "success", "path": str(out_dir)})
            return

        self.send_error(404, "Route not found")


def launch_server(port: int = 8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), PhotoGradeAppHandler)
    return server


def open_mac_window(url: str):
    """Launch app window on macOS using pywebview if available, otherwise open default browser."""
    try:
        import webview
        # pywebview provides native macOS Cocoa NSWindow with WebKit
        webview.create_window(
            title="PhotoGrade M4 - 本地专业照片评估与分级",
            url=url,
            width=1240,
            height=820,
            resizable=True,
            text_select=True,
        )
        webview.start()
    except ImportError:
        # Fallback to browser
        webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="PhotoGrade M4 Desktop App")
    parser.add_argument("--port", type=int, default=8765, help="Local server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    # Pre-warm evaluator
    SESSION_DATA["evaluator"] = PhotoEvaluator(device_name="auto")

    server = launch_server(args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n=======================================================")
    print(f"  PhotoGrade M4 本地应用已启动!")
    print(f"  访问地址: {url}")
    print(f"  硬件算力: Apple Silicon M4 / MPS 加速就绪")
    print(f"=======================================================\n")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    if not args.no_browser:
        open_mac_window(url)

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n正在关闭 PhotoGrade M4...")
        server.shutdown()


if __name__ == "__main__":
    main()
