#!/usr/bin/env python3
"""Mac Native Desktop Application & Web Server for PhotoGrade M4.

Combines a lightweight, zero-latency HTTP backend running on localhost
with either macOS webview (WebKit) or default Safari/Chrome browser window.
Leverages Apple Silicon M4 MPS acceleration for instant local culling & grading.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default
import glob
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

# Auto-detect local virtualenv site-packages so it works out of the box with system/homebrew python
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parent.parent
for _cand in [
    _REPO_ROOT / ".venv",
    Path.cwd() / ".venv",
    Path.home() / ".venv",
]:
    for _site in glob.glob(str(_cand / "lib" / "python*" / "site-packages")):
        if _site not in sys.path:
            sys.path.insert(0, _site)

from PIL import Image

sys.path.insert(0, str(_REPO_ROOT / "shared" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "photo-eval-grade" / "scripts"))

from eval_photo import PhotoEvaluator, organize_files, PRESET_WEIGHTS  # noqa: E402
from look_select import build_look_compare_sheet, suggest_look_detail  # noqa: E402
from looks import ALL_LOOKS, get_look, look_choices, load_prefs, save_prefs  # noqa: E402
from raw_develop import apply_grade, apply_orientation, decode_raw, resize_long_edge, save_image  # noqa: E402
import numpy as np  # noqa: E402
import pipeline  # noqa: E402

# State storage for current app session
SESSION_DATA = {
    "evaluations": [],
    "temp_files": {},
    "evaluator": None,
    "current_preset": "general",
    "current_look": "auto",
    "locked_look": None,
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
            prefs = load_prefs()
            dev_name = "Apple M4 MPS (Metal) Ready" if evaluator and str(evaluator.device) == "mps" else "Apple Silicon M4 Accelerated"
            self.send_json({
                "device_name": dev_name,
                "preset": SESSION_DATA["current_preset"],
                "look": SESSION_DATA.get("current_look", "auto"),
                "brand": prefs.get("brand", "sony"),
                "looks": ["auto"] + look_choices(),
                "prefs": prefs,
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
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            msg_bytes = f"Content-Type: {ctype}\r\n\r\n".encode("utf-8") + raw_body
            msg = BytesParser(policy=default).parsebytes(msg_bytes)

            preset = "general"
            uploaded_files = []

            for part in msg.iter_parts():
                pname = part.get_param("name", header="content-disposition")
                pfilename = part.get_filename()
                if pname == "preset":
                    payload = part.get_payload(decode=True)
                    if payload:
                        preset = payload.decode("utf-8", errors="ignore").strip()
                elif pfilename:
                    raw_bytes = part.get_payload(decode=True)
                    if raw_bytes:
                        uploaded_files.append((pfilename, raw_bytes))

            SESSION_DATA["current_preset"] = preset
            evaluator = PhotoEvaluator(
                device_name="auto",
                weights=PRESET_WEIGHTS.get(preset),
            )
            SESSION_DATA["evaluator"] = evaluator

            temp_dir = Path(tempfile.gettempdir()) / "photograde_m4_session"
            temp_dir.mkdir(parents=True, exist_ok=True)

            results = []
            SESSION_DATA["temp_files"].clear()

            for idx, (filename, raw_bytes) in enumerate(uploaded_files):
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
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            tiers = [t.strip() for t in payload.get("tiers", "S,A").split(",") if t.strip()]
            look_mode = payload.get("look") or SESSION_DATA.get("current_look") or "auto"
            brand = payload.get("brand")
            look_compare = bool(payload.get("look_compare", False))
            sticky = bool(payload.get("sticky", True))
            SESSION_DATA["current_look"] = look_mode
            preview = bool(payload.get("preview", True))

            out_dir = Path.home() / "Pictures" / "PhotoGrade_Export"
            out_dir.mkdir(parents=True, exist_ok=True)
            compare_dir = out_dir / "look_compare"

            keepers = [e for e in SESSION_DATA["evaluations"] if e["tier"] in tiers]
            developed = []
            raw_suffixes = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
            locked_look = SESSION_DATA.get("locked_look") if sticky and look_mode == "auto" else None

            for item in keepers:
                src = Path(item["path"])
                dest = out_dir / f"{src.stem}_graded.jpg"
                try:
                    if src.suffix.lower() in raw_suffixes:
                        rgb = decode_raw(src, bright=1.0, no_auto_bright=False)
                        rgb = apply_orientation(rgb, "auto", src)
                    else:
                        with Image.open(src) as im:
                            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                    if preview or look_compare:
                        rgb = resize_long_edge(rgb, 1600)
                    suggestion = suggest_look_detail(
                        rgb,
                        item.get("details") or {},
                        forced=None if look_mode == "auto" else look_mode,
                        auto=(look_mode == "auto"),
                        brand=brand,
                        locked_look=locked_look,
                        sticky=sticky if look_mode == "auto" else False,
                    )
                    look_name = suggestion.look
                    if look_name not in ALL_LOOKS:
                        look_name = "sony-st"
                    if sticky and look_mode == "auto" and locked_look is None:
                        locked_look = look_name
                        SESSION_DATA["locked_look"] = look_name
                    graded = apply_grade(rgb, get_look(look_name))
                    save_image(graded, dest, quality=92, tiff=False)
                    compare_sheet = None
                    if look_compare and suggestion.candidates:
                        compare_dir.mkdir(parents=True, exist_ok=True)
                        panels = []
                        for cand in suggestion.candidates[:3]:
                            if cand not in ALL_LOOKS:
                                continue
                            alt = apply_grade(rgb, get_look(cand))
                            tag = "PRIMARY" if cand == look_name else "ALT"
                            panels.append((f"{tag}: {cand}", alt))
                        if panels:
                            compare_sheet = str(compare_dir / f"{src.stem}__compare.jpg")
                            build_look_compare_sheet(panels, compare_sheet)
                    developed.append({
                        "source": str(src),
                        "output": str(dest),
                        "look": look_name,
                        "scene_tag": suggestion.scene_tag,
                        "candidates": suggestion.candidates,
                        "reason": suggestion.reason,
                        "compare_sheet": compare_sheet,
                        "tier": item["tier"],
                    })
                except Exception as exc:
                    print(f"Pipeline error {src.name}: {exc}", file=sys.stderr)

            self.send_json({
                "status": "success",
                "total_developed": len(developed),
                "output_dir": str(out_dir),
                "developed": developed,
                "look_mode": look_mode,
                "locked_look": locked_look,
            })
            return

        elif path == "/api/prefs":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            prefs = load_prefs()
            if "default_look" in payload:
                prefs["default_look"] = payload["default_look"]
            if "auto_look" in payload:
                prefs["auto_look"] = bool(payload["auto_look"])
            if "sticky_look" in payload:
                prefs["sticky_look"] = bool(payload["sticky_look"])
                if not prefs["sticky_look"]:
                    SESSION_DATA["locked_look"] = None
            if "brand" in payload:
                brand = str(payload["brand"]).lower()
                if brand in ("sony", "fuji", "nikon"):
                    from looks import BRAND_DEFAULT_LOOK, scene_map_for_brand, scene_pools_for_brand

                    prefs["brand"] = brand
                    prefs["scene_map"] = scene_map_for_brand(brand)
                    prefs["scene_pools"] = scene_pools_for_brand(brand)
                    prefs["default_look"] = BRAND_DEFAULT_LOOK[brand]
                    SESSION_DATA["locked_look"] = None
            if "scene_map" in payload and isinstance(payload["scene_map"], dict):
                prefs["scene_map"].update(payload["scene_map"])
            if "scene_pools" in payload and isinstance(payload["scene_pools"], dict):
                prefs.setdefault("scene_pools", {})
                for scene, lst in payload["scene_pools"].items():
                    if isinstance(lst, list) and lst:
                        prefs["scene_pools"][str(scene)] = [str(x) for x in lst]
            if "look" in payload:
                SESSION_DATA["current_look"] = payload["look"]
                if payload["look"] != "auto":
                    SESSION_DATA["locked_look"] = None
            if payload.get("clear_sticky"):
                SESSION_DATA["locked_look"] = None
            save_prefs(prefs)
            self.send_json({
                "status": "ok",
                "prefs": prefs,
                "look": SESSION_DATA["current_look"],
                "locked_look": SESSION_DATA.get("locked_look"),
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
