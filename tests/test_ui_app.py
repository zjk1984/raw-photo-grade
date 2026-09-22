"""Test PhotoGrade M4 Web/Desktop UI backend endpoints."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "photo-eval-grade" / "ui"))
import app


@pytest.fixture(scope="module")
def app_server():
    server = app.launch_server(port=9876)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)
    yield "http://127.0.0.1:9876"
    server.shutdown()


def test_index_page(app_server):
    req = urllib.request.urlopen(f"{app_server}/")
    assert req.status == 200
    html = req.read().decode("utf-8")
    assert "PhotoGrade M4" in html
    assert "Apple M4 MPS" in html


def test_system_info_endpoint(app_server):
    req = urllib.request.urlopen(f"{app_server}/api/system_info")
    assert req.status == 200
    data = json.loads(req.read().decode("utf-8"))
    assert "device_name" in data
    assert "preset" in data
