#!/bin/bash
# ==============================================================================
# PhotoGrade M4 - macOS Application Launcher & Packager
# ==============================================================================

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/.."

echo "🚀 启动 PhotoGrade M4 本地桌面应用..."

# 检查 Python 依赖
python3 -c "import torch" 2>/dev/null || {
  echo "⚠️ 未检测到 PyTorch，建议先运行: pip3 install torch torchvision"
}

# 启动本地 App 服务并唤起浏览器 / Webview 窗口
python3 photo-eval-grade/ui/app.py "$@"
