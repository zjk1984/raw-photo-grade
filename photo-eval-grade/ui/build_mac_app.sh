#!/bin/bash
# ==============================================================================
# Build macOS Standalone Application: PhotoGrade M4.app
# Designed for Apple Silicon (MacBook Air / Pro M4 / M3 / M2 / M1)
# ==============================================================================

set -e

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   PhotoGrade M4 - macOS Native App Packager          ${NC}"
echo -e "${BLUE}======================================================${NC}"

cd "$REPO_ROOT"

# 1. 检查操作系统
OS="$(uname -s)"
if [ "$OS" != "Darwin" ]; then
    echo -e "${YELLOW}提示: 当前系统不是 macOS ($OS)。正在生成打包配置与脚手架...${NC}"
    echo -e "${YELLOW}要生成可在 macOS 直接双击运行的 .app，请在你的 Mac 电脑上运行此脚本。${NC}"
fi

# 2. 检查 Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3，请先安装 Python 3.10+。${NC}"
    exit 1
fi

PYTHON_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo -e "${GREEN}✓ 检测到 Python 版本: $PYTHON_VER${NC}"

# 3. 检查并安装必备打包依赖
echo -e "${BLUE}==> 检查打包依赖 (pyinstaller, pywebview)...${NC}"
python3 -c "import PyInstaller" 2>/dev/null || {
    echo -e "${YELLOW}正在安装 PyInstaller...${NC}"
    pip3 install pyinstaller
}

python3 -c "import webview" 2>/dev/null || {
    echo -e "${YELLOW}正在安装 pywebview (用于生成原生 macOS WebKit 窗口)...${NC}"
    pip3 install pywebview
}

# 4. 创建应用高清图标 (可选，如系统有 sips/iconutil 则生成 ICNS)
ICON_PATH="$REPO_ROOT/photo-eval-grade/ui/AppIcon.icns"
if [ ! -f "$ICON_PATH" ] && command -v sips &>/dev/null; then
    echo -e "${BLUE}==> 生成应用图标...${NC}"
    ICONSET_DIR="$REPO_ROOT/photo-eval-grade/ui/AppIcon.iconset"
    mkdir -p "$ICONSET_DIR"
    python3 -c "
from PIL import Image, ImageDraw
im = Image.new('RGB', (1024, 1024), (20, 22, 28))
d = ImageDraw.Draw(im)
# Gradient pill
d.rounded_rectangle([128, 128, 896, 896], radius=220, fill=(139, 92, 246))
d.rounded_rectangle([150, 150, 874, 874], radius=200, fill=(30, 32, 42))
d.ellipse([340, 340, 684, 684], fill=(59, 130, 246))
im.save('$ICONSET_DIR/icon_512x512@2x.png')
" 2>/dev/null || true
    if [ -f "$ICONSET_DIR/icon_512x512@2x.png" ] && command -v iconutil &>/dev/null; then
        sips -z 512 512   "$ICONSET_DIR/icon_512x512@2x.png" --out "$ICONSET_DIR/icon_512x512.png" 2>/dev/null || true
        sips -z 256 256   "$ICONSET_DIR/icon_512x512@2x.png" --out "$ICONSET_DIR/icon_256x256.png" 2>/dev/null || true
        sips -z 128 128   "$ICONSET_DIR/icon_512x512@2x.png" --out "$ICONSET_DIR/icon_128x128.png" 2>/dev/null || true
        iconutil -c icns "$ICONSET_DIR" -o "$ICON_PATH" 2>/dev/null || true
        rm -rf "$ICONSET_DIR"
    fi
fi

# 5. 执行 PyInstaller 打包
echo -e "${BLUE}==> 开始编译 macOS Application Bundle...${NC}"
SPEC_FILE="$REPO_ROOT/photo-eval-grade/ui/PhotoGradeM4.spec"

pyinstaller "$SPEC_FILE" --clean -y

APP_DIST="$REPO_ROOT/dist/PhotoGrade M4.app"

if [ -d "$APP_DIST" ]; then
    echo -e "${GREEN}======================================================${NC}"
    echo -e "${GREEN}✓ .app 打包成功！${NC}"
    echo -e "${GREEN}应用路径: $APP_DIST${NC}"
    echo -e "${GREEN}======================================================${NC}"

    # 如果没有指定 --no-dmg 且运行在 macOS 上，自动调用生成 DMG 安装包
    if [ "$1" != "--no-dmg" ] && [ "$OS" = "Darwin" ]; then
        echo -e "${BLUE}==> 开始生成 .dmg 安装镜像...${NC}"
        "$SCRIPT_DIR/build_dmg.sh"
    fi
else
    echo -e "${RED}打包遇到问题，请检查上方日志。${NC}"
fi
