#!/bin/bash
# ==============================================================================
# Build macOS DMG Installer for PhotoGrade M4
# Creates a professional drag-and-drop DMG: [PhotoGrade M4.app] -> [/Applications]
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
echo -e "${BLUE}   PhotoGrade M4 - macOS DMG Installer Packager       ${NC}"
echo -e "${BLUE}======================================================${NC}"

cd "$REPO_ROOT"

APP_NAME="PhotoGrade M4"
APP_BUNDLE="$REPO_ROOT/dist/${APP_NAME}.app"
DMG_OUTPUT_DIR="$REPO_ROOT/dist"
DMG_NAME="PhotoGrade-M4-Installer.dmg"
FINAL_DMG="$DMG_OUTPUT_DIR/$DMG_NAME"

# 1. 检查是否存在 .app
if [ ! -d "$APP_BUNDLE" ]; then
    echo -e "${YELLOW}未在 dist/ 中找到 ${APP_NAME}.app，正在先编译 .app...${NC}"
    "$SCRIPT_DIR/build_mac_app.sh" --no-dmg
    if [ ! -d "$APP_BUNDLE" ]; then
        echo -e "${RED}错误: 无法生成 ${APP_BUNDLE}，DMG 打包中止。${NC}"
        exit 1
    fi
fi

# 2. 检查操作系统与 hdiutil 命令
OS="$(uname -s)"
if [ "$OS" != "Darwin" ]; then
    echo -e "${YELLOW}提示: 当前系统不是 macOS ($OS)，正在使用跨平台生成器构建兼容的 Apple HFS+ DMG...${NC}"
    python3 "$SCRIPT_DIR/make_dmg.py"
    exit 0
fi

if ! command -v hdiutil &>/dev/null; then
    echo -e "${RED}错误: 未检测到 hdiutil 工具，请在 macOS 系统环境下执行。${NC}"
    exit 1
fi

echo -e "${BLUE}==> 准备 DMG 打包临时目录...${NC}"
DMG_STAGING_DIR="$(mktemp -d -t photograde_dmg_staging_XXXXXX)"
mkdir -p "$DMG_STAGING_DIR"

# 清理历史可能存在的 DMG
rm -f "$FINAL_DMG"

# 复制 .app 到临时工作目录
echo -e "${BLUE}==> 复制应用包至临时打包区...${NC}"
cp -R "$APP_BUNDLE" "$DMG_STAGING_DIR/"

# 创建标准 /Applications 软链接，实现经典的拖拽安装体验
echo -e "${BLUE}==> 创建 Applications 目录拖拽软链接...${NC}"
ln -s /Applications "$DMG_STAGING_DIR/Applications"

# 如果存在自定义图标，附加到 DMG 卷标
ICON_PATH="$REPO_ROOT/photo-eval-grade/ui/AppIcon.icns"
if [ -f "$ICON_PATH" ]; then
    cp "$ICON_PATH" "$DMG_STAGING_DIR/.VolumeIcon.icns" 2>/dev/null || true
fi

# 优先检查是否有 create-dmg 命令行工具（可提供更美观的背景图和坐标排版）
if command -v create-dmg &>/dev/null; then
    echo -e "${GREEN}✓ 检测到 create-dmg，使用高级排版生成美化 DMG...${NC}"
    create-dmg \
      --volname "${APP_NAME} Installer" \
      --window-pos 200 120 \
      --window-size 660 400 \
      --icon-size 110 \
      --icon "${APP_NAME}.app" 160 190 \
      --hide-extension "${APP_NAME}.app" \
      --app-drop-link 500 190 \
      "$FINAL_DMG" \
      "$DMG_STAGING_DIR" || {
        echo -e "${YELLOW}create-dmg 执行未完全通过，回退到原生 hdiutil...${NC}"
    }
fi

# 如果 create-dmg 没运行或未生成，使用 macOS 自带的原生 hdiutil 构建高压缩 DMG
if [ ! -f "$FINAL_DMG" ]; then
    echo -e "${BLUE}==> 使用 macOS 原生 hdiutil 生成压缩只读 DMG...${NC}"
    TEMP_DMG="$DMG_OUTPUT_DIR/temp_${DMG_NAME}"
    rm -f "$TEMP_DMG"

    # 生成读写过渡 DMG
    hdiutil create -srcfolder "$DMG_STAGING_DIR" \
                   -volname "${APP_NAME}" \
                   -fs HFS+ \
                   -fsargs "-c c=64,a=16,e=16" \
                   -format UDRW \
                   "$TEMP_DMG"

    # 启用卷标图标展示
    if [ -f "$ICON_PATH" ] && command -v SetFile &>/dev/null; then
        MOUNT_DIR="$(mktemp -d -t photograde_mount_XXXXXX)"
        hdiutil attach "$TEMP_DMG" -mountpoint "$MOUNT_DIR" -nobrowse -quiet || true
        SetFile -a C "$MOUNT_DIR" 2>/dev/null || true
        hdiutil detach "$MOUNT_DIR" -quiet || true
        rm -rf "$MOUNT_DIR"
    fi

    # 转换为只读并应用 UDZO 高压缩
    hdiutil convert "$TEMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$FINAL_DMG"
    rm -f "$TEMP_DMG"
fi

# 清理临时工作目录
rm -rf "$DMG_STAGING_DIR"

if [ -f "$FINAL_DMG" ]; then
    DMG_SIZE="$(du -h "$FINAL_DMG" | cut -f1)"
    echo -e "${GREEN}======================================================${NC}"
    echo -e "${GREEN}✓ DMG 安装包制作成功！${NC}"
    echo -e "${GREEN}文件路径: $FINAL_DMG ($DMG_SIZE)${NC}"
    echo -e "${GREEN}双击该 DMG 即可弹出经典的拖拽安装到 Applications 窗口！${NC}"
    echo -e "${GREEN}======================================================${NC}"
else
    echo -e "${RED}DMG 生成失败，请查看日志排查。${NC}"
    exit 1
fi
