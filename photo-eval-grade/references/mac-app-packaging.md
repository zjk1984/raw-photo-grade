# macOS App 打包与使用指南 (PhotoGrade M4.app)

本指南针对在苹果芯片（Apple Silicon M4 / M3 / M2 / M1，特别针对 MacBook Air M4 32G）上构建和双击运行独立 `.app` 应用程序进行说明。

---

## 1. 为什么推荐将 PhotoGrade M4 打包为原生 Mac App？

1. **摆脱命令行与终端**：双击即可在 Launchpad 或 Dock 栏常驻运行，支持拖放导入照片。
2. **零开销原生渲染（WebKit/Cocoa）**：采用轻量级原生窗口（通过 `pywebview` 基于系统自带 WebKit 引擎驱动），不需要笨重的 Chromium/Electron，内存常驻极低（< 50MB）。
3. **M4 GPU (MPS) 原生直连**：App 运行时通过 Metal Performance Shaders 直通 M4 统一内存 GPU，享受全速张量计算与 Sobel/Laplacian 硬件卷积。

---

## 2. 在你的 MacBook Air M4 本机上一键打包步骤

打开 Mac 终端（Terminal.app），进入项目根目录：

### 步骤 1：安装必要依赖

```bash
# 1. 确保已安装 Homebrew 下的 LibRaw（RAW解码）
brew install libraw exiftool

# 2. 安装 Python 核心及打包依赖
pip3 install torch torchvision rawpy Pillow numpy pyinstaller pywebview
```

### 步骤 2：运行自动化打包脚本

我们在 `photo-eval-grade/ui/` 中提供了全自动打包脚本：

```bash
./photo-eval-grade/ui/build_mac_app.sh
```

脚本将自动执行：
- 检查系统环境与 Python 版本；
- 抓取依赖与前端界面模板 `index.html`；
- 调用 PyInstaller 按照 `photo-eval-grade/ui/PhotoGradeM4.spec` 规范完成编译；
- 生成独立的 macOS Bundle：`dist/PhotoGrade M4.app`。

---

## 3. 安装与使用

打包完成后，在项目的 `dist/` 文件夹下即可看到：

```
dist/
└── PhotoGrade M4.app
```

### 日常运行：
- **安装到系统**：直接将 `PhotoGrade M4.app` 拖入 Finder 左侧的 **“应用程序”（/Applications）** 目录；
- **启动使用**：在 Launchpad（启动台）或 Spotlight（聚焦搜索）中输入 `PhotoGrade M4`，回车即可像原生 Mac 软件一样秒开；
- **拖拽体验**：将 SD 卡、相机文件夹或单张 RAW 照片直接拖入窗口中间，M4 GPU 即可开始并发打分并呈现 S/A/B/C 分级卡片。

---

## 4. 常见问题排查 (FAQ)

### Q1: 首次打开提示“无法打开，因为无法验证开发者”？
**解决方法**：
由于是本地自行编译的 App（未经 Apple 开发者付费证书公证）：
1. 在 Finder 的“应用程序”中找到 `PhotoGrade M4.app`；
2. **按住 Control 键并右键点击** 该应用图标，在弹出菜单中点击 **“打开”**；
3. 弹出对话框后点击“仍要打开”即可，后续双击均可正常启动。
或者在终端中运行：
```bash
xattr -cr "/Applications/PhotoGrade M4.app"
```

### Q2: 想在调试阶段不打包直接启动窗口？
直接运行：
```bash
./photo-eval-grade/ui/launch_app.sh
```
或：
```bash
python3 photo-eval-grade/ui/app.py
```
会自动唤起原生 Native WebKit 窗口。
