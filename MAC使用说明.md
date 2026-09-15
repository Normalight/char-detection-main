# 中文字符标注工具 · Mac 使用说明

## 操作流程

1. 打开「中文字符标注工具」，选择包含图片或 PDF 的输入文件夹。
2. 选择专用输出文件夹，粒度选 `char`（单字）；其余参数可先保持默认，点击「开始」。
3. 等待后台加载模型和预标注。选择框后检查切字结果；可创建、旋转、删除框，放大检查边界。显示缩放不会降低原图切片清晰度。
4. 保存当前页后继续翻页。离开有未保存框的页面时按提示保存；退出时按提示保存需要保留的结果，等待后台保存完成。
5. 输出包括单字 PNG、标注 JSON、每页汇总 Excel。把保存结果目录交给「字形归并工具」继续按字整理。

常用快捷键（先点击画布获取焦点）：

| 按键 | 操作 |
|---|---|
| C / R / O | 创建 / 旋转 / 导航模式 |
| + / - | 放大 / 缩小显示 |
| N / B | 下一个 / 上一个框 |
| A / D | 上一页 / 下一页 |
| X / Delete | 删除框 |
| S | 保存当前页 |
| Q / Esc | 退出并按提示处理保存 |

PDF 使用内置 PyMuPDF 渲染，正常情况下无需另装 Poppler。标注框始终对应原图，导出的 PNG 使用原图裁剪。

## 安装与首次启动（无需 Python）

支持 macOS 14 Sonoma 及以上的 Intel Mac 和 Apple Silicon（M 系列）。更老系统不在支持范围；最低系统版本仍需在对应实机验证。

1. 下载并解压 `*-macOS-universal.zip`。Intel / M 芯片使用同一个压缩包。
2. 将其中的 `.app` 拖到「应用程序」，双击启动。应用已经包含 Python、Qt、ONNX 推理库和模型，使用时无需联网或安装依赖。
3. 没有 Apple 开发者证书与公证，第一次可能被拦截。先尝试打开一次，再到「系统设置 → 隐私与安全性」，找到该应用并选择「仍要打开」，按系统提示确认。之后可正常双击。
4. 解压目录中的 `打开软件.command` 可以双击启动同目录应用；`.sh` 版本可在终端执行 `bash 打开软件.sh`。移动应用后请直接打开应用本身。

请仅对自己构建或确认来源的应用允许打开。若系统提示“已损坏”且确认文件完整，可仅对这个应用清除下载隔离标记：在终端输入 `xattr -dr com.apple.quarantine `（末尾一个空格），把该 `.app` 拖入终端补全路径，再回车。无需 sudo，也无需关闭系统 Gatekeeper。若提示恶意软件，不应按此方式绕过。

可用 `shasum -a 256 压缩包.zip` 核对随包提供的 `.sha256` 校验文件。

## 通用包的实现

一个 `.app` 内含 arm64 和 x86_64 两套独立原生运行环境；外层启动器为真正的 universal2 Mach-O，系统选定架构后启动对应运行环境。Intel 运行 x86_64，M 系列运行 arm64，接收者不需要 Rosetta。

内部 Python/Qt/ONNX 并非全部融合成单个 universal2 文件；双运行环境方案避免单架构依赖混装，代价是磁盘占用更大。不要移走或删改 `.app/Contents` 内的文件。

PyInstaller 和构建脚本使用本地 ad-hoc 签名保证包结构完整；这不代表 Apple Developer ID 签名或公证，不能消除首次启动提示。

## 从源码运行（开发者）

安装项目依赖后执行：

```bash
PYTHON=/你的环境/bin/python bash start_gui.sh
```

脚本以自身位置为工作目录，支持包含空格、中文的路径。源码运行可用 Python 3.10/3.11。

## 重新打包（开发者）

推荐在 M 系列 Mac 上构建双架构版本。需要 Command Line Tools（`xcode-select --install`）、[uv](https://docs.astral.sh/uv/getting-started/installation/)，以及 Rosetta 2 用于**构建和验证 Intel 运行环境**。需要时通过系统安装提示，或执行 `softwareupdate --install-rosetta` 并阅读许可。接收者的 M 系列 Mac 不需要 Rosetta。

```bash
# 在本项目目录执行，默认生成一个通用 .app 及 ZIP
bash build_mac.sh
# 或明确指定
bash build_mac.sh universal
# 可选：只构建单架构包
bash build_mac.sh arm64
bash build_mac.sh x86_64
```

首次构建需要联网下载两个架构的 Python 3.11 与固定版本依赖。构建环境位于 `build/mac-env-*`，不修改已有 conda 环境；模型必须预先存在于项目内，缺失会立即报错。

不使用 uv 时，可分别设置 `PYTHON_ARM64` 和 `PYTHON_X86_64` 为对应 Python 3.11 的绝对路径。Intel Mac 可以构建 x86_64 包，不能执行 arm64 构建与测试；完整通用包请在 M 系列 Mac 上构建。

脚本保留 Windows 的 `app.spec`，Mac 专用配置位于 `packaging/mac.spec`。每次只替换 `dist/应用名-macOS-目标架构` 及其 ZIP，源码和标注结果不受影响。`build/`、`dist/` 不应提交到 Git。

发布目录包含应用、说明、启动脚本、每个架构的依赖清单；旁边的 ZIP 保留应用所需的符号链接，另有 SHA-256 校验文件。请分发 ZIP，不要只分发应用内的单个可执行文件。

## 验证与排错

构建脚本会依次验证每个原生包和通用启动器的 `--smoke-test`，再检查整个应用的 ad-hoc 签名；任何一步失败都会停止，不会生成新的发布 ZIP。

- 提示 `Bad CPU type in executable`：构建 Intel 包的 M 系列机器缺少 Rosetta，或选择了错误架构的 Python。
- 缺少模型：补齐配置引用的 ONNX 与同目录 `inference.yml`，再构建。
- 输出失败：选择自己的文稿等可写目录，不要选择应用包内部或只读位置。
- 启动闪退：终端执行 `"/Applications/应用名.app/Contents/MacOS/launcher"` 查看错误；也可附加 `--smoke-test` 进行诊断。
- 修改依赖版本后需重建环境：删除本项目 `build/mac-env-arm64` / `build/mac-env-x86_64` 后重新打包。

验证覆盖启动与基础推理/输出，不替代使用真实碑文、图片和多页 PDF 的完整业务验收。Intel 路径可在 M 系列的 Rosetta 中验证，仍建议发布前在真实 Intel Mac 和最低支持系统测试。

参考：[Apple 首次打开说明](https://support.apple.com/zh-cn/guide/mac-help/mh40616/mac)、[PyInstaller 架构与签名说明](https://pyinstaller.org/en/stable/feature-notes.html#macos-multi-arch-support)。
