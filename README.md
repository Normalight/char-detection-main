# 字符（单字）检测

图片 / PDF 的中文字符检测、交互式标注与原图切片导出工具。

## 原理

默认流程：

```text
PP-OCRv6-small 检测文本行
  → PP-OCRv6-small 识别
  → 按官方 CTC 列中心计算单字位置
  → 用局部笔画分别估计每字的左、上、右、下边界
  → 保留外伸笔画，允许相邻字框重叠
  → 保存单字 PNG / JSON / Excel
```

PP-OCRv6 的 CTC 时间列与裁剪图横向位置对应。实现采用官方
`return_word_box` 的列中心语义：
`x = (column + 0.5) × crop_width / time_steps`。默认 `stroke-ownership`
把这些位置作为笔画归属的参考：按局部背景的明暗差提取墨迹，避免将泛黄纸面
当作笔画；强墨迹作为种子，保留与其连接的弱笔画；
用整行有序匹配合并分离偏旁；对跨多个字符的粘连区域，在墨迹内部从各字种子
竞争生长。最终框取归属笔画的外接范围，只加 1 个源像素的抗锯齿余量。
规则竖排会先检查各字间是否存在低墨迹谷；全部间隔均有证据时按这些位置
归属像素，再生成实际墨迹框。栏线按长度、位置和厚度排除，搜索区内容必须
得到本列核心墨迹支持，避免带入邻列或页框线。没有固定比例外扩，也没有强制中文方框。笔画像素分开归属后，矩形框仍可能
自然重叠。没有可信墨迹时保留定位估计框，不能将这种回退视作真实轮廓。

为找回检测行框边缘以外的笔画，几何细化会额外读取原图附近上下文；扩大的是
搜索区域，最终框只跟随找到的笔画，仍限制在原图内。识别文本不会因此重跑。
检测置信度低于 0.60 且文字识别置信度同时低于 0.40 的候选不生成字框，
原始识别保留在结果的 `suppressed_candidate` 中供诊断；任一置信度较高时保留。
这是对双低置信度杂点的处理，不是通用印章识别器。
这仍是启发式方案，不是训练过的单字实例分割模型；严重粘连、背景纹理、框线
及识别位置错误可能影响笔画归属，需要人工复核。

旧 `independent-strokes`（10% 留白）、`polarity-aware`、`projection-refined`、
`ctc-midpoint` 保留，可在 `config.py` 切换对照。重启后生效。旧标注的策略、
框算法版本、检测尺寸、DPI 或图像缩放不一致时，选择“重新 OCR”生成新框；
选择使用旧框则保留原标注。

“开始”可重复点击以应用新目录或参数；有未保存内容时可选择保存、不保存
或取消。程序异步等待旧后台任务结束后再启动，新设置不会提前改变旧会话
的保存目录。切换过程中暂时锁定编辑，完成后恢复。

主项目到此结束：保存最终单字 PNG、标注 JSON 和单页 Excel。按字归并已拆到
独立的 [`group-by-character`](https://github.com/Normalight/group-by-character) 项目；它会递归遍历切片
目录，重新识别并生成按字目录、清单和总 Excel。主项目不保存或展示额外的框内
识别元数据，也不会在批量保存后自动启动第二次归并流程。


## 安装和使用

项目按 `conda chardet` 环境验证：

```bash
conda activate chardet
python -m pip install -r requirements.txt
```

> 建议 Python 3.10 / 3.11。PDF 功能还需要系统安装 [poppler](https://poppler.freedesktop.org/)（macOS: `brew install poppler`）。
> GUI 基于 PySide6（Qt），Windows 上**中文输入法激活时快捷键也照常生效**。

### 下载默认模型

**模型不随 Git 仓库上传。** 从官方下载权重及同版本 YAML，并放入下表路径：

| 用途 | 下载 | 本地目录 |
| --- | --- | --- |
| 文本检测 | [ONNX](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx) · [YAML](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.yml) · [官方页面](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/tree/main) | `models/ppocrv6/small/` |
| 文字识别 | [ONNX](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx) · [YAML](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml) · [官方页面](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/tree/main) | `models/ppocrv6_rec/small/` |

文件名保留为 `inference.onnx`、`inference.yml`，后者包含后处理参数或字符字典。


默认是完整 PP-OCRv6-small 流水线：

- `config.det_model = "v6-small"`
- `config.rec_model = "v6-small"`
- `config.char_box_strategy = "stroke-ownership"`

det + rec ONNX 共约 30 MB。先安装下载工具并拉取官方导出的模型：

```bash
conda run -n chardet python -m pip install huggingface_hub
conda activate chardet
python - <<'PY'
from huggingface_hub import hf_hub_download

for repo, target in [
    ("PaddlePaddle/PP-OCRv6_small_det_onnx", "models/ppocrv6/small"),
    ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "models/ppocrv6_rec/small"),
]:
    for filename in ("inference.onnx", "inference.yml"):
        hf_hub_download(repo, filename=filename, local_dir=target)
PY
```

若模型缺失，启动会给出明确路径和下载提示。旧后端可从 [chineseocr_lite 模型目录](https://github.com/DayBreak-u/chineseocr_lite/tree/master/models)
下载 `dbnet.onnx`、`crnn_lite_lstm.onnx`，放入本项目 `models/`。
macOS 打包只收集当前配置选中的模型，默认无需旧后端模型。
开启角度检测时还需同目录的 `angle_net.onnx`。无需 v6 模型的回退组合：

```python
# config.py
det_model = "legacy"
rec_model = "crnn"
```

也可保留旧检测、只换识别：`det_model = "legacy"`、
`rec_model = "v6-small"`。

图形化标注工具（推荐）：

```bash
conda run -n chardet python app.py
```

### Windows exe 打包

项目提供 Windows x64 构建脚本 `build_windows.bat`。由于 PyInstaller 不能把
macOS 二进制交叉编译成 Windows exe，请在 Windows 10/11（x64）上准备
Miniconda/Anaconda，并在项目目录运行：

```bat
conda create -n chardet python=3.10 -y
conda run -n chardet python -m pip install -r requirements.txt pyinstaller
build_windows.bat
```

脚本会检查当前默认的 PP-OCRv6-small det/rec 模型，完成后输出 onedir 目录，
启动程序位于 `dist\\interactive_annotator\\interactive_annotator.exe`。发布时必须
复制整个 `dist\\interactive_annotator` 目录，不能只复制 exe。模型文件不进 Git，需要先按上面的下载方式
放入 `models\\ppocrv6\\small` 和 `models\\ppocrv6_rec\\small`；如果使用
其他档位，需同步修改 `config.py` 和 `app.spec` 的模型选择。legacy 模型是可选
回退项，缺失时脚本只给出警告。

PDF 输入优先使用内置 PyMuPDF；只有回退到 `pdf2image` 时才需要另外安装
Windows 的 Poppler。

## 参考

旧 OCR 引擎来源：
[chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite)。


### 高分辨率图像与重新开始

OCR 使用独立副本（默认最长边 2560 像素，`config.ocr_max_side` 可调整，非正值回退到 2560），检测、识别和笔画细化都在副本完成。结果按实际缩放后的宽高比例映射回原图；显示标注、JSON 和 PNG 切片始终基于原图像素。界面的 OCR 缩放只控制推理副本，不再改变原图大小。旧标注会提示重新 OCR。

重新开始在保存或放弃旧标注后立即切换会话，旧推理取消并在后台安全收尾，旧结果不会覆盖新页面。取消检查位于检测后、逐行识别和细化之间；单次正在执行的模型调用结束后生效。关闭窗口仍等待线程退出，正在写入的保存任务仍需安全结束。


黑底白字碑文的背景极性从多字行的文本核心区域判断，避免周围浅色纸边反转前景。规则竖排行用可靠笔画分量寻找字间空隙，并限制细碎碑面纹理连接到正文；近乎完全重合的检测候选在识别前按多边形 IoU 过滤，普通重叠保留。


对于多列、字距稳定的黑底白字整页，增加页面级字距/字形尺寸估计：合并同列检测碎片，结合真实字间空隙修正局部位置，外围小噪点依据主体字形尺寸排除。常见尺寸不是硬裁剪模板，连通长笔画、窄字与短字仍保留自身形状。识别漏字但有可靠笔画的位置可输出 `unrecognized: true` 的无文本单字框；不会虚构文字标签。OCR 仅在整页几何确定后细化一次。

保存当前页、保存并翻页均在后台逐张写入原图切片、JSON 和 Excel；保存成功后才加载目标页。页面解码、PDF 渲染和 JSON 读取也在后台，保存或加载失败会保留当前页面。保存期间继续编辑会保留新版本并取消自动翻页。

当前页、后台预取与保存兜底 OCR 共用 `ocr_image.py` 的下采样与坐标回映；后台 JPEG 预取优先使用缩小解码，降采样后的检测输入不会再被过大的检测尺寸参数放大。状态栏显示实际原图和推理尺寸。

保存当前页时只保留一张原图切片，顺手生成有尺寸上限的 Excel 缩略图供后续复用；Excel 直接在内存打包，避免每张切片再次解码和为每张缩略图创建临时文件。原始 PNG 不受缩略图尺寸影响。


## macOS 通用应用（Intel / M 系列）

执行 `bash build_mac.sh` 构建一个适配两种芯片的 `.app` 与发布 ZIP；无需开发者证书。使用者无需安装 Python。详见 [Mac 安装、使用与打包说明](MAC使用说明.txt)。源码启动脚本为 `start_gui.sh`。


## macOS

激活 Python 3.10 / 3.11 环境后执行 `bash start_gui.sh`。
构建可使用 `bash build_mac.sh arm64`（Apple Silicon）或 `bash build_mac.sh x86_64`（Intel）；
不带参数构建双架构包。详细说明见 [macOS 使用说明](MAC使用说明.txt)。

## 仓库范围

保留应用源码、打包脚本、自动化测试及必要固定样例。
运行输出、临时试图、缓存和本机环境不纳入版本管理。
MegaHan 训练/复核、历史评测与命令行演示已移除；其测试源码按要求保留于 `tests/archive/`，不参与当前测试收集。

## 测试

```bash
python -m pip install pytest
python -m pytest tests/ -q
```


## GitHub 自动打包

在 GitHub 发布 Release（正式版或预发布版）时，自动运行 **Build desktop apps**，分别构建：

- Windows x64：解压后运行 exe，保留整个文件夹。
- macOS Intel（x86_64）：使用 Intel 原生 runner 构建。
- macOS Apple Silicon（arm64 / M 系列）：使用 ARM 原生 runner 构建。

Mac 需要 macOS 14 或更新版本。每个 Mac ZIP 内都有 `MAC使用说明.txt`，
只说明首次允许打开未使用 Developer ID 签名及公证的应用，以及软件使用方式。
构建脚本保留本地完整性签名，不需要 Apple 开发者账号。

在 Actions 的成功运行页面下载 Artifacts（保留 14 天）；下载到的就是应用 ZIP，解压一次即可使用。
三种构建全部通过后，自动将 应用 ZIP 上传到触发本次构建的 Release。
保存草稿、推送代码或单独推送标签均不触发打包；不会由流水线自动创建另一个 Release。
标签需使用 `v主版本.次版本.修订号`，如 `v1.0.0` 或 `v1.0.0-rc.1`。
请从包含当前工作流的提交创建版本标签；每次构建使用该 Release 标签对应的源码。
也可在 Actions 手动运行验证构建，文件名使用 `dev-提交SHA`，仅上传 Artifacts。
私有仓库的构建产物与 Release 仍需相应仓库访问权限。

CI 使用 Python 3.11 和固定版本依赖；模型由 `packaging/download_models.py`
从官方 Hugging Face 固定提交下载，包含在应用中，不提交到源码仓库。
每个构建会运行应用测试和真实打包程序的 `--smoke-test`；任一步失败都不会向已发布的 Release 上传安装包。

本地构建仍可运行 `bash build_mac.sh arm64`、`bash build_mac.sh x86_64`；
双架构通用包可在具备两种 Python 3.11 与 Rosetta 的 M 系列 Mac 上运行 `bash build_mac.sh universal`。

安装包命名统一为 `切字-版本标签-平台-架构.zip`，不额外生成校验文件或嵌套压缩包。
例如本仓库的 `v1.0.0`：

- `切字-v1.0.0-windows-x64.zip`
- `切字-v1.0.0-macos-x86_64.zip`（Intel）
- `切字-v1.0.0-macos-arm64.zip`（M 系列）

文件名以标签为准，与可自由填写的 Release 标题无关；应用内名称保持中文。

Mac ZIP 根目录只有 `中文字符标注工具.app` 与 `MAC使用说明.txt`，解压后直接打开应用；不包含外层目录、启动脚本或依赖清单。
本地构建通过 `PACKAGE_VERSION=v1.0.0` 指定包名版本，未指定时为 `v0.0.0-dev`。


## 账号登录与保护打包

启动时需要登录统一账号服务，默认 `http://127.0.0.1:8765`，也可在登录窗口填写服务器 IP + 端口。管理员可在该地址的网页后台分配软件权限、停用账号、设置有效期或重置密码。客户端每 60 秒检查授权，断网缓冲 5 分钟；锁定后保留保存当前成果的入口。

在本项目根目录安装 `requirements.txt` 即可安装 `vendor/` 中的共用认证包。认证包与服务端源码位于工作区相邻的 `hanzi-auth/`，完整本机部署说明见该目录的 README.md；此软件仓库的构建本身不依赖相邻目录。

平台打包使用免费开源的 Cython，将主体 Python 模块及账号模块编译成原生扩展（Windows `.pyd`、Mac `.so`），由 PyInstaller 打包。无需付费混淆许可证。构建默认开启保护，生成 `build/code-protection/manifest.json`；编译失败会中止。Windows 构建需安装 Visual Studio C++ Build Tools（MSVC 与 Windows SDK），Mac 需 Xcode Command Line Tools。各架构分别编译。仅开发调试可显式设置 `HANZI_PROTECT=0`。交付包不包含主体 Python 源码或其字节码；保留最小多进程启动入口。编译保护提高逆向成本，不包含 ONNX 模型加密；生成的 C 文件和源码暂存仅保留在 build 中，不应随发布包分发。


## 登录日志和 GitCode 更新

账号后台提供登录日志，记录时间、账号、软件、来源 IP、结果和客户端版本，支持筛选与翻页。密码和会话令牌不会进入日志。

软件登录后自动检查一次更新，也可点击工具栏“检查更新”或“帮助 → 检查更新”。客户端直接查询 GitCode 的 `qq_28599947/hanzi-soft` 中带 `char-detection-main-` 前缀的最新发行版，匹配当前平台，并读取 `hanzi-update.json` 中的大小和 SHA-256。下载并校验后自动准备更新；点击“保存并重启更新”，按现有提示保存工作，实际退出后在原位置替换并重启，旧版本保留在相邻的 `.previous-*` 目录。Windows 更新保留安装目录中未被新版覆盖的用户文件（运行库 `_internal` 使用新版）。原目录不可写时提示迁移到个人可写目录；源码运行或 macOS 临时隔离位置回退为独立目录，直接打开新版。早期发行版没有清单时提供“前往下载”，下载后手动安装。

发布时必须设置 `PACKAGE_VERSION`，构建会生成 `build_info.json`。可用 `python packaging/release_info.py <安装包.zip> --product char-detection-main --target windows-x64 --version v1.0.0` 计算大小和校验值；Mac 的 target 为 `macos-arm64` 或 `macos-x86_64`。GitCode 仓库创建、标签及附件发布步骤见工作区 `hanzi-auth/README.md` 的“GitCode 发布与软件更新”。

### GitCode 发行版同步

发布 GitHub `vMAJOR.MINOR.PATCH` Release 后，Actions 构建三个平台安装包，上传 GitHub，再复用同一批 Artifact 上传 GitCode。需要仓库变量 `GITCODE_OWNER`、`GITCODE_REPO` 和 Secret `GITCODE_TOKEN`。同步可重跑，已存在的同名附件保留；替换已发布内容应使用新版本号。

同步会将三个安装包的公开下载链接、文件大小和 SHA-256 上传为 GitCode 发行版的 `hanzi-update.json`，供客户端读取；同一份信息也保存在 `gitcode-release-metadata` Artifact。GitCode 仓库需允许用户公开下载；凭据仅用于 CI 发布，不放入客户端。

### 自动递增发布版本

在 GitHub Actions 的 **Build desktop apps → Run workflow** 中勾选 **Build and publish a new version**，默认 `patch` 将 `v1.0.0` 递增为 `v1.0.1`；可选 `minor` 或 `major`。流程同时检查源码仓库与公开安装包仓库的已有正式版本，三个平台共用同一新版本，构建成功后发布并同步。未勾选时仅生成开发构建。手动发布 GitHub Release 时沿用其标签。
