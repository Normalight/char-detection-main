# 字符检测项目上下文

## 项目职责

图片/PDF 的中文字符检测、交互式标注与原图切片导出工具。当前默认流程：PP-OCRv6-small 检测文本行 → PP-OCRv6-small 识别 → CTC 单字定位 → stroke-ownership 笔画归属细化 → 人工标注 → PNG、JSON、Excel 导出。导出的单字切片可供独立的字形归并项目使用。

Python 推荐 3.11，界面使用 PySide6 / QPainter，推理使用 onnxruntime，图像处理使用 numpy / OpenCV / Pillow，Excel 使用 xlsxwriter。PDF 优先通过 PyMuPDF 渲染，pdf2image 回退需要系统 Poppler。

## 源码地图

| 文件 | 职责 |
| --- | --- |
| `app.py` | 桌面启动入口 |
| `ui/window.py` | 主窗口、参数、工具栏、进度、登录和更新接入、关闭事件 |
| `annotation/controller.py` | 标注业务、坐标、编辑状态、快捷键、页面和异步任务编排 |
| `ui/canvas.py` | QPainter 绘制、鼠标/滚轮事件、图像缓存 |
| `annotation/workers.py` | 模型加载、页面加载、当前页 OCR、批量预取、当前页及批量保存 |
| `ocr/pipeline.py` | `OcrHandle` 推理编排与结果生成 |
| `ocr/image.py` | OCR 限尺寸副本、图像解码优化与坐标回映 |
| `ocr/detector.py`、`recognizer.py` | ONNX 模型、配套 YAML、检测与识别适配 |
| `ocr/char_boxes.py`、`stroke_ownership.py` | CTC 定位、单字几何和笔画归属 |
| `ocr/page_layout.py` | 整页竖排几何与字距估计 |
| `utils/paths.py` | 标注与 Excel 共用的输出目录、前缀、兼容 JSON 查找 |
| `utils/storage.py` | JSON、原图裁剪、切片持久化、保存兜底 OCR |
| `utils/excel.py` | 版面排列、缩略图、Excel 导出 |
| `utils/pdf.py`、`ocr/geometry.py` | PDF 页数/渲染、OCR 裁剪与坐标回映 |
| `config.py` | 模型、单字框策略、推理尺寸及资源定位 |
| `app.spec`、`packaging/` | Cython 保护、PyInstaller 打包、自检和发布 |

默认 `det_model = rec_model = "v6-small"`、`char_box_strategy = "stroke-ownership"`。模型与配套字典/参数文件位于 `models/ppocrv6/small/` 和 `models/ppocrv6_rec/small/`，每处均需 `inference.onnx`、`inference.yml`；下载方法见 README。权重不进 Git。修改配置时核对模型下载、打包收集与已有标注的上下文兼容。

## 坐标与单字框

- `base_image_bgr` 是原图或 PDF 渲染页，boxes、JSON 和 PNG 切片始终使用此坐标系。
- PDF 经 `utils/pdf.py` 优先提取单张铺满页面的扫描原图并应用页面旋转；含文字（包括隐藏 OCR 层）、图形、批注、裁剪、特殊颜色空间或透明效果等复杂页保守按 DPI 渲染。原图提取失败仍先使用 PyMuPDF 渲染。坐标回归包含 `tests/test_pdf.py`。
- `view_image_bgr`、`view_scale`、`viewport_offset`、`draw_origin` 只用于显示；缩放和平移不改变 boxes 或导出清晰度。
- `image_scale` 作用于 OCR 副本；统一经过 `ocr/image.py` 的尺寸限制，`ocr_max_side` 默认 2560，非正值回退到 2560。推理结果按实际宽高比例回映原图。
- 单字框跟随可信墨迹，可自然重叠。存在可靠笔画而未识别出文字时保留无文本框，不虚构字标签；缺少可信墨迹时保留定位估计，不能声称得到真实轮廓。
- 裁剪使用 `storage_utils.get_rotate_crop_image`，保持竖排切片方向。
- 修改 boxes 或显示状态后调用 `_refresh_canvas()`，绘制集中在画布。图像缓存使用内容/版本指纹，不使用对象地址作为内容标识。

## 异步生命周期与保存

- 模型加载、图像解码、PDF 渲染、OCR、切片和 Excel 导出均在 `annotation/workers.py` 的 QThread 执行，通过 `Signal` 回主线程；worker 不操作 GUI。
- 页面流程：`ModelLoadWorker` → `_load_item` 启动 `PageLoadWorker` → `_apply_loaded_item` → 按需启动 `OcrPrefetchWorker`。加载成功前不提前改写当前页索引。
- 空闲时 `OcrBatchPrefetchWorker` 预取未访问且无 JSON 的页面；前台 OCR 与批量保存可取消预取。结果按请求号、页面标识、编辑版本和会话代数校验，不能覆盖后续编辑或另一会话。
- 重新开始时保留退役推理线程的引用直到其结束，丢弃过期结果；关闭窗口仍检查所有线程，保存任务需安全收尾。
- `save_current` 启动 `SaveCurrentWorker`，按原图像素摘要、框和文件状态复用未变化切片，只暂存变化项，保留整页事务回滚。保存成功且编辑版本未变化才继续目标页导航，失败或取消留在当前页。
- 翻页前对编辑过或存在未落盘框的页面处理保存询问。保存期间继续编辑要保留新版本并取消自动翻页。
- `save_all` 只启动 `SaveAllWorker` 并立即返回；退出由 `quit_app`、`_pending_quit` 和保存结束回调协调。
- 未访问页批量保存优先内存 boxes，再读 JSON，最后才回退 OCR。重存时清理对应旧切片，防止删框后残留污染结果。
- 图片默认以文件主名作为输出目录；收集输入时对同名图片分配 `output_stem`，追加扩展名后缀，后缀仍重名时追加序号。JSON、切片和 Excel 共用该路径。兼容旧标注时核对来源文件名；缺少来源信息时只读取输入中无歧义的旧标注。
- 批量保存只有在切片、JSON 和 Excel 均成功后才发送 `page_saved`；Excel 失败计入未完成，保留该页未保存状态并阻止保存后退出。
- 保存时复用有尺寸上限的 Excel 缩略图，只缓存缩略图，避免同时保存全部原图切片；Excel 使用内存打包。

## GUI 与工程约定

快捷键统一登记在 `annotation.controller.AnnotatorCore.register_hotkeys`，编辑控件提交后恢复画布焦点。修改快捷键时检查中文输入法、文本输入焦点、授权锁定和会话切换状态。

`annotation/controller.py` 的 Qt 导入保持按需加载；日志使用 `logging.getLogger(__name__)`。业务错误抛异常交由调用方处理，避免在模块导入阶段退出进程。PDF 页数通过 `get_pdf_page_count` 获取，参数沿用界面现有上下限校验。

## 开发与验证

在本项目目录的 Python 环境中执行：

```sh
python -m pip install -r requirements-dev.txt
python app.py
QT_QPA_PLATFORM=offscreen python -m pytest tests -q
```

测试包括纯逻辑、静态结构和 Qt/异步回归，直接收集 `tests/` 中当前功能的测试。按改动选择验证：

- 单字几何：stroke ownership、char boxes、page layout、vertical page regression，并查看固定图例的框与切片。
- 坐标和导出：annotator core coords、OCR image/proxy、storage、Excel。
- 任务与窗口：restart session、async page IO、worker lifecycle、background prefetch、exit state、desktop lifecycle。
- 打包与发布：native protection、release assets、next version、public publish、GitCode publish，并执行构建自检。

## 认证、更新与分发

`ui/window.py` 通过 `hanzi_auth` 登录并保护窗口，通过独立的 `hanzi_update.attach_updates` 接入更新。授权锁定后保留保存当前成果的入口；更新重启沿用 `quit_app` 的保存流程，拒绝退出时清除 `restart_requested`。

共享包从本项目 `vendor/` 安装，源码在相邻 `hanzi-auth/`、`hanzi-update/`。公共 `packaging/protection.py`、`release_info.py`、`publish_public.py` 的维护源为 `hanzi-auth/tools/desktop_packaging/`；改动后使用其 `sync_desktop.py` 同步，升级包时维护普通和平台 requirements 引用。

Windows 使用 `build_windows.bat`；macOS 使用 `bash build_mac.sh`（默认 arm64）。保留 Cython 保护、模型收集、`build_info.json`、`--smoke-test` 和完整 onedir 分发。各平台在相应原生环境构建；发布入口见 `.github/workflows/`。

Mac arm64 的 `packaging/mac.spec` 通过 `minimal_opencv.py` 构建固定版本的图像处理模块，通过 `slimming.py` 裁剪已验证版本的 Qt/ONNX 依赖。变更需验证真实图片 OCR、图片编解码、PDF 和实际产物自检；对照开关及缓存说明见 README。

源码、测试、打包配置和必要分发依赖进入版本管理；用户图片、导出结果、模型、缓存及 build/dist 不作为源码提交。入口、默认行为、异步契约变化时同步本文件和 README。

## 提交说明

每次提交必须根据本次实际 diff 编写明确的 commit message。标题使用 `类型(范围): 中文改动摘要`，类型按实际选择 `feat`、`fix`、`refactor`、`docs`、`test`、`build` 或 `chore`；范围填写实际模块。避免只有“更新”“修复问题”或文件列表的标题。

涉及行为变化、多个模块或共享依赖时，正文说明改了什么、为什么改、用户或调用方受到什么影响，以及执行过的验证和结果；未执行的验证及原因如实注明。仅记录该提交包含的内容，不把已有但未纳入提交的工作或计划写成已完成。跨仓库改动分别编写对应提交说明。

## 客户端更新记录

每次客户端发版前，必须在本仓库编写 `RELEASE_NOTES_v<版本号>.md`，随该版代码提交；按需包含新增功能、体验改进、问题修复，以及兼容性、升级步骤或已知限制。使用面向用户的中文描述实际变化，不以 commit 列表或笼统的“优化体验”代替。

更新记录的版本必须与安装包 `build_info.json`、客户端版本和发布标签中的产品版本一致。只记录本版已完成的变化；共享认证、更新包或打包变化若影响用户，也应在本客户端记录。

发布 GitHub / GitCode Release 时使用该版本更新记录作为正文；公开分发仓库的产品前缀标签沿用相同产品版本和说明。发版前核对实际 Release 正文及客户端可见更新说明，确保分发/镜像过程保留记录；工作流自动生成的 notes 不能代替人工编写的版本说明。

## 结构维护

Python 文件使用 `snake_case.py`，按职责命名；外部入口、软件标识、模型目录和导出格式保持稳定。开发测试依赖放在 `requirements-dev.txt`，平台共同构建依赖放在 `packaging/requirements-build.txt`，平台文件只写包含与差异。`packaging/` 中的公共脚本从认证项目 `tools/desktop_packaging/` 同步，不在副本上独立修改。在认证项目运行 `python tools/sync_desktop.py --check` 检查漂移。源码地图见 README。

源码按职责收敛到包，根目录仅保留启动入口和配置。公共工具位于 `utils/`（如适用），业务包不使用模型历史名称。打包保护自动发现根目录下含 `__init__.py` 的源码包，并要求实际 `ui/window.py` 进入保护清单。测试和打包自检直接引用对应包。

## 保存性能

- `utils/save_cache.py` 管理可丢弃的保存缓存。缺失缓存时重建；原图变化、框变化或输出文件变化使相应缓存失效。
- Excel 缩略图使用快速 PNG 无损编码，按页面持久化复用；输入和布局参数不变时保留已有 Excel。Excel 通过临时文件成功替换，失败保留旧文件。
- 保存线程等待已取消的 OCR 线程退出后再计算，等待不阻塞 GUI，仍可取消。新增行为回归见 `tests/test_incremental_save.py`。

## 历史 ZIP 更新暂存

以下暂存行为适用于保留的旧更新协议。

自动更新的下载包、解压内容、脚本、日志及旧程序备份全部位于软件内部 `.hanzi-update-*`；更新保留安装根目录，只替换新版程序项并保留用户文件。新版完成登录后清理匹配的已完成更新。无法定位可写安装目录时提示处理。原生生命周期测试以安装目录作为助手工作目录，验证原位替换。

## 原生更新

`hanzi-update` 0.1.5 的 attach_updates 使用 WinSparkle / Sparkle；保留现有按钮和 closeEvent 保存契约。Windows 当前用户安装器和 Mac 外层 app 的签名/公证辅助源在 hanzi-auth/tools/desktop_packaging/native_*，通过 sync_desktop.py 同步。设置 HANZI_UPDATE_FEED、HANZI_UPDATE_PUBLIC_KEY 后现有平台构建脚本生成原生更新产物；未配置时更新入口提示不可用。PACKAGE_VERSION 派生原生版本信息，配置与已构建产物必须一致。详细流程与尚未完成的真机验收见 ../hanzi-update/docs/update/release.md 和 test_report.md。原生 feed 发布不使用旧 ZIP/SHA-256 镜像器。

Mac 原生更新构建支持 HANZI_MAC_SIGNING=adhoc，无 Apple 签名标识时默认采用此模式；使用本地 ad-hoc 签名并保留更新包 Ed25519 校验。developer-id 模式另需证书及公证 profile。

原生发布使用 `.github/workflows/native-build.yml`，传入已提交更新说明的版本。公开更新配置位于 packaging/native_update_public.json；SDK 从官方固定版本下载并校验 SHA-256，Mac bridge 源从 hanzi-update 同步。仓库变量 NATIVE_UPDATES=true 时旧 build.yml 停用。安装包资产统一为 `<软件>-vX.Y.Z-<平台>`（Windows 原生安装器 `.exe`、Mac `.zip`），与公开仓历史 Release 命名一致；`dist/` 每个平台只留一个安装包，`build_mac.sh` 在设置 HANZI_UPDATE_FEED 后不再另打包，避免同名 ZIP 覆盖签名更新包。默认运行时只构建/验收并上传 Artifact，更新包由本机钥匙串签名后发布；勾选 `publish` 输入后改用仓库 Secrets 的 `HANZI_UPDATE_PRIVATE_KEY` 签名，`packaging/native_publish.py` 发布 `<软件>-vX.Y.Z` 与 `<软件>-stable`，不勾选时私钥不进入 CI。
