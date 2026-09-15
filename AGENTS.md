# AGENTS.md

本文件给 AI 编码助手（和新人）提供本项目的工作地图。先读这一份，再动代码。

## 项目是什么

基于 OCR 的**中文字符 / 单字检测与交互式标注工具**。

- 输入：图片（PNG/JPG/...）或 PDF（多页）
- 流程：DBNet 检测文本框 → CRNN 识别 → 按 char / word / line 三种粒度切分
- 输出：每个字的切片 PNG + 标注 JSON + 汇总 Excel（每页一个 Sheet）
- UI：PySide6（Qt）侧栏 + QPainter 画布（创建 / 旋转 / 导航 / 删除框、快捷键操作）
- 推理后端：onnxruntime，模型在 `models/dbnet.onnx` 与 `models/crnn_lite_lstm.onnx`

> 键盘走 Qt `QShortcut`（直接吃 `WM_KEYDOWN` 的虚拟键码，不经 TranslateMessage / IME），
> 因此**中文输入法激活状态下快捷键也照常触发**。这是项目从 Tk+cv2 迁到 Qt 的主要动机。

## 技术栈

| 维度    | 选型                                            |
| ------- | ----------------------------------------------- |
| 语言    | Python 3.10 / 3.11（旧版依赖在 3.10+ 装不上）   |
| 推理    | onnxruntime + numpy + opencv-python             |
| 几何    | Shapely + pyclipper（DBNet 后处理）             |
| GUI     | **PySide6（Qt）** —— 侧栏 QMainWindow + 画布 QWidget（QPainter）|
| PDF     | PyMuPDF → pdf2image → Pillow 三级 fallback      |
| Excel   | xlsxwriter                                      |
| 打包    | PyInstaller onedir（`app.spec`）                |

## 架构总览

```
app.py  ──QMainWindow、侧栏、工具栏、进度对话框──▶  annotator_core.py  (业务核心 + 交互状态机)
              │                                       │
              │                                       │ 持有 + 触发重绘
              ▼                                       ▼
        canvas_widget.py  ◀── 鼠标/滚轮事件回传 core._on_mouse_* / _on_wheel
        AnnotatorCanvas（QPainter 绘制）
                                                       │
                                                       │ 长任务委派
                                                       ▼
                                                worker.py（QThread）
                                                ModelLoadWorker / PageLoadWorker / OcrPrefetchWorker / SaveCurrentWorker / SaveAllWorker
                                                       │
            ┌──────────────────────────────────────────┴──────────────────────────────┐
            ▼                                                                          ▼
      cutbatch.py / dbnet/ + crnn/                                       storage_utils.py / excel_utils.py
      (ONNX 推理，在 worker 线程)                                         boxes/JSON/切片/Excel 落盘
```

**主线程绝不跑长任务**：模型加载、OCR 预标注、批量保存全部走 worker.py 的 QThread，
结果通过 pyqtSignal 回主线程。任何在主线程直接调 onnxruntime / 大图 IO / xlsxwriter
的改动都要拒绝。

关键约束（动代码前必读）：

- **坐标解耦**：`base_image_bgr` 是原图处理 / 保存坐标，`view_image_bgr` 仅用于显示。
  boxes 永远存 `base_image_bgr` 坐标系，界面放大缩小**不**改写 boxes、**不**影响切片清晰度。
  详见 `annotator_core.py:1` 文件头注释。
- **画布刷新**：业务代码改完 boxes / view_scale 后调 `self._refresh_canvas()`（= `canvas.update()`），
  Qt 会在下一帧调 `paintEvent`。**不要**直接在业务里画——绘制全部集中在 `canvas_widget._draw_boxes`。
- **异步加载流程**：`app._start` → `core.start` → `ModelLoadWorker` 加载模型 →
  `_on_model_ready` → `_load_item`（图片 + JSON 立即显示）→ 若无 boxes，`OcrPrefetchWorker`
  后台跑预标注 → `_on_ocr_done` 合并。切页同理。**`_load_item` 主线程不阻塞**。
- **空闲预取（无感识别）**：当前页就绪后，`OcrBatchPrefetchWorker` 在后台按顺序
  预标注未访问且无 JSON 的页面（worker 自己加载图片、逐页 `page_ocr_done` 回传、
  单页失败跳过不弹窗），切页时结果已在内存、直接显示。当前页 OCR 与批量保存会
  **抢占取消**预取（`_cancel_background_prefetch`）；调度入口
  `_maybe_start_background_prefetch`，结果复用 `_on_ocr_done` 的
  request_id / item_identity / edit_revision 保护。翻页保存询问与旧版一致：
  覆盖**编辑过（dirty）或存在未落盘框**（含纯 OCR / 预取结果）的页面；
  OCR 未返回、画布为空的页不打扰，退出仍走整体保存询问。
- **`save_all` 立即返回**：纯启动 `SaveAllWorker`，不等待。退出流程由 `quit_app` 置
  `_pending_quit`，`_on_save_all_finished` 检测到后调 `_really_quit`。`app.closeEvent`
  监听 worker 是否在跑，强退前会弹确认 + cancel。
- **快捷键**：所有字母 / +/-/Esc/Delete 在 `annotator_core.register_hotkeys` 的 bindings 表里登记，
  并对应 `QShortcut(QKeySequence("X"), main_window)`。大小写各注册一次（`"C"` 和 `"Shift+C"`）。
  新增快捷键时同步改这张表。
- **焦点与 IME**：QLineEdit / QComboBox 获焦时会吞字母快捷键（Qt 把它们当文本输入），
  约定**回车 / 选择后自动 `canvas.setFocus()`**（见 `_make_line_edit` / level_combo 的信号）。
  画布 `focusInEvent` 调 `grabKeyboard` 作为双保险。
- **QImage 缓存指纹**：`canvas_widget._fingerprint` 用 `(shape, dtype, _view_version, 头部字节 hash)`，
  **不能用 id()**——CPython 地址会被复用，缓存误命中会画错图。
- **切片裁剪**走 `storage_utils.get_rotate_crop_image`，**不要**用 `utils.get_rotate_crop_image`——
  后者会对竖排切片 `np.rot90`，把碑文 / 竖排中文压成横向扁图（`utilsre1.py` 就是当年失败的修复尝试）。
- PDF 渲染三级 fallback，任一级失败只 `logger.warning` 后落下一级；全失败才抛 `RuntimeError`。

## 模块清单

### 主线（可改、被 import）

| 文件                     | 作用                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `app.py`                 | PySide6 `QMainWindow`：侧栏（QSplitter + QScrollArea）、参数表单、工具栏、进度对话框、closeEvent |
| `canvas_widget.py`       | `AnnotatorCanvas(QWidget)`：QPainter 绘制 + 鼠标/滚轮事件路由 + QImage 内容指纹缓存 |
| `worker.py`              | `ModelLoadWorker` / `PageLoadWorker` / `OcrPrefetchWorker` / `SaveCurrentWorker` / `SaveAllWorker`（QThread + pyqtSignal），主线程不阻塞的关键 |
| `annotator_core.py`      | `AnnotatorCore`：业务核心（坐标系、boxes、保存）+ Qt 鼠标分发 + QShortcut + 异步任务编排 |
| `cutbatch.py`            | `OcrHandle`：封装 DBNet+CRNN 调用，产出 `{bbox,result,score}` |
| `storage_utils.py`       | 输出目录、JSON 持久化、切片裁剪、OCR 兜底                    |
| `excel_utils.py`         | 横 / 竖排对齐、防重叠、缩略图控体积                          |
| `pdf_utils.py`           | PDF 页渲染 + 页数读取（三级 fallback）                       |
| `utils.py`               | `validate_and_prepare_polygon_points` 等几何小工具            |
| `config.py`              | 模型路径、dbnet/crnn/angle 参数；适配 PyInstaller `_MEIPASS` |

### 推理引擎（外部代码，慎改）

| 文件                  | 作用                                          |
| --------------------- | --------------------------------------------- |
| `dbnet/dbnet_infer.py`| DBNet 文本检测，`process(img, short_size)`    |
| `dbnet/decode.py`     | DBNet 后处理（pyclipper + Shapely）           |
| `crnn/CRNN.py`        | CRNN 识别，`predict_rbg` 返回 char/word/line  |
| `crnn/util.py`        | CTC 解码                                      |
| `crnn/keys.py`        | 字符表 `alphabetChinese`                      |

参考自 [chineseocr_lite](https://github.com/DayBreak-u/chineseocr_lite)。

### 清理后的仓库

旧版单体入口、原型、备份、mayavi 演示和无关产物已删除。
`tests/`、固定样例、`bench/` 中测试依赖的工具以及仍可配置的推理后端保留。

## 常用命令

```bash
# 安装依赖（PDF 功能还需系统装 poppler：brew install poppler）
pip install -r requirements.txt

# 主入口：图形化标注工具（推荐）
python app.py

# 命令行 demo（每条都要 --images）
python model.py --level char  --images images/0.png
python model.py --level word  --images images/0.png
python model.py --level line  --images images/0.png

# 单元测试（只测纯逻辑，不依赖 onnxruntime / QApplication 真实例化）
python -m pytest tests/ -q

# 打包 onedir（输出 dist/interactive_annotator/）
pyinstaller app.spec
```

> 当前环境若未装 `onnxruntime`，`tests/test_annotator_core_coords.py`、
> `test_annotator_core_overlay.py` 会在 import 阶段失败——这是已知的依赖链问题
> （`annotator_core → cutbatch → onnxruntime`），不是测试本身错。
> `tests/test_canvas_widget.py` 和 `tests/test_worker.py` 只做静态源码检查，
> 不触发该依赖链（其中 `test_worker.py` 还会**强制 worker 代码不碰 GUI**）。

## 工程约定

- **日志**：用 `logging.getLogger(__name__)`，不要 `print`、不要裸 `except: print(...)`。
  参考 commit `85f0ed9`。
- **Qt 惰性导入**：`annotator_core.py` 顶层**不** import PySide6，相关 `from PySide6...`
  都写在方法内（`register_hotkeys` / `_msg_box` / `_process_events` / `_start_*`）。这样纯逻辑方法
  可在没有 Qt 的环境里单测。`app.py` / `canvas_widget.py` 顶层可自由 import Qt。
- **主线程不跑长任务**（强约束）：模型加载、OCR、`save_all`、Excel 生成、PDF 渲染
  全部必须走 `worker.py` 的 `QThread`。主线程里直接调 `OcrHandle.text_predict` /
  `save_item_boxes_to_disk` / `build_excel_for_single_item` 会让 Qt 事件循环阻塞，
  界面"未响应"。新增这类调用前先想清楚是同步还是异步。
- **worker 不得碰 GUI**：`worker.py` 内禁止出现 `QPainter` / `QImage` / `QMessageBox` /
  `.setText()` / `.update()` 等。结果只能通过 `pyqtSignal` 回主线程。
  `tests/test_worker.py::WorkerStructureTests::test_ocr_worker_does_not_touch_gui` 会强制检查。
- **错误**：模块顶层禁止 `exit()` / `sys.exit()`（会让 GUI 静默退出），改抛 `RuntimeError`，
  见 `cutbatch.py:35`。
- **保存流程**（批量保存未访问页时）：优先用内存 boxes → 回退 JSON → 再回退 OCR 重跑。
  见 `storage_utils.extract_boxes_from_ocr_results`。
- **清理旧切片**：保存前 `clear_existing_crops`，避免删框后旧切片残留污染 Excel。
- **PDF 页数**：始终通过 `get_pdf_page_count` 取，不要硬编码。
- **参数校验**：`app.py` 的 `_parse_int` / `_parse_float` 已带上下限，新参数沿用同套校验。
- **新增快捷键**：在 `annotator_core.register_hotkeys` 的 bindings 表里加一行；
  若涉及新字母，无需在别处登记（`QShortcut(QKeySequence("X"))` 直接覆盖大小写两种 Shift 状态，
  因为表里同时给了 `"X"` 和 `"Shift+X"`）。

## 仓库布局提醒

`.gitignore` 已忽略以下生成物，**不要**提交：

- `output/` `interactive_output/` `data/` `outputs/` `docs/` `ppt_assets/`（运行 / PPT 任务产物）
- `*.xlsx` `*.xls`（标注结果）
- `build/` `dist/` `*.spec` 产物（`app.spec` 本身要提交）
- `.idea/` `.vscode/` `.zcode/` `.superpowers/` `__pycache__/`（IDE / 工具元数据）

## 坐标与缩放词汇表（避免混用）

| 术语              | 含义                                                         |
| ----------------- | ------------------------------------------------------------ |
| `base_image_bgr`  | 原始图片 / PDF 渲染页，boxes / 切片都基于它  |
| `view_image_bgr`  | 仅显示用，`base * view_scale`                                |
| `view_scale`      | 显示缩放，**不**写入 boxes、**不**影响切片清晰度             |
| `image_scale`     | OCR 副本缩放，**不**进入 `base_image_bgr`，**不**改变原图与切片     |
| `viewport_offset` | 画布在 view image 上的左上角（图像大于窗口时）               |
| `draw_origin`     | 图像小于窗口时的居中偏移                                     |
| `_view_version`   | view_image_bgr 版本号，`_rebuild_view_image` 自增；canvas 用它做 QImage 缓存指纹 |

- OCR 使用最长边受 `config.ocr_max_side` 限制的独立副本，检测/识别/笔画细化完成后按实际宽高比例回映原图；切勿将副本用于保存。
- 重新开始将旧推理 worker 标记为 retired，保留引用至线程结束，旧结果按请求号/会话代数丢弃；新会话只检查当前会话 worker。关闭窗口仍必须检查所有 worker，保存线程不得提前隔离。

- `_load_item(target_index)` 只启动 `PageLoadWorker`，原图解码、PDF 渲染、JSON 读取和 RGB→BGR 都在 worker；`_apply_loaded_item` 只合并内存结果。页面加载期间不提前改写当前页索引。
- `save_current` 只启动 `SaveCurrentWorker`；保存成功且编辑版本未变化才继续 `_navigation_target`，失败/取消留在当前页。切片逐张裁剪并暂存，仍保留整页事务回滚。
- 所有 `OcrHandle` 推理统一经过 `ocr_image.py` 的限尺寸副本和实际宽高比例回映；`text_predict_file` 可用 JPEG 缩小解码。非正 `ocr_max_side` 回退 2560，不能放开超大图推理。
- `SaveCurrentWorker` 在原图切片仍在内存时生成有上限的 Excel 缩略图，供 `build_excel_for_single_item(cached_thumbnails=...)` 复用；仅缓存缩略图，不缓存全部原图切片。Excel 内存打包，减少 Windows 临时文件 IO。
