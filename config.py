# -*- coding: utf-8 -*-
import sys
import os

# -----------------------------------------------------------------------------
# 运行路径适配：支持源码调试 和 PyInstaller onefile 打包后的临时解包目录
# -----------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后会把所有 datas 解压到 _MEIPASS
    father_path = sys._MEIPASS
else:
    # 普通脚本运行时，以当前文件夹为基准
    filt_path    = os.path.abspath(__file__)
    father_path  = os.path.dirname(filt_path)

# -----------------------------------------------------------------------------
# dbnet 参数
# -----------------------------------------------------------------------------
dbnet_max_size = 6000  # 长边最大长度
pad_size       = 0     # 检测时 pad 尺寸，有些文档文字充满整个屏幕检测有误，需要 pad

# -----------------------------------------------------------------------------
# crnn 参数
# -----------------------------------------------------------------------------
crnn_lite       = True
model_path      = os.path.join(father_path, "models", "dbnet.onnx")
is_rgb          = True
crnn_model_path = os.path.join(father_path, "models", "crnn_lite_lstm.onnx")

# -----------------------------------------------------------------------------
# OCR 后端选择
# -----------------------------------------------------------------------------
# det_model = "legacy"   用项目原 DBNet
# det_model = "v6-small" 用 PP-OCRv6 small det（官方 ONNX + inference.yml）
det_model = "v6-small"
# 检测输入缩放策略：
#   short —— 短边缩放到"检测尺寸"（可上采样也可下采样），与 legacy DBNet 语义
#            一致。小图会被上采样，避免低分辨率漏检（默认）。
#   max   —— 最长边压到"检测尺寸"且永不上采样（PaddleOCR 移动端默认）。小图会
#            以原始低分辨率进检测，细字/窄列易漏检，UI 调大尺寸也救不回来。
#   min   —— 只上采样到短边达到"检测尺寸"，不下采样。
ppocrv6_det_limit_type = "short"
ppocrv6_det_max_side_limit = 4000
# 检测框分数阈值覆盖；None = 用官方 inference.yml 的 box_thresh。
# 漏检窄列/淡墨（如碑文小字旁注被合并）时可降到 0.4~0.5 召回，代价是可能多出
# 误检框，改动后需人工核查。
ppocrv6_det_box_thresh = None
ppocrv6_det_model_path = {
    "tiny": os.path.join(
        father_path, "models", "ppocrv6", "tiny", "inference.onnx"
    ),
    "small": os.path.join(
        father_path, "models", "ppocrv6", "small", "inference.onnx"
    ),
    "medium": os.path.join(
        father_path, "models", "ppocrv6", "medium", "inference.onnx"
    ),
}

# rec_model = "crnn"     用项目原 CRNN（crnn_lite_lstm.onnx，4.7MB，最快）
# rec_model = "v6-small" 用 PP-OCRv6 small rec（中文识别明显更准，速度慢 ~6×）
# rec_model = "v6-tiny"  用 PP-OCRv6 tiny rec（最轻量，偶尔重复字）
# rec_model = "v6-medium" 用 PP-OCRv6 medium rec（最准，但 77MB + 慢 15×）
#
# v6 模型文件在 models/ppocrv6[_rec]/<tier>/（gitignored，走 HF 下载，
# 下载方式见 README.md）。polarity-aware 会按背景明暗选择切线证据：
# 亮背景使用 Otsu 墨迹谷微调，纹理复杂的暗背景保留 CTC 中点；
# 首尾框保留检测行完整边缘，避免自适应收缩裁掉外伸笔画；
# projection-refined 保留旧固定极性路径，ctc-midpoint 则只使用中点。
rec_model = "v6-small"
# 按强弱笔画连接与偏旁归属估计四边，只留 1 像素余量，允许框自然重叠。
char_box_strategy = "stroke-ownership"
ppocrv6_rec_model_path = {
    "tiny":   os.path.join(father_path, "models", "ppocrv6_rec", "tiny",   "inference.onnx"),
    "small":  os.path.join(father_path, "models", "ppocrv6_rec", "small",  "inference.onnx"),
    "medium": os.path.join(father_path, "models", "ppocrv6_rec", "medium", "inference.onnx"),
}

# -----------------------------------------------------------------------------
# angle 参数
# -----------------------------------------------------------------------------
angle_detect     = False
angle_detect_num = 30
angle_net_path   = os.path.join(father_path, "models", "angle_net.onnx")

# -----------------------------------------------------------------------------
# 其他
# -----------------------------------------------------------------------------
max_post_time = 100  # IP 访问最大次数
from crnn.keys import alphabetChinese as alphabet
white_ips     = []   # 白名单
version       = 'api/v1'

# OCR proxy only: preserve original pixels for display, coordinates and crops.
ocr_max_side = 2560
