import onnxruntime as ort

ort.set_default_logger_severity(3)  # ERROR level, 忽略 warning
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import copy
import logging
import traceback
import random
import glob  # 用于查找文件
import shutil  # 用于复制文件

logger = logging.getLogger(__name__)

# --- 项目相关导入 ---
try:
    # 确保这些路径和变量在 config.py 中是正确的
    from config import model_path, crnn_model_path
    import config as app_config

    # 为可选参数提供默认值，以防它们在 config.py 中不存在
    angle_detect = getattr(app_config, "angle_detect", False)
    angle_net_path = getattr(app_config, "angle_net_path", "")
    angle_detect_num = getattr(app_config, "angle_detect_num", 10)
    # 检测模型选择："legacy" / "v6-tiny" / "v6-small" / "v6-medium"
    det_model = getattr(app_config, "det_model", "legacy")
    ppocrv6_det_model_path = getattr(
        app_config, "ppocrv6_det_model_path", {}
    )
    ppocrv6_det_limit_type = getattr(
        app_config, "ppocrv6_det_limit_type", "short"
    )
    ppocrv6_det_max_side_limit = getattr(
        app_config, "ppocrv6_det_max_side_limit", 4000
    )
    ppocrv6_det_box_thresh = getattr(
        app_config, "ppocrv6_det_box_thresh", None
    )
    # 识别模型选择："crnn" / "v6-tiny" / "v6-small" / "v6-medium"（见 config.py 注释）
    rec_model = getattr(app_config, "rec_model", "crnn")
    char_box_strategy = getattr(
        app_config, "char_box_strategy", "ctc-midpoint"
    )
    ppocrv6_rec_model_path = getattr(
        app_config, "ppocrv6_rec_model_path", {}
    )

    from crnn import CRNNHandle
    # from angnet import AngleNetHandle # 如果有角度检测，取消注释
    from utils import get_rotate_crop_image, validate_and_prepare_polygon_points  # 确保这两个函数在 utils.py 中且导入正确
    from dbnet.dbnet_infer import DBNET
    from ppocrv6 import PPOcrV6Detector, PPOcrV6RecHandle
except ImportError as e:
    # 注意：模块顶层不能调用 exit()，否则在 import 链中（例如 GUI 入口 app.py
    # 通过 annotator_core → cutbatch 间接 import）会让整个进程静默退出，
    # 上层无法捕获。改为抛 RuntimeError，让上层显示清晰的错误提示。
    raise RuntimeError(
        f"cutbatch 导入失败：{e}。"
        f"请确保 config.py / crnn.py / utils.py / dbnet/dbnet_infer.py 存在且路径正确，"
        f"并已安装 onnxruntime / opencv-python / numpy 等依赖。"
    ) from e


# --- 项目相关导入结束 ---


def _build_det_handle():
    """根据 ``config.det_model`` 构造检测器并返回 ``(handle, name)``。"""

    if det_model == "legacy":
        return DBNET(model_path), "legacy-dbnet"
    if det_model.startswith("v6-"):
        tier = det_model.split("-", 1)[1]
        path = ppocrv6_det_model_path.get(tier)
        if not path:
            raise RuntimeError(
                f"未知 v6 检测档位：{det_model}（可选 tiny/small/medium）"
            )
        if not os.path.exists(path):
            raise RuntimeError(
                f"v6 det 模型文件不存在：{path}\n"
                "下载方式见 bench/README.md"
                f"（huggingface_hub 拉取 PP-OCRv6_{tier}_det_onnx）"
            )
        return (
            PPOcrV6Detector(
                path,
                limit_type=ppocrv6_det_limit_type,
                max_side_limit=ppocrv6_det_max_side_limit,
                box_thresh=ppocrv6_det_box_thresh,
            ),
            f"v6-{tier}",
        )
    raise RuntimeError(
        f"未知 det_model：{det_model}"
        "（可选 legacy / v6-tiny / v6-small / v6-medium）"
    )


def _build_rec_handle():
    """根据 config.rec_model 选择识别模型 handle。

    返回 (handle, name) —— name 仅用于日志/调试。
    所有 handle 共享同一接口：predict_rbg(crop_pil, bbox, **rec_options) -> dict
    （schema: {'raw', 'line_pred', 'char_list', 'words'}）。
    """
    if rec_model == "crnn":
        return CRNNHandle(crnn_model_path), "CRNN"
    if rec_model.startswith("v6-"):
        tier = rec_model.split("-", 1)[1]   # tiny / small / medium
        path = ppocrv6_rec_model_path.get(tier)
        if not path:
            raise RuntimeError(f"未知 v6 档位：{rec_model}（可选 tiny/small/medium）")
        if not os.path.exists(path):
            raise RuntimeError(
                f"v6 rec 模型文件不存在：{path}\n"
                f"下载方式见 bench/README.md（huggingface_hub 拉取 PP-OCRv6_{tier}_rec_onnx）"
            )
        return (
            PPOcrV6RecHandle(
                path, char_box_strategy=char_box_strategy
            ),
            f"v6-{tier}",
        )
    raise RuntimeError(f"未知 rec_model：{rec_model}（可选 crnn / v6-tiny / v6-small / v6-medium）")


def _deduplicate_text_rows(boxes, scores):
    """Suppress near-identical detector proposals, keeping legitimate overlap.

    Polygon IoU and similar area are both required; shared strokes or one
    region contained in a much larger line are not treated as duplicates.
    """
    kept = []
    polygons = [cv2.convexHull(np.asarray(box, np.float32).reshape(-1, 2)) for box in boxes]
    areas = [abs(cv2.contourArea(polygon)) for polygon in polygons]
    for index in sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True):
        duplicate = False
        for previous in kept:
            small, large = min(areas[index], areas[previous]), max(areas[index], areas[previous])
            if small <= 0 or small / large < .85:
                continue
            intersection, _ = cv2.intersectConvexConvex(polygons[index], polygons[previous])
            union = areas[index] + areas[previous] - intersection
            if union > 0 and intersection / union >= .85:
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
    kept.sort()
    return [boxes[i] for i in kept], [scores[i] for i in kept]


class OcrHandle(object):
    def __init__(self):
        self.text_handle, self.det_name = _build_det_handle()
        self.crnn_handle, self.rec_name = _build_rec_handle()
        logger.info(
            "OCR 后端：det=%s, rec=%s, char_box=%s",
            self.det_name,
            self.rec_name,
            char_box_strategy if self.rec_name.startswith("v6-") else "legacy",
        )
        # if angle_detect:
        #     self.angle_handle = AngleNetHandle(angle_net_path)

    def crnnRecWithBox(self, im_opencv_bgr, boxes_list_from_dbnet, score_list_from_dbnet, cancel_check=None):
        """CRNN 批量识别。

        性能：把所有 crop 出来的行一次性塞进 predict_rbg_batch，
        按宽度分桶后每个桶一次 ONNX 推理。一张 N 行文字的图，
        相比旧的 for 循环逐行推理快 3-10×（固定开销被摊掉）。

        返回结构与历史版本一致：[{'bbox', 'result', 'score'}, ...]
        """
        # 1) 先把所有行裁出来；失败的或空的，记下索引后续填空结果。
        valid_items = []   # list of (orig_index, crop_pil_rgb, bbox_deepcopy)
        empty_indices = set()
        for index, (box_coords_quad, _score) in enumerate(zip(boxes_list_from_dbnet, score_list_from_dbnet)):
            cropped_text_region_np_bgr = None
            try:
                cropped_text_region_np_bgr = get_rotate_crop_image(
                    im_opencv_bgr, box_coords_quad.astype(np.float32))
            except cv2.error as e_cv_crop:
                logger.warning("crnnRecWithBox get_rotate_crop_image OpenCV 错误 (box %d): %s", index, e_cv_crop)
            except Exception as e_crop:
                logger.warning("crnnRecWithBox get_rotate_crop_image 未知错误 (box %d): %s", index, e_crop)

            if cropped_text_region_np_bgr is None or cropped_text_region_np_bgr.size == 0:
                empty_indices.add(index)
                continue

            try:
                # cropped 是 BGR；CRNN 要 RGB。用 [:, :, ::-1] 零拷贝反转通道，
                # 比 cvtColor 快（不发生内存复制之外的颜色矩阵计算）。
                cropped_rgb = cropped_text_region_np_bgr[:, :, ::-1]
                crop_pil_rgb = Image.fromarray(np.ascontiguousarray(cropped_rgb))
                valid_items.append((index, crop_pil_rgb, copy.deepcopy(box_coords_quad)))
            except Exception as e_conv:
                logger.warning("行 %d 转 PIL 失败：%s", index, e_conv)
                empty_indices.add(index)

        # 2) 批量推理。某一行的退化几何不应让整页失败：批量路径异常时
        # 降级成逐行识别，并只跳过真正失败的那一行。
        crnn_inputs = [(pil, bbox) for _idx, pil, bbox in valid_items]
        rec_options = ({'defer_stroke_refinement': True}
                       if isinstance(self.crnn_handle, PPOcrV6RecHandle) else {})
        results_by_index = {}
        if crnn_inputs:
            try:
                if cancel_check is None:
                    crnn_outputs = self.crnn_handle.predict_rbg_batch(crnn_inputs, **rec_options)
                else:
                    crnn_outputs = []
                    for entry in crnn_inputs:
                        if cancel_check():
                            return []
                        crnn_outputs.extend(self.crnn_handle.predict_rbg_batch([entry], **rec_options))
            except Exception as batch_error:
                logger.warning(
                    "批量识别失败，降级为逐行隔离：%s", batch_error
                )
                for idx, crop_pil, bbox in valid_items:
                    if cancel_check is not None and cancel_check():
                        return []
                    try:
                        results_by_index[idx] = (
                            self.crnn_handle.predict_rbg(crop_pil, bbox, **rec_options)
                        )
                    except Exception as row_error:
                        logger.warning(
                            "跳过识别失败的文本行 %d：%s",
                            idx,
                            row_error,
                        )
            else:
                for (idx, _pil, _bbox), result in zip(
                    valid_items, crnn_outputs
                ):
                    results_by_index[idx] = result

        # 3) 装回原顺序；空 crop / 单行失败的位置用默认空结果。
        # 用 dict 按 orig_index 存，避免 batch 返回顺序与原 boxes 顺序耦合。
        page_rows = [
            {'bbox': box, 'score': score,
             'result': results_by_index.get(index, {'line_pred':'','char_list':[],'words':[]})}
            for index, (box, score) in enumerate(zip(boxes_list_from_dbnet, score_list_from_dbnet))
        ]
        if cancel_check is not None and cancel_check():
            return []
        from ppocrv6.page_layout import refine_regular_rubbing_page
        # Regular pages use a shared size/spacing estimate before expensive
        # row-level ownership. Other layouts keep the existing path.
        handled = refine_regular_rubbing_page(im_opencv_bgr, page_rows)
        results = []
        for index, (box_coords_quad, score) in enumerate(zip(boxes_list_from_dbnet, score_list_from_dbnet)):
            if cancel_check is not None and cancel_check():
                return []
            result = page_rows[index]['result']
            if result.get("char_box_strategy") == "stroke-ownership" and index not in handled:
                # A region rejected by both models is often a seal fragment or
                # ornament. Keep its raw recognition for diagnostics, but do
                # not manufacture an annotation from that weak agreement.
                recognition_score = result.get("score")
                if (recognition_score is not None and np.isfinite(recognition_score)
                        and float(score) < 0.60 and float(recognition_score) < 0.40):
                    result["suppressed_candidate"] = {
                        "reason": "low_detection_and_recognition_confidence",
                        "text": result.get("line_pred", ""),
                        "det_score": float(score),
                        "rec_score": float(recognition_score),
                    }
                    result["char_list"] = []
                    result["words"] = []
                    result["line_pred"] = ""
                    logger.info("忽略双低置信度文本候选：det=%.3f, rec=%.3f", score, recognition_score)
                from ppocrv6.stroke_ownership import refine_stroke_row
                try:
                    refine_stroke_row(im_opencv_bgr, box_coords_quad, result)
                except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                    logger.warning("行 %d 的原图笔画细化失败，保留行内字框：%s", index, error)
            if result.get("char_box_strategy") in {"independent-strokes", "stroke-ownership"}:
                # Padding may extend beyond the detector row, but must stay
                # inside the original image, in the same base coordinates.
                image_h, image_w = im_opencv_bgr.shape[:2]
                for obj in result.get("char_list", []) + result.get("words", []):
                    polygon = np.asarray(obj["bbox"], dtype=np.float32).copy()
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_w)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_h)
                    obj["bbox"] = polygon
            results.append({'bbox': box_coords_quad, 'result': result, 'score': score})
        return results

    def text_predict(self, pil_image_rgb, dbnet_short_size, *, cancel_check=None):
        from ocr_image import prepare_ocr_image
        if cancel_check is not None and cancel_check():
            return []
        prepared = prepare_ocr_image(pil_image_rgb, getattr(self,'ocr_image_scale',1.0),
                                     getattr(app_config,'ocr_max_side',2560))
        return self._predict_prepared(prepared, dbnet_short_size, cancel_check)

    def text_predict_file(self, path, dbnet_short_size, *, cancel_check=None):
        from ocr_image import load_ocr_image
        if cancel_check is not None and cancel_check():
            return []
        prepared = load_ocr_image(path, getattr(self,'ocr_image_scale',1.0),
                                  getattr(app_config,'ocr_max_side',2560))
        return self._predict_prepared(prepared, dbnet_short_size, cancel_check)

    def _predict_prepared(self, prepared, dbnet_short_size, cancel_check=None):
        from ocr_image import map_results_to_source
        image_rgb = np.asarray(prepared.image,dtype=np.uint8)
        if prepared.source_size != prepared.target_size:
            # Do not upscale a deliberately bounded proxy back to a huge
            # detector tensor when a legacy short-side value is oversized.
            dbnet_short_size=min(dbnet_short_size,min(prepared.target_size))
        logger.info('OCR 原图 %s -> 推理 %s，检测短边 %s',
                    prepared.source_size, prepared.target_size, dbnet_short_size)
        # 旧 DBNET.process 的公开输入是 BGR，函数内部再转 RGB；v6 检测器
        # 则明确接收 RGB。这里显式分流，消除历史上的双重通道歧义。
        if self.det_name == "legacy-dbnet":
            detector_input = np.ascontiguousarray(image_rgb[:, :, ::-1])
        else:
            detector_input = image_rgb
        text_boxes_detected, text_scores_detected = self.text_handle.process(
            detector_input, dbnet_short_size
        )

        text_boxes_detected, text_scores_detected = _deduplicate_text_rows(
            text_boxes_detected, text_scores_detected)
        if cancel_check is not None and cancel_check():
            return []
        # OCR 副本给识别 crop 用：历史是 BGR。把 RGB 反转通道得到 BGR，
        # 用 ::-1 view 代替 cvtColor，零内存拷贝、结果完全一致。
        img_opencv_bgr_original = np.ascontiguousarray(image_rgb[:, :, ::-1])
        options = {"cancel_check": cancel_check} if cancel_check is not None else {}
        ocr_full_results = self.crnnRecWithBox(
            img_opencv_bgr_original, text_boxes_detected, text_scores_detected, **options)
        return map_results_to_source(ocr_full_results, prepared)


def get_random_bgr_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# validate_and_prepare_polygon_points 函数应该在 utils.py 中，并从那里导入
# from utils import validate_and_prepare_polygon_points

if __name__ == "__main__":
    # ---------- 参数配置区 ----------
    INPUT_IMAGE_FOLDER = "/home/asus/Oracle/cut/cut1"
    OUTPUT_ROOT_DIRECTORY = "/home/asus/Oracle/cut/cut_result12"
    PROCESSING_LEVEL = "char"
    DBNET_SHORT_SIDE = 630 # big:630 small:1280!!!!!
    SAVE_DEBUG_IMAGE = True
    DEBUG_FONT_PATH = "仿宋_GB2312.ttf"
    DEBUG_FONT_SIZE_DEFAULT = 15
    # ---------- 参数配置区结束 ----------

    print(f"--- OCR 图像处理开始 ---")
    print(f"输入图片文件夹: {os.path.abspath(INPUT_IMAGE_FOLDER)}")
    print(f"输出根目录: {os.path.abspath(OUTPUT_ROOT_DIRECTORY)}")
    print(f"处理级别: {PROCESSING_LEVEL}")
    print(f"DBNET 短边尺寸: {DBNET_SHORT_SIDE}")

    os.makedirs(OUTPUT_ROOT_DIRECTORY, exist_ok=True)
    if not os.path.isdir(INPUT_IMAGE_FOLDER):
        print(f"错误: 输入文件夹 '{INPUT_IMAGE_FOLDER}' 未找到。请创建该文件夹并将图片放入其中。")
        exit()

    supported_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_file_paths = []
    for ext in supported_extensions:
        image_file_paths.extend(glob.glob(os.path.join(INPUT_IMAGE_FOLDER, ext)))

    if not image_file_paths:
        print(f"在文件夹 '{INPUT_IMAGE_FOLDER}' 中未找到支持的图片文件。")
        exit()

    print(f"找到 {len(image_file_paths)} 张图片进行处理...")

    try:
        ocr_handler_instance = OcrHandle()
    except Exception as e_init:  # 捕获OcrHandle初始化时的所有错误
        print(f"初始化 OcrHandle 失败: {e_init}")
        traceback.print_exc()
        exit()

    debug_pil_font = None
    if SAVE_DEBUG_IMAGE:
        try:
            debug_pil_font = ImageFont.truetype(DEBUG_FONT_PATH, size=DEBUG_FONT_SIZE_DEFAULT)
        except IOError:
            print(f"警告: 字体文件 '{DEBUG_FONT_PATH}' 未找到。调试图像中的文本将使用默认字体。")
            debug_pil_font = ImageFont.load_default()

    total_images_processed_count = 0
    for img_idx_loop, current_image_full_path in enumerate(image_file_paths):
        print(f"\n[{img_idx_loop + 1}/{len(image_file_paths)}] 正在处理图片: {current_image_full_path}")

        pil_rgb_image_original = None
        opencv_bgr_img_original = None
        image_processing_failed_fatally = False  # 标记当前图片处理是否遭遇严重错误

        # 步骤 1: 加载和初步转换图片
        try:
            pil_rgb_image_original = Image.open(current_image_full_path).convert('RGB')
            opencv_bgr_img_original = cv2.cvtColor(np.array(pil_rgb_image_original), cv2.COLOR_RGB2BGR)
            if opencv_bgr_img_original is None or opencv_bgr_img_original.size == 0:
                print(
                    f"  错误: OpenCV 图像 (opencv_bgr_img_original) 在转换后为空或无效. 图片: {current_image_full_path}.")
                image_processing_failed_fatally = True
        except FileNotFoundError:
            print(f"  错误: 图片文件未找到于 {current_image_full_path}. 跳过此图片的处理.")
            continue  # 直接跳到下一张图片，不尝试复制
        except Exception as e_open_convert:
            print(f"  错误: 打开或转换图片 {current_image_full_path} 失败: {e_open_convert}.")
            image_processing_failed_fatally = True  # 标记为失败，后续会尝试复制

        # 创建当前图片的输出子目录（无论处理是否成功）
        img_file_basename_no_ext = os.path.splitext(os.path.basename(current_image_full_path))[0]
        output_subdir_current_img = os.path.join(OUTPUT_ROOT_DIRECTORY,
                                                 f"{img_file_basename_no_ext}_processed")  # 统一命名
        os.makedirs(output_subdir_current_img, exist_ok=True)

        # 步骤 2: 只有在图片加载成功时才进行 OCR 和裁剪
        if not image_processing_failed_fatally:
            try:
                # 步骤 2a: OCR 预测
                all_ocr_block_results = ocr_handler_instance.text_predict(pil_rgb_image_original, DBNET_SHORT_SIDE)

                if not all_ocr_block_results:
                    print(f"  未在图片 {current_image_full_path} 中检测到任何文本内容（DBNET未返回结果）。")
                    # 这种情况不标记为 fatal error，但也不会有字符被切割

                # 准备调试图像绘制上下文 (如果需要)
                pil_img_for_debug = None
                pillow_draw_ctx = None
                if SAVE_DEBUG_IMAGE and pil_rgb_image_original:
                    pil_img_for_debug = pil_rgb_image_original.copy()
                    pillow_draw_ctx = ImageDraw.Draw(pil_img_for_debug)

                items_cropped_this_image = 0

                # 步骤 2b: 字符/词/行 提取和保存 (这是最可能发生底层错误的地方)
                # 将这个核心循环包裹在 try-except 中
                try:
                    if PROCESSING_LEVEL == "char":
                        for text_block_res in all_ocr_block_results:  # 遍历 DBNET 检测到的每个文本块
                            recognition_res_details = text_block_res.get('result', {})
                            if 'char_list' in recognition_res_details and recognition_res_details['char_list']:
                                for char_item_data in recognition_res_details['char_list']:  # 遍历块内的每个字符
                                    char_text_val = char_item_data.get('char', '')
                                    char_poly_pts_data = char_item_data.get('bbox')

                                    char_poly_np_validated = validate_and_prepare_polygon_points(char_poly_pts_data,
                                                                                                 f"字符 '{char_text_val}'")
                                    if char_poly_np_validated is None:
                                        continue

                                        # 调用 get_rotate_crop_image
                                    cropped_char_bgr = get_rotate_crop_image(opencv_bgr_img_original,
                                                                             char_poly_np_validated)

                                    if cropped_char_bgr is None or cropped_char_bgr.size == 0:
                                        # print(f"    警告: 字符 '{char_text_val}' 裁剪结果为空或失败。")
                                        continue

                                    items_cropped_this_image += 1
                                    output_filename = f"char_{items_cropped_this_image:04d}.png"
                                    output_path = os.path.join(output_subdir_current_img, output_filename)
                                    cv2.imwrite(output_path, cropped_char_bgr)

                                    if SAVE_DEBUG_IMAGE and pillow_draw_ctx and debug_pil_font:
                                        try:
                                            color_bgr_tuple = get_random_bgr_color()  # OpenCV BGR tuple
                                            pillow_draw_ctx.polygon(char_poly_np_validated.flatten().tolist(),
                                                                    outline=(0, 238, 238), width=20)
                                        except Exception as e_draw_char:
                                            print(f"    绘制字符bbox时出错: {e_draw_char}")

                    elif PROCESSING_LEVEL in ["word", "line"]:
                        # ... (word/line 级别的处理逻辑，也需要类似的错误处理) ...
                        # (为了简洁，这里省略，但您需要确保这部分也有对 get_rotate_crop_image 的调用保护)
                        # 例如:
                        for text_block_res in all_ocr_block_results:
                            polygons_to_crop = []  # ... (填充 polygons_to_crop) ...
                            if PROCESSING_LEVEL == "line":  # ...
                                if text_block_res.get('bbox') is not None: polygons_to_crop.append(
                                    text_block_res['bbox'])
                            # ... (word 级别逻辑) ...

                            for poly_pts_data in polygons_to_crop:
                                poly_np_validated = validate_and_prepare_polygon_points(poly_pts_data, PROCESSING_LEVEL)
                                if poly_np_validated is None: continue
                                cropped_item_bgr = get_rotate_crop_image(opencv_bgr_img_original, poly_np_validated)
                                if cropped_item_bgr is None or cropped_item_bgr.size == 0: continue
                                items_cropped_this_image += 1
                                # ... (保存和绘制) ...

                    if items_cropped_this_image > 0:
                        print(f"  已成功切割并保存 {items_cropped_this_image} 个 '{PROCESSING_LEVEL}' 级别的图像块。")
                    elif all_ocr_block_results:  # 有检测结果但没切出字符
                        print(
                            f"  在级别 '{PROCESSING_LEVEL}' 下，未切割出任何图像块 (可能所有字符裁剪均失败或无有效字符bbox)。")
                    # 如果 all_ocr_block_results 为空，则之前已经打印过无检测内容

                except cv2.error as e_cv_processing:  # 捕获在字符/块处理循环中发生的 OpenCV 特定错误
                    print(
                        f"  错误 (致命): 在处理字符/块时发生 OpenCV 错误: {e_cv_processing}. 图片 '{current_image_full_path}' 的切割中止。")
                    traceback.print_exc()
                    image_processing_failed_fatally = True  # 标记此图片处理失败
                except Exception as e_processing_general:  # 捕获其他 Python 错误
                    print(
                        f"  错误 (致命): 在处理字符/块时发生未知错误: {e_processing_general}. 图片 '{current_image_full_path}' 的切割中止。")
                    traceback.print_exc()
                    image_processing_failed_fatally = True  # 标记此图片处理失败
                # -------- 字符/块处理循环的 try-except 结束 --------

                # 保存调试图 (仅当未发生致命错误，并且有东西可画时)
                if SAVE_DEBUG_IMAGE and pil_img_for_debug and not image_processing_failed_fatally:
                    debug_output_filename = f"{img_file_basename_no_ext}_ed.jpg"
                    if items_cropped_this_image == 0 and all_ocr_block_results:  # 有检测框但没切出字符
                        debug_output_filename = f"{img_file_basename_no_ext}_debug_no_crop.jpg"

                    debug_output_path = os.path.join(OUTPUT_ROOT_DIRECTORY, debug_output_filename)
                    try:
                        pil_img_for_debug.save(debug_output_path, format='JPEG')
                        if items_cropped_this_image > 0 or not all_ocr_block_results:  # 正常或无检测结果
                            print(f"  已保存调试图像到: {debug_output_path}")
                        else:  # 有检测框但没切出
                            print(f"  已保存调试图像（无成功裁剪）到: {debug_output_path}")
                    except Exception as e_save_dbg:
                        print(f"  错误: 保存调试图像 '{debug_output_path}' 失败: {e_save_dbg}")

            except Exception as e_text_predict:  # 捕获 ocr_handler_instance.text_predict 本身的错误
                print(
                    f"  错误 (致命): 对图片 {current_image_full_path} 进行 ocr_predict 时发生严重错误: {e_text_predict}. 将复制原图。")
                traceback.print_exc()
                image_processing_failed_fatally = True

        # 步骤 3: 如果当前图片的处理过程中发生了致命错误，复制原图
        if image_processing_failed_fatally:
            print(f"  由于处理 '{current_image_full_path}' 过程中发生错误，将直接复制原图片。")
            failed_image_copy_name = f"{img_file_basename_no_ext}_FAILED_ORIGINAL{os.path.splitext(current_image_full_path)[1]}"  # 保留原扩展名
            failed_image_copy_path = os.path.join(output_subdir_current_img, failed_image_copy_name)
            try:
                shutil.copyfile(current_image_full_path, failed_image_copy_path)
                print(f"  已将问题图片复制到: {failed_image_copy_path}")
            except Exception as e_copy_failed:
                print(
                    f"  错误: 复制问题图片 '{current_image_full_path}' 到 '{failed_image_copy_path}' 失败: {e_copy_failed}")

        total_images_processed_count += 1

    print(f"\n--- 所有图片处理完毕 ---")
    print(f"总共尝试处理了 {total_images_processed_count} / {len(image_file_paths)} 张图片。")
    print(f"结果保存在根目录: {os.path.abspath(OUTPUT_ROOT_DIRECTORY)}")
