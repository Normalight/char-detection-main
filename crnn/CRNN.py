from PIL import  Image
import numpy as np
import math
import cv2
import re
import os
import logging
from .keys import alphabetChinese as alphabet
# from keys import alphabetChinese as alphabet

import onnxruntime as rt
# from util import strLabelConverter, resizeNormalize
from .util import strLabelConverter, resizeNormalize
# from ..utils import reverse_rotate_crop_image

logger = logging.getLogger(__name__)


def _make_session_opts():
    """CPU 多线程 + 图优化，不开 GPU。CRNN 单行很快，但批量推理时仍能受益。"""
    opts = rt.SessionOptions()
    cpu = os.cpu_count() or 1
    physical = max(1, cpu // 2)
    opts.intra_op_num_threads = physical
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts

# 绕pointx,pointy顺时针旋转
def Srotate(angle,valuex,valuey,pointx,pointy):
  valuex = np.array(valuex)
  valuey = np.array(valuey)
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return [sRotatex,sRotatey]
# ————————————————
# 版权声明：本文为CSDN博主「星夜孤帆」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_38826019/article/details/84233397

def reverse_rotate_crop_image(img, part_img, points, part_points, origin):
    """
    get_rotate_crop_image的逆操作
    img为原图
    part_img为crop后的图
    bbox_points为part_img中对应在原图的bbox, 四个点，左上，右上，右下，左下
    part_points为在part_img中的点[(x, y), (x, y)]
    """
    # np.rot
    # 关键：必须 copy，不能直接 np.float32(points)。
    # 当 points 已经是 float32 时，np.float32() 返回原数组（不复制），
    # 后面 points[:, 0] -= left 会改坏调用方传入的 bbox，导致同一 bbox
    # 第二次进本函数时已经被减成局部坐标（典型表现：字框全堆到图像原点）。
    # int16/float64 输入因 dtype 不同恰好触发了复制，掩盖了这个 bug。
    points = np.array(points, dtype=np.float32)
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    # img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],\
        [img_crop_width, img_crop_height], [0, img_crop_height]])
    # print(points, points.shape, pts_std, pts_std.shape)


    M = cv2.getPerspectiveTransform(points, pts_std)
    _, IM = cv2.invert(M)
    raw_points = []
    # if img_crop_height * 1.0 / img_crop_width >= 1.5:
    #     part_points.reverse()

    for point in part_points:
        new_point = point
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            new_point = Srotate(math.radians(-90), new_point[0], new_point[1], 0 , 0)
            new_point[0] = new_point[0] + img_crop_width

        p = np.float32(new_point + [1])
        x, y, z = np.dot(IM, p)
        new_point = [x/z , y/z]

        new_point = [int(new_point[0] + left), int(new_point[1] + top)]

        raw_points.append(new_point)
    return raw_points


def expand_line_to_box(line, w):
    """
    line [(x1, y1), (x2, y2)]
    """
    # radio =



converter = strLabelConverter(''.join(alphabet))

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def order_points(box):
    '''矩形框顺序排列
    :param box: numpy.array, shape=(4, 2)
    :return:
    '''
    box = np.array(box).reshape((-1, 2))
    center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
    if np.any(box[:, 0] == center_x) and np.any(box[:, 1] == center_y):  # 有两点横坐标相等，有两点纵坐标相等，菱形
        p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
        p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
        p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
        p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
    elif np.all(box[:, 0] == center_x):  # 四个点的横坐标都相同
        y_sort = np.argsort(box[:, 1])
        p1 = box[y_sort[0]]
        p2 = box[y_sort[1]]
        p3 = box[y_sort[2]]
        p4 = box[y_sort[3]]
    elif np.any(box[:, 0] == center_x) and np.all(box[:, 1] != center_y):  # 只有两点横坐标相等，先上下再左右
        p12, p34 = box[np.where(box[:, 1] < center_y)], box[np.where(box[:, 1] > center_y)]
        p1, p2 = p12[np.where(p12[:, 0] == np.min(p12[:, 0]))], p12[np.where(p12[:, 0] == np.max(p12[:, 0]))]
        p3, p4 = p34[np.where(p34[:, 0] == np.max(p34[:, 0]))], p34[np.where(p34[:, 0] == np.min(p34[:, 0]))]
    else:  # 只有两点纵坐标相等，或者是没有相等的，先左右再上下
        p14, p23 = box[np.where(box[:, 0] < center_x)], box[np.where(box[:, 0] > center_x)]
        p1, p4 = p14[np.where(p14[:, 1] == np.min(p14[:, 1]))], p14[np.where(p14[:, 1] == np.max(p14[:, 1]))]
        p2, p3 = p23[np.where(p23[:, 1] == np.min(p23[:, 1]))], p23[np.where(p23[:, 1] == np.max(p23[:, 1]))]

    return np.array([p1, p2, p3, p4]).reshape((-1, 2)).tolist()
class CRNNHandle:
    def __init__(self, model_path):
        # CPU 多线程 + 图优化；providers 显式声明 CPU，避免误用 CUDA。
        self.sess = rt.InferenceSession(
            model_path,
            sess_options=_make_session_opts(),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, image):
        """
        预测
        """
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))

        image = transformer(image)

        image = image.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)

        preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})

        preds = preds[0]


        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)


        sim_pred = converter.decode(preds, length, raw=False)

        return sim_pred



    def _preprocess(self, crop_im):
        """单行预处理：返回 (transformed_image(N,3,32,w), scale, w)。

        分离出来便于批量：预处理是纯 CPU/IO，可以逐个做；
        只有 sess.run 必须拼成 batch 一次性发出去。
        """
        scale = crop_im.size[1] * 1.0 / 32
        w = crop_im.size[0] / scale
        w = int(w)

        img = crop_im.resize((w, 32), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        img -= 127.5
        img /= 127.5
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        return transformed_image.astype(np.float32), scale, w

    def predict_rbg(self, crop_im, bbox):
        """单行预测。

        历史性能瓶颈：DBNet 检测出的每行单独调本方法，ONNX 推理的固定开销
        （kernel launch、内存拷贝）被乘以行数。本可把多行拼成 batch 一次推理，
        但**实测此 ONNX 模型在 batch>1 时会跨样本互相影响**（疑为某算子被融合
        到 batch 维），导致单字末尾标点错位。为保识别正确性，本项目仍走逐行推理；
        真要批量需换一个 batch-safe 重导出的 crnn.onnx。
        """
        transformed_image, scale, w = self._preprocess(crop_im)
        preds = self.sess.run(["out"], {"input": transformed_image})[0]
        return self._postprocess(preds, scale, w, crop_im, bbox)

    def predict_rbg_batch(self, items):
        """批量预测（兼容接口）。

        当前实现等价于逐行调 predict_rbg：见 predict_rbg 文档说明的 batch 不安全
        问题。保留方法签名以便未来换 batch-safe 模型后直接改成真批量，调用方
        （cutbatch.crnnRecWithBox）不需要改。
        """
        return [self.predict_rbg(crop, bbox) for crop, bbox in items]

    def _postprocess(self, preds, scale, w, crop_im, bbox):
        """CTC 解码 + 单字坐标还原。从 predict_rbg 抽出，供单行/批量共用。"""
        length = preds.shape[0]
        raw_scores = preds.reshape(length, -1).astype(np.float64)
        class_indexes = np.argmax(raw_scores, axis=1).reshape(-1)

        # 兼容模型直接输出概率、log-probability 或 logits。旧实现只保留
        # argmax，导致 CRNN 后端没有字符置信度，无法安全判断是否启用
        # 低置信度几何兜底。
        row_sums = raw_scores.sum(axis=1)
        if (
            np.all(raw_scores >= 0.0)
            and np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-4)
        ):
            probabilities = raw_scores
        else:
            shifted = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / np.maximum(
                exp_scores.sum(axis=1, keepdims=True),
                np.finfo(np.float64).tiny,
            )

        result = converter.decode(class_indexes, length, raw=True)
        raw, line_pred, char_list = result['raw'], result['line_pred'], result['char_list']
        for char_item in char_list:
            time_index = int(char_item.get('idx', -1))
            if 0 <= time_index < length:
                class_index = int(class_indexes[time_index])
                char_item['score'] = float(
                    probabilities[time_index, class_index]
                )
        result['score'] = (
            float(np.mean([item['score'] for item in char_list]))
            if char_list
            else 0.0
        )

        prev_line = [[0, 0], [0, crop_im.size[1]]]
        prev_line = reverse_rotate_crop_image(None, None, bbox, prev_line, (0, 0))
        ws = []
        for i, char_item in enumerate(char_list):  # 添加原始坐标
            crop_left = char_item['idx'] * 4 * scale  # 在裁剪中的坐标
            line = [[crop_left, 0], [crop_left, crop_im.size[1]]]
            line = reverse_rotate_crop_image(None, None, bbox, line, (0, 0))
            char_item['line'] = line

            w_char = ((line[0][0] - prev_line[0][0]) ** 2 + (line[0][1] - prev_line[0][1]) ** 2) ** 0.5
            # Bug 1（历史）：旧版 w_char = min(w_char, crop_im.size[1] / 2) 把字框宽度
            # 钳成"行高的一半"，但 crop_im.size[1] 是 crop 图高（横排=行高，竖排 rot90 后=字宽），
            # 在横排矮行（行高 ~24px）场景会把窄字母挤成 1-9px，叠加下面的旧版重叠消除
            # 逻辑后会退化成竖条。改成宽松钳制：不超过行高本身（保留足够宽度给字母）。
            w_char = min(w_char, crop_im.size[1])
            ws.append([crop_left - w_char, crop_left + w_char])

            prev_line = line
        # 调整 bbox 有重叠的地方。
        # Bug 2（历史）：旧版对重叠的两个框各动 4 个边界（cur 左右、nxt 左右各缩 distance/2），
        # 实际重叠只有 distance 一份，结果两边都被多缩一倍、挤成竖条。
        # 正确做法：只在两个框相邻的边界取中点作为分界（cur[1] 和 nxt[0] 的中点），
        # cur[0] / nxt[1] 不动，保留各自宽度。
        for i in range(len(ws) - 1):
            cur, nxt = ws[i], ws[i+1]
            if cur[1] > nxt[0]:  # 有交集
                mid = (cur[1] + nxt[0]) / 2
                cur[1] = mid
                nxt[0] = mid

        for (left, right), char_item in zip(ws, char_list):
            rect = [[left, 0], [left, crop_im.size[1]], [right, crop_im.size[1]], [right, 0]]
            rect = reverse_rotate_crop_image(None, None, bbox, rect, (0, 0))
            rect = order_points(rect)
            char_item['bbox'] = rect

        def group_char_to_word(char_items):
            bboxs = [x['bbox'] for x in char_items]
            # np.int0 已 deprecated（NumPy 1.20+），改成 np.intp（等价于原 np.int0 的索引 dtype）。
            points = np.array(bboxs, dtype=np.intp).reshape((-1, 1, 2))
            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect)
            bbox = order_points(bbox)
            bbox = np.array(bbox, dtype=np.intp)
            return {'value': ''.join(x['char'] for x in char_items), 'bbox': bbox}

        words = []
        # 使用中文 \u4e00-\u9fa5 和英文符号 ,.?;\()[] 进行分割
        # raw string 避免 SyntaxWarning（'\\\(' 等无效转义）。
        matchs = list(re.finditer(r'[\u4e00-\u9fa5 ,.?:/\\()\[\]]', line_pred))
        prev_idx = 0
        for match in matchs:
            left, right = match.span()
            if prev_idx != left:
                words.append(group_char_to_word(char_list[prev_idx: left]))
            words.append(group_char_to_word(char_list[left: right]))
            prev_idx = right

        if prev_idx != len(char_list):
            words.append(group_char_to_word(char_list[prev_idx:]))

        result['words'] = words
        return result



if __name__ == "__main__":
    im = Image.open("471594277244_.pic.jpg")
    crnn_handle = CRNNHandle(model_path="../models/crnn_lite_lstm_bk.onnx")
    print(crnn_handle.predict(im))
