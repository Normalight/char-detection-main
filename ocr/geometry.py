"""OCR 行裁剪、坐标回映与点序；保存原图切片使用 storage_utils。"""
import math

import cv2
import numpy as np

def get_rotate_crop_image(img, points):
    """透视裁剪 OCR 文本行；高宽比 >= 1.5 时转为横向供识别器使用。

    保存用户切片时使用 utils.storage.get_rotate_crop_image 保留原始方向。
    """
    # --- 关键修复：复制一份坐标，避免修改原始传入的数组 ---
    points = points.copy()
    # ---------------------------------------------------

    # 确保 points 是 float32 类型的 numpy 数组，以进行后续计算
    points = np.asarray(points, dtype=np.float32)

    # 1. 先进行一个大致的矩形裁剪，减少后续处理的计算量
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))

    # 确保裁剪坐标不会超出图像边界
    left = max(0, left)
    top = max(0, top)
    right = min(img.shape[1], right)
    bottom = min(img.shape[0], bottom)

    img_crop = img[top:bottom, left:right, :].copy()

    if img_crop.size == 0:
        return None

    # 将世界坐标转换为相对于裁剪出的小图的坐标
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top

    # 2. 精确计算目标图像的宽度和高度
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))

    if img_crop_width <= 0 or img_crop_height <= 0:
        return None

    # 3. 执行透视变换
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)

    dst_img = cv2.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE)

    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img


# 旋转公式来源：星夜孤帆，CC BY-SA 4.0
# https://blog.csdn.net/qq_38826019/article/details/84233397
def rotate_point_clockwise(angle,valuex,valuey,pointx,pointy):
  valuex = np.array(valuex)
  valuey = np.array(valuey)
  sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
  sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
  return [sRotatex,sRotatey]


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
            new_point = rotate_point_clockwise(math.radians(-90), new_point[0], new_point[1], 0 , 0)
            new_point[0] = new_point[0] + img_crop_width

        p = np.float32(new_point + [1])
        x, y, z = np.dot(IM, p)
        new_point = [x/z , y/z]

        new_point = [int(new_point[0] + left), int(new_point[1] + top)]

        raw_points.append(new_point)
    return raw_points


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

