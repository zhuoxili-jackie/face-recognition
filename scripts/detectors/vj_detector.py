"""Viola-Jones 人脸检测器（OpenCV 自带 Haar 级联）。

OpenCV 的 cv2.CascadeClassifier 就是原始 VJ 方法（积分图 + 类 Haar 特征 +
AdaBoost + 级联）的现成实现，开箱即用、无需训练。

VJ 本身不输出概率分数。这里用 detectMultiScale3(outputRejectLevels=True) 返回的
levelWeights（窗口通过级联的层级权重，越大越可信）作为分数，供下游排序与画 ROC。
"""
import cv2
import numpy as np

from .base import FaceDetector, empty_result


class VJOpenCVDetector(FaceDetector):
    def __init__(self, cascade_xml=None, scale_factor=1.1, min_neighbors=3, min_size=24):
        """
        参数：
          cascade_xml:   Haar 级联 xml 路径，默认用 OpenCV 自带的正脸级联。
          scale_factor:  图像金字塔每层的缩放比例（越接近 1 越慢越全）。
          min_neighbors: 合并候选框所需的最少邻居数（越大越严、误检越少）。
          min_size:      可检测的最小人脸边长（像素），小于此尺寸的脸会被忽略。
        """
        path = cascade_xml or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cc = cv2.CascadeClassifier(path)
        if self.cc.empty():
            raise FileNotFoundError(f"无法加载 Haar 级联：{path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 直方图均衡，提升对光照的鲁棒性
        rects, _, weights = self.cc.detectMultiScale3(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
            outputRejectLevels=True,
        )
        if len(rects) == 0:
            return empty_result()
        # (x, y, w, h) -> (x1, y1, x2, y2)
        rects = np.asarray(rects, dtype=np.float32)
        boxes = np.column_stack([
            rects[:, 0],
            rects[:, 1],
            rects[:, 0] + rects[:, 2],
            rects[:, 1] + rects[:, 3],
        ]).astype(np.float32)
        confs = np.asarray(weights, dtype=np.float32).ravel()  # levelWeights 作为分数
        return boxes, confs
