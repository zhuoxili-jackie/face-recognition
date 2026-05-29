"""检测器统一接口。

整条流水线（DBSCAN 密度聚类 / 可视化 / eval_recognition 的 ROC、Recall、Prec）
只依赖检测器吐出的「人脸框 + 分数」。把检测器抽象成本接口后，YOLO 与 VJ
（Viola-Jones）可互换，下游代码无需改动。
"""
import numpy as np


class FaceDetector:
    """统一检测器接口：输入 BGR 图(np.ndarray)，输出人脸框与分数。"""

    def detect(self, img):
        """检测单张图中的人脸。

        参数：
          img: np.ndarray，BGR 格式（cv2 读出的原图），形状 (H, W, 3)。

        返回 (boxes, confs):
          boxes: np.ndarray (N, 4) float —— 每张脸的方框 [x1, y1, x2, y2]，像素坐标。
          confs: np.ndarray (N,)  float —— 对应分数（越大越像人脸），
                 用于排序 / 按阈值扫描画 ROC。无脸时返回空数组。
        """
        raise NotImplementedError


def empty_result():
    """无检测结果时的统一空返回，避免下游对形状做特判。"""
    return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
