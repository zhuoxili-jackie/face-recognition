"""可插拔人脸检测器。

通过 build_detector(name, ...) 按名字构造检测器，下游脚本只认 FaceDetector 接口。
目前支持：
  - "yolo": Ultralytics YOLO（默认，行为与改造前一致）
"""
from .base import FaceDetector, empty_result


def build_detector(name, weights=None, conf=0.25, imgsz=640):
    """按名字构造检测器。

    参数：
      name:    检测器名（"yolo"）。
      weights: 模型权重路径（YOLO 需要）。
      conf:    检测置信度阈值。
      imgsz:   推理输入尺寸（YOLO 用）。
    """
    name = (name or "yolo").lower()
    if name == "yolo":
        from .yolo_detector import YoloDetector

        return YoloDetector(weights=weights, conf=conf, imgsz=imgsz)
    raise ValueError(f"未知检测器：{name}（可选：yolo）")


__all__ = ["FaceDetector", "empty_result", "build_detector"]
