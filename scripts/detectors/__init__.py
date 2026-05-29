"""可插拔人脸检测器。

通过 build_detector(name, ...) 按名字构造检测器，下游脚本只认 FaceDetector 接口。
目前支持：
  - "yolo": Ultralytics YOLO（默认，行为与改造前一致）
  - "vj":   Viola-Jones（OpenCV Haar 级联），免训练
"""
from .base import FaceDetector, empty_result


def build_detector(name, weights=None, conf=0.25, imgsz=640):
    """按名字构造检测器。

    参数：
      name:    检测器名（"yolo" / "vj"）。
      weights: 模型权重路径（YOLO 需要；vj 可传 Haar 级联 xml 路径，None 用自带正脸级联）。
      conf:    检测置信度阈值（YOLO 用；vj 不使用，按 levelWeights 给分）。
      imgsz:   推理输入尺寸（YOLO 用）。
    """
    name = (name or "yolo").lower()
    if name == "yolo":
        from .yolo_detector import YoloDetector

        return YoloDetector(weights=weights, conf=conf, imgsz=imgsz)
    if name == "vj":
        from .vj_detector import VJOpenCVDetector

        # 对 vj 而言 weights 若指向 .xml 则作为自定义级联，否则用自带正脸级联。
        cascade_xml = weights if (weights and str(weights).lower().endswith(".xml")) else None
        return VJOpenCVDetector(cascade_xml=cascade_xml)
    raise ValueError(f"未知检测器：{name}（可选：yolo / vj）")


__all__ = ["FaceDetector", "empty_result", "build_detector", "AVAILABLE"]

AVAILABLE = ("yolo", "vj")
