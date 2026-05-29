"""YOLO 人脸检测器（Ultralytics）。

封装原 predict_face.py / eval_recognition.py 里的 YOLO 调用逻辑，
使其符合 FaceDetector 接口：输入 BGR 图，输出 (boxes, confs)。
"""
from .base import FaceDetector, empty_result


class YoloDetector(FaceDetector):
    def __init__(self, weights, conf=0.25, imgsz=640):
        # 延迟导入，避免没装 ultralytics 时影响其它检测器（如 VJ）的使用。
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, img):
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if len(r.boxes):
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return empty_result()
