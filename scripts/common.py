"""定位 YOLOv5 源码与基础权重的公共工具。

本项目不重复存放体积庞大的 YOLOv5 源码，默认复用 ref/train-yolo-v5/yolov5。
也可以通过环境变量 YOLOV5_DIR 指定其它位置（例如自己 git clone 的 yolov5）。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def find_yolov5_dir():
    candidates = []
    env = os.environ.get("YOLOV5_DIR")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(ROOT, "yolov5"))                       # 项目内自带
    candidates.append(os.path.join(ROOT, "ref", "train-yolo-v5", "yolov5"))  # 复用 ref
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "train.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "找不到 YOLOv5 源码。请设置环境变量 YOLOV5_DIR 指向 yolov5 目录，"
        "或在项目根目录执行：git clone https://github.com/ultralytics/yolov5"
    )


def find_base_weights():
    """训练用的预训练基础权重 yolov5s.pt。"""
    yolov5_dir = find_yolov5_dir()
    p = os.path.join(yolov5_dir, "yolov5s.pt")
    return p  # 不存在时 yolov5 会自动下载


def default_trained_weight():
    """优先用本项目训练出的权重，否则回退到 ref 的人体检测权重作占位。"""
    cands = [
        os.path.join(ROOT, "runs", "face_yolov5s", "weights", "best.pt"),
        os.path.join(ROOT, "model", "weights", "face_detector.pt"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return cands[0]  # 返回期望路径，提示用户先训练
