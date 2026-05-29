"""训练人脸检测模型（封装 yolov5/train.py）。

用法：
  python scripts/train.py                 # 默认 img640 / batch16 / epochs100
  python scripts/train.py --epochs 50 --batch 8

结果保存到 runs/face_yolov5s/，最佳权重为 runs/face_yolov5s/weights/best.pt。
"""
import os
import sys
import subprocess
import argparse

from common import find_yolov5_dir, find_base_weights, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--name", default="face_yolov5s")
    args = ap.parse_args()

    yolov5_dir = find_yolov5_dir()
    data_yaml = os.path.join(ROOT, "data.yaml")
    train_script = os.path.join(yolov5_dir, "train.py")

    cmd = [
        sys.executable, train_script,
        "--cache", "ram",
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", data_yaml,
        "--cfg", os.path.join(yolov5_dir, "models", "yolov5s.yaml"),
        "--weights", find_base_weights(),
        "--name", args.name,
        "--project", os.path.join(ROOT, "runs"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
