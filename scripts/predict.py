"""人脸检测 + 人脸密度统计 + 异常聚集可视化。

流程（与 ref 的人群版一致，只是检测对象从「整个人」换成「人脸」）：
  1. 调用 yolov5/detect.py 对 SOURCE 目录里的图片做人脸检测，导出 YOLO 标签；
  2. 取每张人脸框中心点，用 DBSCAN 做密度聚类；
  3. 簇内人脸数 ≥ 阈值 的判为「异常聚集」，用黄色圆圈圈出；
  4. 在图上标注总人脸数 / 聚集区数量，并保存结果图。

颜色约定：
  绿色实心点 = 普通人脸中心   红色实心点 = 属于异常聚集的人脸   黄色圆圈 = 异常聚集区

用法：
  python scripts/predict.py
  python scripts/predict.py --source datasets/images/test --R 60 --min-cluster 5
"""
import os
import argparse
import subprocess

import numpy as np
import cv2
from sklearn.cluster import DBSCAN

from common import find_yolov5_dir, default_trained_weight, ROOT


def parse_yolo_label(lbl_path):
    """读取 YOLO 标签，返回 [(cls, cx, cy, w, h), ...]（归一化坐标）。"""
    if not lbl_path or not os.path.exists(lbl_path):
        return None
    boxes = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cls, x, y, w, h))
    return boxes


def run_detection(source_abs, weight, yolov5_dir, detect_name):
    cmd = [
        "python", os.path.join(yolov5_dir, "detect.py"),
        "--weights", weight,
        "--source", source_abs,
        "--img", "640",
        "--conf", "0.25",
        "--iou-thres", "0.15",
        "--save-txt",
        "--save-conf",
        "--project", os.path.join(ROOT, "runs"),
        "--name", detect_name,
        "--exist-ok",
        "--nosave",
    ]
    print("Running detection:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/images/test", help="待检测图片目录")
    ap.add_argument("--weights", default=None, help="模型权重，默认用训练得到的 best.pt")
    ap.add_argument("--out", default="image_results/latest", help="结果图保存目录")
    ap.add_argument("--R", type=float, default=80, help="DBSCAN 邻域半径（像素）")
    ap.add_argument("--min-pts", type=int, default=3, help="DBSCAN 核心点最少邻居数")
    ap.add_argument("--min-cluster", type=int, default=5, help="判为异常聚集的最小人脸数")
    ap.add_argument("--detect-name", default="face_yolov5s_detect", help="detect 输出子目录名")
    args = ap.parse_args()

    source_abs = os.path.abspath(os.path.join(ROOT, args.source)) if not os.path.isabs(args.source) else args.source
    if not os.path.isdir(source_abs):
        raise FileNotFoundError(f"找不到图片目录：{source_abs}")

    weight = os.path.abspath(args.weights) if args.weights else default_trained_weight()
    if not os.path.isfile(weight):
        raise FileNotFoundError(
            f"找不到模型权重：{weight}\n请先运行 python scripts/train.py 进行训练，"
            "或用 --weights 指定一个已有的人脸检测权重。"
        )

    yolov5_dir = find_yolov5_dir()
    label_dir = os.path.join(ROOT, "runs", args.detect_name, "labels")
    out_dir = os.path.abspath(os.path.join(ROOT, args.out)) if not os.path.isabs(args.out) else args.out
    os.makedirs(out_dir, exist_ok=True)

    print("\033[35m------ 人脸密度估计 ------\033[0m")
    print(f"  R={args.R}, MinPts={args.min_pts}, 异常聚集阈值={args.min_cluster}")
    print(f"  weights: {weight}")
    print(f"  source:  {source_abs}")
    print(f"  output:  {out_dir}")

    run_detection(source_abs, weight, yolov5_dir, args.detect_name)
    print("\n\033[35m------ 检测完成，开始密度聚类与可视化 ------\033[0m")

    images = [
        os.path.join(source_abs, f)
        for f in sorted(os.listdir(source_abs))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    summary = []
    for img_path in images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        parsed = parse_yolo_label(os.path.join(label_dir, stem + ".txt"))

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] 无法读取图片 {img_path}")
            continue
        h, w = img.shape[:2]

        n_faces = 0 if not parsed else len(parsed)
        abnormal_points = set()
        n_clusters = 0

        if parsed:
            points = np.array([(x * w, y * h) for _, x, y, _, _ in parsed])

            labels = DBSCAN(eps=args.R, min_samples=args.min_pts).fit_predict(points)
            clusters = {}
            for i, lab in enumerate(labels):
                clusters.setdefault(lab, []).append(points[i])

            abnormal_clusters = [
                pts for lab, pts in clusters.items()
                if lab != -1 and len(pts) >= args.min_cluster
            ]
            n_clusters = len(abnormal_clusters)
            for cluster in abnormal_clusters:
                abnormal_points.update(tuple(pt) for pt in cluster)
                pts = np.array(cluster, dtype=np.float32)
                (cx, cy), radius = cv2.minEnclosingCircle(pts)
                radius *= 1.05
                cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 255, 255), 3)

            for pt in points:
                color = (0, 0, 255) if tuple(pt) in abnormal_points else (0, 255, 0)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1)

        # 在左上角标注总人脸数与聚集区数量
        text = f"faces: {n_faces}  clusters: {n_clusters}"
        cv2.rectangle(img, (0, 0), (10 + 11 * len(text), 34), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ext = os.path.splitext(img_path)[1]
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imencode(ext, img)[1].tofile(out_path)
        summary.append((os.path.basename(img_path), n_faces, n_clusters))
        print(f"  {os.path.basename(img_path):40s} faces={n_faces:4d}  clusters={n_clusters}")

    # 写一份汇总 csv
    csv_path = os.path.join(out_dir, "_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,face_count,cluster_count\n")
        for name, nf, nc in summary:
            f.write(f"{name},{nf},{nc}\n")
    total = sum(nf for _, nf, _ in summary)
    print(f"\n共处理 {len(summary)} 张图，累计检测人脸 {total} 张。")
    print(f"汇总已写入：{csv_path}")


if __name__ == "__main__":
    main()
