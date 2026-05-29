"""人脸密度估计（免训练版）。

直接加载一个**预训练的人脸检测模型**（Ultralytics YOLO，如 yolov10n-face.pt），
对图片检测人脸 -> 取人脸框中心点 -> DBSCAN 密度聚类 -> 圈出异常聚集 + 标注人脸数。

与 scripts/predict.py 的区别：
  - predict.py 走 yolov5/detect.py 子进程，需要自己训练出的 YOLOv5 权重；
  - 本脚本用 ultralytics 包直接加载现成的人脸权重，**无需训练，开箱即出人脸结果**。

颜色约定：绿点=普通人脸  红点=异常聚集内人脸  黄圈=异常聚集区

用法：
  python scripts/predict_face.py
  python scripts/predict_face.py --source datasets/images/test --conf 0.3 --R 60 --min-cluster 5
"""
import os
import argparse

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from common import ROOT

DEFAULT_WEIGHT = os.path.join(ROOT, "model", "weights", "yolov10n-face.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/images/test", help="待检测图片目录")
    ap.add_argument("--weights", default=DEFAULT_WEIGHT, help="预训练人脸权重")
    ap.add_argument("--out", default="image_results/face_latest", help="结果图保存目录")
    ap.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    ap.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    ap.add_argument("--R", type=float, default=80, help="DBSCAN 邻域半径（像素）")
    ap.add_argument("--min-pts", type=int, default=3, help="DBSCAN 核心点最少邻居数")
    ap.add_argument("--min-cluster", type=int, default=5, help="判为异常聚集的最小人脸数")
    args = ap.parse_args()

    source_abs = args.source if os.path.isabs(args.source) else os.path.abspath(os.path.join(ROOT, args.source))
    if not os.path.isdir(source_abs):
        raise FileNotFoundError(f"找不到图片目录：{source_abs}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"找不到权重：{args.weights}")

    out_dir = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(out_dir, exist_ok=True)

    print("\033[35m------ 人脸密度估计（免训练）------\033[0m")
    print(f"  weights: {args.weights}")
    print(f"  source:  {source_abs}")
    print(f"  output:  {out_dir}")
    print(f"  conf={args.conf}, R={args.R}, MinPts={args.min_pts}, 异常聚集阈值={args.min_cluster}")

    model = YOLO(args.weights)

    images = [
        os.path.join(source_abs, f)
        for f in sorted(os.listdir(source_abs))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    summary = []
    for img_path in images:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] 无法读取 {img_path}")
            continue

        result = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        # 人脸框中心点（像素坐标，已是原图尺度）
        centers = result.boxes.xywh.cpu().numpy()[:, :2] if len(result.boxes) else np.empty((0, 2))

        n_faces = len(centers)
        abnormal_points = set()
        n_clusters = 0

        if n_faces:
            labels = DBSCAN(eps=args.R, min_samples=args.min_pts).fit_predict(centers)
            clusters = {}
            for i, lab in enumerate(labels):
                clusters.setdefault(lab, []).append(centers[i])
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

            for pt in centers:
                color = (0, 0, 255) if tuple(pt) in abnormal_points else (0, 255, 0)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1)

        text = f"faces: {n_faces}  clusters: {n_clusters}"
        cv2.rectangle(img, (0, 0), (10 + 11 * len(text), 34), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ext = os.path.splitext(img_path)[1]
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imencode(ext, img)[1].tofile(out_path)
        summary.append((os.path.basename(img_path), n_faces, n_clusters))
        print(f"  {os.path.basename(img_path):40s} faces={n_faces:4d}  clusters={n_clusters}")

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
