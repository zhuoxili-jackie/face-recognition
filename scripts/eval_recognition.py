"""人脸识别率评估。

把模型检测到的人脸框，与标签文件里的真实人脸框做 IoU 匹配，统计：
  GT      = 标签中的真实人脸数
  Det     = 模型检测到的人脸数
  TP      = 命中的真实人脸数（与某个 GT 框 IoU ≥ 阈值）
  Recall  = TP / GT      —— 识别率（应识别的里识别出了多少）
  Prec    = TP / Det     —— 精确率（识别出的里有多少是对的）

标签来源：优先 datasets/labels/<split>/<stem>.txt，回退 archive/labels/<stem>.txt。
标签为 YOLO 归一化格式：cls cx cy w h。

用法：
  python scripts/eval_recognition.py --source datasets/images/test
  python scripts/eval_recognition.py --images wider_3784 wider_9691
  python scripts/eval_recognition.py --source datasets/images/test --conf 0.25 --iou 0.5
"""
import os
import argparse

import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

from common import ROOT

DEFAULT_WEIGHT = os.path.join(ROOT, "model", "weights", "yolov10n-face.pt")
ARCHIVE_LABELS = os.path.join(ROOT, "archive", "labels")


def find_label(stem):
    """按文件名找标签：先在 datasets/labels/* 里找，再回退到 archive/labels。"""
    for split in ("test", "val", "train"):
        p = os.path.join(ROOT, "datasets", "labels", split, stem + ".txt")
        if os.path.isfile(p):
            return p
    p = os.path.join(ARCHIVE_LABELS, stem + ".txt")
    return p if os.path.isfile(p) else None


def load_gt_boxes(label_path, w, h):
    """读取 YOLO 标签，返回像素坐标的 (x1,y1,x2,y2) 列表。"""
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)


def iou(box, boxes):
    """单框 box 对一组 boxes 的 IoU。"""
    if len(boxes) == 0:
        return np.empty((0,), np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0).astype(np.float32)


def match(pred, conf, gt, iou_thr):
    """按置信度从高到低贪心匹配，返回命中的预测框下标集合（即 TP 的那些预测）。"""
    matched = set()
    if len(pred) == 0 or len(gt) == 0:
        return matched
    order = np.argsort(-conf)
    used = np.zeros(len(gt), dtype=bool)
    for i in order:
        ious = iou(pred[i], gt)
        ious[used] = -1
        j = int(np.argmax(ious)) if len(ious) else -1
        if j >= 0 and ious[j] >= iou_thr:
            used[j] = True
            matched.add(int(i))
    return matched


def dense_clusters(pred, R, min_pts, min_cluster):
    """对人脸框中心做 DBSCAN。

    返回 (circles, dense_idx)：
      circles   = 密集区外接圆列表 [(cx,cy,radius), ...]
      dense_idx = 落在密集区内的人脸框下标集合（这些框画红色，其余画绿色）
    """
    if len(pred) == 0:
        return [], set()
    centers = np.stack([(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2], axis=1)
    labels = DBSCAN(eps=R, min_samples=min_pts).fit_predict(centers)
    circles, dense_idx = [], set()
    for lab in set(labels):
        if lab == -1:
            continue
        idxs = np.where(labels == lab)[0]
        if len(idxs) >= min_cluster:
            pts = centers[idxs].astype(np.float32)
            (cx, cy), radius = cv2.minEnclosingCircle(pts)
            circles.append((cx, cy, radius * 1.05))
            dense_idx.update(int(i) for i in idxs)
    return circles, dense_idx


def draw_annotated(img, pred, conf, dense_idx, gt_n, prec, circles):
    """画人群密集区黄圈 + 人脸框（圈内=红, 圈外=绿）+ 置信度，顶部标注 GT/Det/Prec/Clusters。"""
    # 先画黄圈（密集区），让框叠在上面
    for cx, cy, radius in circles:
        cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 255, 255), 3)

    for i, (x1, y1, x2, y2) in enumerate(pred):
        color = (0, 0, 255) if i in dense_idx else (0, 255, 0)  # 红=密集区内, 绿=密集区外
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{conf[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(int(y1), th + 3)
        cv2.rectangle(img, (p1[0], ty - th - 3), (p1[0] + tw + 2, ty), color, -1)
        cv2.putText(img, label, (p1[0] + 1, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    header = f"GT: {gt_n}  Det: {len(pred)}  Prec: {prec:.1%}  Clusters: {len(circles)}"
    (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (0, 0), (tw + 12, th + 14), (0, 0, 0), -1)
    cv2.putText(img, header, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/images/test", help="待评估图片目录")
    ap.add_argument("--images", nargs="*", default=None,
                    help="只评估这些图（文件名，可省略扩展名），从 archive/images 取图")
    ap.add_argument("--weights", default=DEFAULT_WEIGHT)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5, help="判为命中的 IoU 阈值")
    ap.add_argument("--R", type=float, default=80, help="DBSCAN 邻域半径（像素）")
    ap.add_argument("--min-pts", type=int, default=3, help="DBSCAN 核心点最少邻居数")
    ap.add_argument("--min-cluster", type=int, default=5, help="判为人群密集区的最小人脸数")
    ap.add_argument("--out", default="image_results/eval", help="保存 csv 的目录")
    args = ap.parse_args()

    # 组织待评估图片列表
    if args.images:
        img_paths = []
        for name in args.images:
            stem = os.path.splitext(name)[0]
            for d in (os.path.join(ROOT, "archive", "images"),
                      os.path.join(ROOT, "datasets", "images", "test")):
                cand = os.path.join(d, stem + ".jpg")
                if os.path.isfile(cand):
                    img_paths.append(cand)
                    break
            else:
                print(f"  [warn] 找不到图片 {name}")
    else:
        src = args.source if os.path.isabs(args.source) else os.path.abspath(os.path.join(ROOT, args.source))
        img_paths = [os.path.join(src, f) for f in sorted(os.listdir(src))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    model = YOLO(args.weights)
    out_dir = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"权重: {args.weights}  conf={args.conf}  IoU匹配阈值={args.iou}\n")
    header = f"{'image':28s} {'GT':>4s} {'Det':>4s} {'TP':>4s} {'Recall':>8s} {'Prec':>8s} {'Clu':>4s}"
    print(header)
    print("-" * len(header))

    rows = []
    tot_gt = tot_det = tot_tp = 0
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = find_label(stem)
        if not label_path:
            print(f"{stem:28s} 无标签，跳过")
            continue
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        gt = load_gt_boxes(label_path, w, h)

        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        if len(r.boxes):
            pred = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
        else:
            pred = np.empty((0, 4), np.float32)
            conf = np.empty((0,), np.float32)

        matched = match(pred, conf, gt, args.iou)
        tp = len(matched)
        recall = tp / len(gt) if len(gt) else 0.0
        prec = tp / len(pred) if len(pred) else 0.0
        circles, dense_idx = dense_clusters(pred, args.R, args.min_pts, args.min_cluster)
        n_clusters = len(circles)
        rows.append((stem, len(gt), len(pred), tp, recall, prec, n_clusters))
        tot_gt += len(gt); tot_det += len(pred); tot_tp += tp
        print(f"{stem:28s} {len(gt):4d} {len(pred):4d} {tp:4d} {recall:8.1%} {prec:8.1%} {n_clusters:4d}")

        # 保存标注图：黄圈(密集区) + 框(圈内红/圈外绿) + 置信度 + 顶部 GT/Det/Prec/Clusters
        draw_annotated(img, pred, conf, dense_idx, len(gt), prec, circles)
        ext = os.path.splitext(img_path)[1]
        cv2.imencode(ext, img)[1].tofile(os.path.join(out_dir, stem + ext))

    print("-" * len(header))
    o_rec = tot_tp / tot_gt if tot_gt else 0.0
    o_prec = tot_tp / tot_det if tot_det else 0.0
    print(f"{'总计/overall':28s} {tot_gt:4d} {tot_det:4d} {tot_tp:4d} {o_rec:8.1%} {o_prec:8.1%}")

    csv_path = os.path.join(out_dir, "recognition_rate.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,gt_faces,detected,tp,recall,precision,clusters\n")
        for stem, g, d, tp, rec, pr, nclu in rows:
            f.write(f"{stem},{g},{d},{tp},{rec:.4f},{pr:.4f},{nclu}\n")
        f.write(f"overall,{tot_gt},{tot_det},{tot_tp},{o_rec:.4f},{o_prec:.4f},\n")
    print(f"\n已写入：{csv_path}")


if __name__ == "__main__":
    main()
