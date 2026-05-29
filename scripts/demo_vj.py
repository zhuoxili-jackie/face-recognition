"""VJ 进度演示脚本（给老师看「我已经开始了 + 我理解 VJ 在做什么」）。

一条命令产出三样东西：

  1) 对比图  image_results/demo/compare_vj_vs_yolo.png
     —— 同几张图，左列 VJ（实验室方法：OpenCV Haar 级联），右列 YOLO（对照）。
        直观证明检测器已被抽象成可插拔模块，VJ 已接入同一条流水线。

  2) 指标表（终端打印 + csv）
     —— VJ 与 YOLO 在同一批图上的 Recall / Precision 对比，体现两者差异。

  3) 原理图  image_results/demo/vj_principle.png
     —— 证明理解 VJ 框架的三大件：
        (a) 积分图：演示任意矩形和可 O(1) 计算（用暴力求和数值校验，结果完全一致）；
        (b) 类 Haar 特征：画出 VJ 经典的 2/3/4 矩形特征模板；
        (c) 把经典「双矩形」Haar 特征贴到真实人脸上（眼睛区比颧骨区暗）——
            这正是 VJ 用来区分人脸/非人脸的最基础线索。

用法：
  python scripts/demo_vj.py
  python scripts/demo_vj.py --n 4 --source datasets/images/test
"""
import os
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import ROOT
from detectors import build_detector
# 复用评估脚本里现成的标签读取 / IoU 匹配逻辑
from eval_recognition import find_label, load_gt_boxes, match

# 让 matplotlib 能正常显示中文（Windows 常见字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_WEIGHT = os.path.join(ROOT, "model", "weights", "yolov10n-face.pt")


def draw_boxes(img_bgr, boxes, color):
    """在图副本上画框，返回 RGB 图（供 matplotlib 显示）。"""
    vis = img_bgr.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def pick_images(source_abs, n, vj):
    """优先挑 VJ 能检出人脸的图，让演示更直观；不足则用前几张补齐。"""
    files = [f for f in sorted(os.listdir(source_abs))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    hit, rest = [], []
    for f in files:
        p = os.path.join(source_abs, f)
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        boxes, _ = vj.detect(img)
        (hit if len(boxes) else rest).append(p)
        if len(hit) >= n:
            break
    chosen = hit[:n]
    if len(chosen) < n:
        chosen += rest[: n - len(chosen)]
    return chosen


def make_compare_figure(img_paths, vj, yolo, out_path):
    """左列 VJ、右列 YOLO 的并排检测对比图，并顺便算 Recall/Prec 指标。"""
    n = len(img_paths)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    rows = []  # 指标行
    for i, p in enumerate(img_paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        gt = load_gt_boxes(find_label(stem), w, h) if find_label(stem) else np.empty((0, 4), np.float32)

        for col, (det, name, color) in enumerate([
            (vj, "VJ（Haar 级联）", (0, 200, 0)),
            (yolo, "YOLO（对照）", (255, 80, 0)),
        ]):
            boxes, confs = det.detect(img)
            tp = len(match(boxes, confs, gt, 0.5))
            recall = tp / len(gt) if len(gt) else 0.0
            prec = tp / len(boxes) if len(boxes) else 0.0
            rows.append((stem, name, len(gt), len(boxes), tp, recall, prec))

            ax = axes[i, col]
            ax.imshow(draw_boxes(img, boxes, color))
            ax.set_title(f"{name}\n检出 {len(boxes)} / 真值 {len(gt)}  "
                         f"Recall {recall:.0%}  Prec {prec:.0%}", fontsize=11)
            ax.axis("off")

    fig.suptitle("人脸检测：实验室 VJ 方法 vs YOLO（同一条流水线，检测器可插拔）",
                 fontsize=14, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return rows


def haar_templates():
    """VJ 经典类 Haar 特征模板：白区(+)减黑区(-)。返回 [(名称, 2D数组0/1), ...]。"""
    W, B = 1.0, 0.0  # 白=1（加），黑=0（减）
    two_h = np.array([[W, B]])                 # 双矩形·水平（边缘特征）
    two_v = np.array([[W], [B]])               # 双矩形·垂直
    three_h = np.array([[W, B, W]])            # 三矩形·水平（线特征）
    three_v = np.array([[W], [B], [W]])        # 三矩形·垂直
    four = np.array([[W, B], [B, W]])          # 四矩形（对角特征）
    return [
        ("双矩形·水平", two_h),
        ("双矩形·垂直", two_v),
        ("三矩形·水平", three_h),
        ("三矩形·垂直", three_v),
        ("四矩形", four),
    ]


def make_principle_figure(face_crop_bgr, sample_gray, out_path):
    """VJ 原理图：积分图 O(1) 求和校验 + Haar 模板 + Haar 特征贴脸。"""
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.05, 1])

    # ---- (a) 积分图：演示任意矩形和 O(1) 计算 ----
    gray = sample_gray.astype(np.float64)
    ii = cv2.integral(sample_gray).astype(np.float64)  # (H+1, W+1)
    H, Wd = gray.shape
    # 取图像中部一个矩形
    x, y = Wd // 4, H // 4
    w, h = Wd // 3, H // 3
    A = ii[y, x]; Bc = ii[y, x + w]; C = ii[y + h, x]; D = ii[y + h, x + w]
    sum_ii = D - Bc - C + A             # 4 次查表
    sum_brute = gray[y:y + h, x:x + w].sum()  # 暴力逐像素求和

    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.imshow(sample_gray, cmap="gray")
    ax0.add_patch(Rectangle((x, y), w, h, edgecolor="red", facecolor="none", lw=2))
    ax0.set_title("(a) 原始灰度图 + 待求和矩形", fontsize=11)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 2:4])
    im = ax1.imshow(ii[1:, 1:], cmap="viridis")
    ax1.add_patch(Rectangle((x, y), w, h, edgecolor="red", facecolor="none", lw=2))
    # 标出参与计算的四个角点 A,B,C,D
    for (px, py), name in [((x, y), "A"), ((x + w, y), "B"),
                           ((x, y + h), "C"), ((x + w, y + h), "D")]:
        ax1.plot(px, py, "rs", ms=6)
        ax1.text(px + 4, py + 4, name, color="white", fontsize=11, fontweight="bold")
    ax1.set_title("(b) 积分图（每点=左上所有像素和）", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 4])
    ax2.axis("off")
    txt = (
        "积分图：矩形和只需 4 次查表\n"
        "S = D − B − C + A\n"
        "（与位置/大小无关，恒 O(1)）\n\n"
        f"积分图法  S = {sum_ii:,.0f}\n"
        f"暴力求和  S = {sum_brute:,.0f}\n"
        f"两者一致：{np.isclose(sum_ii, sum_brute)}"
    )
    ax2.text(0, 0.5, txt, fontsize=11, va="center",
             bbox=dict(boxstyle="round", fc="#f0f0f0", ec="gray"))

    # ---- (b) 类 Haar 特征模板 ----
    templates = haar_templates()
    for j, (name, tpl) in enumerate(templates):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(tpl, cmap="gray", vmin=0, vmax=1, extent=[0, tpl.shape[1], tpl.shape[0], 0])
        # 在白/黑块上标 + / -
        for r in range(tpl.shape[0]):
            for c in range(tpl.shape[1]):
                ax.text(c + 0.5, r + 0.5, "+" if tpl[r, c] == 1 else "−",
                        ha="center", va="center",
                        color="black" if tpl[r, c] == 1 else "white",
                        fontsize=16, fontweight="bold")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("VJ 框架原理：积分图 O(1) 求矩形和  +  类 Haar 特征（白区减黑区）",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- 额外：把经典双矩形 Haar 特征贴到真实人脸上，单独存一张 ----
    if face_crop_bgr is not None:
        fig2, ax = plt.subplots(figsize=(4.5, 4.8))
        face_rgb = cv2.cvtColor(cv2.resize(face_crop_bgr, (160, 160)), cv2.COLOR_BGR2RGB)
        ax.imshow(face_rgb)
        # 经典特征：上半（眉/额，较亮，+） vs 下半眼睛区（较暗，−）
        fh, fw = 160, 160
        ax.add_patch(Rectangle((0.18 * fw, 0.22 * fh), 0.64 * fw, 0.16 * fh,
                               facecolor="white", alpha=0.45, edgecolor="k"))
        ax.add_patch(Rectangle((0.18 * fw, 0.38 * fh), 0.64 * fw, 0.16 * fh,
                               facecolor="black", alpha=0.45, edgecolor="k"))
        ax.text(0.5 * fw, 0.30 * fh, "+ 较亮(额/眉)", ha="center", color="k", fontsize=9)
        ax.text(0.5 * fw, 0.46 * fh, "− 较暗(眼睛)", ha="center", color="w", fontsize=9)
        ax.set_title("双矩形 Haar 特征：眼睛区比上方暗\n→ VJ 区分人脸的最基础线索", fontsize=10)
        ax.axis("off")
        fig2.tight_layout()
        face_out = out_path.replace(".png", "_haar_on_face.png")
        fig2.savefig(face_out, dpi=130)
        plt.close(fig2)

    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="datasets/images/test", help="演示用图片目录")
    ap.add_argument("--weights", default=DEFAULT_WEIGHT, help="YOLO 对照权重")
    ap.add_argument("--n", type=int, default=4, help="对比图用几张图")
    ap.add_argument("--out", default="image_results/demo", help="输出目录")
    args = ap.parse_args()

    source_abs = args.source if os.path.isabs(args.source) else os.path.abspath(os.path.join(ROOT, args.source))
    out_dir = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(ROOT, args.out))
    os.makedirs(out_dir, exist_ok=True)

    print("\033[35m------ VJ 进度演示 ------\033[0m")
    vj = build_detector("vj")
    yolo = build_detector("yolo", weights=args.weights)

    # 1) 选图 + 对比图 + 指标
    img_paths = pick_images(source_abs, args.n, vj)
    compare_path = os.path.join(out_dir, "compare_vj_vs_yolo.png")
    rows = make_compare_figure(img_paths, vj, yolo, compare_path)
    print(f"[1] 对比图已保存：{compare_path}")

    print("\n  指标对比（IoU≥0.5 判命中）")
    print(f"  {'image':16s} {'detector':14s} {'GT':>3s} {'Det':>4s} {'TP':>3s} {'Recall':>7s} {'Prec':>7s}")
    print("  " + "-" * 60)
    for stem, name, g, d, tp, rec, pr in rows:
        print(f"  {stem:16s} {name:14s} {g:3d} {d:4d} {tp:3d} {rec:7.0%} {pr:7.0%}")
    csv_path = os.path.join(out_dir, "compare_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,detector,gt,det,tp,recall,precision\n")
        for stem, name, g, d, tp, rec, pr in rows:
            f.write(f"{stem},{name},{g},{d},{tp},{rec:.4f},{pr:.4f}\n")
    print(f"  指标已写入：{csv_path}")

    # 2) 原理图：用第一张图做积分图演示，并在所有演示图里挑「最大的一张 VJ 检出脸」做 Haar 贴脸
    first = cv2.imdecode(np.fromfile(img_paths[0], np.uint8), cv2.IMREAD_COLOR)
    sample_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    face_crop, best_area = None, 0
    for p in img_paths:
        im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        boxes, _ = vj.detect(im)
        for x1, y1, x2, y2 in boxes.astype(int):
            area = max(0, x2 - x1) * max(0, y2 - y1)
            crop = im[max(0, y1):y2, max(0, x1):x2]
            if area > best_area and crop.size:
                best_area, face_crop = area, crop
    if face_crop is not None:
        # 适度提亮，让眼睛/额头明暗对比在贴图上更清楚
        face_crop = cv2.convertScaleAbs(face_crop, alpha=1.3, beta=25)
    principle_path = os.path.join(out_dir, "vj_principle.png")
    make_principle_figure(face_crop, sample_gray, principle_path)
    print(f"\n[2] 原理图已保存：{principle_path}")
    print(f"    （含积分图 O(1) 求和校验 + 类 Haar 特征模板 + Haar 特征贴脸）")


if __name__ == "__main__":
    main()
