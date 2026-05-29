"""
把 archive/ 原始数据（WIDER FACE，已是 YOLO 格式）整理成 YOLOv5 训练所需的
datasets/images/{train,val,test} + datasets/labels/{train,val,test} 结构，
并自动改写项目根目录的 data.yaml 中的 path 为绝对路径。

原始数据约定：
  archive/images/wider_xxx.jpg   图片
  archive/labels/wider_xxx.txt   标签（每行：cls cx cy w h，归一化；cls 恒为 0=face）

用法：
  python scripts/prepare_dataset.py                 # 默认 90% train / 8% val / 2% test
  python scripts/prepare_dataset.py --val 0.1 --test 0.05
  python scripts/prepare_dataset.py --copy          # 复制文件（默认是创建硬链接，省空间）
"""
import os
import shutil
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

ARCHIVE_IMAGES = os.path.join(ROOT, "archive", "images")
ARCHIVE_LABELS = os.path.join(ROOT, "archive", "labels")
DATASETS_DIR = os.path.join(ROOT, "datasets")
DATA_YAML = os.path.join(ROOT, "data.yaml")

IMG_EXTS = (".jpg", ".jpeg", ".png")


def link_or_copy(src, dst, copy):
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)  # 硬链接，几乎不占额外空间
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.08, help="验证集比例")
    ap.add_argument("--test", type=float, default=0.02, help="测试集比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="复制文件而非创建硬链接")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 张（快速调试）")
    args = ap.parse_args()

    if not os.path.isdir(ARCHIVE_IMAGES):
        raise FileNotFoundError(f"找不到图片目录：{ARCHIVE_IMAGES}")

    stems = []
    for f in sorted(os.listdir(ARCHIVE_IMAGES)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXTS:
            continue
        lbl = os.path.join(ARCHIVE_LABELS, stem + ".txt")
        if os.path.exists(lbl):
            stems.append((stem, ext))
    if not stems:
        raise RuntimeError("没有找到任何 图片+标签 配对。")

    random.seed(args.seed)
    random.shuffle(stems)
    if args.limit > 0:
        stems = stems[: args.limit]

    n = len(stems)
    n_val = int(n * args.val)
    n_test = int(n * args.test)
    n_train = n - n_val - n_test

    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(DATASETS_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASETS_DIR, "labels", split), exist_ok=True)

    for split, items in splits.items():
        for stem, ext in items:
            link_or_copy(
                os.path.join(ARCHIVE_IMAGES, stem + ext),
                os.path.join(DATASETS_DIR, "images", split, stem + ext),
                args.copy,
            )
            link_or_copy(
                os.path.join(ARCHIVE_LABELS, stem + ".txt"),
                os.path.join(DATASETS_DIR, "labels", split, stem + ".txt"),
                args.copy,
            )
        print(f"  {split:5s}: {len(items)} 张")

    # 改写 data.yaml 的 path 为绝对路径，确保从任何目录启动训练都能找到数据
    _rewrite_data_yaml(DATASETS_DIR)

    print(f"\n完成。共 {n} 张  ->  train {n_train} / val {n_val} / test {n_test}")
    print(f"数据集目录：{DATASETS_DIR}")
    print(f"已更新：{DATA_YAML}")


def _rewrite_data_yaml(datasets_dir):
    path_line = f"path: {datasets_dir.replace(os.sep, '/')}\n"
    if os.path.exists(DATA_YAML):
        with open(DATA_YAML, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out, replaced = [], False
        for line in lines:
            if line.lstrip().startswith("path:"):
                out.append(path_line)
                replaced = True
            else:
                out.append(line)
        if not replaced:
            out.append(path_line)
    else:
        out = [
            path_line,
            "train: images/train\n",
            "val: images/val\n",
            "test: images/test\n\n",
            "nc: 1\n",
            "names: ['face']\n",
        ]
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.writelines(out)


if __name__ == "__main__":
    main()
