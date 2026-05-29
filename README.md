# face-recognition — 人脸密度估计与异常聚集检测

基于 **YOLOv5** 的人脸检测 + **DBSCAN** 密度聚类，用于在图像中**检测所有人脸、统计人脸密度，并圈出人脸异常聚集的区域**。

本项目是 `ref/train-yolo-v5`（人群异常聚集检测，检测对象为「整个人」）的「人脸版」：
整体思路完全一致，区别在于**检测对象从行人身体换成了人脸**，标注框也是人脸的长宽。

---

## 1. 这是干啥的（What）

整体流程分两步：

1. **检测**：用 YOLOv5s 在图像里检测所有「人脸」（只有一个类别 `face`），输出每张人脸框的中心点。
2. **密度统计 + 判异常**：对所有人脸中心点做 DBSCAN 聚类——
   - 统计每张图的**总人脸数**和**异常聚集区数量**；
   - 若某个簇内的人脸数 ≥ 阈值，就判定为「异常聚集」，在结果图上用**黄色圆圈**框出来。

可视化颜色约定：

| 元素 | 颜色 |
| --- | --- |
| 普通人脸中心点 | 绿色实心点 |
| 属于异常聚集的人脸中心点 | 红色实心点 |
| 异常聚集区域 | 黄色外接圆（半径放大 5%） |
| 左上角文字 | 当前图的 `faces`（总人脸数）/ `clusters`（聚集区数） |

可调参数（`scripts/predict.py` 命令行参数）：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--R` | DBSCAN 邻域半径（像素） | 80 |
| `--min-pts` | 成为核心点所需的最少邻居数 | 3 |
| `--min-cluster` | 判定为异常聚集的最小人脸数 | 5 |

---

## 2. 数据集（archive）

`archive/` 是 **WIDER FACE** 数据集，已经是 YOLO 标注格式：

```
archive/
├── images/   wider_xxx.jpg     共 12880 张
└── labels/   wider_xxx.txt     每行：cls cx cy w h（归一化 0~1，cls 恒为 0=face）
```

每个标签文件里有若干行，每行对应一张人脸框（中心点 + 长宽，均已归一化）。

---

## 3. 怎么用（How）

### 环境准备

```bash
pip install -r requirements.txt
```

> 依赖：numpy、opencv-python、matplotlib、scikit-learn、torch、pyyaml、tqdm。
> 训练/检测还需要 YOLOv5 自身的依赖（见下）。

### 🚀 免训练快速出结果（推荐）

如果你**不想训练，只想直接出人脸密度结果**，用预训练的人脸模型即可：

```bash
# 1. 下载预训练人脸权重（约 5.7MB，开箱即用，类别就是 face）
#    放到 model/weights/yolov10n-face.pt
#    来源：https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov10n-face.pt

# 2. 直接推理（用 ultralytics 包加载，无需 YOLOv5 源码，无需训练）
python scripts/predict_face.py --source datasets/images/test --out image_results/face_demo
```

`predict_face.py` 与下面的 `predict.py` 逻辑一致（检测 → DBSCAN 密度聚类 → 可视化 + csv），
区别在于它**直接用 ultralytics 加载现成的人脸权重**，所以不需要训练、也不需要 YOLOv5 源码。
其余参数同样可调：`--conf 0.3 --R 60 --min-cluster 5` 等。

> 也可换用更大的模型（精度更高、速度更慢），如
> `yolov11m-face.pt` / `yolov11l-face.pt`，见 [yolo-face releases](https://github.com/akanametov/yolo-face/releases)。

---

下面是**自己训练**的完整路线（数据来自 `archive`，需要 GPU 才实际可行）：

### YOLOv5 源码

为避免重复存放体积庞大的源码，本项目**默认复用 `ref/train-yolo-v5/yolov5`** 里的 YOLOv5。
脚本会按以下顺序自动查找：

1. 环境变量 `YOLOV5_DIR` 指定的目录；
2. 本项目根目录下的 `yolov5/`；
3. `ref/train-yolo-v5/yolov5/`（默认命中）。

如需独立一份，可在项目根目录执行：

```bash
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

### 第一步：整理数据集

把 `archive/` 划分成 YOLOv5 训练所需的 train/val/test 结构（默认 90% / 8% / 2%）：

```bash
python scripts/prepare_dataset.py
```

它会在项目下生成 `datasets/images/{train,val,test}` 与 `datasets/labels/{train,val,test}`
（默认用**硬链接**，几乎不占额外磁盘），并把 `data.yaml` 里的 `path` 改写成绝对路径。

> 常用选项：`--val 0.1 --test 0.05` 调整比例；`--copy` 复制而非硬链接；`--limit 100` 只取前 100 张快速调试。

### 第二步：训练人脸检测模型

```bash
python scripts/train.py
```

封装 `yolov5/train.py`，默认 `--img 640 --batch 16 --epochs 100`，
基础权重为 `yolov5s.pt`，结果保存到 `runs/face_yolov5s/`，
最佳权重为 `runs/face_yolov5s/weights/best.pt`。

> 常用选项：`--epochs 50 --batch 8`。

### 第三步：推理 + 人脸密度统计 + 可视化

```bash
python scripts/predict.py
```

默认对 `datasets/images/test` 里的图片做检测，并：

1. 调 `yolov5/detect.py` 检测人脸，导出标签到 `runs/face_yolov5s_detect/labels/`；
2. 读取人脸框中心点，做 DBSCAN 聚类，圈出异常聚集；
3. 把可视化结果与一份 `_summary.csv`（每张图的人脸数、聚集区数）写到 `image_results/latest/`。

> 常用选项：`--source <目录>` 指定待检测图片；`--weights <权重>` 指定模型；
> `--R 60 --min-cluster 5` 调密度参数；`--out image_results/v1` 改输出目录。

---

## 4. 目录结构

```
face-recognition/
├── archive/                  # 原始数据集（WIDER FACE，YOLO 格式）
│   ├── images/               #   12880 张图片
│   └── labels/               #   对应标签 txt（人脸 cx cy w h）
├── data.yaml                 # 数据集配置：path + train/val/test，nc=1，names=['face']
├── requirements.txt
├── scripts/
│   ├── prepare_dataset.py    #   划分 archive -> datasets/{train,val,test}
│   ├── train.py              #   训练入口（封装 yolov5/train.py）
│   ├── predict.py            #   推理 + DBSCAN 密度聚类 + 可视化
│   └── common.py             #   定位 YOLOv5 源码 / 权重的公共工具
├── datasets/                 # prepare_dataset.py 生成（YOLO 训练结构）
├── runs/                     # YOLOv5 训练/检测输出（含 best.pt）
├── image_results/            # 人脸密度可视化结果 + _summary.csv
└── ref/                      # 参考项目：人群（行人）异常聚集检测
```

---

## 5. 与 ref 项目的对应关系

| | ref（人群版） | 本项目（人脸版） |
| --- | --- | --- |
| 检测对象 | 整个人（行人身体） | 人脸 |
| 类别 | `person` | `face` |
| 标注框含义 | 人体长宽 | 人脸长宽 |
| 密度聚类 | DBSCAN（人体中心点） | DBSCAN（人脸中心点） |
| 输出 | 异常聚集圈 | 异常聚集圈 + 总人脸数 + 汇总 csv |

---

## 6. 下一阶段：用 Viola-Jones（VJ）框架改造检测器

本项目正在引入实验室的 **Viola-Jones 框架**（积分图 + 类 Haar 特征 + AdaBoost + 级联）
作为可替换的检测器。检测器已抽象成可插拔模块，`--detector yolo` / `--detector vj` 一键切换。

进度与路线（详见 [`docs/VJ_改造方案.md`](docs/VJ_改造方案.md)、演示图见 [`docs/demo/`](docs/demo/)）：

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 抽象 `FaceDetector` 接口，下游改走接口（yolo 行为不变） | ✅ 已完成 |
| M1 | 接入 OpenCV Haar 级联（原始 VJ），`--detector vj` 出框 | ✅ 已完成 |
| M2 | 从零自研 VJ：积分图 / 类 Haar 特征 / AdaBoost / 级联 | ⏳ 下一步 |
| M3 | 两项改进：分散矩形特征、双阈值弱分类器 | ⏳ 待做 |
| M4 | ROC/FROC 对比：原始 VJ vs 改进 VJ vs YOLO | ⏳ 待做 |

---

## 致谢

人脸/行人检测基于 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)（AGPL-3.0）。
数据集为 [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)。
