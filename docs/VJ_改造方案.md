# 用 Viola-Jones（VJ）框架替换 YOLO 的改造方案

> 目标：把现有「YOLO 人脸检测 + DBSCAN 密度聚类」流水线里的**检测器**，
> 从 YOLO 替换为实验室的 **VJ 框架**（积分图 + 类 Haar 特征 + AdaBoost + 级联分类器），
> 并实现实验室要求的两项改进（分散矩形特征、双阈值弱分类器），
> 最后用 **ROC 曲线**对比 原始 VJ / 改进 VJ / YOLO。

---

## 0. 可行性结论

**可行。** 原因是当前架构里，检测器与下游是松耦合的：

```
                 ┌──────────────┐
   图像  ──────► │   检测器     │ ──► (boxes Nx4 xyxy, confs N)
                 └──────────────┘            │
                                             ▼
                          DBSCAN 密度聚类 / 可视化 / eval_recognition.py（ROC、Recall、Prec）
```

下游所有代码（`predict_face.py`、`eval_recognition.py`）只依赖检测器吐出的
`boxes (x1,y1,x2,y2)` 和 `confs`。只要 VJ 检测器也产出同样的两样东西，下游一行都不用改。

改造分两种力度，按需选：

| 力度 | 内容 | 工作量 | 是否满足实验室要求 |
| --- | --- | --- | --- |
| **A. 开箱即用的 VJ** | 用 OpenCV 自带的 Haar 级联（`cv2.CascadeClassifier`），它就是原始 VJ 的实现 | 半天 | 满足「基础框架」，但**没有**两项改进 |
| **B. 自研 VJ + 两项改进** | 自己实现积分图 / Haar 特征 / AdaBoost / 级联，并加上分散矩形特征 + 双阈值弱分类器 | 2~4 周 | **完整满足**实验室课题要求 |

建议路线：**先做 A 打通管线拿到基线，再做 B 实现论文级改进**，最后用同一套 `eval_recognition.py` 出 ROC 对比图。

---

## 1. 现状分析（解耦点在哪）

当前检测器调用集中在两处，且接口高度一致：

- `scripts/predict_face.py:71-80`
  ```python
  result = model.predict(img, conf=..., imgsz=..., verbose=False)[0]
  boxes  = result.boxes.xyxy.cpu().numpy()   # (N,4) x1y1x2y2
  confs  = result.boxes.conf.cpu().numpy()   # (N,)
  centers= result.boxes.xywh.cpu().numpy()[:, :2]
  ```
- `scripts/eval_recognition.py:193-199`
  ```python
  r = model.predict(img, conf=..., imgsz=..., verbose=False)[0]
  pred = r.boxes.xyxy.cpu().numpy()
  conf = r.boxes.conf.cpu().numpy()
  ```

**结论**：只要定义一个统一的检测器接口 `detect(img) -> (boxes, confs)`，
把这两处的 `model.predict(...)` 解包逻辑替换成 `detector.detect(img)` 即可。

> 注意：VJ 的「置信度」不像 YOLO 是天然概率。OpenCV 的
> `detectMultiScale3(..., outputRejectLevels=True)` 会返回 `levelWeights`
> （通过级联的层数权重），可作为分数代理，**ROC 曲线正是靠它来扫阈值的**。
> 自研版（方案 B）则用最终强分类器的加权得分 `Σ αₜ hₜ(x)` 作为分数。

---

## 2. 总体策略：把检测器抽象成可插拔模块

新增一个检测器抽象层，YOLO 和 VJ 都实现它。下游只认接口。

### 2.1 新增 `scripts/detectors/base.py`

```python
import numpy as np

class FaceDetector:
    """统一检测器接口：输入 BGR 图(np.ndarray)，输出人脸框与分数。"""
    def detect(self, img):
        """返回 (boxes, confs):
        boxes: np.ndarray (N,4)  float  [x1,y1,x2,y2] 像素坐标
        confs: np.ndarray (N,)   float  分数(越大越像人脸)，用于排序/扫阈值出 ROC
        """
        raise NotImplementedError
```

### 2.2 新增 `scripts/detectors/yolo_detector.py`（把现有逻辑搬进来）

```python
from ultralytics import YOLO
import numpy as np
from .base import FaceDetector

class YoloDetector(FaceDetector):
    def __init__(self, weights, conf=0.25, imgsz=640):
        self.model = YOLO(weights); self.conf = conf; self.imgsz = imgsz
    def detect(self, img):
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if len(r.boxes):
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return np.empty((0,4), np.float32), np.empty((0,), np.float32)
```

### 2.3 新增 `scripts/detectors/vj_detector.py`

- **方案 A（OpenCV Haar 级联）**：

  ```python
  import cv2, numpy as np
  from .base import FaceDetector

  class VJOpenCVDetector(FaceDetector):
      def __init__(self, cascade_xml=None, scale_factor=1.1, min_neighbors=3, min_size=24):
          path = cascade_xml or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
          self.cc = cv2.CascadeClassifier(path)
          self.sf, self.mn, self.ms = scale_factor, min_neighbors, min_size
      def detect(self, img):
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          gray = cv2.equalizeHist(gray)
          rects, _, weights = self.cc.detectMultiScale3(
              gray, scaleFactor=self.sf, minNeighbors=self.mn,
              minSize=(self.ms, self.ms), outputRejectLevels=True)
          if len(rects) == 0:
              return np.empty((0,4), np.float32), np.empty((0,), np.float32)
          boxes = np.array([[x, y, x+w, y+h] for (x,y,w,h) in rects], np.float32)
          confs = np.array(weights, np.float32).ravel()   # levelWeights 作为分数
          return boxes, confs
  ```

- **方案 B（自研 VJ + 改进）**：见第 4 节，封装成同样的 `detect(img)`。

### 2.4 改造下游脚本（改动很小）

- `predict_face.py` / `eval_recognition.py`：把 `model = YOLO(...)` 换成根据
  `--detector {yolo,vj,vj_custom}` 选择构造对应 `FaceDetector`，并把
  `r = model.predict(...)` 那几行替换为 `boxes, confs = detector.detect(img)`。
  `centers` 由 `boxes` 现算：`centers = (boxes[:, :2] + boxes[:, 2:]) / 2`。

---

## 3. 分阶段路线图

### 阶段 0：抽象检测器接口（0.5 天）
- 建 `scripts/detectors/`，写 `base.py` + `yolo_detector.py`。
- 改 `predict_face.py`、`eval_recognition.py` 走接口，新增 `--detector` 参数（默认 `yolo`，保证旧行为不变）。
- **验收**：`--detector yolo` 跑出来的结果与改造前完全一致。

### 阶段 1：接入 OpenCV Haar 级联（方案 A，0.5 天）
- 写 `vj_detector.py` 的 `VJOpenCVDetector`。
- `python scripts/predict_face.py --detector vj`、`eval_recognition.py --detector vj` 跑通。
- **验收**：能在图上画出 VJ 检出的人脸框，并跑出 Recall/Precision。
- **预期**：在 WIDER FACE 上 VJ 的 Recall 会明显低于 YOLO（VJ 只擅长**正脸、较大、无遮挡**；WIDER 里大量小脸/侧脸/遮挡）。这个差距正是后面 ROC 对比要呈现的，属正常现象，不是 bug。

### 阶段 2：从零实现原始 VJ（方案 B 第一步，1~2 周）
实现论文 *Rapid Object Detection using a Boosted Cascade of Simple Features* 的四大件：

1. **积分图（Integral Image）** `vj/integral.py`
   - `ii[y,x] = Σ img[≤y,≤x]`，用 `np.cumsum` 两次即可。
   - 任意矩形和 O(1) 计算：`S = ii[D] - ii[B] - ii[C] + ii[A]`。

2. **类 Haar 特征** `vj/haar.py`
   - 2-矩形（边缘）、3-矩形（线）、4-矩形（对角）特征。
   - 在固定窗口（如 24×24）枚举所有位置/尺度的特征（约 16 万个）。
   - 每个特征值 = 白区像素和 − 黑区像素和（用积分图算）。

3. **AdaBoost + 弱分类器** `vj/adaboost.py`
   - 弱分类器：单特征 + 单阈值 + 极性 `h(x)=p·sign(f(x)−θ)`。
   - 每轮选使加权错误率最小的特征，更新样本权重，累计 `αₜ`。
   - 强分类器分数 `F(x)=Σ αₜ hₜ(x)`（**这就是 ROC 要扫的连续分数**）。

4. **级联（Cascade）** `vj/cascade.py`
   - 多级强分类器串联，每级调阈值保证高召回（如 99.5%）、适度拒真率。
   - 早期级用极少特征快速排除背景窗口 → 提速。
   - 推理：多尺度滑窗 + 每窗过级联 + NMS 合并框 → 输出 `(boxes, confs)`。

5. **训练数据准备** `vj/prepare_vj_data.py`
   - 正样本：从 `archive/` 的人脸框裁剪并缩放到 24×24（灰度、直方图均衡）。
   - 负样本：从无脸区域随机裁剪 + 训练中 hard-negative mining（用当前级联在背景图上找误检，补进负集）。
   - 因 WIDER 脸多为小/侧脸，建议先**筛选较大的正脸子集**作为正样本，VJ 才训得动。

   > 训练自研 VJ 计算量大，建议先用**子集 + 较小特征池**跑通，再逐步放大。
   > 也可用 OpenCV 的 `opencv_traincascade` 工具训练 `.xml`，再用方案 A 的加载方式接入——能省下大量造轮子时间，但**两项改进就无法体现**，所以仅作对照/兜底。

- **验收**：自研 VJ 在小测试集上能检出正脸；ROC 与方案 A 的 OpenCV 级联大致同档。

### 阶段 3：实现实验室的两项改进（方案 B 第二步，1~2 周）

**改进一：分散矩形特征（Scattered / Disjoint Rectangle Features）**
- 原始 Haar 要求黑白矩形**相邻**；分散特征去掉相邻限制，黑白矩形可分布在窗口任意位置。
- 实现：在 `vj/haar.py` 旁新增 `vj/scattered.py`，特征定义从「相邻矩形组」改为「任意若干带符号(+/−)矩形的线性组合」，特征值 `f(x)=Σ sᵢ·RectSum(rᵢ)`（每个 `RectSum` 仍用积分图 O(1)）。
- 特征池更大，需控制规模：可随机采样 / 限定矩形个数（2~4 个）来生成候选特征池，再交给 AdaBoost 选优。
- **预期收益**：表达能力更强，同样级数下错误率更低（论文/实验室结论）。

**改进二：双阈值弱分类器（Dual-threshold Weak Classifier）**
- 原始弱分类器是单阈值（一刀切）。双阈值把特征值轴划成三段，给中间「模糊带」更细的决策：
  ```
  f(x) < θ_low            -> 输出 a
  θ_low ≤ f(x) ≤ θ_high   -> 输出 b   (模糊带，单独取值)
  f(x) > θ_high           -> 输出 c
  ```
- 实现：在 `vj/adaboost.py` 里把弱分类器从「阈值+极性」换成「双阈值+分段取值」，每轮训练时对每个特征搜索最优 `(θ_low, θ_high)` 及三段输出，使加权错误率最小。
- **预期收益**：每个特征决策更精细，分类错误更少 → 强分类器更强。

> 工程建议：把弱分类器做成策略类（`SingleThresholdWeak` / `DualThresholdWeak`），
> 特征做成（`HaarFeature` / `ScatteredFeature`），通过开关组合，便于做**消融实验**
> （单阈值+Haar、双阈值+Haar、单阈值+分散、双阈值+分散 四种组合各出一条 ROC）。

### 阶段 4：ROC 评估与对比（2~3 天）
- 复用 `eval_recognition.py` 的 IoU 匹配逻辑，**新增按分数扫阈值的 ROC 绘制脚本** `scripts/eval_roc.py`：
  - 对每张图收集所有预测框 `(conf, 是否命中GT)`，汇总后按 `conf` 从高到低扫描；
  - 横轴用 **每图平均误检数(FPPI)** 或 FPR，纵轴用 **召回率/TPR**（人脸检测惯用 FPPI–Recall，即 FROC）；
  - 同一张图叠加多条曲线：YOLO / OpenCV-VJ / 自研原始 VJ / 改进 VJ（四象限消融）。
- **验收**：得到一张 ROC/FROC 对比图，显示「改进 VJ 优于原始 VJ」（实验室的核心结论），YOLO 作为上界参照。

---

## 4. 文件改动清单

```
scripts/
├── detectors/
│   ├── __init__.py
│   ├── base.py              # 新增：FaceDetector 抽象接口
│   ├── yolo_detector.py     # 新增：把现有 YOLO 逻辑搬进来
│   └── vj_detector.py       # 新增：VJOpenCVDetector(方案A) + VJCustomDetector(方案B)
├── vj/                      # 新增：自研 VJ（方案 B）
│   ├── integral.py          #   积分图
│   ├── haar.py              #   类 Haar 特征
│   ├── scattered.py         #   改进一：分散矩形特征
│   ├── adaboost.py          #   AdaBoost + 弱分类器(单/双阈值)
│   ├── cascade.py           #   级联训练 + 多尺度滑窗推理
│   ├── prepare_vj_data.py   #   从 archive 造 24x24 正/负样本 + hard negative
│   └── train_vj.py          #   训练入口，产出 model/weights/vj_cascade.pkl
├── predict_face.py          # 改：加 --detector，走 FaceDetector 接口
├── eval_recognition.py      # 改：加 --detector，走 FaceDetector 接口
└── eval_roc.py              # 新增：扫分数阈值，画多检测器 ROC/FROC 对比

model/weights/
└── vj_cascade.pkl           # 新增：自研 VJ 训练产物（方案 B）

requirements.txt             # 方案A仅需 opencv(已含)；方案B 纯 numpy 可行，可选 numba 提速
docs/VJ_改造方案.md          # 本文档
```

> 下游 `predict.py`（走 yolov5 子进程那版）可暂不动；VJ 路线统一走
> `predict_face.py` / `eval_recognition.py` 这条「直接加载检测器」的管线即可。

---

## 5. 关键技术难点与风险

| 难点 | 说明 | 对策 |
| --- | --- | --- |
| VJ 在 WIDER 上召回偏低 | VJ 只擅长正脸/大脸，WIDER 多小脸侧脸遮挡 | 训练正样本筛大正脸子集；评估时也可分难度子集报告；ROC 对比时如实呈现 |
| 自研训练计算量大 | 16 万特征 × 上万样本 × 多轮 boosting | 先小特征池/子集跑通；用积分图向量化；可选 `numba`/多进程加速 |
| 分散特征池爆炸 | 任意矩形组合数极大 | 限定矩形数(2~4)、随机采样候选池，再交给 AdaBoost 选 |
| VJ 分数语义 | 不是概率，但 ROC 需要连续分数 | 方案A用 `levelWeights`；方案B用 `Σαₜhₜ(x)` |
| 多尺度滑窗慢 | 推理需大量窗口 | 级联早退 + 适当步长/缩放因子；这正是级联存在的意义 |
| NMS | 多尺度会重复检出同一张脸 | 推理后接非极大值抑制合并框 |

---

## 6. 里程碑与工期估计

| 里程碑 | 交付物 | 估时 |
| --- | --- | --- |
| M0 检测器接口抽象 | `--detector yolo` 行为不变 | 0.5 天 |
| M1 OpenCV VJ 打通 | `--detector vj` 出框 + Recall/Prec | 0.5 天 |
| M2 自研原始 VJ | 积分图/Haar/AdaBoost/级联，能检正脸 | 1~2 周 |
| M3 两项改进 | 分散矩形特征 + 双阈值弱分类器 + 消融开关 | 1~2 周 |
| M4 ROC 对比 | `eval_roc.py` 出多曲线对比图 | 2~3 天 |

> 只要交差「用上 VJ 框架」：做到 **M1** 即可（半天到一天）。
> 要满足实验室课题的**改进与 ROC 对比**：需做到 **M4**。

---

## 7. 验收标准

1. `--detector yolo` 与改造前结果一致（回归不破坏）。
2. `--detector vj`（OpenCV）能出框、出 Recall/Precision。
3. 自研 VJ 能训练并在测试集检出正脸。
4. 四种消融组合（Haar/分散 × 单/双阈值）各能产出一条 ROC。
5. 最终 ROC/FROC 对比图显示：**改进 VJ > 原始 VJ**，YOLO 作为上界参照——即实验室课题的核心结论可被复现。
