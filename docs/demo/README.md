# VJ 进度演示（阶段性成果）

本目录是用实验室 **Viola-Jones（VJ）框架**做人脸检测的阶段性演示，
由 `scripts/demo_vj.py` 一键生成（`python scripts/demo_vj.py`）。

当前进度：已把检测器抽象成可插拔模块，**实验室的 VJ 方法（积分图 + 类 Haar 特征 +
AdaBoost + 级联）已通过 OpenCV Haar 级联接入同一条流水线**，与原 YOLO 检测器可一键切换
（`--detector vj` / `--detector yolo`）。下一步将从零自研 VJ 并实现实验室的两项改进
（分散矩形特征、双阈值弱分类器）。详见 `../VJ_改造方案.md`。

## 图说明

| 文件 | 内容 |
| --- | --- |
| `compare_vj_vs_yolo.png` | 同一批图：左列 **VJ（实验室方法）**、右列 YOLO（对照），各自检出框 + Recall/Precision。证明 VJ 已接入流水线。 |
| `vj_principle.png` | VJ 框架原理：**(a/b) 积分图**——任意矩形和可 O(1) 计算，并用暴力求和数值校验（结果完全一致）；**类 Haar 特征**——2/3/4 矩形模板（白区减黑区）。 |
| `vj_principle_haar_on_face.png` | 经典「双矩形」Haar 特征贴到真实人脸上：眼睛区比上方额头暗——这正是 VJ 区分人脸/非人脸的最基础线索。 |
| `compare_metrics.csv` | VJ 与 YOLO 在演示图上的 GT/检出/TP/Recall/Precision 明细。 |

## 现象说明

VJ 的 Recall 明显低于 YOLO 属**预期**：经典 VJ 只擅长正脸、较大、无遮挡的人脸，
而 WIDER FACE 含大量小脸/侧脸/遮挡。这一差距正是后续用 ROC 曲线对比、并通过两项改进去缩小的目标。
