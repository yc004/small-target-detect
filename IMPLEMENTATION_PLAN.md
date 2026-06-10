# 小目标交通标志检测 — 渐进式实现方案

## 总览

```
Stage 1 (基线)          →  Stage 2 (+P2)       →  Stage 3 (+ECA)
  ─── 数据准备              ─── 高分辨率检测头        ─── 通道注意力
  ─── 基线训练              ─── 小目标召回验证         ─── 特征表达增强
  ─── 基准指标              ─── mAP 对比              ─── 小目标 AP 提升

Stage 4 (+CopyPaste)    →  Stage 5 (+SoftNMS)   →  Stage 6 (集成+交付)
  ─── 数据增强              ─── 后处理替换            ─── 消融实验
  ─── 泛化能力提升           ─── 减少误检/漏检         ─── 实时 Demo
  ─── 鲁棒性验证             ─── 精度微调              ─── 最终报告
```

每一个 Stage 都是**可独立验证的里程碑**，产出可量化的指标，下一阶段建立在上一阶段稳定的代码之上。

---

## 项目文件结构（最终形态）

```
small_target_detect/
├── data/
│   ├── TT100K/                    # 原始数据集（需下载）
│   ├── processed/                 # 转换后的 YOLO 格式标签
│   ├── dataset.yaml               # YOLO 数据集配置
│   └── prepare_dataset.py         # 数据预处理脚本
├── models/
│   ├── yolov8s_p2.yaml            # Stage 2: +P2 检测层
│   ├── yolov8s_p2_eca.yaml        # Stage 3: +P2 + ECA
│   ├── attention.py               # ECA / CBAM 等注意力模块
│   └── model_builder.py           # 模型构建工具
├── augment/
│   └── copy_paste.py              # Stage 4: Copy-Paste 增强
├── utils/
│   ├── soft_nms.py                # Stage 5: Soft-NMS 实现
│   ├── visualization.py           # 检测结果可视化
│   └── metrics.py                 # 小目标专项评估
├── configs/
│   ├── baseline.yaml              # 基线训练配置
│   └── improved.yaml              # 改进模型训练配置
├── experiments/
│   ├── stage1_baseline/           # 各阶段实验输出
│   ├── stage2_p2/
│   ├── stage3_eca/
│   ├── stage4_copypaste/
│   ├── stage5_softnms/
│   └── stage6_full/
├── scripts/
│   ├── train.py                   # 统一训练入口
│   ├── detect.py                  # 图片/视频批量推理
│   ├── eval.py                    # 评估脚本
│   └── demo.py                    # 实时摄像头演示
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Stage 1: 基线搭建（Baseline）

**目标**：跑通数据流，获得基线指标作为对比基准。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 1.1 | 下载 TT100K 数据集，筛选 5 类小目标 (`i2, i4, i5, io, p10`) | `data/TT100K/` 原始数据 |
| 1.2 | 编写 [data/prepare_dataset.py](data/prepare_dataset.py)：标注格式转换 → YOLO 归一化，7:2:1 划分 | `data/processed/` + `dataset.yaml` |
| 1.3 | 数据分析：统计小目标占比（面积 < 32×32 px），类别分布，anchor 尺寸分布 | `experiments/stage1_baseline/data_analysis.md` |
| 1.4 | 使用官方 `yolov8s.yaml` 训练基线模型，100 epoch，imgsz=1280 | `experiments/stage1_baseline/weights/best.pt` |
| 1.5 | 评估基线：mAP@0.5, mAP@0.5:0.95, AP_S, FPS | `experiments/stage1_baseline/metrics.json` |
| 1.6 | 编写评估脚本 [scripts/eval.py](scripts/eval.py)：支持按 COCO 小/中/大目标分组统计 | 可复用评估工具 |

### 关键代码设计

```python
# data/prepare_dataset.py 核心逻辑
def convert_tt100k_to_yolo(ann_file, class_map, img_size=(2048, 2048)):
    """
    TT100K 原始标注 → YOLO 格式
    - 输入: TT100K 的 JSON 标注
    - 输出: 每张图一个 .txt, class_id cx cy w h (归一化)
    - 筛选: 只保留 class_map 中 5 类，过滤掉非目标类别
    """
    ...

def split_dataset(image_dir, label_dir, ratio=(0.7, 0.2, 0.1)):
    """
    按 7:2:1 划分 train/val/test
    - 生成 train.txt, val.txt, test.txt（图片路径列表）
    - 生成 dataset.yaml（路径 + 类别名）
    """
    ...

# configs/baseline.yaml
train:
  model: yolov8s.yaml       # 官方未修改
  data: data/dataset.yaml
  epochs: 100
  imgsz: 1280
  batch: 8
  device: 0
  workers: 8
  patience: 20               # 早停
  cos_lr: true
  close_mosaic: 10           # 最后 10 epoch 关闭 mosaic
```

### 验证标准
- [ ] 数据集正确加载，5 类目标标签完整
- [ ] 小目标（<32×32）占总标注框的 __% (需实测)
- [ ] 基线 mAP@0.5 记录在案
- [ ] 推理 FPS 在 RTX 3060 上记录

### 预期指标记录模板

| 指标 | 基线 YOLOv8s |
|------|-------------|
| mAP@0.5 | ? |
| mAP@0.5:0.95 | ? |
| AP_S (small) | ? |
| AP_M (medium) | ? |
| FPS (1280×1280) | ? |
| 参数量 | 11.1M |
| FLOPs | 28.6G |

---

## Stage 2: P2 高分辨率检测层

**改进原理**：YOLOv8s 默认有 P3(80×80), P4(40×40), P5(20×20) 三个检测头。P3 的下采样倍数=8，对于 1280×1280 输入，P3 特征图上每个 cell 对应 8×8 像素区域。如果一个标志只有 16×16 像素，在 P3 上仅占 2×2 个 cell，特征表达极弱。增加 P2(160×160) 头使下采样倍数为 4，同样 16×16 的标志在 P2 上占 4×4 个 cell，特征表达能力翻倍。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 2.1 | 编写 [models/yolov8s_p2.yaml](models/yolov8s_p2.yaml)：修改 head 结构，添加 P2 层 | 模型配置文件 |
| 2.2 | 修改 Neck：将 backbone 第 2 层 (P2, stride=4) 的特征引入 FPN，与 P3 上采样拼接 | 配置文件改动 |
| 2.3 | 训练 P2 模型，其他配置与 Stage 1 完全相同（控制变量） | `experiments/stage2_p2/` |
| 2.4 | 对比 Stage 1：重点观察 AP_S 变化、小目标召回率 Recall@S | 对比报告 |
| 2.5 | 分析 P2 检测头的计算开销（参数量增量、FPS 变化） | 开销分析 |

### YAML 核心改动

```yaml
# models/yolov8s_p2.yaml (关键部分)
# YOLOv8s backbone (不变)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]                  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]                 # 1-P2/4  ← 新增引入
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]                 # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]                 # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]               # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]                  # 9

head:
  # 上采样阶段（自顶向下）
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]                  # cat P4
  - [-1, 3, C2f, [512]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]                  # cat P3
  - [-1, 3, C2f, [256]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # ← 新增：继续上采样
  - [[-1, 2], 1, Concat, [1]]                  # ← 新增：cat P2 (layer 2)
  - [-1, 3, C2f, [128]]                        # ← 新增：P2 特征融合

  # 下采样阶段（自底向上）
  - [-1, 1, Conv, [128, 3, 2]]                 # ← 新增：P2 下采样
  - [[-1, 15], 1, Concat, [1]]                 # ← 新增：cat 上采样 P3 输出
  - [-1, 3, C2f, [256]]                        # P3 增强
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, C2f, [512]]                        # P4 增强
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]]                       # P5 增强

  # 检测头（4 个尺度）
  - [[19, 22, 25, 28], 1, Detect, [nc]]
  #   ↑ P2  ↑ P3  ↑ P4  ↑ P5
```

### 关键注意事项
- YOLOv8 的 `Detect` 模块输出 channel 数 = `(nc + 4) × reg_max`（anchor-free），4 个检测头会自动适配
- 需要确保 Concat 的索引号正确（layer 编号需要精确数算）
- 引入 P2 后训练速度大约下降 20-30%（多了一个大特征图的处理），这是正常的

### 验证标准
- [ ] P2 模型成功训练不报错
- [ ] AP_S 相比基线提升 ≥ 3%（保守预期）
- [ ] 肉眼对比：小目标的检测框更完整、漏检减少
- [ ] 计算开销在可接受范围（FPS 下降 < 30%）

---

## Stage 3: ECA 通道注意力

**改进原理**：YOLOv8 的 Neck（FPN+PAN）在各尺度特征图之间传递信息，但 C2f 模块内部没有显式的通道重要性建模。ECA（Efficient Channel Attention）通过 1D 卷积捕获局部跨通道交互，以极小的计算开销（几乎可忽略）让网络自适应地关注重要通道，抑制噪声通道。对小目标而言，通道中选择性地增强精细纹理特征尤为有效。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 3.1 | 实现 ECA 模块 ([models/attention.py](models/attention.py))：支持可配 kernel_size | 可复用注意力模块 |
| 3.2 | 编写 [models/yolov8s_p2_eca.yaml](models/yolov8s_p2_eca.yaml)：在各检测层前的 C2f 后插入 ECA | 模型配置 |
| 3.3 | 实现模型构建工具 ([models/model_builder.py](models/model_builder.py))：解析 yaml 并注入 ECA 模块 | 构建工具 |
| 3.4 | 训练 P2+ECA 模型 | `experiments/stage3_eca/` |
| 3.5 | 对比 Stage 2：验证 ECA 是否带来精度的进一步提升 | 对比报告 |

### ECA 实现细节

```python
# models/attention.py
import torch.nn as nn

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA-Net, CVPR 2020)
    通过 1D 卷积实现局部跨通道交互，无降维，几乎无额外计算开销。
    
    kernel_size 自适应公式: k = |(log2(C) / γ + b/γ)|_odd
    其中 γ=2, b=1 是推荐参数
    """
    def __init__(self, channels, k_size=None, gamma=2, b=1):
        super().__init__()
        if k_size is None:
            # 自适应 kernel size
            t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) 
                         / gamma + b / gamma)))
            k_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        y = self.avg_pool(x)                    # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)     # (B, 1, C)
        y = self.conv(y)                        # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)   # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """
    (可选) CBAM 作为对比方案 — 同时使用通道+空间注意力
    在消融实验中可与 ECA 对比
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca = self.channel_attn(x)
        x = x * ca
        sa = self.spatial_attn(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        )
        return x * sa
```

### ECA 插入位置

```python
# models/model_builder.py
def insert_eca(model, positions='detect_before'):
    """
    在 YOLOv8 Neck 中插入 ECA 模块
    
    策略：在每个检测层 (P2/P3/P4/P5) 的最后一个 C2f 之后插入 ECA，
    让网络在输出预测之前做最后的通道重要性校准。
    
    可扩展为在每个 C2f 后插入（需对比实验验证最优策略）
    """
    ...

# 伪代码：遍历 model.model 的 layer 列表
# 找到 Detect 层前的 4 个 C2f，各插入一个 ECA(ch) 和对应的 Conv 适配
# 用 nn.Sequential 包装 C2f + ECA
```

### 验证标准
- [ ] ECA 模块正向/反向传播正确（梯度测试）
- [ ] P2+ECA 模型的 mAP@0.5 和 AP_S 均不低于 Stage 2（ECA 至少无害）
- [ ] 理想情况：AP_S 再提升 1-2%
- [ ] 参数量增量 < 0.01M（ECA 极轻量）

---

## Stage 4: Copy-Paste 数据增强

**改进原理**：小目标检测的核心瓶颈之一是**正样本数量不足**。TT100K 中每张图的小标志数量有限，且分布可能不均匀。Copy-Paste 增强将不同图中的小目标实例"粘贴"到其他图的合理位置，人工增加每张图的小目标密度，让模型在训练中看到更多样化的小目标-背景组合，尤其对小众类别（如 `p10` 禁止行人通行）的样本数有显著补充作用。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 4.1 | 实现 Copy-Paste 核心逻辑 ([augment/copy_paste.py](augment/copy_paste.py)) | 可复用增强模块 |
| 4.2 | 集成到训练流程：通过 `albumentations` 自定义 transform 或训练 loop 中回调 | 训练钩子 |
| 4.3 | 编写可视化脚本：抽查增强后的图片，验证粘贴位置合理、标签正确 | 质量检查 |
| 4.4 | 训练 P2+ECA+CopyPaste 模型 | `experiments/stage4_copypaste/` |
| 4.5 | 对比 Stage 3：验证泛化能力提升（train/val mAP gap 收窄） | 对比报告 |

### Copy-Paste 实现细节

```python
# augment/copy_paste.py
import random
import cv2
import numpy as np

class CopyPasteAugmentation:
    """
    小目标 Copy-Paste 增强
    
    算法流程：
    1. 随机选择一个 batch 中的源图像和目标图像
    2. 从源图像中随机选取小目标框（面积 < threshold）
    3. 提取目标像素（mask 或 bbox crop）
    4. 随机缩放 (0.8~1.2x) 和轻微旋转 (±10°)
    5. 粘贴到目标图像的空旷位置：
       - 用目标图像现有 bbox 的 IoU 判断是否"空旷"
       - 候选位置尝试 N 次，找 IoU < max_iou 的位置
    6. 更新目标图像的标签列表
    """
    
    def __init__(self, p=0.5, scale_range=(0.8, 1.2), max_iou=0.1, 
                 max_attempts=20, small_area_thresh=32*32):
        self.p = p
        self.scale_range = scale_range
        self.max_iou = max_iou
        self.max_attempts = max_attempts
        self.small_area_thresh = small_area_thresh

    def __call__(self, image, bboxes, all_images, all_bboxes):
        """
        Args:
            image: np.ndarray (H, W, 3) 目标图像
            bboxes: np.ndarray (N, 5) [cls, x1, y1, x2, y2] 绝对坐标
            all_images: list of np.ndarray — batch 中所有图像
            all_bboxes: list of np.ndarray — batch 中所有图像的框
        
        Returns:
            image_aug, bboxes_aug
        """
        if random.random() > self.p or len(all_images) < 2:
            return image, bboxes

        h, w = image.shape[:2]
        
        # 选源图像（与目标图像不同 & 有足够小目标）
        candidates = [
            (img, bb) for img, bb in zip(all_images, all_bboxes)
            if id(img) != id(image) and len(bb) > 0
        ]
        if not candidates:
            return image, bboxes
        
        src_img, src_bboxes = random.choice(candidates)
        
        # 从源图像中筛选小目标
        src_h, src_w = src_img.shape[:2]
        small_bboxes = []
        for bb in src_bboxes:
            x1, y1, x2, y2 = bb[1:]
            area = (x2 - x1) * (y2 - y1)
            if area < self.small_area_thresh:
                small_bboxes.append(bb)
        
        if not small_bboxes:
            return image, bboxes

        # 随机选 N 个小目标粘贴 (1~3 个)
        num_paste = min(random.randint(1, 3), len(small_bboxes))
        selected = random.sample(small_bboxes, num_paste)
        
        new_bboxes = list(bboxes)
        
        for bb in selected:
            cls_id = int(bb[0])
            x1, y1, x2, y2 = bb[1:].astype(int)
            crop = src_img[y1:y2, x1:x2].copy()
            
            # 随机缩放
            scale = random.uniform(*self.scale_range)
            new_h, new_w = int((y2 - y1) * scale), int((x2 - x1) * scale)
            if new_h < 5 or new_w < 5:
                continue
            crop_resized = cv2.resize(crop, (new_w, new_h))
            
            # 找粘贴位置
            pasted = False
            for _ in range(self.max_attempts):
                px = random.randint(0, max(0, w - new_w))
                py = random.randint(0, max(0, h - new_h))
                
                paste_box = np.array([px, py, px + new_w, py + new_h])
                
                # 检查与现有框的 IoU
                has_overlap = False
                for existing in new_bboxes:
                    existing_box = existing[1:5]
                    iou = self._compute_iou(paste_box, existing_box)
                    if iou > self.max_iou:
                        has_overlap = True
                        break
                
                if not has_overlap:
                    # 简单的直接粘贴（也可用 Poisson blending 让边缘更自然）
                    image[py:py + new_h, px:px + new_w] = crop_resized
                    new_bboxes.append(np.array([cls_id, px, py, px + new_w, py + new_h]))
                    pasted = True
                    break
            
        return image, np.array(new_bboxes)
    
    def _compute_iou(self, box1, box2):
        """计算两个 bbox 的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)
```

### 集成到训练流程

```python
# 方案 A: 通过自定义 DataLoader collate_fn 在 batch 组装时做 Copy-Paste
# 优点: 不侵入 ultralytics 代码
# 缺点: 需要控制变量，确保其他增强不冲突

# 方案 B: 改造 ultralytics 的 BaseDataset.__getitem__
# 优点: 与 mosaic/mixup 等增强在同一 pipeline 中
# 缺点: 需要了解 ultralytics 内部实现

# 推荐方案 A 起步，方案 B 作为优化
```

### 验证标准
- [ ] 增强后图片质量检查：粘贴痕迹不突兀，坐标无越界
- [ ] 训练过程中，每个 batch 平均小目标数量显著增加
- [ ] val mAP 不低于 Stage 3（增强不应降低验证精度）
- [ ] 过拟合程度降低（train/val gap 缩小）
- [ ] 对小众类别（`io`, `p10`）的 AP 提升明显

---

## Stage 5: Soft-NMS 后处理

**改进原理**：标准 NMS 对重叠度高的低分框直接置零，这在**密集小目标场景**下容易误杀真阳性。例如两个交通标志在图像中距离很近时，标准 NMS 可能只保留一个。Soft-NMS 采用"惩罚而非删除"策略：当两个框 IoU 高时，降低低分框的置信度（用高斯衰减函数），而不是直接去掉。这样密集排列的小目标有更多机会被保留下来。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 5.1 | 实现 Soft-NMS ([utils/soft_nms.py](utils/soft_nms.py))：支持 linear 和 gaussian 两种衰减 | 可复用 NMS 模块 |
| 5.2 | 在 [scripts/detect.py](scripts/detect.py) 中替换默认 NMS 为 Soft-NMS | 推理脚本改进 |
| 5.3 | 在 [scripts/eval.py](scripts/eval.py) 中支持 Soft-NMS 评估（val set 上跑） | 评估脚本改进 |
| 5.4 | Grid Search Soft-NMS 的超参数（σ, score_threshold） | 超参数调优 |
| 5.5 | 对比 Stage 4 + Soft-NMS vs Stage 4 + 标准 NMS | 对比报告 |

### Soft-NMS 实现

```python
# utils/soft_nms.py
import torch
import numpy as np

def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, 
             score_threshold=0.001, method='gaussian'):
    """
    Soft-NMS 实现
    
    Args:
        boxes: (N, 4) tensor [x1, y1, x2, y2]
        scores: (N,) tensor — 置信度
        iou_threshold: IoU 阈值
        sigma: 高斯衰减的 sigma 参数
        score_threshold: 最低分数阈值，低于此值的框直接丢弃
        method: 'linear' | 'gaussian'
    
    Returns:
        keep_indices: 保留的框索引
    
    算法 (对每个类别独立执行):
    1. 按 scores 降序排列
    2. 取最高分框 M
    3. 对剩余框 b_i 计算 IoU(M, b_i)
    4. 如果 method=='linear':
         score_i = score_i * (1 - IoU)    if IoU > threshold
    5. 如果 method=='gaussian':
         score_i = score_i * exp(-IoU^2 / sigma)
    6. 移除 score_i < score_threshold 的框
    7. 重复 2-6 直到所有框处理完
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    # 转成左上+右下格式
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    _, order = scores.sort(descending=True)
    keep = []
    
    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)
        
        if order.numel() == 1:
            break
        
        # 计算当前最高分框与剩余框的 IoU
        xx1 = torch.max(x1[idx], x1[order[1:]])
        yy1 = torch.max(y1[idx], y1[order[1:]])
        xx2 = torch.min(x2[idx], x2[order[1:]])
        yy2 = torch.min(y2[idx], y2[order[1:]])
        
        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        iou = inter / (areas[idx] + areas[order[1:]] - inter)
        
        # Soft-NMS 分数衰减
        if method == 'linear':
            weight = torch.where(
                iou > iou_threshold, 
                1.0 - iou,
                torch.ones_like(iou)
            )
        elif method == 'gaussian':
            weight = torch.exp(-(iou * iou) / sigma)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scores[order[1:]] *= weight
        
        # 过滤低分框
        keep_mask = scores[order[1:]] > score_threshold
        order = order[1:][keep_mask]
        
        # 重新排序（因为分数变了）
        _, order = scores[order].sort(descending=True)
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_soft_nms(predictions, iou_threshold=0.5, sigma=0.5, 
                     score_threshold=0.001, method='gaussian'):
    """
    对 ultralytics 的预测结果做 Soft-NMS
    
    Args:
        predictions: ultralytics Results 对象的原始输出
        ...
    
    Returns:
        过滤后的预测结果
    """
    # 提取 boxes, scores, classes
    # 按类别分组调用 soft_nms
    # 合并结果
    ...
```

### 超参数调优

```python
# 建议 Grid Search 范围
sigma_candidates = [0.3, 0.5, 0.7, 0.9]
iou_threshold_candidates = [0.45, 0.5, 0.55, 0.6]
# 在 val set 上评估每种组合，选最优 mAP
```

### 验证标准
- [ ] Soft-NMS 输出格式与标准 NMS 兼容
- [ ] 在密集小目标场景（多标志近距离排列），召回率提升
- [ ] 推理速度下降 < 5%（Soft-NMS 的计算开销主要在 IoU 计算，增加有限）
- [ ] 与标准 NMS 的 PR 曲线对比

---

## Stage 6: 系统集成、消融实验与交付

**目标**：完成消融实验表格，输出最终模型，交付 Demo。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 6.1 | 完整消融实验：逐一移除各改进点，量化每个模块的贡献 | `experiments/stage6_full/ablation.md` |
| 6.2 | 编写可视化工具 ([utils/visualization.py](utils/visualization.py))：PR 曲线、检测结果对比、特征图可视化 | 可视化工具 |
| 6.3 | 实现实时摄像头 Demo ([scripts/demo.py](scripts/demo.py))：支持 GPU 加速推理 | Demo 应用 |
| 6.4 | 导出 ONNX / TensorRT（可选）用于部署验证 | 部署模型 |
| 6.5 | 编写最终 README 和实验报告 | 文档 |

### 消融实验设计

| 实验 | P2 | ECA | Copy-Paste | Soft-NMS | mAP@0.5 | mAP@0.5:0.95 | AP_S | FPS |
|------|----|-----|------------|----------|---------|--------------|------|-----|
| A (基线) | ✗ | ✗ | ✗ | ✗ | ? | ? | ? | ? |
| B | ✓ | ✗ | ✗ | ✗ | ? | ? | ? | ? |
| C | ✓ | ✓ | ✗ | ✗ | ? | ? | ? | ? |
| D | ✓ | ✓ | ✓ | ✗ | ? | ? | ? | ? |
| E (完整) | ✓ | ✓ | ✓ | ✓ | ? | ? | ? | ? |

**消融分析**：
- B - A = P2 层的贡献
- C - B = ECA 的贡献
- D - C = Copy-Paste 的贡献
- E - D = Soft-NMS 的贡献

### 可视化工具

```python
# utils/visualization.py
def plot_detection_comparison(img, baseline_results, improved_results, save_path):
    """
    并排对比基线和改进模型的检测结果
    - 用不同颜色标注两个模型的框
    - FP/FN 差异可视化
    """
    ...

def plot_pr_curves(metrics_dict, save_path):
    """
    绘制多个模型的 PR 曲线对比
    - 每个类别一条线
    - 小/中/大目标分开
    """
    ...

def plot_feature_map(model, img, layer_names, save_path):
    """
    可视化特征图 — 查看 P2 检测层是否真的关注了小目标
    """
    ...
```

### Demo 应用

```python
# scripts/demo.py
"""
实时交通标志检测 Demo
- 支持: 图片文件 / 视频文件 / 摄像头 / RTSP 流
- 显示: 检测框 + 类别名 + 置信度 + FPS
- 快捷键: q=退出, s=截图, p=暂停
"""
```

### 验证标准
- [ ] 消融实验表格完整，每个改进点的贡献量化
- [ ] Demo 在摄像头输入下流畅运行（>30 FPS）
- [ ] 异常场景处理：无目标帧、低光照、遮挡场景
- [ ] 代码结构清晰，可复现训练

---

## 关键技术风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| P2 层导致显存不够 (batch > 8 → 4 或更低) | 训练不稳定 | 使用 gradient accumulation 补偿；降低 imgsz 到 1024 |
| ECA 无效果甚至掉点 | 白做 | 尝试其他 attention (CBAM, SE)；或仅在 P2/P3 层加 ECA |
| Copy-Paste 粘贴位置不合理导致 FP 增多 | 精度下降 | 严格 IoU 限制；加入边缘区域排除 |
| Soft-NMS 超参数敏感 | 效果不稳定 | 在 val 上充分 Grid Search；提供 fallback 回标准 NMS |
| TT100K 数据下载/标注质量问题 | 数据不可用 | 标注清洗脚本；标注可视化检查 |

---

## 里程碑时间线（建议）

```
Week 1: Stage 1 (数据 + 基线) ───────────────  ████████
Week 2: Stage 2 (P2) ───────────────────────── ████████
Week 3: Stage 3 (ECA) + Stage 4 (CopyPaste) ──  ████░░░░ (并行实验)
Week 4: Stage 5 (SoftNMS) + Stage 6 (集成) ───  ████████
```

P2 是最关键的改进（直接提升小目标特征分辨率），建议优先确保 P2 稳定后再叠加后续改进。ECA 和 Copy-Paste 可以并行实验（各自基于 Stage 2 分支），最后在 Stage 6 合并。

---

## 代码规范

- 遵循现有项目风格（Python 3.12+, 类型标注）
- 所有路径使用 `pathlib.Path`
- 训练配置通过 yaml 文件管理，不硬编码
- 每个 Stage 的训练产物放在独立目录下
- 使用 `logging` 而非 `print`
- 关键函数有 docstring 和类型标注
