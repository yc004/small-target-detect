# 基于改进 YOLO26 的小目标交通标志检测 — 渐进式实现方案

## 总览

```
Stage 1 (基线)        → Stage 2 (+官方P2)     → Stage 3 (+ARF-Head)
  官方 YOLO26s            官方 P2 高分辨率          自适应感受野检测头
  数据流验证              小目标召回验证             尺度感知的特征提取

Stage 4 (+BCEM)       → Stage 5 (+HJ-Loss)     → Stage 6 (集成+交付)
  双向上下文增强           分层联合损失              消融实验 + Demo
  前景/背景判别            小目标定位精度             完整系统交付
```

每一个 Stage 都是**可独立验证的里程碑**，下一阶段建立在上一阶段稳定的代码之上。

---

## 项目文件结构（最终形态）

```
small_target_detect/
├── data/
│   ├── TT100K/                        # 原始数据集（需手动下载）
│   ├── processed/                     # 转换后的 YOLO 格式标签
│   │   ├── dataset.yaml               # YOLO 数据集配置
│   │   ├── images/{train,val,test}/   # 划分后的图片
│   │   └── labels/{train,val,test}/   # YOLO 格式标注
│   ├── prepare_dataset.py             # 数据预处理脚本
│   └── extract_dataset.py             # 备选：从预打包 ZIP 提取
├── models/
│   ├── yolo26s_p2_arf.yaml            # Stage 3: P2 + ARF-Head 模型配置
│   ├── yolo26s_p2_arf_bcem.yaml       # Stage 4: + BCEM
│   ├── arf_head.py                    # ARF-Head 模块实现
│   ├── bcem.py                        # BCEM 模块实现
│   ├── hj_loss.py                     # HJ-Loss 分层联合损失函数
│   └── model_builder.py               # 模型构建工具（注入自定义模块）
├── utils/
│   ├── soft_nms.py                    # Soft-NMS 实现
│   ├── visualization.py               # 检测结果可视化 + PR 曲线
│   └── metrics.py                     # COCO 小/中/大目标分类评估
├── configs/
│   ├── stage1_baseline.yaml           # Stage 1: YOLO26s 基线
│   ├── stage2_p2.yaml                 # Stage 2: YOLO26s-P2 官方
│   ├── stage3_arf_head.yaml           # Stage 3: + ARF-Head
│   ├── stage4_bcem.yaml               # Stage 4: + BCEM
│   ├── stage5_hjloss.yaml             # Stage 5: + HJ-Loss
│   └── stage6_full.yaml               # Stage 6: 最终集成配置
├── experiments/                       # 各阶段实验输出
│   ├── stage1_baseline/
│   ├── stage2_p2/
│   ├── stage3_arf_head/
│   ├── stage4_bcem/
│   ├── stage5_hjloss/
│   └── stage6_full/
├── scripts/
│   ├── __init__.py
│   ├── train.py                       # 统一训练入口
│   ├── detect.py                      # 图片/视频批量推理
│   ├── eval.py                        # COCO 评估脚本
│   ├── demo.py                        # 实时摄像头演示
│   ├── analyze_data.py                # 数据集统计分析
│   ├── setup_server.sh                # Linux 一键部署脚本
│   └── setup_server.bat               # Windows 一键部署脚本
├── docs/
│   ├── generate_report.py             # Word 报告自动生成
│   ├── report.tex                     # LaTeX 报告源码
│   ├── 作品报告_*.docx                # 生成的 Word 报告
│   └── 作品报告模板.docx              # 报告模板
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## Stage 1: YOLO26s 基线

**目标**：用官方 YOLO26s 跑通完整数据流，获得所有基准指标。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 1.1 | 确保数据集就绪：`data/processed/` 下有正确的 YOLO 格式标注和 `dataset.yaml` | 可训练的数据集 |
| 1.2 | 编写 `configs/stage1_baseline.yaml`，使用 `yolo26s.yaml` | 训练配置 |
| 1.3 | 执行 `python scripts/train.py --config configs/stage1_baseline.yaml` | `experiments/stage1_baseline/weights/best.pt` |
| 1.4 | 执行 `python scripts/eval.py --weights best.pt --data data/processed/dataset.yaml --analyze_sizes` | `metrics.json` (含 AP_S/AP_M/AP_L) |
| 1.5 | 执行 `python scripts/analyze_data.py` 获取数据集统计 | 小目标占比、类别分布报告 |

### 关键配置

```yaml
# configs/stage1_baseline.yaml
model: "yolo26s.yaml"          # 官方 YOLO26 small
pretrained: true
data: "data/processed/dataset.yaml"
imgsz: 640
epochs: 100
batch: 16
device: "0"
optimizer: "auto"              # SGD (YOLO26s 属于 small 模型)
lr0: 0.01
cos_lr: true
patience: 20                   # 早停
close_mosaic: 10               # 最后 10 epoch 关闭 mosaic
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
project: "experiments/stage1_baseline"
```

### 验证标准
- [ ] 训练成功收敛（无 OOM/NaN）
- [ ] 记录 mAP@0.5, mAP@0.5:0.95, AP_S, FPS
- [ ] 数据集统计确认：小目标 (< 32×32 px) 占比 > 40%
- [ ] 基线模型参数量 & FLOPs 记录在案

### 预期指标记录模板

| 指标 | YOLO26s (Stage 1) | 备注 |
|------|:---:|------|
| mAP@0.5 | ? | COCO 标准评估 |
| mAP@0.5:0.95 | ? | |
| AP_S (small, <32²) | ? | 核心关注指标 |
| AP_M (medium) | ? | |
| AP_L (large) | ? | |
| FPS (640×640, GPU) | ? | |
| 参数量 | 10.01M | 官方数据 |
| GFLOPs | 22.8 | 官方数据 |

---

## Stage 2: 官方 YOLO26s-P2 高分辨率检测层

**目标**：使用 ultralytics 官方 `yolo26s-p2.yaml` 训练，量化 P2 层对小目标检测的贡献。

### 原理

YOLO26 默认 3 个检测头 (P3/8, P4/16, P5/32)。增加 P2/4 层后：
- 16×16 px 的目标在 P3 上仅占 2×2 cells → P2 上占 4×4 cells
- 特征分辨率的翻倍使得小目标的定位精度和分类置信度都能提升
- 官方 P2 变体经过 ultralytics 充分测试，稳定性有保障

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 2.1 | 创建 `configs/stage2_p2.yaml`，指向 `yolo26s-p2.yaml` | 训练配置 |
| 2.2 | 执行训练（batch 减半以容纳 P2 层的额外计算） | `experiments/stage2_p2/weights/best.pt` |
| 2.3 | 评估 + 与 Stage 1 对比 | 对比报告 |
| 2.4 | 分析 P2 层的计算开销 | 开销分析 |

### 配置改动

```yaml
# configs/stage2_p2.yaml
model: "yolo26s-p2.yaml"       # 官方 P2 变体
batch: 8                        # 减半（P2 增加 ~25% FLOPs）
project: "experiments/stage2_p2"
# 其余与 baseline 完全相同
```

### YOLO26s-P2 架构关键信息
- 4 个检测头：P2(160×160), P3(80×80), P4(40×40), P5(20×20)
- Neck 中相比标准版增加：1 次额外的上采样 + 1 次额外的下采样 + 对应的 C3k2 特征融合
- P2 层通道数 = 128（最轻量，不会过度增加计算量）
- 参数量：~9.77M (s 变体)，GFLOPs：~27.8

### 验证标准
- [ ] P2 模型成功训练不报错
- [ ] AP_S 相比 Stage 1 提升 ≥ 5%（P2 对小目标的直接增益）
- [ ] 小目标漏检率显著降低（肉眼对比检测结果）
- [ ] FPS 下降 < 30%

---

## Stage 3: ARF-Head — 自适应感受野检测头 ⭐ 原创

**改进原理**：不同检测层负责不同尺度的目标，但对感受野的需求不同。P2 检测极小目标时，需要略大的感受野来捕获上下文（区分目标 vs 噪声）；P5 检测大目标时，感受野已经很大，再增加反而有害。标准 YOLO26 所有检测头使用相同结构，缺乏这种尺度感知能力。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 3.1 | 实现 ARF 模块 (`models/arf_head.py`) | 可复用的 ARF 模块 |
| 3.2 | 编写 `models/yolo26s_p2_arf.yaml`：在 P2/P3/P4/P5 各检测层前插入 ARF | 模型配置 |
| 3.3 | 编写 `models/model_builder.py`：解析 yaml 并注入 ARF 模块 | 模型构建工具 |
| 3.4 | 编写 `configs/stage3_arf_head.yaml` | 训练配置 |
| 3.5 | 训练 + 对比 Stage 2 | 对比报告 |

### ARF 模块实现

```python
# models/arf_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ARF(nn.Module):
    """
    Adaptive Receptive Field Module (ARF)
    
    通过多分支空洞卷积 + 可学习 softmax 融合，
    为不同检测层提供尺度感知的自适应感受野。
    
    空洞率根据检测层自适应配置：
      - P2 (stride=4):  [1, 3, 5] — 适度扩大 RF
      - P3 (stride=8):  [1, 2, 3] — 轻微扩大
      - P4 (stride=16): [1, 1, 2] — 几乎不变
      - P5 (stride=32): [1, 1, 1] — 标准卷积
    """
    
    def __init__(self, channels, dilations=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                # 深度可分离卷积：空间维度用 depthwise，通道维度用 pointwise
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
            )
            for d in dilations
        ])
        
        # 可学习的分支权重
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, len(dilations), 1, bias=False),
            nn.Softmax(dim=1),
        )
        
        # 输出投影
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        # 各分支输出
        branch_outs = [branch(x) for branch in self.branches]  # list of (B,C,H,W)
        
        # 可学习权重
        weights = self.weight_net(x)  # (B, num_branches, 1, 1)
        
        # 加权融合
        fused = sum(w * out for w, out in zip(
            weights.split(1, dim=1), branch_outs
        ))
        
        # 残差连接 + 投影
        return self.proj(fused) + x


# 不同检测层的空洞率配置
ARF_CONFIGS = {
    'P2': (1, 3, 5),   # stride=4: 需要更多上下文
    'P3': (1, 2, 3),   # stride=8: 轻度扩大
    'P4': (1, 1, 2),   # stride=16: 几乎不变
    'P5': (1, 1, 1),   # stride=32: 标准
}


def insert_arf_heads(model_yaml_dict, arf_configs=ARF_CONFIGS):
    """
    在模型 YAML 配置的每个检测层之前插入 ARF 模块。
    
    解析 head 部分，找到 Detect 层引用的各层索引（即 P2/P3/P4/P5 的特征层），
    在这些层之后、Detect 之前插入对应的 ARF 模块。
    
    由于 YOLO26 的 Detect 层是 `[[16, 19, 22], 1, Detect, [nc]]`，
    我们需要在索引 16, 19, 22 之后各插入一个 ARF。
    
    Args:
        model_yaml_dict: 已解析的模型配置字典
        arf_configs: 各检测层对应的空洞率配置
    
    Returns:
        修改后的模型配置字典
    """
    import copy
    result = copy.deepcopy(model_yaml_dict)
    head = result['head']
    
    # 找到 Detect 层的索引
    detect_idx = None
    detect_entry = None
    for i, entry in enumerate(head):
        if entry[-1] == 'Detect' or (isinstance(entry, list) and len(entry) >= 2 and entry[1] == 'Detect'):
            detect_idx = i
            detect_entry = head[i]
            break
    
    if detect_entry is None:
        raise ValueError("Cannot find Detect layer in head config")
    
    # Detect 层引用的特征层索引
    detect_from = detect_entry[0]  # e.g. [16, 19, 22] or [19, 22, 25, 28]
    
    # 按照从 P2 到 P5 的顺序分配配置
    layer_names = ['P2', 'P3', 'P4', 'P5'] if len(detect_from) == 4 else ['P3', 'P4', 'P5']
    
    # 为每个检测层插入 ARF
    for i, (layer_idx, layer_name) in enumerate(zip(detect_from, layer_names)):
        dilations = arf_configs[layer_name]
        ch = head[layer_idx][3][0]  # 该层 C3k2 的输出通道数
        # 在该层之后插入 ARF
        # 策略：增加一层 ARF，from = [layer_idx], 1, ARF, [ch, dilations]
        pass
    
    return result
```

### 模型配置 (`yolo26s_p2_arf.yaml`)

基于 `yolo26s-p2.yaml` 的 head 部分，在各检测层前插入 ARF 模块：

```yaml
# models/yolo26s_p2_arf.yaml
# 基于 yolo26s-p2.yaml，在 head 尾部检测层前插入 ARF 模块

nc: 5
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# Backbone (同 yolo26s-p2.yaml, 不变)
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]

# Head (同 yolo26s-p2.yaml + ARF 模块)
head:
  # FPN 自顶向下
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, True]]       # 19: P2 特征
  
  # PAN 自底向上
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]       # 22: P3 特征
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]       # 25: P4 特征
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]  # 28: P5 特征
  
  # ARF 模块 (新增) + 检测层
  - [[19], 1, ARF, [128, [1, 3, 5]]]   # 29: ARF-P2
  - [[22], 1, ARF, [256, [1, 2, 3]]]   # 30: ARF-P3
  - [[25], 1, ARF, [512, [1, 1, 2]]]   # 31: ARF-P4
  - [[28], 1, ARF, [1024, [1, 1, 1]]]  # 32: ARF-P5
  
  - [[29, 30, 31, 32], 1, Detect, [nc]]
```

### 模型构建工具

```python
# models/model_builder.py
import torch.nn as nn
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
from models.arf_head import ARF, ARF_CONFIGS

# 注册自定义模块到 ultralytics
# ultralytics 在解析 yaml 时会动态查找模块，需要在 ultralytics.nn.modules 中注册
import ultralytics.nn.modules as ult_nn
ult_nn.ARF = ARF  # 注册 ARF 模块


def create_model(cfg_path, nc=5):
    """
    加载自定义模型配置并构建模型。
    
    Args:
        cfg_path: yaml 配置文件路径 (如 'models/yolo26s_p2_arf.yaml')
        nc: 类别数
    
    Returns:
        DetectionModel 实例
    """
    # 读取 yaml 并覆盖 nc
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg['nc'] = nc
    
    # 使用 DetectionModel 构建
    model = DetectionModel(cfg, ch=3, nc=nc)
    return model
```

### 验证标准
- [ ] ARF 模块正向/反向传播测试通过
- [ ] 自定义 yaml 可被 ultralytics 正确加载（不报错）
- [ ] 训练收敛正常
- [ ] AP_S 相比 Stage 2 提升 ≥ 2%
- [ ] 参数量增量 < 0.2M

---

## Stage 4: BCEM — 双向上下文增强模块 ⭐ 原创

**改进原理**：PANet 的跳跃连接直接将 Backbone 浅层特征拼接到 Neck，但这些浅层特征包含大量与前景目标无关的背景纹理。小目标（如交通标志）极易与道路标志线、文字、建筑边缘等背景元素混淆。BCEM 在跳跃连接处进行双向（通道 + 空间）的上下文增强，显式抑制背景、增强前景。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 4.1 | 实现 BCEM 模块 (`models/bcem.py`) | 可复用的 BCEM 模块 |
| 4.2 | 编写 `models/yolo26s_p2_arf_bcem.yaml`：在 Backbone→Neck 跳跃连接前插入 BCEM | 模型配置 |
| 4.3 | 编写 `configs/stage4_bcem.yaml` | 训练配置 |
| 4.4 | 训练 + 对比 Stage 3 | 对比报告 |

### BCEM 模块实现

```python
# models/bcem.py
import torch
import torch.nn as nn
import math


class BCEM(nn.Module):
    """
    Bi-directional Context Enhancement Module (BCEM)
    
    在 Backbone → Neck 的跳跃连接处进行通道 + 空间双向上下文增强。
    - 通道分支：通过 1D 卷积捕获局部跨通道交互（类似 ECA，但核更大）
    - 空间分支：通过 7×7 大核深度卷积进行前景/背景判别
    - 双分支乘法融合：确保两个维度双重确认才增强
    - 残差连接：Output = Input + Input × ChAttn × SpAttn
    
    Args:
        channels: 输入通道数
        k_size: 1D 卷积核大小（None 则自适应计算）
        gamma, b: 自适应核大小的参数
    """
    
    def __init__(self, channels, k_size=None, gamma=2, b=1):
        super().__init__()
        
        # 自适应 1D 卷积核大小
        if k_size is None:
            t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
            k_size = t if t % 2 == 1 else t + 1
            k_size = max(3, k_size)  # 至少 3
        
        # 通道分支：1D 卷积实现局部跨通道交互
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # (B, C, 1, 1)
            nn.Conv1d(1, 1, k_size, padding=k_size//2, bias=False),  # 1D 跨通道扫描
            nn.Sigmoid(),
        )
        
        # 空间分支：7×7 大核深度卷积 + 1×1 pointwise
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),  # pointwise 压缩
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False),  # 大核 depthwise
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 1, 1, bias=False),         # 空间注意力图 (B, 1, H, W)
            nn.Sigmoid(),
        )
        
        # 初始化：初始时 BCEM 输出接近恒等映射
        nn.init.constant_(self.spatial_branch[-1].weight, 0.0)
    
    def forward(self, x):
        # 通道注意力
        b, c, h, w = x.shape
        ca = self.channel_branch(x)                # (B, C, 1, 1)
        
        # 空间注意力
        sa = self.spatial_branch(x)                 # (B, 1, H, W)
        
        # 乘法融合 + 残差连接
        # ca 和 sa 通过广播相乘 → (B, C, H, W)
        enhanced = x * ca * sa
        return x + enhanced


# 注册到 ultralytics
# from ultralytics.nn import modules as ult_nn
# ult_nn.BCEM = BCEM
```

### 插入位置

BCEM 模块插入在 Backbone 各层输出到 Neck 跳跃连接之间。具体来说，在 `yolo26s_p2_arf.yaml` 的 head 部分，每个 `Concat` 从 Backbone 引入特征之前，对 Backbone 特征先过 BCEM：

```yaml
# models/yolo26s_p2_arf_bcem.yaml 的关键改动
head:
  # ... 前半部分不变 ...
  
  # FPN 阶段：对每个从 Backbone 拼接的特征先做 BCEM
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[6], 1, BCEM, [512]]           # ← 新增: backbone P4 特征过 BCEM
  - [[-2, -1], 1, Concat, [1]]      # 拼接 BCEM 增强后的特征
  - [-1, 2, C3k2, [512, True]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[4], 1, BCEM, [256]]           # ← 新增: backbone P3 特征过 BCEM
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]
  
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[2], 1, BCEM, [128]]           # ← 新增: backbone P2 特征过 BCEM
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, True]]
  
  # ... PAN 路径和 ARF 模块保持不变 ...
```

### 与 BCEM 配合的注意事项
- 由于 ultralytics 的 yaml 解析限制，BCEM 的插入可能需要在 `model_builder.py` 中通过代码方式实现
- 备选方案：直接在构建后的 `model.model` 上遍历和 wrap 特定层
- 通道数需根据 scale (n/s/m/l/x) 动态匹配

### 验证标准
- [ ] BCEM 模块正向/反向传播测试通过（梯度不消失）
- [ ] 插入 BCEM 后模型加载不报错
- [ ] 训练收敛，损失下降正常
- [ ] mAP@0.5 相比 Stage 3 提升 ≥ 1.5%
- [ ] AP_S 提升 ≥ 2%，证明上下文增强对小目标有效

---

## Stage 5: HJ-Loss — 分层联合损失函数 ⭐ 原创

**改进原理**：
- IoU Loss 对小目标的位置偏移极度敏感 → 需要尺度鲁棒的定位监督
- NWD（Normalized Wasserstein Distance）将 bbox 建模为高斯分布，对尺度不敏感，特别适合小目标
- Wise-IoU v3 通过动态聚焦机制自动关注中等质量样本
- **关键创新**：不同检测层使用不同的损失组合 —— P2/P3（小目标层）使用 WIoUv3 + NWD 联合监督，P4/P5 仅使用 WIoUv3

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 5.1 | 实现 WIoU v3 Loss (`models/hj_loss.py` 中的 `WIoUv3Loss`) | WIoU v3 损失模块 |
| 5.2 | 实现 NWD Loss (`models/hj_loss.py` 中的 `NWDLoss`) | NWD 损失模块 |
| 5.3 | 实现 HJ-Loss 调度器：管理不同层的损失分配和权重 | 分层损失调度器 |
| 5.4 | 修改训练流程：通过回调或自定义 loss 注入 HJ-Loss | 训练集成 |
| 5.5 | 编写 `configs/stage5_hjloss.yaml` | 训练配置 |
| 5.6 | 训练 + 对比 Stage 4 | 对比报告 |

### HJ-Loss 实现

```python
# models/hj_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WIoUv3Loss(nn.Module):
    """
    Wise-IoU v3: 动态非单调聚焦机制
    
    相比 CIoU, WIoUv3 通过离群度 β 动态调整梯度增益，
    自动关注中等质量的 anchor，抑制极端离群样本。
    
    公式:
      L_WIoUv3 = r * R_WIoU * L_IoU
      其中:
        R_WIoU = exp((x-x_gt)²+(y-y_gt)²/(Wg²+Hg²))  # 惩罚框中心距离
        r = β / (δ * α^(β-δ))                          # 动态梯度增益
        β = L_IoU* / L_IoU_mean                         # 离群度 (detach)
    
    Reference: Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing", 2023
    """
    
    def __init__(self, momentum=0.99, delta=2.0):
        super().__init__()
        self.momentum = momentum
        self.delta = delta
        self.register_buffer('running_mean', torch.tensor(1.0))
    
    def forward(self, pred_boxes, gt_boxes):
        """
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] normalized
            gt_boxes:   (N, 4) [cx, cy, w, h] normalized
        Returns:
            loss: scalar
        """
        # 计算 IoU
        px1, py1, px2, py2 = self._xywh_to_xyxy(pred_boxes)
        gx1, gy1, gx2, gy2 = self._xywh_to_xyxy(gt_boxes)
        
        inter_w = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
        inter_h = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
        inter = inter_w * inter_h
        
        area_p = (px2 - px1) * (py2 - py1)
        area_g = (gx2 - gx1) * (gy2 - gy1)
        union = area_p + area_g - inter + 1e-7
        iou = inter / union
        
        # R_WIoU: 惩罚中心点距离
        px_c, py_c = (px1 + px2) / 2, (py1 + py2) / 2
        gx_c, gy_c = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        wg, hg = gx2 - gx1, gy2 - gy1
        r_wioU = torch.exp(
            ((px_c - gx_c) ** 2 + (py_c - gy_c) ** 2) / 
            ((wg ** 2 + hg ** 2) + 1e-7)
        )
        
        # 基础损失
        L_iou = 1.0 - iou
        
        # 动态聚焦因子: r = β / (δ * α^(β-δ))
        # β = L_iou / L_iou_mean (离群度)
        with torch.no_grad():
            beta = L_iou.detach() / (self.running_mean + 1e-7)
        
        r = beta / (self.delta * torch.pow(1.0, beta - self.delta))
        
        # 更新 running mean (仅在训练时)
        if self.training:
            self.running_mean = (
                self.momentum * self.running_mean + 
                (1 - self.momentum) * L_iou.detach().mean()
            )
        
        loss = (r.detach() * r_wioU * L_iou).mean()
        return loss
    
    @staticmethod
    def _xywh_to_xyxy(boxes):
        cx, cy, w, h = boxes.chunk(4, dim=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2


class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss
    
    将边界框建模为 2D 高斯分布，计算 Wasserstein 距离。
    对小尺度偏移不敏感，专为小目标定位设计。
    
    公式:
      NWD(N_a, N_b) = exp(-√W_2²(N_a, N_b) / C)
      其中 C 是归一化常数（通常设为数据集平均目标尺寸）
    
    Reference: Wang et al., "A Normalized Gaussian Wasserstein Distance 
               for Tiny Object Detection", 2021
    """
    
    def __init__(self, c=12.0):
        """
        Args:
            c: 归一化常数，与数据集平均目标尺寸相关
               对于 TT100K，建议 c = 12.0 (匹配平均 16×16 小目标的 2D 高斯)
        """
        super().__init__()
        self.c = c
    
    def forward(self, pred_boxes, gt_boxes):
        """
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] normalized
            gt_boxes:   (N, 4) [cx, cy, w, h] normalized
        Returns:
            loss: scalar
        """
        # 将 bbox 转换为 2D 高斯参数: (μ_x, μ_y, σ_w, σ_h)
        # 高斯分布建模: μ = (cx, cy); Σ = diag(w²/4, h²/4)
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes.chunk(4, dim=-1)
        gt_cx, gt_cy, gt_w, gt_h = gt_boxes.chunk(4, dim=-1)
        
        # Wasserstein 距离平方
        # W_2² = ||μ_p - μ_g||² + ||σ_p - σ_g||_F²
        mu_dist = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        sigma_dist = (
            ((pred_w - gt_w) / 2) ** 2 + 
            ((pred_h - gt_h) / 2) ** 2
        )
        w2_sq = mu_dist + sigma_dist
        
        # 归一化
        nwd = torch.exp(-torch.sqrt(w2_sq + 1e-7) / self.c)
        
        loss = (1.0 - nwd).mean()
        return loss


class HierarchicalJointLoss(nn.Module):
    """
    分层联合损失 (HJ-Loss)
    
    为不同检测层提供差异化的损失函数：
    - P2/P3 (小目标): WIoUv3 + 0.3 * NWD
    - P4/P5 (大目标): WIoUv3
    
    这种设计避免了 NWD 对大目标引入不必要的噪声。
    """
    
    def __init__(self, nwd_weight=0.3, nwd_layers=['P2', 'P3'], c_nwd=12.0):
        super().__init__()
        self.wiouv3 = WIoUv3Loss()
        self.nwd = NWDLoss(c=c_nwd)
        self.nwd_weight = nwd_weight
        self.nwd_layers = nwd_layers  # 使用 NWD 的检测层
    
    def forward(self, pred_boxes, gt_boxes, layer_name='P3'):
        """
        Args:
            pred_boxes: 预测框
            gt_boxes: 真实框
            layer_name: 检测层标识 ('P2', 'P3', 'P4', 'P5')
        """
        loss_wiou = self.wiouv3(pred_boxes, gt_boxes)
        
        if layer_name in self.nwd_layers:
            loss_nwd = self.nwd(pred_boxes, gt_boxes)
            return loss_wiou + self.nwd_weight * loss_nwd
        else:
            return loss_wiou
```

### ultralytics 集成方案

HJ-Loss 需要集成到训练流程中。ultralytics 的损失函数在 `ultralytics.utils.loss` 中定义。集成方案：

```python
# 方案：通过修改 ultralytics 的 v8DetectionLoss 来注入自定义损失
# 在 models/hj_loss.py 中提供替换函数

def apply_hj_loss(trainer):
    """
    在 ultralytics Trainer 中注入 HJ-Loss。
    
    通过 monkey-patch trainer.loss 来替换默认的 CIoU loss，
    同时保留分类损失和 DFL (reg_max=1 时自动跳过 DFL)。
    """
    from models.hj_loss import HierarchicalJointLoss
    
    hj_loss = HierarchicalJointLoss()
    original_loss_fn = trainer.loss
    
    def custom_loss(preds, batch):
        # 调用原始损失获取分类部分
        # 替换回归部分
        ...
    
    return hj_loss
```

### 验证标准
- [ ] WIoUv3 和 NWD 模块的数值正确性（与参考实现对比）
- [ ] HJ-Loss 能正常训练（不 NaN，收敛）
- [ ] AP_S 相比 Stage 4 提升 ≥ 3%
- [ ] 高 IoU 阈值下（AP@0.75）的小目标精度提升显著（NWD 的优势）

---

## Stage 6: 系统集成、消融实验与交付

**目标**：完成完整消融实验，输出最终模型，交付所有代码和文档。

### 工作内容

| 序号 | 任务 | 产出 |
|------|------|------|
| 6.1 | 完整消融实验：逐一移除各模块，量化贡献 | `experiments/stage6_full/ablation.md` |
| 6.2 | 实现 Soft-NMS (`utils/soft_nms.py`) + 集成到推理 | 优化后的后处理 |
| 6.3 | 多尺度 TTA（测试时增强）评估 | 最终精度指标 |
| 6.4 | ONNX 导出验证 | 部署就绪模型 |
| 6.5 | 实时摄像头 Demo | 演示应用 |
| 6.6 | 更新 `docs/generate_report.py`，生成最终 Word 报告 | 竞赛报告 |

### 消融实验设计

| 实验 | P2 | ARF-Head | BCEM | HJ-Loss | mAP@0.5 | mAP@0.5:0.95 | AP_S | FPS |
|:---:|:--:|:---:|:--:|:--:|:---:|:---:|:---:|:---:|
| A (基线) | ✗ | ✗ | ✗ | ✗ | ? | ? | ? | ? |
| B | ✓ | ✗ | ✗ | ✗ | ? | ? | ? | ? |
| C | ✓ | ✓ | ✗ | ✗ | ? | ? | ? | ? |
| D | ✓ | ✓ | ✓ | ✗ | ? | ? | ? | ? |
| E (完整) | ✓ | ✓ | ✓ | ✓ | ? | ? | ? | ? |

**贡献分析**：
- B - A = P2 检测层的净贡献
- C - B = ARF-Head 的净贡献
- D - C = BCEM 的净贡献
- E - D = HJ-Loss 的净贡献

### 可视化验证清单
- [ ] PR 曲线对比（Baseline vs Stage 6）
- [ ] 小目标热力图：BCEM 的显著性图可视化（证明 BCEM 确实增强了前景区域）
- [ ] 错误分析：对比 Baseline 和 Full Model 的 FN/FP 案例
- [ ] 推理速度 Benchmark（GPU + CPU）

---

## 关键技术风险与对策

| 风险 | 影响 | 概率 | 对策 |
|------|------|:---:|------|
| YOLO26s 预训练权重与自定义 head 不兼容 | ARF/BCEM 模块无预训练权重 | 中 | 仅 load backbone 预训练，head 部分随机初始化后 warmup |
| ARF 模块在 yaml 中注册失败 | 模型无法构建 | 中 | 备选：用 model_builder.py 代码方式手动插入模块 |
| BCEM 插入后维度不匹配 | 训练报错 | 低 | 编写自动化通道验证脚本；支持自适应通道适配 |
| HJ-Loss 与 ultralytics 训练流程不兼容 | 无法使用标准 train() | 中 | 通过 callback hook 或 monkey-patch 注入；备选：自定义训练循环 |
| P2 + ARF 导致显存不足（batch=8 → OOM） | 训练失败 | 中 | 降至 batch=4 + gradient_accumulation=2 |
| ultralytics 版本升级导致 API 变动 | 代码不兼容 | 低 | 锁定 ultralytics==8.4.65 版本 |

---

## 里程碑时间线

```
Phase 1 (2天):  Stage 1 — 数据 + 基线训练 + 基线评估
Phase 2 (1天):  Stage 2 — YOLO26s-P2 官方 P2 变体
Phase 3 (2天):  Stage 3 — ARF-Head 模块开发 + 训练 + 验证
Phase 4 (2天):  Stage 4 — BCEM 模块开发 + 训练 + 验证
Phase 5 (1天):  Stage 5 — HJ-Loss 实现 + 训练 + 验证
Phase 6 (2天):  Stage 6 — 消融实验 + Demo + 报告
───────────────────────────────────────────────────
总计: 10 天
```

---

## 代码规范

- 所有新增模块在 `models/` 下，遵循现有项目的 docstring + 类型标注风格
- 模块通过 `model_builder.py` 注册到 ultralytics，不直接修改 ultralytics 源码
- 每个 Stage 的配置和产物独立管理，确保可复现
- 训练配置通过 YAML 文件管理，与 `scripts/train.py` 兼容
- 所有自定义 nn.Module 提供 `forward` 的显式 shape 注释
