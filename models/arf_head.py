# ────────────────────────────────────────────────────────────────────────────
# ARF-Head: Adaptive Receptive Field Detection Head
# ────────────────────────────────────────────────────────────────────────────
# Stage 3 原创模块 — 为不同检测层提供尺度感知的自适应感受野。
#
# 核心思想：
#   P2 层检测极小目标时需要略大的感受野来捕获上下文（区分目标 vs 噪声），
#   P5 层检测大目标时感受野已经足够甚至过大。
#   标准 YOLO26 所有检测头使用相同结构，缺乏这种尺度感知能力。
#
# 实现：
#   - 多分支空洞卷积 (dilation branches) 提供不同感受野
#   - 可学习 softmax 权重进行自适应融合
#   - 深度可分离卷积控制计算开销 (< 0.2M 额外参数)
# ────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 不同检测层的空洞率配置
# ---------------------------------------------------------------------------
# P2 (stride=4):  检测极小目标，需要更多上下文 → 大空洞率
# P3 (stride=8):  检测小目标，轻度扩大感受野
# P4 (stride=16): 检测中目标，几乎不需要额外上下文
# P5 (stride=32): 检测大目标，标准卷积即可

ARF_CONFIGS = {
    "P2": (1, 3, 5),
    "P3": (1, 2, 3),
    "P4": (1, 1, 2),
    "P5": (1, 1, 1),
}


class ARF(nn.Module):
    """
    Adaptive Receptive Field Module.

    通过多分支空洞卷积 + 可学习 softmax 融合，为每个检测层
    提供自适应感受野。插入位置：检测层 C3k2 之后、Detect 之前。

    Args:
        channels: 输入/输出通道数（保持不变）
        dilations: 空洞率元组，默认 (1, 3, 5)
    """

    def __init__(self, channels: int, dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.dilations = dilations
        self.num_branches = len(dilations)

        # 多分支空洞深度可分离卷积
        self.branches = nn.ModuleList([
            nn.Sequential(
                # Depthwise: 空间维度卷积（空洞率可变）
                nn.Conv2d(
                    channels, channels, kernel_size=3,
                    padding=d, dilation=d, groups=channels, bias=False
                ),
                # Pointwise: 通道维度融合
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
            )
            for d in dilations
        ])

        # 可学习的分支权重网络
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, self.num_branches, kernel_size=1, bias=False),
            nn.Softmax(dim=1),
        )

        # 输出投影（残差前做一次精调）
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 特征图

        Returns:
            (B, C, H, W) 增强后的特征图（同 shape）
        """
        # 各分支并行计算
        branch_outs = [branch(x) for branch in self.branches]

        # 动态分支权重
        weights = self.weight_net(x)  # (B, num_branches, 1, 1)
        weights = weights.split(1, dim=1)  # list of (B, 1, 1, 1)

        # 加权融合
        fused = sum(w * out for w, out in zip(weights, branch_outs))

        # 残差连接
        return self.proj(fused) + x

    def __repr__(self):
        return f"ARF(ch={self.branches[0][0].in_channels}, dilations={self.dilations})"
