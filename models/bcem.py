# ────────────────────────────────────────────────────────────────────────────
# BCEM: Bi-directional Context Enhancement Module
# ────────────────────────────────────────────────────────────────────────────
# Stage 4 原创模块 — 在 Backbone → Neck 跳跃连接处进行通道+空间双向上下文增强。
#
# 核心思想：
#   PANet 跳跃连接直接将 Backbone 浅层特征拼接到 Neck，但这些浅层特征
#   包含大量与前景目标无关的背景纹理（道路标志线、文字、建筑边缘等）。
#   小目标极易与这些背景元素混淆。
#
#   BCEM 在跳跃连接前对 Backbone 特征进行"提纯"：
#   - 通道分支：通过 1D 卷积捕获局部跨通道交互（保留重要通道）
#   - 空间分支：通过 7×7 大核深度卷积进行前景/背景判别
#   - 乘法融合：双维度双重确认才增强（更保守，更精准）
#   - 残差连接：Output = Input + Input × ChAttn × SpAttn
#
# 插入位置：Backbone 各层输出 → Neck 中对应 Concat 之前
# ────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn


class BCEM(nn.Module):
    """
    Bi-directional Context Enhancement Module.

    通道注意力 + 空间注意力的双向增强，乘法融合 + 残差连接。
    初始状态输出接近恒等映射，训练过程中逐渐学会增强前景区域。

    Args:
        channels: 输入通道数
        k_size: 1D 卷积核大小，None 则根据通道数自适应计算
        gamma, b: 自适应核大小的参数（默认 CVPR 2020 ECA 推荐值）
    """

    def __init__(
        self,
        channels: int,
        k_size: int | None = None,
        gamma: int = 2,
        b: int = 1,
    ):
        super().__init__()

        # ── 通道分支 ──────────────────────────────────────────────
        # 自适应 1D 卷积核大小：k = |log2(C)/γ + b/γ|_odd
        if k_size is None:
            t = int(abs((math.log2(channels) / gamma) + (b / gamma)))
            k_size = t if t % 2 == 1 else t + 1
            k_size = max(3, k_size)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) → (B, C, 1, 1)
        )
        # 1D 卷积沿通道维度扫描（需要 reshape）
        self.ch_conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=k_size // 2, bias=False
        )
        self.ch_sigmoid = nn.Sigmoid()
        self.k_size = k_size

        # ── 空间分支 ──────────────────────────────────────────────
        # 7×7 大核 DWConv 提供足够感受野进行前景/背景判别
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.Conv2d(
                channels // 2, channels // 2,
                kernel_size=7, padding=3, groups=channels // 2, bias=False,
            ),
            nn.BatchNorm2d(channels // 2),
            nn.Conv2d(channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 初始化：使 BCEM 初始状态接近恒等映射
        self._init_weights()

    def _init_weights(self):
        """初始化空间分支末层权重为 0，使初始输出 = 恒等映射。"""
        # 空间分支最后一层 Conv 的 weight 初始化为 0
        last_conv = self.spatial_attn[-2]  # 1×1 conv before sigmoid
        nn.init.constant_(last_conv.weight, 0.0)
        if last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) Backbone 特征图

        Returns:
            (B, C, H, W) 增强后的特征图（同 shape）
        """
        b, c, h, w = x.shape

        # ── 通道注意力 ──
        ca = self.channel_attn(x)              # (B, C, 1, 1)
        # reshape 为 (B, 1, C) 做 1D 卷积
        ca = ca.squeeze(-1).transpose(-1, -2)   # (B, 1, C)
        ca = self.ch_conv(ca)                   # (B, 1, C)
        ca = ca.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        ca = self.ch_sigmoid(ca)

        # ── 空间注意力 ──
        sa = self.spatial_attn(x)               # (B, 1, H, W)

        # ── 乘法融合 + 残差 ──
        # ca 和 sa 广播相乘得到 (B, C, H, W)
        enhanced = x * ca * sa
        return x + enhanced

    def __repr__(self):
        return f"BCEM(ch={self.ch_conv.in_channels}, k_size={self.k_size})"
