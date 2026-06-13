# ────────────────────────────────────────────────────────────────────────────
# HJ-Loss: Hierarchical Joint Loss Function
# ────────────────────────────────────────────────────────────────────────────
# Stage 5 原创模块 — 分层联合损失函数，为不同检测层提供差异化的损失策略。
#
# 核心思想：
#   小目标对 IoU 偏移极度敏感 → P2/P3 层使用 WIoUv3 + NWD 联合监督
#   大目标 IoU 本身已稳定 → P4/P5 层仅使用 WIoUv3
#   避免 NWD 对大目标引入不必要的噪声。
#
# 参考文献：
#   WIoU v3: Tong et al., "Wise-IoU: Bounding Box Regression Loss with
#            Dynamic Focusing Mechanism", 2023
#   NWD:     Wang et al., "A Normalized Gaussian Wasserstein Distance for
#            Tiny Object Detection", 2021
# ────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class WIoUv3Loss(nn.Module):
    """
    Wise-IoU v3 — 动态非单调聚焦机制。

    相比 CIoU，WIoUv3 通过离群度 β 动态调整每个样本的梯度增益：
    - β 小（高质量 anchor）→ 正常梯度
    - β 中等（需关注的 anchor）→ 放大梯度
    - β 大（极端离群）→ 抑制梯度

    使用 running mean 跟踪 L_IoU 均值以计算离群度。
    """

    def __init__(self, momentum: float = 0.99, delta: float = 2.0):
        """
        Args:
            momentum: running mean 动量
            delta: 聚焦强度参数（越大，中等质量样本的增益越大）
        """
        super().__init__()
        self.momentum = momentum
        self.delta = delta
        self.register_buffer("running_mean", torch.tensor(1.0))

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] — 归一化坐标
            gt_boxes:   (N, 4) [cx, cy, w, h] — 归一化坐标

        Returns:
            loss: scalar tensor
        """
        # ── 转换为 xyxy 格式并计算 IoU ──
        px1, py1, px2, py2 = self._xywh_to_xyxy(pred_boxes).chunk(4, dim=-1)
        gx1, gy1, gx2, gy2 = self._xywh_to_xyxy(gt_boxes).chunk(4, dim=-1)

        # 修复维度：确保每个是 (N,) 或 (N,1) 的标量
        px1, py1, px2, py2 = px1.squeeze(-1), py1.squeeze(-1), px2.squeeze(-1), py2.squeeze(-1)
        gx1, gy1, gx2, gy2 = gx1.squeeze(-1), gy1.squeeze(-1), gx2.squeeze(-1), gy2.squeeze(-1)

        inter_w = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
        inter_h = (torch.min(py2, gy2) - torch.max(px1, gy1)).clamp(min=0)
        inter = inter_w * inter_h

        area_p = (px2 - px1) * (py2 - py1)
        area_g = (gx2 - gx1) * (gy2 - gy1)
        union = area_p + area_g - inter + 1e-7
        iou = inter / union

        # ── R_WIoU: 中心点距离惩罚项 ──
        px_cx, px_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        gx_cx, gy_cx = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
        w_gt, h_gt = gx2 - gx1, gy2 - gy1

        r_wious = torch.exp(
            ((px_cx - gx_cx) ** 2 + (px_cy - gy_cx) ** 2)
            / ((w_gt ** 2 + h_gt ** 2) + 1e-7)
        )

        # ── 基础 IoU 损失 ──
        L_iou = 1.0 - iou

        # ── 动态聚焦因子 r = β / (δ * α^(β-δ)) ──
        with torch.no_grad():
            beta = L_iou.detach() / (self.running_mean + 1e-7)

        r = beta / (self.delta * torch.pow(torch.ones_like(beta), beta - self.delta))

        # 更新 running mean（仅训练时）
        if self.training:
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * L_iou.detach().mean()
            )

        loss = (r.detach() * r_wious * L_iou).mean()
        return loss

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """[cx, cy, w, h] → [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes.chunk(4, dim=-1)
        return torch.cat([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dim=-1)


class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss。

    将边界框建模为 2D 高斯分布 N(μ, Σ)，其中：
      μ = (cx, cy)
      Σ = diag(w²/4, h²/4)

    计算两个高斯分布之间的 Wasserstein 距离并归一化。

    特点：对小目标的微小位置偏移不敏感，提供更平滑的定位监督信号。

    Args:
        c: 归一化常数（与数据集平均目标尺寸相关）
           对于 TT100K，建议 c = 12.0
    """

    def __init__(self, c: float = 12.0):
        super().__init__()
        self.c = c

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] — 归一化坐标
            gt_boxes:   (N, 4) [cx, cy, w, h] — 归一化坐标

        Returns:
            loss: scalar
        """
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes.chunk(4, dim=-1)
        gt_cx, gt_cy, gt_w, gt_h = gt_boxes.chunk(4, dim=-1)

        # Wasserstein 距离平方
        # W_2² = ||μ_p - μ_g||² + ||Σ_p^(1/2) - Σ_g^(1/2)||_F²
        mu_dist_sq = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2
        sigma_dist_sq = ((pred_w - gt_w) / 2.0) ** 2 + ((pred_h - gt_h) / 2.0) ** 2
        w2_sq = mu_dist_sq + sigma_dist_sq

        # 归一化 + 指数映射到 [0, 1]
        nwd = torch.exp(-torch.sqrt(w2_sq + 1e-7).squeeze(-1) / self.c)

        loss = (1.0 - nwd).mean()
        return loss


class HierarchicalJointLoss(nn.Module):
    """
    分层联合损失 (HJ-Loss)。

    为不同检测层提供差异化的损失函数组合：
      P2, P3 (小目标层): L = WIoUv3 + λ_nwd × NWD
      P4, P5 (大目标层): L = WIoUv3

    关键设计决策：NWD 仅用于小目标层，避免对大目标引入噪声。

    Args:
        nwd_weight: NWD 损失的权重系数
        nwd_layers: 使用 NWD 的检测层名称列表
        c_nwd: NWD 归一化常数
    """

    def __init__(
        self,
        nwd_weight: float = 0.3,
        nwd_layers: list | None = None,
        c_nwd: float = 12.0,
    ):
        super().__init__()
        self.wiouv3 = WIoUv3Loss()
        self.nwd = NWDLoss(c=c_nwd)
        self.nwd_weight = nwd_weight
        self.nwd_layers = nwd_layers or ["P2", "P3"]

    def forward(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        layer_name: str = "P3",
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] 归一化
            gt_boxes:   (N, 4) [cx, cy, w, h] 归一化
            layer_name: "P2" | "P3" | "P4" | "P5"

        Returns:
            联合损失 (scalar)
        """
        loss_wiou = self.wiouv3(pred_boxes, gt_boxes)

        if layer_name in self.nwd_layers:
            loss_nwd = self.nwd(pred_boxes, gt_boxes)
            return loss_wiou + self.nwd_weight * loss_nwd

        return loss_wiou

    def __repr__(self):
        return (
            f"HJ-Loss(nwd_weight={self.nwd_weight}, "
            f"nwd_layers={self.nwd_layers})"
        )
