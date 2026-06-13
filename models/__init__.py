# ────────────────────────────────────────────────────────────────────────────
# Model Definitions — YOLO26-based Small Target Detection
# ────────────────────────────────────────────────────────────────────────────
# Custom modules for improving small target detection on TT100K.
#
# Modules:
#   arf_head.py  — Adaptive Receptive Field Detection Head (Stage 3)
#   bcem.py      — Bi-directional Context Enhancement Module (Stage 4)
#   hj_loss.py   — Hierarchical Joint Loss: WIoUv3 + NWD (Stage 5)
#   model_builder.py — Model construction utilities
# ────────────────────────────────────────────────────────────────────────────

from models.arf_head import ARF, ARF_CONFIGS
from models.bcem import BCEM
from models.hj_loss import WIoUv3Loss, NWDLoss, HierarchicalJointLoss
from models.model_builder import create_model, register_custom_modules

__all__ = [
    "ARF", "ARF_CONFIGS",
    "BCEM",
    "WIoUv3Loss", "NWDLoss", "HierarchicalJointLoss",
    "create_model", "register_custom_modules",
]
