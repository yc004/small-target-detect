# ────────────────────────────────────────────────────────────────────────────
# Model Builder — YOLO26 Custom Model Construction
# ────────────────────────────────────────────────────────────────────────────
# 工具函数：
#   1. 将自定义模块 (ARF, BCEM) 注册到 ultralytics 的模块命名空间
#   2. 加载 YAML 配置并构建 DetectionModel
#   3. 支持代码方式注入模块（当 YAML 解析有局限时的备选方案）
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

from ultralytics.nn.tasks import DetectionModel

from models.arf_head import ARF, ARF_CONFIGS
from models.bcem import BCEM


# ──────────────────────────────────────────────────────────────────────────
# 注册自定义模块到 ultralytics
# ──────────────────────────────────────────────────────────────────────────

def register_custom_modules() -> None:
    """
    将自定义模块 (ARF, BCEM) 注册到 ultralytics 的模块解析命名空间。

    ultralytics 在 parse_model() 中通过 ``globals()[m]`` (即
    ``ultralytics.nn.tasks.__dict__``) 来解析 YAML 中的非内置模块名。
    因此需要将 ARF 和 BCEM 注入到 tasks 模块的全局命名空间中。

    同时也注册到 ultralytics.nn.modules（兼容其他可能的 lookup 路径）。

    用法：
        from models.model_builder import register_custom_modules
        register_custom_modules()
        model = YOLO("models/yolo26s_p2_arf.yaml")  # ARF 可被解析
    """
    # 主注册路径：ultralytics.nn.tasks（parse_model 使用 globals()[m] 解析）
    try:
        import ultralytics.nn.tasks as ult_tasks
        ult_tasks.ARF = ARF
        ult_tasks.BCEM = BCEM
    except ImportError:
        pass

    # 辅助注册路径：ultralytics.nn.modules（其他可能的 lookup）
    try:
        import ultralytics.nn.modules as ult_nn
        ult_nn.ARF = ARF
        ult_nn.BCEM = BCEM
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────
# Model Creation
# ──────────────────────────────────────────────────────────────────────────

def create_model(
    cfg_path: str | Path,
    nc: int = 5,
    pretrained: bool = False,
) -> DetectionModel:
    """
    加载自定义 YAML 配置并创建 YOLO26 DetectionModel。

    自动完成：
      1. 注册自定义模块（ARF, BCEM）
      2. 读取 YAML 配置
      3. 覆写 nc (类别数)
      4. 构建 DetectionModel

    Args:
        cfg_path: yaml 配置文件路径
        nc: 类别数量（TT100K 子集 = 5）
        pretrained: 是否加载预训练权重（通常用 ultralytics 的 YOLO().train()
                    来加载，此参数保留用于未来扩展）

    Returns:
        DetectionModel 实例

    Example:
        >>> from models.model_builder import create_model
        >>> model = create_model("models/yolo26s_p2_arf.yaml", nc=5)
    """
    cfg_path = Path(cfg_path)

    # 确保自定义模块已注册
    register_custom_modules()

    # 加载 yaml 配置
    with open(cfg_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # 覆写类别数
    cfg["nc"] = nc

    # 构建模型
    model = DetectionModel(cfg, ch=3, nc=nc)

    return model


# ──────────────────────────────────────────────────────────────────────────
# 编程式模块注入（备选方案）
# ──────────────────────────────────────────────────────────────────────────

def inject_arf_heads(model: DetectionModel) -> DetectionModel:
    """
    在已构建的 DetectionModel 中通过代码方式注入 ARF 模块。

    当 YAML 解析路径遇到问题时，可用此函数作为备选方案。
    遍历模型的 head 部分，在每个 Detect 层引用的特征层之后插入 ARF。

    Args:
        model: 已构建的 DetectionModel 实例

    Returns:
        修改后的 DetectionModel（原地修改）
    """
    model_list = model.model  # nn.Sequential
    detect_layer = None

    # 找到 Detect 层
    for i, layer in enumerate(model_list):
        if hasattr(layer, "stride"):  # Detect 层有 stride 属性
            detect_layer = layer
            detect_idx = i
            break

    if detect_layer is None:
        raise ValueError("Cannot find Detect layer in model")

    # Detect 层引用的特征层索引（例如 [16, 19, 22] 或 [19, 22, 25, 28]）
    # 在 ultralytics 源码中这存储在 detect_layer 的 cv2/cv3 之前
    # 对于 YOLO26-P2：4 个检测头，对应 P2/P3/P4/P5
    num_heads = 3 if "p2" not in str(model.yaml.get("model", "")) else 4

    # ARF 配置分配
    layer_keys = ["P2", "P3", "P4", "P5"] if num_heads == 4 else ["P3", "P4", "P5"]

    # 找到 C3k2 特征层（Detect 层引用的前几层）
    # 这部分需要根据实际模型结构调整，这里提供框架
    return model


def inject_bcem_modules(model: DetectionModel) -> DetectionModel:
    """
    在已构建的 DetectionModel 中通过代码方式注入 BCEM 模块。

    在 Backbone → Neck 的跳跃连接处插入 BCEM。
    遍历 head 部分的 Concat 层，在引用 Backbone 层索引的位置前插入 BCEM。

    Args:
        model: 已构建的 DetectionModel 实例

    Returns:
        修改后的 DetectionModel（原地修改）
    """
    # 参见 IMPLEMENTATION_PLAN.md Stage 4 的详细设计
    # 主要思路：找到 head 中的 Concat 层，对引用的 backbone 层插入 BCEM
    return model
