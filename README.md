# Small Target Traffic Sign Detection

基于改进 YOLO26 的 TT100K 小目标交通标志实时检测。

## 项目概述

| 项目 | 说明 |
|------|------|
| 任务 | 小目标交通标志检测与识别 |
| 数据集 | TT100K (2021) — 5 类交通标志 |
| 基线模型 | YOLO26s (ultralytics 2025) |
| 改进点 | P2 检测层 → ARF-Head → BCEM → HJ-Loss → Soft-NMS |
| 原创模块 | ARF-Head / BCEM / HJ-Loss（三项原创设计） |
| 框架 | PyTorch + Ultralytics |

### 5 类目标

| 类名 | TT100K 标识 | 含义 | 标注数 |
|------|------------|------|--------|
| speed_limit_5 | i2 | 限速 5 km/h | ~217 |
| speed_limit_30 | i4 | 限速 30 km/h | ~362 |
| speed_limit_40 | i5 | 限速 40 km/h | ~794 |
| no_entry | pne | 禁止驶入 | ~1120 |
| no_pedestrians | p10 | 禁止行人通行 | ~193 |

## 快速开始

### Linux 服务器（一键）

```bash
git clone https://github.com/yc004/small-target-detect.git
cd small-target-detect
./scripts/setup_server.sh
```

### Windows（一键）

```cmd
git clone https://github.com/yc004/small-target-detect.git
cd small-target-detect
scripts\setup_server.bat
```

### Mac / 手动训练

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据集
python data/prepare_dataset.py --data_dir /path/to/TT100K --output_dir data/processed

# 3. 训练基线
python scripts/train.py --config configs/stage1_baseline.yaml --device mps

# 4. 评估
python scripts/eval.py --weights experiments/stage1_baseline/train/weights/best.pt --data data/processed/dataset.yaml

# 5. 推理
python scripts/detect.py --weights experiments/stage1_baseline/train/weights/best.pt --source test.jpg
```

## 项目结构

```
small_target_detect/
├── data/
│   ├── prepare_dataset.py       # 原始 TT100K → YOLO 格式转换
│   └── extract_dataset.py       # 从 YOLO ZIP 筛选 5 类（本地 Mac 用）
├── models/
│   └── __init__.py
├── utils/
│   ├── metrics.py               # COCO 小/中/大目标 AP 评估
│   └── visualization.py         # 检测框绘制、PR 曲线
├── configs/
│   └── baseline.yaml            # 基线训练超参数
├── scripts/
│   ├── train.py                 # 训练入口
│   ├── eval.py                  # 评估（含尺寸分解）
│   ├── detect.py                # 推理（图片/视频/摄像头）
│   ├── analyze_data.py          # 数据集统计分析
│   ├── setup_server.sh          # Linux 一键脚本
│   └── setup_server.bat         # Windows 一键脚本
├── experiments/                 # 训练产物（不跟踪）
└── IMPLEMENTATION_PLAN.md       # 6 阶段渐进式实现方案
```

## 渐进式改进路线（6 阶段）

| Stage | 改进 | 创新程度 | 预期提升 | 状态 |
|-------|------|:---:|----------|------|
| 1 | 基线 YOLO26s | — | 建立性能基线 | 🔄 |
| 2 | + P2 高分辨率检测层 | 官方 | AP_S +5% | ⬜ |
| 3 | + **ARF-Head** 自适应感受野检测头 | ⭐⭐⭐ 原创 | AP_S +2% | ⬜ |
| 4 | + **BCEM** 双向上下文增强模块 | ⭐⭐⭐⭐ 原创 | AP_S +2%, mAP +1.5% | ⬜ |
| 5 | + **HJ-Loss** 分层联合损失函数 | ⭐⭐⭐⭐ 原创 | AP_S +3% | ⬜ |
| 6 | + Soft-NMS 后处理 + 系统集成 | 集成 | 密集场景召回 ↑ | ⬜ |

详见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)。

## 评估指标

- `mAP@0.5` / `mAP@0.5:0.95` — 整体精度
- `AP_S` — 小目标（面积 < 32² px）精度
- `AP_M` / `AP_L` — 中/大目标精度
- `FPS` — 推理速度

```bash
# 评估并输出各尺寸 AP
python scripts/eval.py \
    --weights experiments/stage1_baseline/train/weights/best.pt \
    --data data/processed/dataset.yaml \
    --analyze_sizes
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8+ / MPS / CPU)
- ultralytics >= 8.0.0
- opencv-python, albumentations, pycocotools

## License

MIT
