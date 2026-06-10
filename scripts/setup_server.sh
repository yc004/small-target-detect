#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# TT100K Small Target Detection — Server One-Click Training
# ═══════════════════════════════════════════════════════════════════════════════
# Prerequisites: conda env with PyTorch + CUDA + ultralytics already active.
# Dataset: /public/data/image/TT100K
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────

DATASET_PATH="${TT100K_PATH:-/public/data/image/TT100K}"
OUTPUT_DIR="data/processed"
EXPERIMENT_DIR="experiments/stage1_baseline"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-1280}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

EVAL_ONLY=false; RESUME=false
for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=true ;;
        --resume)    RESUME=true ;;
        *)           err "Unknown argument: $arg" ;;
    esac
done

# ─── Step 1: Check environment ───────────────────────────────────────────────

log "═══ Step 1/4: Checking environment ═══"

PYTHON=$(command -v python || command -v python3) || err "Python not found"
log "  Python: $($PYTHON --version 2>&1)"

$PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}')" || err "PyTorch not available"
$PYTHON -c "import ultralytics; print(f'  ultralytics: {ultralytics.__version__}')" || err "ultralytics not installed"

[ -d "$DATASET_PATH" ] || err "Dataset not found at $DATASET_PATH"
log "  Dataset: $DATASET_PATH ($(find "$DATASET_PATH" -type f 2>/dev/null | wc -l) files)"

# ─── Step 2: Prepare dataset ─────────────────────────────────────────────────

log "═══ Step 2/4: Preparing dataset ═══"

if [ -f "$OUTPUT_DIR/dataset.yaml" ]; then
    log "  Already prepared, skipping."
else
    # Detect format and prepare accordingly
    YOLO_ZIP=$(find "$DATASET_PATH" -maxdepth 2 \( -name "*YOLO*.zip" -o -name "*yolo*.zip" \) 2>/dev/null | head -1 || true)

    if [ -n "$YOLO_ZIP" ] && [ -f "$YOLO_ZIP" ]; then
        log "  Found YOLO ZIP, extracting & filtering 5 target classes..."
        $PYTHON data/extract_dataset.py --zip_path "$YOLO_ZIP" --output_dir "$OUTPUT_DIR"

    elif [ -f "$DATASET_PATH/annotations.json" ]; then
        log "  Found annotations.json, converting to YOLO format..."
        $PYTHON data/prepare_dataset.py --data_dir "$DATASET_PATH" --output_dir "$OUTPUT_DIR"

    elif [ -d "$DATASET_PATH/images" ] && [ -d "$DATASET_PATH/labels" ]; then
        log "  YOLO-format directory detected, linking..."
        mkdir -p "$OUTPUT_DIR/images" "$OUTPUT_DIR/labels"
        cp -rn "$DATASET_PATH/images/"* "$OUTPUT_DIR/images/" 2>/dev/null || true
        cp -rn "$DATASET_PATH/labels/"* "$OUTPUT_DIR/labels/" 2>/dev/null || true
        $PYTHON -c "
import yaml, pathlib
out = pathlib.Path('$OUTPUT_DIR')
test = str(out / 'images' / 'test') if (out / 'images' / 'test').exists() else str(out / 'images' / 'val')
config = {'path': str(out.absolute()), 'train': str(out / 'images' / 'train'),
          'val': str(out / 'images' / 'val'), 'test': test,
          'nc': 5, 'names': ['speed_limit_5','speed_limit_30','speed_limit_40','no_entry','no_pedestrians']}
with open(out / 'dataset.yaml', 'w') as f: yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('  dataset.yaml generated')
"
    else
        err "Cannot determine dataset format. Expected: YOLO .zip, annotations.json, or images/ + labels/ directories"
    fi
    log "  ✓ Done"
fi

# ─── Step 3: Quick stats ─────────────────────────────────────────────────────

log "═══ Step 3/4: Dataset statistics ═══"

$PYTHON -c "
from pathlib import Path
out = Path('$OUTPUT_DIR')
for split in ['train', 'val', 'test']:
    img_dir = out / 'images' / split
    lbl_dir = out / 'labels' / split
    if img_dir.exists() and lbl_dir.exists():
        n_img = len([f for f in img_dir.glob('*') if f.suffix in ('.jpg','.jpeg','.png')])
        n_lbl = len(list(lbl_dir.glob('*.txt')))
        print(f'  {split}: {n_img} images, {n_lbl} labels')
" 2>/dev/null || true

# ─── Step 4: Train / Evaluate ────────────────────────────────────────────────

log "═══ Step 4/4: Training ═══"
mkdir -p "$EXPERIMENT_DIR"

if $EVAL_ONLY; then
    BEST=$(find "$EXPERIMENT_DIR" -name "best.pt" 2>/dev/null | head -1)
    [ -z "$BEST" ] && err "No best.pt found for evaluation"
    log "  Evaluating: $BEST"
    $PYTHON scripts/eval.py --weights "$BEST" --data "$OUTPUT_DIR/dataset.yaml" --split test --analyze_sizes --device "$DEVICE"
    exit 0
fi

if $RESUME; then
    LAST=$(find "$EXPERIMENT_DIR" -name "last.pt" 2>/dev/null | head -1)
    if [ -n "$LAST" ]; then
        log "  Resuming from $LAST"
        MODEL_ARG="--model $LAST"
    else
        warn "  No last.pt found, starting fresh"
        MODEL_ARG=""
    fi
else
    MODEL_ARG=""
fi

log "  Config: configs/baseline.yaml"
log "  Model: yolov8s.yaml | Epochs: $EPOCHS | Batch: $BATCH_SIZE | ImgSz: $IMG_SIZE | Device: $DEVICE"
echo ""

$PYTHON scripts/train.py \
    --config configs/baseline.yaml \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE" \
    --project "$EXPERIMENT_DIR" \
    --name train \
    $MODEL_ARG \
    2>&1 | tee "$EXPERIMENT_DIR/train.log"

BEST_PT="$EXPERIMENT_DIR/train/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  ✓ Training complete!  Best model: $BEST_PT"
    log "  Evaluate: python scripts/eval.py --weights $BEST_PT --data $OUTPUT_DIR/dataset.yaml"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    warn "  best.pt not found — check $EXPERIMENT_DIR/train.log"
fi
