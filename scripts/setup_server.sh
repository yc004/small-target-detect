#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# TT100K Small Target Detection — Linux Server One-Click Setup & Train
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash scripts/setup_server.sh                  # train from scratch
#   bash scripts/setup_server.sh --eval-only      # only evaluate existing model
#   bash scripts/setup_server.sh --resume         # resume interrupted training
#
# Prerequisites:
#   - Linux with NVIDIA GPU + CUDA 11.8+
#   - Python 3.10+ available
#   - Dataset at /public/data/image/TT100K
#   - Git repo already cloned
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Config (edit these if needed) ───────────────────────────────────────────

DATASET_PATH="${TT100K_PATH:-/public/data/image/TT100K}"
OUTPUT_DIR="data/processed"
EXPERIMENT_DIR="experiments/stage1_baseline"
CONFIG_FILE="configs/baseline.yaml"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-1280}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"          # CUDA device index

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Parse args ──────────────────────────────────────────────────────────────

EVAL_ONLY=false; RESUME=false
for arg in "$@"; do
    case "$arg" in
        --eval-only) EVAL_ONLY=true ;;
        --resume)    RESUME=true ;;
        *)           err "Unknown argument: $arg" ;;
    esac
done

# ─── Step 1: Check environment ───────────────────────────────────────────────

log "═══ Step 1/6: Checking environment ═══"

# Python
PYTHON=$(command -v python3 || command -v python) || err "Python3 not found"
PY_VER=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
log "  Python: $PY_VER ($PYTHON)"

# CUDA
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -3
    log "  CUDA: $(nvidia-smi | grep 'CUDA Version' | grep -oP '\d+\.\d+')"
else
    warn "  nvidia-smi not found — will use CPU if no MPS available"
    DEVICE="cpu"
fi

# Dataset
if [ -d "$DATASET_PATH" ]; then
    log "  Dataset: $DATASET_PATH ($(find "$DATASET_PATH" -type f | wc -l) files)"
else
    err "  Dataset not found at $DATASET_PATH"
fi

# ─── Step 2: Check if the dataset is a pre-made YOLO ZIP ─────────────────────

log "═══ Step 2/6: Detecting dataset format ═══"

# Find the main annotation source
HAS_ZIP=false; HAS_JSON=false; HAS_YOLO_DIR=false

# Check for YOLO ZIP (like Datasets-TT100K/TT100K-YOLO格式.zip)
YOLO_ZIP=$(find "$DATASET_PATH" -maxdepth 2 -name "*YOLO*.zip" -o -name "*yolo*.zip" 2>/dev/null | head -1 || true)
if [ -n "$YOLO_ZIP" ] && [ -f "$YOLO_ZIP" ]; then
    log "  Found YOLO-format ZIP: $YOLO_ZIP"
    HAS_ZIP=true
fi

# Check for annotations.json (raw TT100K format)
if [ -f "$DATASET_PATH/annotations.json" ]; then
    log "  Found raw annotations.json"
    HAS_JSON=true
fi

# Check for already-processed YOLO directory (images/ + labels/)
if [ -d "$DATASET_PATH/images" ] && [ -d "$DATASET_PATH/labels" ]; then
    log "  Found pre-existing YOLO directory structure"
    HAS_YOLO_DIR=true
fi

# ─── Step 3: Create venv & install dependencies ──────────────────────────────

log "═══ Step 3/6: Setting up Python environment ═══"

if [ -d ".venv" ]; then
    log "  Using existing .venv"
else
    log "  Creating virtual environment..."
    $PYTHON -m venv .venv
fi
source .venv/bin/activate

log "  Upgrading pip..."
pip install --upgrade pip -q

log "  Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q 2>&1 | tail -1

log "  Installing project dependencies..."
pip install ultralytics opencv-python-headless matplotlib seaborn pyyaml tqdm pillow pycocotools albumentations -q 2>&1 | tail -1

log "  ✓ Environment ready"

# ─── Step 4: Prepare dataset ────────────────────────────────────────────────

log "═══ Step 4/6: Preparing dataset ═══"

if [ -f "$OUTPUT_DIR/dataset.yaml" ]; then
    log "  Dataset already prepared at $OUTPUT_DIR"
else
    if $HAS_ZIP; then
        log "  Extracting from YOLO ZIP with class filter..."
        # Check if our class-filtering extract script exists
        if [ -f "data/extract_dataset.py" ]; then
            $PYTHON data/extract_dataset.py --zip_path "$YOLO_ZIP" --output_dir "$OUTPUT_DIR"
        else
            # Fallback: unzip everything then filter with prepare_dataset.py
            log "  extract_dataset.py not found, falling back to generic prepare..."
            $PYTHON data/prepare_dataset.py --data_dir "$DATASET_PATH" --output_dir "$OUTPUT_DIR"
        fi

    elif $HAS_JSON; then
        log "  Converting from raw TT100K annotations.json → YOLO format..."
        $PYTHON data/prepare_dataset.py \
            --data_dir "$DATASET_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --ann_file "$DATASET_PATH/annotations.json"

    elif $HAS_YOLO_DIR; then
        log "  Dataset already in YOLO format, creating dataset.yaml..."
        # Create symlinks and generate dataset.yaml directly
        mkdir -p "$OUTPUT_DIR"
        cp -r "$DATASET_PATH/images" "$OUTPUT_DIR/images" 2>/dev/null || true
        cp -r "$DATASET_PATH/labels" "$OUTPUT_DIR/labels" 2>/dev/null || true

        # Write dataset.yaml
        $PYTHON -c "
import yaml, pathlib
out = pathlib.Path('$OUTPUT_DIR')
config = {
    'path': str(out.absolute()),
    'train': str(out / 'images' / 'train'),
    'val': str(out / 'images' / 'val'),
    'test': str(out / 'images' / 'test') if (out / 'images' / 'test').exists() else str(out / 'images' / 'val'),
    'nc': 5,
    'names': ['speed_limit_5','speed_limit_30','speed_limit_40','no_entry','no_pedestrians'],
}
with open(out / 'dataset.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_flow=False, sort_keys=False)
print('  dataset.yaml generated')
"
    else
        err "Cannot determine dataset format. Expected one of:\n  - YOLO .zip file\n  - annotations.json\n  - images/ + labels/ directories"
    fi

    log "  ✓ Dataset prepared"
fi

# ─── Step 5: Quick data stats ────────────────────────────────────────────────

log "═══ Step 5/6: Dataset statistics ═══"

$PYTHON -c "
from pathlib import Path
out = Path('$OUTPUT_DIR')
for split in ['train', 'val', 'test']:
    img_dir = out / 'images' / split
    lbl_dir = out / 'labels' / split
    if img_dir.exists() and lbl_dir.exists():
        n_img = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        n_lbl = len(list(lbl_dir.glob('*.txt')))
        print(f'  {split}: {n_img} images, {n_lbl} labels')
print(f'  Config: {out / \"dataset.yaml\"}')
" 2>/dev/null || warn "  (stats skipped)"

# ─── Step 6: Train / Evaluate ────────────────────────────────────────────────

log "═══ Step 6/6: Training ═══"

mkdir -p "$EXPERIMENT_DIR"

if $EVAL_ONLY; then
    WEIGHTS=$(find "$EXPERIMENT_DIR" -name "best.pt" 2>/dev/null | head -1)
    if [ -z "$WEIGHTS" ]; then
        err "No trained weights found for evaluation."
    fi
    log "  Evaluating: $WEIGHTS"
    $PYTHON scripts/eval.py \
        --weights "$WEIGHTS" \
        --data "$OUTPUT_DIR/dataset.yaml" \
        --split test \
        --analyze_sizes \
        --device "$DEVICE"
else
    if $RESUME; then
        LAST_PT=$(find "$EXPERIMENT_DIR" -name "last.pt" 2>/dev/null | head -1)
        if [ -z "$LAST_PT" ]; then
            warn "  No last.pt found, starting fresh"
            RESUME=false
        else
            log "  Resuming from $LAST_PT"
        fi
    fi

    log "  Model: yolov8s.yaml (baseline)"
    log "  Epochs: $EPOCHS | Batch: $BATCH_SIZE | ImgSz: $IMG_SIZE | Device: $DEVICE"

    # Build training command
    TRAIN_CMD="$PYTHON scripts/train.py \
        --config $CONFIG_FILE \
        --epochs $EPOCHS \
        --batch $BATCH_SIZE \
        --imgsz $IMG_SIZE \
        --device $DEVICE \
        --project $EXPERIMENT_DIR \
        --name train"

    if $RESUME; then
        TRAIN_CMD="$TRAIN_CMD --model $LAST_PT"
    fi

    log "  Running: $TRAIN_CMD"
    echo ""

    $TRAIN_CMD 2>&1 | tee "$EXPERIMENT_DIR/train.log"

    # Print final model path
    BEST_PT="$EXPERIMENT_DIR/train/weights/best.pt"
    if [ -f "$BEST_PT" ]; then
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "  ✓ Training complete!"
        log "  Best model: $BEST_PT"
        log "  Evaluate:   python scripts/eval.py --weights $BEST_PT --data $OUTPUT_DIR/dataset.yaml"
        log "  Detect:     python scripts/detect.py --weights $BEST_PT --source <image_or_video>"
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    else
        warn "  Training finished but best.pt not found — check $EXPERIMENT_DIR/train.log"
    fi
fi
