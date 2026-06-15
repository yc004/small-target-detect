#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Linux Server One-Click Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# Usage:
#   bash scripts/server_train.sh                           # train stage1 (default)
#   bash scripts/server_train.sh stage2_p2                 # train specified stage
#   bash scripts/server_train.sh stage3_arf_head --epochs 50  # custom epochs
#   bash scripts/server_train.sh --skip-data                # skip dataset prep
#   bash scripts/server_train.sh --skip-env                 # skip env check
#
# Env vars (optional):
#   CONDA_ENV=myenv   bash scripts/server_train.sh          # custom conda env
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."
SCRIPT_START=$(date +%s)

# ─── Config ──────────────────────────────────────────────────────────────────
STAGE="${1:-stage1_baseline}"             # training stage name
CONDA_ENV="${CONDA_ENV:-base}"            # conda environment name
DATASET_PATH="${TT100K_PATH:-/public/data/image/TT100K}"
OUTPUT_DIR="data/processed"
MNT_DIR="/mnt"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-640}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"
SKIP_DATA=false
SKIP_ENV=false

# Parse optional flags
for arg in "$@"; do
    case "$arg" in
        --skip-data) SKIP_DATA=true ;;
        --skip-env)  SKIP_ENV=true  ;;
        --epochs)    EPOCHS="$2"    ;;
        --batch)     BATCH_SIZE="$2" ;;
        --imgsz)     IMG_SIZE="$2"   ;;
        --device)    DEVICE="$2"     ;;
    esac
done

# Stage → config mapping
declare -A STAGE_CONFIG
STAGE_CONFIG[stage1_baseline]="configs/stage1_baseline.yaml"
STAGE_CONFIG[stage2_p2]="configs/stage2_p2.yaml"
STAGE_CONFIG[stage3_arf_head]="configs/stage3_arf_head.yaml"
STAGE_CONFIG[stage4_bcem]="configs/stage4_bcem.yaml"
STAGE_CONFIG[stage5_hjloss]="configs/stage5_hjloss.yaml"
STAGE_CONFIG[stage6_full]="configs/stage6_full.yaml"

CONFIG="${STAGE_CONFIG[$STAGE]:-}"
if [ -z "$CONFIG" ]; then
    echo "❌ Unknown stage: $STAGE"
    echo "   Valid stages: ${!STAGE_CONFIG[*]}"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "❌ Config file not found: $CONFIG"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="experiments/${STAGE}"

log()  { echo -e "[$(date '+%H:%M:%S')] $*"; }
step() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  📌 $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# ─── Sanity checks ───────────────────────────────────────────────────────────
log "🔍 Pre-flight checks ..."

if [ ! -f "$DATASET_PATH/tt100k_2021.zip" ]; then
    echo "❌ Dataset not found: $DATASET_PATH/tt100k_2021.zip"
    echo "   Expected: /public/data/image/TT100K/tt100k_2021.zip"
    echo "   Contents of $DATASET_PATH:"
    ls -lh "$DATASET_PATH" 2>/dev/null || echo "   (directory not accessible)"
    exit 1
fi

# Activate conda (use shell hook — the only reliable way in non-interactive scripts)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || {
        echo "⚠️  Could not activate conda env '$CONDA_ENV'. Trying base ..."
        conda activate base 2>/dev/null || true
    }
    log "  Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
else
    echo "⚠️  conda not found. Continuing with system Python."
fi

log "  Python:  $(python --version 2>&1)"
log "  CUDA:    $(python -c 'import torch; print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")' 2>&1 || echo 'N/A')"
log "  Stage:   $STAGE  →  $CONFIG"
log "  Device:  $DEVICE"
log "  Epochs:  $EPOCHS  |  Batch: $BATCH_SIZE  |  ImgSz: $IMG_SIZE"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Prepare Dataset
# ═══════════════════════════════════════════════════════════════════════════════
if $SKIP_DATA; then
    log "⏭  Step 1: Dataset preparation SKIPPED (--skip-data)"
else
    step "Step 1/4: Preparing dataset from $DATASET_PATH"

    if [ -f "$OUTPUT_DIR/dataset.yaml" ]; then
        log "  ✓ dataset.yaml already exists — skipping preparation."
        log "    To force re-prepare: rm -rf $OUTPUT_DIR"
    else
        # ── The dataset path contains two zips: tt100k_2016.zip and tt100k_2021.zip ──
        ZIP_2021="$DATASET_PATH/tt100k_2021.zip"

        if [ ! -f "$ZIP_2021" ]; then
            echo "❌ tt100k_2021.zip not found at $ZIP_2021"
            echo "   Contents of $DATASET_PATH:"
            ls -lh "$DATASET_PATH" 2>/dev/null || echo "   (directory not accessible)"
            exit 1
        fi

        log "  📦 Found tt100k_2021.zip ($(du -h "$ZIP_2021" | cut -f1))"

        # Extract to a temp directory under data/
        EXTRACT_DIR="data/TT100K_2021"
        if [ ! -f "$EXTRACT_DIR/.extracted" ]; then
            log "  ⏳ Extracting (this may take a few minutes) ..."
            rm -rf "$EXTRACT_DIR"
            mkdir -p "$EXTRACT_DIR"
            unzip -qo "$ZIP_2021" -d "$EXTRACT_DIR/"
            touch "$EXTRACT_DIR/.extracted"
            log "  ✓ Extracted to $EXTRACT_DIR"
        else
            log "  ✓ Already extracted at $EXTRACT_DIR"
        fi

        # Handle case where zip extracts into a single subdirectory
        SUBDIRS=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '.extracted' 2>/dev/null | wc -l)
        if [ "$SUBDIRS" -eq 1 ]; then
            EXTRACT_DIR=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)
            log "  📁 Dataset root: $EXTRACT_DIR"
        fi

        # Find annotation file
        ANN_FILE=$(find "$EXTRACT_DIR" -maxdepth 2 -name "annotations_all.json" 2>/dev/null | head -1 || true)
        [ -z "$ANN_FILE" ] && ANN_FILE=$(find "$EXTRACT_DIR" -maxdepth 2 -name "annotations.json" 2>/dev/null | head -1 || true)

        if [ -n "$ANN_FILE" ] && [ -f "$ANN_FILE" ]; then
            log "  📋 Converting $ANN_FILE → YOLO format ..."
            python data/prepare_dataset.py \
                --data_dir "$EXTRACT_DIR" \
                --ann_file "$ANN_FILE" \
                --output_dir "$OUTPUT_DIR"
            log "  ✓ Conversion complete"
        else
            echo "❌ No annotation file (annotations_all.json / annotations.json) found in $EXTRACT_DIR"
            echo "   Top-level contents:"
            ls -l "$EXTRACT_DIR" 2>/dev/null | head -20
            exit 1
        fi

        log "  ✓ Dataset prepared at $OUTPUT_DIR"
    fi

    # Quick stats
    log "  Dataset statistics:"
    python -c "
from pathlib import Path
out = Path('$OUTPUT_DIR')
for split in ['train', 'val', 'test']:
    img_dir = out / 'images' / split
    if img_dir.exists():
        n = len([f for f in img_dir.glob('*') if f.suffix in ('.jpg','.jpeg','.png')])
        print(f'    {split}: {n} images')
"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Environment Check (skip torch)
# ═══════════════════════════════════════════════════════════════════════════════
if $SKIP_ENV; then
    log "⏭  Step 2: Environment check SKIPPED (--skip-env)"
else
    step "Step 2/4: Checking Python dependencies (excluding torch)"

    # (module_name, pip_package, optional_fix_hint)
    REQUIRED_PKGS=(
        "ultralytics:ultralytics"
        "cv2:opencv-python-headless"
        "albumentations:albumentations"
        "pycocotools:pycocotools"
        "numpy:numpy"
        "matplotlib:matplotlib"
        "seaborn:seaborn"
        "yaml:pyyaml"
        "tqdm:tqdm"
        "PIL:pillow"
    )

    MISSING=()
    MISSING_HINT=()
    for entry in "${REQUIRED_PKGS[@]}"; do
        mod="${entry%%:*}"
        pkg="${entry##*:}"
        if python -c "import $mod" 2>/dev/null; then
            :
        else
            MISSING+=("$pkg")
            MISSING_HINT+=("$mod→$pkg")
        fi
    done

    if [ ${#MISSING[@]} -gt 0 ]; then
        log "  ⚠️  Missing packages: ${MISSING_HINT[*]}"

        # Special handling for opencv: two variants conflict
        if [[ "${MISSING[*]}" =~ opencv ]]; then
            log "  🔧 opencv conflict? Uninstalling all variants first ..."
            pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
        fi

        log "  Installing: ${MISSING[*]}"
        pip install "${MISSING[@]}" 2>&1 | tail -5
        log "  ✓ Packages installed"
    else
        log "  ✓ All required packages present"
    fi

    # Quick verify
    python -c "
import cv2; print(f'  ✓ cv2 {cv2.__version__}')
from ultralytics import YOLO; print('  ✓ ultralytics OK')
from PIL import Image; print('  ✓ PIL OK')
"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Training
# ═══════════════════════════════════════════════════════════════════════════════
step "Step 3/4: Training — $STAGE"

mkdir -p "$EXPERIMENT_DIR"

log "  Config:   $CONFIG"
log "  Epochs:   $EPOCHS"
log "  Batch:    $BATCH_SIZE"
log "  ImgSz:    $IMG_SIZE"
log "  Device:   $DEVICE"
log "  Log:      $EXPERIMENT_DIR/train_${TIMESTAMP}.log"
echo ""

TRAIN_START=$(date +%s)

python scripts/train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE" \
    --project "$EXPERIMENT_DIR" \
    --name train \
    2>&1 | tee "$EXPERIMENT_DIR/train_${TIMESTAMP}.log"

TRAIN_END=$(date +%s)
TRAIN_MINS=$(( (TRAIN_END - TRAIN_START) / 60 ))

BEST_PT="$EXPERIMENT_DIR/train/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  ✅ Training complete!  Duration: ${TRAIN_MINS} min"
    log "  📦 Best model: $BEST_PT"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ Training failed — best.pt not found."
    echo "   Check log: $EXPERIMENT_DIR/train_${TIMESTAMP}.log"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Copy results to /mnt
# ═══════════════════════════════════════════════════════════════════════════════
step "Step 4/4: Copying results to $MNT_DIR"

MNT_DEST="$MNT_DIR/${STAGE}_${TIMESTAMP}"

if [ ! -d "$MNT_DIR" ]; then
    log "  ⚠️  $MNT_DIR does not exist — creating ..."
    mkdir -p "$MNT_DIR" 2>/dev/null || {
        echo "❌ Cannot create $MNT_DIR. Check permissions or mount status."
        exit 1
    }
fi

mkdir -p "$MNT_DEST"

# Copy essential files only (skip intermediate epochs to save space)
log "  Copying to $MNT_DEST ..."

# Weights (best + last)
cp -r "$EXPERIMENT_DIR/train/weights/best.pt" "$MNT_DEST/" 2>/dev/null || true
cp -r "$EXPERIMENT_DIR/train/weights/last.pt" "$MNT_DEST/" 2>/dev/null || true

# Training curves & plots
for f in results.png results.csv confusion_matrix.png confusion_matrix_normalized.png \
         BoxPR_curve.png BoxF1_curve.png BoxP_curve.png BoxR_curve.png \
         labels.jpg args.yaml; do
    cp "$EXPERIMENT_DIR/train/$f" "$MNT_DEST/" 2>/dev/null || true
done

# Validation batch samples
mkdir -p "$MNT_DEST/val_samples"
cp "$EXPERIMENT_DIR/train/val_batch0_pred.jpg" "$MNT_DEST/val_samples/" 2>/dev/null || true
cp "$EXPERIMENT_DIR/train/val_batch1_pred.jpg" "$MNT_DEST/val_samples/" 2>/dev/null || true
cp "$EXPERIMENT_DIR/train/val_batch2_pred.jpg" "$MNT_DEST/val_samples/" 2>/dev/null || true

# Training log
cp "$EXPERIMENT_DIR/train_${TIMESTAMP}.log" "$MNT_DEST/" 2>/dev/null || true

# Write run metadata
cat > "$MNT_DEST/run_info.txt" << EOF
Stage:       $STAGE
Config:      $CONFIG
Timestamp:   $TIMESTAMP
Epochs:      $EPOCHS
Batch size:  $BATCH_SIZE
Image size:  $IMG_SIZE
Device:      $DEVICE
Duration:    ${TRAIN_MINS} min
Server:      $(hostname)
Date:        $(date)
EOF

log "  ✓ Results copied to $MNT_DEST"
log "  📁 Contents:"
ls -lh "$MNT_DEST/" | tail -20

# ─── Summary ──────────────────────────────────────────────────────────────────
SCRIPT_END=$(date +%s)
TOTAL_MINS=$(( (SCRIPT_END - SCRIPT_START) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ Pipeline complete!"
echo "     Stage:       $STAGE"
echo "     Total time:  ${TOTAL_MINS} min"
echo "     Best model:  $BEST_PT"
echo "     Results:     $MNT_DEST"
echo "════════════════════════════════════════════════════════════"
echo ""
