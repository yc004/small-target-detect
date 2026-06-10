#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# TT100K Small Target Detection — Server One-Click Training
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

log()  { echo -e "[$(date +%H:%M:%S)] $*"; }

# ─── Step 1: Prepare dataset ─────────────────────────────────────────────────

log "═══ Step 1/3: Preparing dataset ═══"

if [ -f "$OUTPUT_DIR/dataset.yaml" ]; then
    log "  Already prepared, skipping."
else
    YOLO_ZIP=$(find "$DATASET_PATH" -maxdepth 2 \( -name "*YOLO*.zip" -o -name "*yolo*.zip" \) 2>/dev/null | head -1 || true)

    if [ -n "$YOLO_ZIP" ] && [ -f "$YOLO_ZIP" ]; then
        log "  Found YOLO ZIP, extracting & filtering..."
        python data/extract_dataset.py --zip_path "$YOLO_ZIP" --output_dir "$OUTPUT_DIR"
    elif [ -f "$DATASET_PATH/annotations.json" ]; then
        log "  Found annotations.json, converting..."
        python data/prepare_dataset.py --data_dir "$DATASET_PATH" --output_dir "$OUTPUT_DIR"
    elif [ -d "$DATASET_PATH/images" ] && [ -d "$DATASET_PATH/labels" ]; then
        log "  YOLO-format directory detected, linking..."
        mkdir -p "$OUTPUT_DIR/images" "$OUTPUT_DIR/labels"
        cp -rn "$DATASET_PATH/images/"* "$OUTPUT_DIR/images/" 2>/dev/null || true
        cp -rn "$DATASET_PATH/labels/"* "$OUTPUT_DIR/labels/" 2>/dev/null || true
        python -c "
import yaml
from pathlib import Path
out = Path('$OUTPUT_DIR')
test = str(out / 'images' / 'test') if (out / 'images' / 'test').exists() else str(out / 'images' / 'val')
config = {'path': str(out.absolute()), 'train': str(out / 'images' / 'train'),
          'val': str(out / 'images' / 'val'), 'test': test, 'nc': 5,
          'names': ['speed_limit_5','speed_limit_30','speed_limit_40','no_entry','no_pedestrians']}
with open(out / 'dataset.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('  dataset.yaml written')
"
    else
        echo "ERROR: Cannot detect dataset format at $DATASET_PATH"
        exit 1
    fi
    log "  ✓ Done"
fi

# ─── Step 2: Stats ───────────────────────────────────────────────────────────

log "═══ Step 2/3: Dataset statistics ═══"

python -c "
from pathlib import Path
out = Path('$OUTPUT_DIR')
for split in ['train', 'val', 'test']:
    img_dir = out / 'images' / split
    lbl_dir = out / 'labels' / split
    if img_dir.exists() and lbl_dir.exists():
        n_img = len([f for f in img_dir.glob('*') if f.suffix in ('.jpg','.jpeg','.png')])
        n_lbl = len(list(lbl_dir.glob('*.txt')))
        print(f'  {split}: {n_img} images, {n_lbl} labels')
"

# ─── Step 3: Train ───────────────────────────────────────────────────────────

log "═══ Step 3/3: Training ═══"
mkdir -p "$EXPERIMENT_DIR"

log "  Epochs: $EPOCHS | Batch: $BATCH_SIZE | ImgSz: $IMG_SIZE | Device: $DEVICE"
echo ""

python scripts/train.py \
    --config configs/baseline.yaml \
    --epochs "$EPOCHS" \
    --batch "$BATCH_SIZE" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE" \
    --project "$EXPERIMENT_DIR" \
    --name train \
    2>&1 | tee "$EXPERIMENT_DIR/train.log"

BEST_PT="$EXPERIMENT_DIR/train/weights/best.pt"
if [ -f "$BEST_PT" ]; then
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  ✓ Done! Best model: $BEST_PT"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
