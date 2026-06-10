#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# TT100K Small Target Detection — Server One-Click Training
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Always run from project root (regardless of where script is called from)
cd "$(dirname "$0")/.."

# ─── Config ──────────────────────────────────────────────────────────────────

DATASET_PATH="${TT100K_PATH:-/public/data/image/TT100K}"
OUTPUT_DIR="data/processed"
EXPERIMENT_DIR="experiments/stage1_baseline"
UNZIP_DIR="data/TT100K_raw"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-1280}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-0}"

log() { echo -e "[$(date +%H:%M:%S)] $*"; }

# ─── Step 1: Prepare dataset ─────────────────────────────────────────────────

log "═══ Step 1/3: Preparing dataset ═══"

if [ -f "$OUTPUT_DIR/dataset.yaml" ]; then
    log "  Already prepared, skipping."
else
    # --- 1a: Extract tt100k_2021.zip ---
    RAW_ZIP=$(find "$DATASET_PATH" -maxdepth 3 -name "tt100k_2021.zip" 2>/dev/null | head -1 || true)

    if [ -n "$RAW_ZIP" ] && [ -f "$RAW_ZIP" ] && [ ! -f "$UNZIP_DIR/.extracted" ]; then
        log "  Extracting tt100k_2021.zip..."
        rm -rf "$UNZIP_DIR"
        mkdir -p "$UNZIP_DIR"
        unzip -qo "$RAW_ZIP" -d "$UNZIP_DIR/"
        touch "$UNZIP_DIR/.extracted"
        log "  ✓ Extracted to $UNZIP_DIR"
    fi

    # Use extracted dir if available, otherwise use original path
    if [ -d "$UNZIP_DIR" ] && [ -f "$UNZIP_DIR/.extracted" ]; then
        SRC="$UNZIP_DIR"
        # If extraction created a single subdirectory, use that as root
        SUBDIRS=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ "$SUBDIRS" -eq 1 ]; then
            SRC=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d | head -1)
            log "  Dataset root: $SRC"
        fi
    else
        SRC="$DATASET_PATH"
    fi

    # --- 1b: Detect format and convert to YOLO ---
    YOLO_ZIP=$(find "$SRC" -maxdepth 2 \( -name "*YOLO*.zip" -o -name "*yolo*.zip" \) 2>/dev/null | head -1 || true)

    if [ -n "$YOLO_ZIP" ] && [ -f "$YOLO_ZIP" ]; then
        log "  YOLO-format ZIP detected, filtering 5 target classes..."
        python data/extract_dataset.py --zip_path "$YOLO_ZIP" --output_dir "$OUTPUT_DIR"

    elif [ -f "$SRC/annotations_all.json" ] || [ -f "$SRC/annotations.json" ]; then
        ANN_FILE=$( [ -f "$SRC/annotations_all.json" ] && echo "$SRC/annotations_all.json" || echo "$SRC/annotations.json" )
        log "  Found $ANN_FILE, converting..."
        python data/prepare_dataset.py --data_dir "$SRC" --ann_file "$ANN_FILE" --output_dir "$OUTPUT_DIR"

    elif [ -d "$SRC/images" ] && [ -d "$SRC/labels" ]; then
        log "  YOLO directory structure detected, linking..."
        mkdir -p "$OUTPUT_DIR/images" "$OUTPUT_DIR/labels"
        cp -rn "$SRC/images/"* "$OUTPUT_DIR/images/" 2>/dev/null || true
        cp -rn "$SRC/labels/"* "$OUTPUT_DIR/labels/" 2>/dev/null || true
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
        log "  Content of $SRC:"
        ls -l "$SRC" 2>/dev/null || true
        echo ""
        echo "ERROR: Cannot detect dataset format."
        echo "Expected: annotations.json, YOLO .zip, or images/ + labels/ directories"
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
