#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# COCO Evaluation — One-click script
# ────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./scripts/eval_coco.sh stage1_baseline          # evaluate one stage
#   ./scripts/eval_coco.sh stage1_baseline --cpu    # force CPU
#   ./scripts/eval_coco.sh                          # evaluate ALL completed stages
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──
IMGSZ=1280
SPLIT="test"
CONF=0.001
BATCH=8

# ── Device auto-detect ──
detect_device() {
    python3 -c "
import torch
if torch.cuda.is_available():
    print('0')
elif torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null || echo "cpu"
}

DEVICE="${DEVICE:-$(detect_device)}"

# ── Helper: eval one stage ──
eval_stage() {
    local stage="$1"
    local exp_dir="experiments/${stage}/train"
    local weights="${exp_dir}/weights/best.pt"

    if [ ! -f "$weights" ]; then
        echo "⚠  Skipping ${stage}: no best.pt found at ${weights}"
        return 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Evaluating: ${stage}"
    echo "  Weights:    ${weights}"
    echo "  Device:     ${DEVICE}"
    echo "  Image size: ${IMGSZ}"
    echo "════════════════════════════════════════════════════════════"

    python3 scripts/eval.py \
        --weights "$weights" \
        --data data/processed/dataset.yaml \
        --split "$SPLIT" \
        --imgsz "$IMGSZ" \
        --batch "$BATCH" \
        --device "$DEVICE" \
        --conf "$CONF" \
        --analyze_sizes \
        --save_dir "experiments/${stage}/eval"

    echo ""
    echo "  ✅ ${stage} done"
    echo "     Metrics:  experiments/${stage}/eval/metrics.json"
    echo "     Size analysis: experiments/${stage}/eval/size_analysis.json"
}

# ── Main ──
if [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; then
    # Specific stage(s) given
    for stage in "$@"; do
        [[ "$stage" =~ ^-- ]] && break
        eval_stage "$stage"
    done
else
    # Evaluate all stages that have a best.pt
    echo "Scanning for trained stages..."
    FOUND=0
    for exp_dir in experiments/*/ ; do
        stage=$(basename "$exp_dir")
        if [ "$stage" = "eval_results" ]; then continue; fi
        if [ -f "${exp_dir}train/weights/best.pt" ]; then
            ((FOUND++))
            eval_stage "$stage"
        fi
    done
    if [ "$FOUND" -eq 0 ]; then
        echo "No trained stages found (no best.pt under experiments/*/train/weights/)."
        exit 1
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All evaluations complete."
echo "  Device used: ${DEVICE}"
echo "════════════════════════════════════════════════════════════"
