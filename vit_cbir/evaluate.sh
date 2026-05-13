#!/usr/bin/env bash
# ============================================================
# evaluate.sh — Chạy đánh giá ViT CBIR trên từng dataset
#
# Usage:
#   ./evaluate.sh                   # tất cả dataset
#   ./evaluate.sh oxford5k          # chỉ Oxford5K
#   ./evaluate.sh ukbench paris6k   # nhiều dataset
#
# Options (env vars):
#   MODEL=vit_b16|vit_b32|vit_l16|vit_l32  (default: vit_b16)
#   NORM=robust|l2_axis1|l1_axis1|...       (default: robust)
#   METRIC=cosine|euclidean|cityblock|...   (default: cosine)
#   BATCH=16                                (default: 16)
#   NO_CACHE=1                              (bỏ qua cache, trích xuất lại)
#
# Ví dụ:
#   MODEL=vit_b32 ./evaluate.sh oxford5k
#   METRIC=euclidean ./evaluate.sh oxford5k ukbench
#   NO_CACHE=1 ./evaluate.sh oxford5k
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------- config ----------
MODEL="${MODEL:-vit_b16}"
NORM="${NORM:-robust}"
METRIC="${METRIC:-cosine}"
BATCH="${BATCH:-16}"
DATA_ROOT="data"
CACHE_DIR="cache"
mkdir -p "$CACHE_DIR"

# ---------- màu sắc ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ---------- helpers ----------
header() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}${CYAN}  $1${RESET}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

check_data() {
    local dataset="$1"
    local img_dir="$DATA_ROOT/$dataset/images"
    case "$dataset" in
        ukbench) img_dir="$DATA_ROOT/$dataset/full" ;;
        inria)   img_dir="$DATA_ROOT/$dataset/jpg"  ;;
    esac
    if [ ! -d "$img_dir" ] || [ -z "$(ls "$img_dir"/*.jpg 2>/dev/null | head -1)" ]; then
        echo -e "${RED}  [SKIP] $dataset — không tìm thấy ảnh trong $img_dir${RESET}"
        return 1
    fi
    return 0
}

run_eval() {
    local dataset="$1"
    local cache_flag=""

    if [ "${NO_CACHE:-0}" != "1" ]; then
        cache_flag="--cache $CACHE_DIR/${dataset}_${MODEL}.npy"
    fi

    echo -e "${YELLOW}  model=$MODEL  norm=$NORM  metric=$METRIC${RESET}"

    python3 run_evaluate.py \
        --dataset   "$dataset" \
        --data_dir  "$DATA_ROOT/$dataset" \
        --model     "$MODEL" \
        --norm      "$NORM" \
        --metric    "$METRIC" \
        --batch_size "$BATCH" \
        $cache_flag \
        2>&1 | grep -v "UserWarning\|pkg_resources\|FutureWarning\|Disabling\|^$" \
             | grep --color=never -E "^\[|mAP|N-S score|images|queries|groups|cache|Loaded|Error|Traceback" \
        && echo -e "${GREEN}  ✓ Hoàn thành${RESET}" \
        || echo -e "${RED}  ✗ Lỗi${RESET}"
}

# ---------- danh sách dataset ----------
ALL_DATASETS=(oxford5k paris6k ukbench inria)
TARGETS=("${@:-${ALL_DATASETS[@]}}")

# ---------- banner ----------
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       ViT CBIR Evaluation  (arXiv:2101.03771)   ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo -e "  Model : ${BOLD}$MODEL${RESET}   Norm : ${BOLD}$NORM${RESET}   Metric : ${BOLD}$METRIC${RESET}"
echo -e "  Targets: ${BOLD}${TARGETS[*]}${RESET}"

START=$(date +%s)

for dataset in "${TARGETS[@]}"; do
    header "$dataset"
    if check_data "$dataset"; then
        run_eval "$dataset"
    fi
done

END=$(date +%s)
echo ""
echo -e "${BOLD}${GREEN}Tổng thời gian: $((END - START))s${RESET}"
echo ""
