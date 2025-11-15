#!/bin/bash
#
# Profile GenTbD with py-spy to identify performance bottlenecks
#
# Usage:
#   bash benchmarks/profile_with_pyspy.sh
#   bash benchmarks/profile_with_pyspy.sh mps
#   bash benchmarks/profile_with_pyspy.sh cpu 100
#

set -e

# Default values
DEVICE=${1:-mps}
FRAMES=${2:-50}
VIDEO=${3:-data/TownCent.mp4}
OUTPUT_DIR="benchmarks/profiles"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}GenTbD Performance Profiling with py-spy${NC}"
echo -e "${GREEN}===========================================================${NC}"
echo ""

# Check if py-spy is installed
if ! command -v py-spy &> /dev/null; then
    echo -e "${RED}Error: py-spy not installed${NC}"
    echo ""
    echo "Install with:"
    echo "  pip install py-spy"
    echo ""
    exit 1
fi

# Check if video exists
if [ ! -f "$VIDEO" ]; then
    echo -e "${RED}Error: Video not found: $VIDEO${NC}"
    echo ""
    echo "Usage: $0 [device] [frames] [video_path]"
    echo "  device: cpu, mps, or cuda (default: mps)"
    echo "  frames: number of frames to process (default: 50)"
    echo "  video_path: path to video file (default: data/TownCent.mp4)"
    echo ""
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FLAMEGRAPH="${OUTPUT_DIR}/flamegraph_${DEVICE}_${TIMESTAMP}.svg"
SPEEDSCOPE="${OUTPUT_DIR}/speedscope_${DEVICE}_${TIMESTAMP}.json"

echo "Configuration:"
echo "  Device:     ${DEVICE}"
echo "  Frames:     ${FRAMES}"
echo "  Video:      ${VIDEO}"
echo "  Output:     ${OUTPUT_DIR}/"
echo ""

# Construct command
CMD="python -m src.main --video $VIDEO --device $DEVICE --max-dimension 640"

echo -e "${YELLOW}Starting profiler...${NC}"
echo "Command: $CMD"
echo ""

# Run py-spy with flamegraph output
echo "Generating flamegraph..."
py-spy record \
    --output "$FLAMEGRAPH" \
    --format flamegraph \
    --rate 100 \
    --subprocesses \
    -- $CMD &

PYSPY_PID=$!

# Let it run for a bit then kill
sleep 10
kill $PYSPY_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}Flamegraph generated: $FLAMEGRAPH${NC}"
echo ""

# Also generate speedscope format (interactive)
echo "Generating speedscope profile..."
py-spy record \
    --output "$SPEEDSCOPE" \
    --format speedscope \
    --rate 100 \
    --subprocesses \
    -- $CMD &

PYSPY_PID=$!
sleep 10
kill $PYSPY_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}Speedscope profile generated: $SPEEDSCOPE${NC}"
echo ""

# Summary
echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}Profiling Complete!${NC}"
echo -e "${GREEN}===========================================================${NC}"
echo ""
echo "View results:"
echo "  1. Open flamegraph: open $FLAMEGRAPH"
echo "  2. View speedscope: https://www.speedscope.app/"
echo "     → Upload: $SPEEDSCOPE"
echo ""
echo "What to look for:"
echo "  - Wide bars = functions taking most time"
echo "  - Look for .cpu() calls (indicates GPU→CPU transfers)"
echo "  - Look for tight loops in postprocess"
echo "  - Compare CPU vs MPS profiles"
echo ""
