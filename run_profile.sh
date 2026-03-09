#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CUDSS_CMAKE_DIR="/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${SCRIPT_DIR}/ncu_results_${TIMESTAMP}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo " cuDSS NCU Profile Script"
echo " Output directory: ${OUTPUT_DIR}"
echo "============================================"

# --- Build ---
CMAKE_EXTRA_ARGS=""
if [ -d "${CUDSS_CMAKE_DIR}" ]; then
    CMAKE_EXTRA_ARGS="-Dcudss_DIR=${CUDSS_CMAKE_DIR}"
fi

echo ""
echo "[1/4] Building profile_cudss..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" ${CMAKE_EXTRA_ARGS} > /dev/null 2>&1
cmake --build "${BUILD_DIR}" --target profile_cudss -j$(nproc) 2>&1 | tail -3

PROFILE_BIN="${BUILD_DIR}/profile_cudss"

# --- Profile each phase ---
PHASES=("analysis" "factorization" "solve")

for phase in "${PHASES[@]}"; do
    echo ""
    echo "============================================"
    echo "[${phase}] Profiling..."
    echo "============================================"

    # 1) SASS disassembly (assembly code)
    echo "  -> Collecting SASS disassembly..."
    ncu --set full \
        --import-source yes \
        --source-folders "${SCRIPT_DIR}" \
        --print-source sass \
        -o "${OUTPUT_DIR}/${phase}_sass_tmp" \
        "${PROFILE_BIN}" "${phase}" \
        > "${OUTPUT_DIR}/${phase}_sass.txt" 2>&1 || true

    # Remove temp ncu-rep from SASS dump (we only want the text)
    rm -f "${OUTPUT_DIR}/${phase}_sass_tmp.ncu-rep"

    echo "  -> SASS output: ${OUTPUT_DIR}/${phase}_sass.txt"

    # 2) Full ncu report (.ncu-rep)
    echo "  -> Collecting full NCU report..."
    ncu --set full \
        --import-source yes \
        --source-folders "${SCRIPT_DIR}" \
        -o "${OUTPUT_DIR}/${phase}_report" \
        "${PROFILE_BIN}" "${phase}" \
        > "${OUTPUT_DIR}/${phase}_ncu_stdout.txt" 2>&1 || true

    echo "  -> NCU report:  ${OUTPUT_DIR}/${phase}_report.ncu-rep"

    # 3) Also export a CSV summary
    if [ -f "${OUTPUT_DIR}/${phase}_report.ncu-rep" ]; then
        ncu --import "${OUTPUT_DIR}/${phase}_report.ncu-rep" \
            --csv \
            > "${OUTPUT_DIR}/${phase}_summary.csv" 2>/dev/null || true
        echo "  -> CSV summary: ${OUTPUT_DIR}/${phase}_summary.csv"
    fi
done

echo ""
echo "============================================"
echo " Profile complete!"
echo " All results in: ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Files per phase:"
echo "  <phase>_sass.txt        - SASS disassembly (assembly code)"
echo "  <phase>_report.ncu-rep  - NCU report (open with Nsight Compute UI)"
echo "  <phase>_summary.csv     - Kernel metrics CSV summary"
echo ""
ls -lh "${OUTPUT_DIR}/"
