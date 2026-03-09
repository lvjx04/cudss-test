#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CUDSS_CMAKE_DIR="/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss"

echo "============================================"
echo " cuDSS Benchmark Build & Run Script"
echo "============================================"

# Check cuDSS cmake dir, fall back to default find_package path
CMAKE_EXTRA_ARGS=""
if [ -d "${CUDSS_CMAKE_DIR}" ]; then
    CMAKE_EXTRA_ARGS="-Dcudss_DIR=${CUDSS_CMAKE_DIR}"
    echo "Using cuDSS at: ${CUDSS_CMAKE_DIR}"
else
    echo "Using system default cuDSS (find_package)"
fi

echo ""
echo "[1/3] Configuring..."
cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" ${CMAKE_EXTRA_ARGS}

echo ""
echo "[2/3] Building..."
cmake --build "${BUILD_DIR}" --target benchmark -j$(nproc)

echo ""
echo "[3/3] Running benchmark..."
echo ""
"${BUILD_DIR}/benchmark"

echo ""
echo "Done. Results saved to ${SCRIPT_DIR}/benchmark_results.txt"
