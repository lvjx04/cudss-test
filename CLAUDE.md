# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cuDSS (NVIDIA CUDA Direct Sparse Solver) test/example project. It demonstrates solving a sparse linear system `Ax = b` using cuDSS APIs, where A is a symmetric positive-definite (SPD) matrix stored in CSR format.

## Build Commands

```bash
# Configure (out-of-source build)
cmake -B build -S .

# Build
cmake --build build

# Build with static linking
cmake -B build -S . -DBUILD_STATIC=ON
cmake --build build
```

The built executable is `build/simple_example` (and `build/simple_example_static` if `BUILD_STATIC=ON`).

## Dependencies

- CUDA toolkit (CMake languages: CXX, CUDA)
- cuDSS >= 0.6.0 (found via `find_package(cudss)`)
- C++11 standard (for CUDA)

## Architecture

Single-file project (`simple.cpp` compiled as CUDA via `set_source_files_properties`). The solver workflow follows the standard cuDSS three-phase pattern:

1. **Analysis** (`CUDSS_PHASE_ANALYSIS`) — symbolic factorization / reordering
2. **Factorization** (`CUDSS_PHASE_FACTORIZATION`) — numeric factorization
3. **Solve** (`CUDSS_PHASE_SOLVE`) — triangular solve

Key cuDSS objects: `cudssHandle_t`, `cudssConfig_t`, `cudssData_t`, `cudssMatrix_t`.

Error handling uses macros `CUDA_CALL_AND_CHECK` and `CUDSS_CALL_AND_CHECK` that print diagnostics and return early on failure.
