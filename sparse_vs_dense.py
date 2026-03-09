#!/usr/bin/env python3
"""
Dense vs Sparse solver benchmark.

Compares PyTorch dense solver (torch.linalg.solve) against cuDSS sparse solver
for complex fp64 symmetric matrices of varying sizes and sparsity levels.

Usage:
    python dense_vs_sparse.py

The cuDSS results are read from the most recent benchmark_results_*.txt file.
Dense solver timings are measured here using PyTorch on GPU.
"""

import os
import sys
import glob
import time
import random
from datetime import datetime

import torch

def generate_symmetric_complex(n, sparsity, seed=42):
    """
    Generate an n×n complex fp64 symmetric dense matrix with given sparsity.
    Matches the structure from benchmark.cpp:
      - Dominant diagonal for non-singularity
      - Random off-diagonal entries with target sparsity = nnz_full / n^2
    Returns (A_dense, b) on GPU, where b = A @ ones(n).
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # Start with diagonal
    A = torch.zeros(n, n, dtype=torch.complex128, device='cuda')
    diag_vals = n + torch.rand(n, dtype=torch.float64, device='cuda') * 10.0 \
                + 1j * torch.rand(n, dtype=torch.float64, device='cuda') * 0.1
    A.diagonal().copy_(diag_vals)

    # Off-diagonal: target count
    target_offdiag = int((sparsity * n * n - n) / 2.0)
    max_offdiag = n * (n - 1) // 2
    target_offdiag = max(0, min(target_offdiag, max_offdiag))

    if target_offdiag > 0:
        fill_ratio = target_offdiag / max_offdiag if max_offdiag > 0 else 0.0

        if fill_ratio < 0.3:
            # Sparse: random sampling
            chosen = set()
            attempts = 0
            max_attempts = target_offdiag * 10 + 1000
            indices_i = []
            indices_j = []
            while len(chosen) < target_offdiag and attempts < max_attempts:
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                key = i * n + j
                if key not in chosen:
                    chosen.add(key)
                    indices_i.append(i)
                    indices_j.append(j)
                attempts += 1

            nfill = len(indices_i)
            vals_re = torch.rand(nfill, dtype=torch.float64, device='cuda') * 2.0 - 1.0
            vals_im = torch.rand(nfill, dtype=torch.float64, device='cuda') * 2.0 - 1.0
            vals = torch.complex(vals_re, vals_im)

            idx_i = torch.tensor(indices_i, device='cuda')
            idx_j = torch.tensor(indices_j, device='cuda')
            A[idx_i, idx_j] = vals
            A[idx_j, idx_i] = vals  # symmetric
        else:
            # Dense-ish: generate full upper triangle mask
            mask = torch.rand(n, n, dtype=torch.float64, device='cuda') < fill_ratio
            mask = torch.triu(mask, diagonal=1)
            re = (torch.rand(n, n, dtype=torch.float64, device='cuda') * 2.0 - 1.0) * mask
            im = (torch.rand(n, n, dtype=torch.float64, device='cuda') * 2.0 - 1.0) * mask
            upper = torch.complex(re, im)
            A = A + upper + upper.T

    # b = A @ ones
    x_exact = torch.ones(n, dtype=torch.complex128, device='cuda')
    b = A @ x_exact

    return A, b, x_exact


def benchmark_dense_solve(n, sparsity, warmup=1, repeat=3):
    """Benchmark torch.linalg.solve for an n×n complex symmetric dense matrix."""
    A, b, x_exact = generate_symmetric_complex(n, sparsity)

    # Warmup
    for _ in range(warmup):
        x = torch.linalg.solve(A, b)
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = torch.linalg.solve(A, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    # Check correctness
    err = (x - x_exact).abs().max().item()

    # GPU memory used by matrix A (approximate)
    mem_A_mb = A.element_size() * A.numel() / (1024 * 1024)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    return {
        'n': n,
        'sparsity': sparsity,
        'avg_ms': avg_ms,
        'min_ms': min_ms,
        'max_error': err,
        'mem_matrix_mb': mem_A_mb,
    }


def parse_cudss_results(filepath):
    """Parse the cuDSS benchmark results file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-') or line.startswith('N') or 'cuDSS' in line:
                continue
            parts = line.split()
            if len(parts) >= 11:
                try:
                    n = int(parts[0])
                    sparsity_str = parts[3].rstrip('%')
                    sparsity_pct = float(sparsity_str)
                    total_ms = float(parts[7])
                    vram_mb = int(parts[8])
                    max_err = float(parts[10])
                    results.append({
                        'n': n,
                        'sparsity_pct': sparsity_pct,
                        'total_ms': total_ms,
                        'vram_mb': vram_mb,
                        'max_error': max_err,
                    })
                except (ValueError, IndexError):
                    continue
    return results


def main():
    print("=" * 100)
    print("  Dense (PyTorch) vs Sparse (cuDSS) Solver Benchmark")
    print("  Matrix type: complex fp64 symmetric, nrhs=1")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 100)

    # Find most recent cuDSS results
    cudss_files = sorted(glob.glob("benchmark_results_*.txt"))
    cudss_results = {}
    if cudss_files:
        latest = cudss_files[-1]
        print(f"\n  cuDSS results from: {latest}")
        for r in parse_cudss_results(latest):
            key = (r['n'], r['sparsity_pct'])
            cudss_results[key] = r
    else:
        print("\n  WARNING: No cuDSS benchmark results found. Run ./run_benchmark.sh first.")

    # Test configurations — same as benchmark.cpp, plus large sizes
    # For large n, dense solver will OOM or be extremely slow
    tests = [
        # Vary size, fixed sparsity 1%
        (500, 0.01),
        (1000, 0.01),
        (2000, 0.01),
        (5000, 0.01),
        (10000, 0.01),

        # Vary sparsity, fixed n=1000
        (1000, 0.001),
        (1000, 0.005),
        (1000, 0.05),
        (1000, 0.10),
        (1000, 0.30),

        # Vary sparsity, fixed n=5000
        (5000, 0.001),
        (5000, 0.005),
        (5000, 0.05),

        # Large matrices — dense will OOM, sparse can handle
        (20000, 0.001),
        (20000, 0.005),
        (50000, 0.001),
    ]

    print(f"\nRunning dense solver benchmarks...\n")

    dense_results = []
    for n, sparsity in tests:
        print(f"  n = {n:<8d}  sparsity = {sparsity*100:<6.1f}% ... ", end='', flush=True)
        # Check if dense matrix would exceed GPU memory
        matrix_bytes = n * n * 16  # complex128 = 16 bytes
        free_mem, total_mem = torch.cuda.mem_get_info()
        if matrix_bytes * 3 > free_mem:  # need ~3x for A, workspace, etc.
            needed_gb = matrix_bytes * 3 / (1024**3)
            avail_gb = free_mem / (1024**3)
            print(f"SKIPPED (need ~{needed_gb:.1f} GB, avail {avail_gb:.1f} GB)")
            dense_results.append({
                'n': n, 'sparsity': sparsity,
                'avg_ms': float('nan'), 'min_ms': float('nan'),
                'max_error': float('nan'),
                'mem_matrix_mb': matrix_bytes / (1024 * 1024),
                'status': 'OOM',
            })
            continue
        try:
            r = benchmark_dense_solve(n, sparsity, warmup=1, repeat=3)
            r['status'] = 'OK'
            dense_results.append(r)
            print(f"done  (avg = {r['avg_ms']:10.3f} ms, err = {r['max_error']:.2e})")
        except Exception as e:
            print(f"FAILED ({e})")
            dense_results.append({
                'n': n, 'sparsity': sparsity,
                'avg_ms': float('nan'), 'min_ms': float('nan'),
                'max_error': float('nan'),
                'mem_matrix_mb': matrix_bytes / (1024 * 1024),
                'status': 'FAILED',
            })

    # Build comparison table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"dense_vs_sparse_{timestamp}.txt"

    header = (
        f"{'N':>8s} {'Sparsity':>10s} "
        f"{'Dense(ms)':>12s} {'Sparse(ms)':>12s} {'Speedup':>10s} "
        f"{'DenseMem(MB)':>14s} {'SparseMem(MB)':>14s} {'MemRatio':>10s} "
        f"{'DenseErr':>12s} {'SparseErr':>12s}"
    )
    sep = "-" * len(header)

    lines = []
    lines.append("=" * len(header))
    lines.append("  Dense (PyTorch torch.linalg.solve) vs Sparse (cuDSS) Solver Comparison")
    lines.append(f"  Matrix: complex fp64 symmetric | GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(sep)

    for dr in dense_results:
        n = dr['n']
        sp = dr['sparsity']
        sp_pct = sp * 100.0
        key = (n, sp_pct)

        dense_ms = dr['avg_ms']
        dense_mem = dr['mem_matrix_mb']
        dense_err = dr['max_error']
        dense_status = dr.get('status', 'OK')

        cr = cudss_results.get(key)
        if cr:
            sparse_ms = cr['total_ms']
            sparse_mem = cr['vram_mb']
            sparse_err = cr['max_error']
            if sparse_ms > 0 and dense_ms == dense_ms:  # not nan
                speedup = f"{dense_ms / sparse_ms:.2f}x"
            else:
                speedup = "N/A"
            if dense_mem > 0 and sparse_mem > 0 and dense_status == 'OK':
                mem_ratio = f"{dense_mem / sparse_mem:.2f}x"
            else:
                mem_ratio = "N/A"
            sparse_ms_s = f"{sparse_ms:.3f}"
            sparse_mem_s = f"{sparse_mem}"
            sparse_err_s = f"{sparse_err:.2e}"
        else:
            sparse_ms_s = "N/A"
            sparse_mem_s = "N/A"
            sparse_err_s = "N/A"
            speedup = "N/A"
            mem_ratio = "N/A"

        if dense_status == 'OOM':
            dense_ms_s = "OOM"
            dense_err_s = "N/A"
            speedup = "sparse only"
        elif dense_status == 'FAILED':
            dense_ms_s = "FAILED"
            dense_err_s = "N/A"
            speedup = "sparse only"
        else:
            dense_ms_s = f"{dense_ms:.3f}"
            dense_err_s = f"{dense_err:.2e}"

        lines.append(
            f"{n:>8d} {sp_pct:>9.3f}% "
            f"{dense_ms_s:>12s} {sparse_ms_s:>12s} {speedup:>10s} "
            f"{dense_mem:>14.1f} {sparse_mem_s:>14s} {mem_ratio:>10s} "
            f"{dense_err_s:>12s} {sparse_err_s:>12s}"
        )

    lines.append("=" * len(header))
    lines.append("")
    lines.append("Speedup = Dense_time / Sparse_time  (>1 means sparse is faster)")
    lines.append("MemRatio = Dense_matrix_mem / Sparse_VRAM  (>1 means sparse uses less)")
    lines.append("DenseMem = memory for the full n×n matrix only (actual GPU usage is higher)")
    lines.append("SparseMem = total GPU VRAM consumed by cuDSS (including factors)")

    table_str = "\n".join(lines)

    # Print to stdout
    print(f"\n{table_str}")

    # Write to file
    with open(outfile, 'w') as f:
        f.write(table_str + "\n")
    print(f"\nResults written to {outfile}")


if __name__ == "__main__":
    main()
