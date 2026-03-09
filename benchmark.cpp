/*
 * cuDSS benchmark: complex fp64 symmetric sparse matrices.
 * Tests varying matrix sizes and sparsity levels.
 * Results (including GPU memory consumption) are written to a timestamped file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <set>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cudss.h"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                      \
            return -1;                                                           \
        }                                                                        \
    } while (0)

#define CUDSS_CHECK(call)                                                        \
    do {                                                                         \
        cudssStatus_t s = call;                                                  \
        if (s != CUDSS_STATUS_SUCCESS) {                                         \
            printf("cuDSS error at %s:%d: status = %d\n", __FILE__, __LINE__, s);\
            return -1;                                                           \
        }                                                                        \
    } while (0)

/*
 * Generate an n×n sparse symmetric matrix in CSR format (upper triangular view)
 * with approximately the target sparsity ratio (nnz_full / n^2).
 * Data type: cuDoubleComplex (complex fp64).
 *
 * Strategy:
 *   - Always include all diagonal entries (dominant, to keep the matrix non-singular).
 *   - Randomly sample off-diagonal entries in the upper triangle to reach target sparsity.
 *   - sparsity = nnz_full / n^2, where nnz_full = n + 2 * nnz_offdiag_upper.
 *
 * The matrix is symmetric (A = A^T, NOT Hermitian), stored as upper triangle only.
 */
static void generate_symmetric_complex(int n, double sparsity, unsigned int seed,
                                        std::vector<int> &offsets,
                                        std::vector<int> &columns,
                                        std::vector<cuDoubleComplex> &values,
                                        std::vector<cuDoubleComplex> &rhs,
                                        int &nnz_upper, int &nnz_full)
{
    srand(seed);

    /* How many off-diagonal entries in upper triangle?
       nnz_full = n (diag) + 2 * nnz_offdiag
       target: sparsity = nnz_full / n^2
       => nnz_offdiag = (sparsity * n^2 - n) / 2 */
    int64_t max_offdiag = (int64_t)n * (n - 1) / 2;
    int64_t target_offdiag = (int64_t)((sparsity * (double)n * n - n) / 2.0);
    if (target_offdiag < 0) target_offdiag = 0;
    if (target_offdiag > max_offdiag) target_offdiag = max_offdiag;

    /* For each row, collect sorted column indices (upper triangle, including diagonal) */
    std::vector<std::vector<int>> row_cols(n);
    for (int i = 0; i < n; i++) {
        row_cols[i].push_back(i); /* diagonal always present */
    }

    /* Randomly sample off-diagonal upper entries.
       For moderate sparsity and moderate n, use random rejection.
       For high sparsity (>30% of upper triangle), fill all then remove. */
    double fill_ratio = (max_offdiag > 0) ? (double)target_offdiag / max_offdiag : 0.0;

    if (fill_ratio < 0.5) {
        /* Sparse: random sampling with a set for dedup */
        /* Use a simple approach: for each desired entry, pick random (i,j) with i<j */
        std::set<int64_t> chosen;
        int64_t attempts = 0;
        int64_t max_attempts = target_offdiag * 10 + 1000;
        while ((int64_t)chosen.size() < target_offdiag && attempts < max_attempts) {
            int i = rand() % n;
            int j = rand() % n;
            if (i >= j) continue;
            int64_t key = (int64_t)i * n + j;
            if (chosen.insert(key).second) {
                row_cols[i].push_back(j);
            }
            attempts++;
        }
    } else {
        /* Dense: include all upper entries then randomly remove some */
        /* Actually just include all for simplicity when fill_ratio >= 0.5 */
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                /* Keep with probability fill_ratio */
                if ((rand() / (double)RAND_MAX) < fill_ratio || fill_ratio >= 0.99) {
                    row_cols[i].push_back(j);
                }
            }
        }
    }

    /* Sort columns within each row */
    for (int i = 0; i < n; i++) {
        std::sort(row_cols[i].begin(), row_cols[i].end());
    }

    /* Build CSR arrays */
    int total_nnz = 0;
    for (int i = 0; i < n; i++) total_nnz += (int)row_cols[i].size();
    nnz_upper = total_nnz;

    /* nnz_full: diagonal counted once, off-diag counted twice */
    int offdiag_count = total_nnz - n;
    nnz_full = n + 2 * offdiag_count;

    offsets.resize(n + 1);
    columns.resize(total_nnz);
    values.resize(total_nnz);

    int idx = 0;
    for (int i = 0; i < n; i++) {
        offsets[i] = idx;
        /* Track row sum for RHS computation */
        for (int k = 0; k < (int)row_cols[i].size(); k++) {
            int j = row_cols[i][k];
            columns[idx] = j;
            if (i == j) {
                /* Diagonal: dominant to ensure non-singularity.
                   Make |a_ii| large relative to off-diagonal sum. */
                double re = (double)n + (rand() / (double)RAND_MAX) * 10.0;
                double im = (rand() / (double)RAND_MAX) * 0.1;
                values[idx] = make_cuDoubleComplex(re, im);
            } else {
                double re = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
                double im = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
                values[idx] = make_cuDoubleComplex(re, im);
            }
            idx++;
        }
    }
    offsets[n] = idx;

    /* Compute RHS = A * x_exact, where x_exact = (1+0i, 1+0i, ..., 1+0i)
       Since A is symmetric stored as upper triangle:
       b_i = sum_j A(i,j) = sum over upper entries in row i
                           + sum over entries (k,i) where k < i (symmetric part) */
    rhs.resize(n);
    /* First pass: accumulate from upper-triangle rows */
    std::vector<cuDoubleComplex> b(n, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < n; i++) {
        for (int k = offsets[i]; k < offsets[i + 1]; k++) {
            int j = columns[k];
            cuDoubleComplex v = values[k];
            /* A(i,j) contributes to b[i] */
            b[i] = make_cuDoubleComplex(cuCreal(b[i]) + cuCreal(v),
                                        cuCimag(b[i]) + cuCimag(v));
            if (i != j) {
                /* Symmetric: A(j,i) = A(i,j), contributes to b[j] */
                b[j] = make_cuDoubleComplex(cuCreal(b[j]) + cuCreal(v),
                                            cuCimag(b[j]) + cuCimag(v));
            }
        }
    }
    rhs = b;
}

struct BenchResult {
    int    n;
    int    nnz_upper;
    int    nnz_full;
    double sparsity;     /* actual nnz_full / n^2 */
    float  analysis_ms;
    float  factor_ms;
    float  solve_ms;
    float  total_ms;
    double max_error;
    size_t mem_before_mb; /* GPU memory used before this test (MB) */
    size_t mem_peak_mb;   /* GPU memory used at peak (after factor, MB) */
    size_t mem_delta_mb;  /* memory consumed by this solve (MB) */
};

/* Return currently used GPU memory in bytes */
static size_t gpu_mem_used()
{
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return total_bytes - free_bytes;
}

static int run_benchmark(int n, double target_sparsity, BenchResult &result)
{
    std::vector<int>             h_offsets, h_columns;
    std::vector<cuDoubleComplex> h_values, h_b;
    int nnz_upper, nnz_full;

    generate_symmetric_complex(n, target_sparsity, 42,
                               h_offsets, h_columns, h_values, h_b,
                               nnz_upper, nnz_full);

    double actual_sparsity = (double)nnz_full / ((double)n * n);

    /* Allocate device memory */
    int             *d_offsets = nullptr, *d_columns = nullptr;
    cuDoubleComplex *d_values = nullptr, *d_b = nullptr, *d_x = nullptr;

    CUDA_CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_columns, nnz_upper * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,  nnz_upper * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), (n + 1) * sizeof(int),                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns, h_columns.data(), nnz_upper * sizeof(int),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,  h_values.data(),  nnz_upper * sizeof(cuDoubleComplex),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,       h_b.data(),       n * sizeof(cuDoubleComplex),          cudaMemcpyHostToDevice));

    /* Record GPU memory before cuDSS operations */
    size_t mem_before = gpu_mem_used();

    /* cuDSS setup */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudssHandle_t handle;
    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssSetStream(handle, stream));

    cudssConfig_t config;
    cudssData_t   data;
    CUDSS_CHECK(cudssConfigCreate(&config));
    CUDSS_CHECK(cudssDataCreate(handle, &data));

    cudssMatrix_t matA, matX, matB;
    CUDSS_CHECK(cudssMatrixCreateCsr(&matA, (int64_t)n, (int64_t)n, (int64_t)nnz_upper,
                d_offsets, nullptr, d_columns, d_values,
                CUDA_R_32I, CUDA_C_64F,
                CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));
    CUDSS_CHECK(cudssMatrixCreateDn(&matB, (int64_t)n, 1, (int64_t)n, d_b,
                CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
    CUDSS_CHECK(cudssMatrixCreateDn(&matX, (int64_t)n, 1, (int64_t)n, d_x,
                CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));

    /* CUDA events for timing */
    cudaEvent_t ev_start, ev_analysis, ev_factor, ev_solve;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_analysis));
    CUDA_CHECK(cudaEventCreate(&ev_factor));
    CUDA_CHECK(cudaEventCreate(&ev_solve));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, matA, matX, matB));
    CUDA_CHECK(cudaEventRecord(ev_analysis, stream));

    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matA, matX, matB));
    CUDA_CHECK(cudaEventRecord(ev_factor, stream));

    /* Record peak GPU memory after factorization */
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t mem_peak = gpu_mem_used();

    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matA, matX, matB));
    CUDA_CHECK(cudaEventRecord(ev_solve, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float t_analysis, t_factor, t_solve;
    CUDA_CHECK(cudaEventElapsedTime(&t_analysis, ev_start,    ev_analysis));
    CUDA_CHECK(cudaEventElapsedTime(&t_factor,   ev_analysis, ev_factor));
    CUDA_CHECK(cudaEventElapsedTime(&t_solve,    ev_factor,   ev_solve));

    /* Verify: exact solution should be (1+0i, 1+0i, ...) */
    std::vector<cuDoubleComplex> h_x(n);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err_re = fabs(cuCreal(h_x[i]) - 1.0);
        double err_im = fabs(cuCimag(h_x[i]));
        double err = sqrt(err_re * err_re + err_im * err_im);
        if (err > max_err) max_err = err;
    }

    result.n           = n;
    result.nnz_upper   = nnz_upper;
    result.nnz_full    = nnz_full;
    result.sparsity    = actual_sparsity;
    result.analysis_ms = t_analysis;
    result.factor_ms   = t_factor;
    result.solve_ms    = t_solve;
    result.total_ms    = t_analysis + t_factor + t_solve;
    result.max_error   = max_err;
    result.mem_before_mb = mem_before / (1024 * 1024);
    result.mem_peak_mb   = mem_peak   / (1024 * 1024);
    result.mem_delta_mb  = (mem_peak > mem_before) ? (mem_peak - mem_before) / (1024 * 1024) : 0;

    /* Cleanup */
    CUDSS_CHECK(cudssMatrixDestroy(matA));
    CUDSS_CHECK(cudssMatrixDestroy(matB));
    CUDSS_CHECK(cudssMatrixDestroy(matX));
    CUDSS_CHECK(cudssDataDestroy(handle, data));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_analysis));
    CUDA_CHECK(cudaEventDestroy(ev_factor));
    CUDA_CHECK(cudaEventDestroy(ev_solve));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_offsets); cudaFree(d_columns); cudaFree(d_values);
    cudaFree(d_b); cudaFree(d_x);

    return 0;
}

static void print_table(FILE *fp, const std::vector<BenchResult> &results)
{
    fprintf(fp, "===========================================================================================================================================\n");
    fprintf(fp, "  cuDSS Benchmark Results  (complex fp64, Symmetric, nrhs=1)\n");
    fprintf(fp, "===========================================================================================================================================\n");
    fprintf(fp, "%-8s %10s %12s %10s %13s %13s %13s %13s %10s %10s %10s\n",
            "N", "NNZ(upper)", "NNZ(full)", "Sparsity",
            "Analysis(ms)", "Factor(ms)", "Solve(ms)", "Total(ms)",
            "VRAM(MB)", "Peak(MB)", "MaxError");
    fprintf(fp, "-------------------------------------------------------------------------------------------------------------------------------------------\n");
    for (auto &r : results) {
        fprintf(fp, "%-8d %10d %12d %9.4f%% %13.3f %13.3f %13.3f %13.3f %10zu %10zu %10.2e\n",
                r.n, r.nnz_upper, r.nnz_full, r.sparsity * 100.0,
                r.analysis_ms, r.factor_ms, r.solve_ms, r.total_ms,
                r.mem_delta_mb, r.mem_peak_mb, r.max_error);
    }
    fprintf(fp, "===========================================================================================================================================\n");
}

int main(int argc, char *argv[])
{
    /* Test configurations: (matrix_size, target_sparsity_ratio) */
    struct TestCase { int n; double sparsity; };
    std::vector<TestCase> tests = {
        /* Vary size with fixed sparsity ~1% */
        {  500, 0.01 },
        { 1000, 0.01 },
        { 2000, 0.01 },
        { 5000, 0.01 },
        {10000, 0.01 },

        /* Vary sparsity with fixed size n=1000 */
        { 1000, 0.001 },
        { 1000, 0.005 },
        { 1000, 0.01  },
        { 1000, 0.05  },
        { 1000, 0.10  },
        { 1000, 0.30  },

        /* Vary sparsity with fixed size n=5000 */
        { 5000, 0.001 },
        { 5000, 0.005 },
        { 5000, 0.01  },
        { 5000, 0.05  },

        /* Larger matrices, sparser */
        { 20000, 0.001 },
        { 20000, 0.005 },
        { 50000, 0.001 },
    };

    /* Warm-up */
    {
        BenchResult dummy;
        run_benchmark(64, 0.1, dummy);
    }

    std::vector<BenchResult> results;

    printf("Running cuDSS benchmarks (complex fp64, Symmetric, nrhs=1)...\n\n");

    for (auto &tc : tests) {
        printf("  n = %-8d  sparsity = %-6.1f%% ... ", tc.n, tc.sparsity * 100.0);
        fflush(stdout);
        BenchResult r;
        int ret = run_benchmark(tc.n, tc.sparsity, r);
        if (ret != 0) {
            printf("FAILED\n");
            continue;
        }
        results.push_back(r);
        printf("done  (total = %10.3f ms, VRAM = %zu MB, max_err = %.2e)\n",
               r.total_ms, r.mem_delta_mb, r.max_error);
    }

    /* Print to stdout */
    printf("\n");
    print_table(stdout, results);

    /* Write to file with timestamp */
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char filename[128];
    snprintf(filename, sizeof(filename), "benchmark_results_%04d%02d%02d_%02d%02d%02d.txt",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);
    FILE *fp = fopen(filename, "w");
    if (fp) {
        print_table(fp, results);
        fclose(fp);
        printf("\nResults written to %s\n", filename);
    } else {
        printf("\nError: could not open %s for writing\n", filename);
    }

    return 0;
}
