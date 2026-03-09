/*
 * cuDSS profiling target: runs a single phase (analysis / factorization / solve)
 * so that ncu can profile each phase independently.
 *
 * Usage: ./profile_cudss <phase>
 *   phase = analysis | factorization | solve | all
 *
 * Matrix: 2000x2000 complex fp64 symmetric, sparsity ~1%
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define CUDSS_CHECK(call)                                                        \
    do {                                                                         \
        cudssStatus_t s = call;                                                  \
        if (s != CUDSS_STATUS_SUCCESS) {                                         \
            fprintf(stderr, "cuDSS error at %s:%d: status = %d\n",               \
                    __FILE__, __LINE__, s);                                       \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

static void generate_symmetric_complex(int n, double sparsity, unsigned int seed,
                                        std::vector<int> &offsets,
                                        std::vector<int> &columns,
                                        std::vector<cuDoubleComplex> &values,
                                        std::vector<cuDoubleComplex> &rhs,
                                        int &nnz_upper)
{
    srand(seed);
    int64_t max_offdiag = (int64_t)n * (n - 1) / 2;
    int64_t target_offdiag = (int64_t)((sparsity * (double)n * n - n) / 2.0);
    if (target_offdiag < 0) target_offdiag = 0;
    if (target_offdiag > max_offdiag) target_offdiag = max_offdiag;

    std::vector<std::vector<int>> row_cols(n);
    for (int i = 0; i < n; i++) row_cols[i].push_back(i);

    double fill_ratio = (max_offdiag > 0) ? (double)target_offdiag / max_offdiag : 0.0;
    if (fill_ratio < 0.5) {
        std::set<int64_t> chosen;
        int64_t attempts = 0, max_attempts = target_offdiag * 10 + 1000;
        while ((int64_t)chosen.size() < target_offdiag && attempts < max_attempts) {
            int i = rand() % n, j = rand() % n;
            if (i >= j) { attempts++; continue; }
            int64_t key = (int64_t)i * n + j;
            if (chosen.insert(key).second) row_cols[i].push_back(j);
            attempts++;
        }
    } else {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if ((rand() / (double)RAND_MAX) < fill_ratio || fill_ratio >= 0.99)
                    row_cols[i].push_back(j);
    }

    for (int i = 0; i < n; i++) std::sort(row_cols[i].begin(), row_cols[i].end());

    int total_nnz = 0;
    for (int i = 0; i < n; i++) total_nnz += (int)row_cols[i].size();
    nnz_upper = total_nnz;

    offsets.resize(n + 1); columns.resize(total_nnz); values.resize(total_nnz);
    int idx = 0;
    for (int i = 0; i < n; i++) {
        offsets[i] = idx;
        for (int k = 0; k < (int)row_cols[i].size(); k++) {
            int j = row_cols[i][k];
            columns[idx] = j;
            if (i == j) {
                values[idx] = make_cuDoubleComplex((double)n + (rand()/(double)RAND_MAX)*10.0,
                                                   (rand()/(double)RAND_MAX)*0.1);
            } else {
                values[idx] = make_cuDoubleComplex((rand()/(double)RAND_MAX)*2.0 - 1.0,
                                                   (rand()/(double)RAND_MAX)*2.0 - 1.0);
            }
            idx++;
        }
    }
    offsets[n] = idx;

    rhs.resize(n);
    std::vector<cuDoubleComplex> b(n, make_cuDoubleComplex(0.0, 0.0));
    for (int i = 0; i < n; i++) {
        for (int k = offsets[i]; k < offsets[i+1]; k++) {
            int j = columns[k];
            cuDoubleComplex v = values[k];
            b[i] = make_cuDoubleComplex(cuCreal(b[i])+cuCreal(v), cuCimag(b[i])+cuCimag(v));
            if (i != j)
                b[j] = make_cuDoubleComplex(cuCreal(b[j])+cuCreal(v), cuCimag(b[j])+cuCimag(v));
        }
    }
    rhs = b;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <analysis|factorization|solve|all>\n", argv[0]);
        return 1;
    }

    const char *phase_str = argv[1];
    bool do_analysis = (strcmp(phase_str, "analysis") == 0 || strcmp(phase_str, "all") == 0);
    bool do_factor   = (strcmp(phase_str, "factorization") == 0 || strcmp(phase_str, "all") == 0);
    bool do_solve    = (strcmp(phase_str, "solve") == 0 || strcmp(phase_str, "all") == 0);

    /* For factorization/solve, analysis must run first (but we skip profiling it) */
    bool need_analysis = do_analysis || do_factor || do_solve;
    bool need_factor   = do_factor || do_solve;

    int n = 2000;
    double sparsity = 0.01;

    printf("Profile target: n=%d, sparsity=%.1f%%, phase=%s\n", n, sparsity*100.0, phase_str);

    std::vector<int> h_offsets, h_columns;
    std::vector<cuDoubleComplex> h_values, h_b;
    int nnz_upper;
    generate_symmetric_complex(n, sparsity, 42, h_offsets, h_columns, h_values, h_b, nnz_upper);

    int *d_offsets, *d_columns;
    cuDoubleComplex *d_values, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc(&d_offsets, (n+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_columns, nnz_upper*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,  nnz_upper*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_x, n*sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_columns, h_columns.data(), nnz_upper*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values,  h_values.data(),  nnz_upper*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudssHandle_t handle;
    CUDSS_CHECK(cudssCreate(&handle));
    CUDSS_CHECK(cudssSetStream(handle, stream));

    cudssConfig_t config;
    cudssData_t data;
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

    /* Always run analysis (prerequisite) */
    if (need_analysis) {
        printf("  Running analysis...\n");
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, matA, matX, matB));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("  Analysis done.\n");
    }

    if (need_factor) {
        printf("  Running factorization...\n");
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matA, matX, matB));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("  Factorization done.\n");
    }

    if (do_solve) {
        printf("  Running solve...\n");
        CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matA, matX, matB));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("  Solve done.\n");
    }

    /* Cleanup */
    CUDSS_CHECK(cudssMatrixDestroy(matA));
    CUDSS_CHECK(cudssMatrixDestroy(matB));
    CUDSS_CHECK(cudssMatrixDestroy(matX));
    CUDSS_CHECK(cudssDataDestroy(handle, data));
    CUDSS_CHECK(cudssConfigDestroy(config));
    CUDSS_CHECK(cudssDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_offsets); cudaFree(d_columns); cudaFree(d_values);
    cudaFree(d_b); cudaFree(d_x);

    printf("Done.\n");
    return 0;
}
