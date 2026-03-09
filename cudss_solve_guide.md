# cuDSS 稀疏直接求解流程详解

本文档基于 `benchmark.cpp` 中的代码，介绍 cuDSS 求解稀疏线性方程组 `Ax = b` 的完整流程。

## 1. 问题背景

cuDSS (CUDA Direct Sparse Solver) 是 NVIDIA 提供的 GPU 稀疏直接求解器。
与迭代法不同，直接法通过矩阵分解精确求解（受浮点精度限制），适用于：

- 多次求解同一结构不同数值的线性系统（分解可复用）
- 迭代法难以收敛的病态问题
- 需要高精度解的场景

## 2. 矩阵类型与分解方法

cuDSS 根据矩阵类型自动选择分解策略：

| 矩阵类型 (cudssMatrixType_t) | 分解方法 | 说明 |
|------------------------------|---------|------|
| `CUDSS_MTYPE_GENERAL`       | PA = LU | 一般矩阵，带行置换的 LU 分解 |
| `CUDSS_MTYPE_SYMMETRIC`     | A = LDL^T | 对称矩阵（A = A^T），LDL^T 分解 |
| `CUDSS_MTYPE_HERMITIAN`     | A = LDL^H | Hermitian 矩阵（A = A^H），LDL^H 分解 |
| `CUDSS_MTYPE_SPD`           | A = LL^T (Cholesky) | 对称正定矩阵，Cholesky 分解 |
| `CUDSS_MTYPE_HPD`           | A = LL^H | Hermitian 正定矩阵 |

**本 benchmark 使用 `CUDSS_MTYPE_SYMMETRIC` + `CUDA_C_64F`（complex fp64），
因此 cuDSS 执行的是复数对称矩阵的 LDL^T 分解，而非 LU 分解。**

若要测试真正的 LU 分解，需将矩阵类型改为 `CUDSS_MTYPE_GENERAL` 并存储完整矩阵。

## 3. 三阶段求解流程

cuDSS 将求解过程分为三个阶段，对应 `cudssExecute` 的三次调用：

### 3.1 Analysis（符号分析）— `CUDSS_PHASE_ANALYSIS`

```cpp
cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, matA, matX, matB);
```

**功能**：仅根据矩阵的稀疏结构（非零元位置），不涉及具体数值，完成：

1. **填充元(fill-in)预测**：分解过程中原本为零的位置可能变为非零，称为 fill-in。
   分析阶段确定 L 因子的非零结构。
2. **重排序(reordering)**：通过行列置换减少 fill-in，常用算法包括 METIS、AMD 等。
   重排序质量直接影响后续分解的计算量和内存消耗。
3. **超节点(supernode)检测**：识别消去树中结构相似的列，合并为超节点，
   使后续分解可利用 dense BLAS（如 cuBLAS）加速。
4. **内存预分配**：根据分析结果预分配 factorization 所需的 GPU 内存。

**特点**：
- 这是 CPU 密集阶段（大量图算法在 host 侧执行），从 benchmark 数据看耗时与 n 成正比。
- 若矩阵结构不变仅数值变化，此阶段只需执行一次。

### 3.2 Factorization（数值分解）— `CUDSS_PHASE_FACTORIZATION`

```cpp
cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, matA, matX, matB);
```

**功能**：基于分析阶段确定的非零结构，执行实际的数值分解。

对于本 benchmark 的 `CUDSS_MTYPE_SYMMETRIC`：
- 计算 A = LDL^T，其中 L 是下三角矩阵，D 是分块对角矩阵（1×1 和 2×2 块）。
- 2×2 块用于处理数值不稳定的 pivot（Bunch-Kaufman 类策略）。

**GPU 加速核心**：
- 超节点内部的密集运算通过 cuBLAS 的 `gemm`/`syrk`/`trsm` 在 GPU 上执行。
- 这也是 benchmark 中 cuDSS 依赖 cublas 库的原因。
- 从 benchmark 数据看，这是大矩阵场景下最耗时的阶段（n=20000 时占 95%+ 时间）。

**显存消耗**：
- 因子 L 的存储是主要的显存消耗来源。
- fill-in 越多，L 越密集，显存消耗越大。
- 从 benchmark 数据：n=10000 sparsity=1% 时消耗 ~1.5GB，n=50000 sparsity=0.1% 消耗 ~29GB。

### 3.3 Solve（三角求解）— `CUDSS_PHASE_SOLVE`

```cpp
cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, matA, matX, matB);
```

**功能**：利用已有的分解结果，通过前代/回代求解 Ax = b：

对于 LDL^T 分解（对称情况）：
1. 前代求解 Ly = Pb（L 为下三角，P 为置换）
2. 对角求解 Dz = y
3. 回代求解 L^T w = z
4. 反置换 x = P^T w

**特点**：
- 这是最快的阶段，即使 n=50000 也仅需 ~48ms。
- 若需求解多个不同的 b，只需重复此阶段。
- 计算量与 L 的非零元数成正比，远小于 factorization。

## 4. 数据结构

### CSR 格式

矩阵 A 以 CSR (Compressed Sparse Row) 格式存储：

```cpp
cudssMatrixCreateCsr(&matA, n, n, nnz_upper,
    d_offsets,    // int[n+1]: 每行起始偏移
    nullptr,      // end offsets (nullptr = 使用 offsets[i+1])
    d_columns,    // int[nnz]: 列索引
    d_values,     // cuDoubleComplex[nnz]: 非零值
    CUDA_R_32I,   // 索引类型: 32位整数
    CUDA_C_64F,   // 值类型: complex fp64
    CUDSS_MTYPE_SYMMETRIC,  // 对称矩阵
    CUDSS_MVIEW_UPPER,      // 仅存储上三角
    CUDSS_BASE_ZERO);       // 0-based 索引
```

对称矩阵只需存储上三角（`CUDSS_MVIEW_UPPER`），cuDSS 内部自动处理对称性。

### 稠密向量

右端项 b 和解 x 以列主序稠密矩阵存储：

```cpp
cudssMatrixCreateDn(&matB, n, nrhs, ldb, d_b, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR);
```

## 5. 关键对象的生命周期

```
cudssCreate(&handle)              // 创建库句柄
  cudssConfigCreate(&config)      // 创建求解器配置
  cudssDataCreate(handle, &data)  // 创建求解器数据（存储分解结果）
    cudssExecute(ANALYSIS)        // 分析
    cudssExecute(FACTORIZATION)   // 分解（可对不同数值重复调用）
    cudssExecute(SOLVE)           // 求解（可对不同 b 重复调用）
  cudssDataDestroy(handle, data)
  cudssConfigDestroy(config)
cudssDestroy(handle)
```

`cudssData_t` 持有分解的全部中间数据（消去树、因子等），是显存消耗的主体。

## 6. Benchmark 数据解读

从 benchmark 结果可以看到几个典型规律：

**Analysis 阶段**：耗时与 n 近似线性，与 nnz 关系不大（因为主要是图结构算法）。

**Factorization 阶段**：耗时增长最快，因为超节点内的密集计算量与 fill-in 的立方相关。
n=20000 sparsity=0.5% 时 factor 耗时 26s，是 analysis 的 31 倍。

**Solve 阶段**：始终最快，与 factor 的比值通常在 100:1 以上。

**显存**：主要由 factorization 的 fill-in 决定。同样 n=5000，
sparsity 从 0.1% 到 5% 时显存从 96MB 增长到 474MB。
