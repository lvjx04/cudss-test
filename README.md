# cuDSS Benchmark & Profiling Suite

> GPU 稀疏直接求解器 [cuDSS](https://docs.nvidia.com/cuda/cudss/index.html) 的性能测试与 profiling 工具集。

## 功能概览

- **benchmark** — 对不同矩阵规模和稀疏度进行自动化计时，记录各阶段耗时、显存消耗和求解精度
- **dense_vs_sparse** — 稠密求解器 (PyTorch `torch.linalg.solve`) 与稀疏求解器 (cuDSS) 的性能/显存/精度对比
- **profile** — 使用 NVIDIA Nsight Compute (ncu) 对三个求解阶段分别采集 SASS 汇编和性能报告
- **simple** — cuDSS 官方示例：5×5 SPD 矩阵求解

## 环境要求

- CUDA Toolkit ≥ 12.x
- cuDSS ≥ 0.6.0（通过 `find_package(cudss)` 查找）
- CMake ≥ 3.19
- NVIDIA GPU（建议显存 ≥ 8GB，大规模测试需更多）
- Nsight Compute (`ncu`)（仅 profiling 需要）
- Python 3 + PyTorch with CUDA（仅 dense vs sparse 对比需要，`conda activate hpc-ops`）

## 快速开始

### 构建

```bash
cmake -B build -S .
cmake --build build -j$(nproc)
```

如果系统上同时存在多个 cuDSS 版本（如 CUDA 12 和 13 共存），需显式指定：

```bash
cmake -B build -S . -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss
```

### 运行 Benchmark

```bash
# 一键构建并运行
./run_benchmark.sh

# 或手动运行
./build/benchmark
```

结果自动写入带时间戳的文件，如 `benchmark_results_20260309_152417.txt`。

### 运行 NCU Profiling

```bash
./run_profile.sh
```

输出目录 `ncu_results_<timestamp>/`，每个阶段生成：

| 文件 | 内容 |
|------|------|
| `<phase>_sass.txt` | SASS 汇编代码（GPU 指令级） |
| `<phase>_report.ncu-rep` | NCU 完整报告（用 Nsight Compute UI 打开） |
| `<phase>_summary.csv` | Kernel 指标 CSV 摘要 |

### 运行 Dense vs Sparse 对比

```bash
conda activate hpc-ops
python dense_vs_sparse.py
```

需要先运行过 `./run_benchmark.sh` 生成 cuDSS 结果文件，脚本会自动读取最新的 `benchmark_results_*.txt` 与 PyTorch dense 求解进行对比。结果写入 `dense_vs_sparse_<timestamp>.txt`。

### 运行原始示例

```bash
./build/simple_example
```

## Benchmark 测试配置

测试矩阵为随机生成的 **复数 fp64 对称矩阵**（`CUDSS_MTYPE_SYMMETRIC` + `CUDA_C_64F`），CSR 格式存储上三角。

| 测试维度 | 配置 |
|---------|------|
| 固定稀疏度变大小 | n = 500 ~ 50000, sparsity = 1% |
| 固定大小变稀疏度 (n=1000) | sparsity = 0.1% ~ 30% |
| 固定大小变稀疏度 (n=5000) | sparsity = 0.1% ~ 5% |
| 大矩阵稀疏 | n = 20000 ~ 50000, sparsity = 0.1% ~ 0.5% |

## 示例结果

> 以下数据在单卡 GPU 上测得，仅供参考。

**矩阵大小 vs 耗时 (sparsity=1%)**

| N | NNZ(full) | Analysis | Factor | Solve | Total | VRAM |
|---|-----------|----------|--------|-------|-------|------|
| 500 | 2,500 | 10.9 ms | 2.0 ms | 0.4 ms | 13.3 ms | 68 MB |
| 1,000 | 10,000 | 16.5 ms | 11.2 ms | 1.6 ms | 29.4 ms | 72 MB |
| 5,000 | 250,000 | 101 ms | 211 ms | 7.0 ms | 319 ms | 392 MB |
| 10,000 | 1,000,000 | 328 ms | 3,116 ms | 10.3 ms | 3,454 ms | 1.4 GB |

**稀疏度 vs 耗时 (n=5000)**

| Sparsity | NNZ(full) | Analysis | Factor | Solve | Total | VRAM |
|----------|-----------|----------|--------|-------|-------|------|
| 0.1% | 25,000 | 44 ms | 15 ms | 1.4 ms | 61 ms | 96 MB |
| 0.5% | 125,000 | 78 ms | 137 ms | 3.6 ms | 219 ms | 318 MB |
| 1.0% | 250,000 | 98 ms | 204 ms | 4.7 ms | 306 ms | 390 MB |
| 5.0% | 1,250,000 | 201 ms | 333 ms | 4.5 ms | 538 ms | 474 MB |

## Dense vs Sparse 对比

> 测试环境：NVIDIA H20, PyTorch 2.10.0+cu128, cuDSS 0.7.1

| N | Sparsity | Dense (PyTorch) | Sparse (cuDSS) | Speedup | Dense Mem | Sparse Mem | 备注 |
|---|----------|----------------|----------------|---------|-----------|------------|------|
| 1,000 | 0.1% | 8.9 ms | 9.8 ms | 0.91x | 15 MB | 70 MB | 接近持平 |
| 1,000 | 1.0% | 8.9 ms | 35 ms | 0.25x | 15 MB | 72 MB | dense 更快 |
| 5,000 | 0.1% | 51 ms | 61 ms | 0.83x | 382 MB | 96 MB | 接近持平，sparse 更省显存 |
| 5,000 | 1.0% | 51 ms | 306 ms | 0.17x | 382 MB | 390 MB | dense 更快 |
| 10,000 | 1.0% | 126 ms | 3,454 ms | 0.04x | 1.5 GB | 1.4 GB | dense 更快 |
| 20,000 | 0.1% | 314 ms | 11,921 ms | 0.03x | 6.0 GB | 3.3 GB | dense 更快，sparse 更省显存 |
| 50,000 | 0.1% | **OOM** | 342,575 ms | - | 37 GB | 29 GB | dense 无法运行 |

**结论：**

- **小规模（n ≤ 10000）**：PyTorch 的 dense solver（cuSOLVER/cuBLAS 后端）全面更快。dense solver 的 O(n³) 计算在 GPU 上被 BLAS 高度优化，而 cuDSS 的 analysis 阶段有固定的 CPU 开销。
- **大规模极稀疏**：dense solver 在 n=50000 时 OOM（需 37GB 仅矩阵存储），而 sparse solver 仍可运行。这是稀疏求解器的核心价值——处理 dense solver 无法触及的问题规模。
- **精度**：cuDSS sparse solver 精度更高（1e-14 ~ 1e-16），dense solver 在矩阵变大/稀疏度增加时精度下降（1e-4 ~ 1e-8），因为条件数增大。
- **显存**：n ≤ 5000 极稀疏时 sparse 显著省内存；高稀疏度大矩阵时两者接近。

## cuDSS 求解流程

cuDSS 将稀疏直接求解分为三个阶段：

```
Analysis (符号分析)  →  Factorization (数值分解)  →  Solve (三角求解)
```

1. **Analysis** — 基于稀疏结构做重排序、fill-in 预测和超节点检测，CPU 密集
2. **Factorization** — 在 GPU 上执行数值分解（对称矩阵: LDL^T），调用 cuBLAS 加速密集子块运算，是大矩阵场景的主要瓶颈
3. **Solve** — 利用分解结果做前代/回代求解，速度最快

详见 [cudss_solve_guide.md](cudss_solve_guide.md)。

## 项目结构

```
├── CMakeLists.txt          # 构建配置（simple_example / benchmark / profile_cudss）
├── simple.cpp              # cuDSS 官方 5×5 SPD 示例
├── benchmark.cpp           # 多规模/多稀疏度 benchmark（含显存记录和时间戳输出）
├── dense_vs_sparse.py      # Dense (PyTorch) vs Sparse (cuDSS) 对比脚本
├── profile_cudss.cpp       # NCU profiling 目标程序（支持按阶段运行）
├── run_benchmark.sh        # 一键构建并运行 benchmark
├── run_profile.sh          # 一键构建并运行 NCU profile（三阶段）
├── cudss_solve_guide.md    # cuDSS 求解流程技术文档
└── CLAUDE.md               # Claude Code 辅助指引
```

## License

示例代码基于 NVIDIA cuDSS samples，受 NVIDIA 软件许可协议约束。详见源文件中的版权声明。
