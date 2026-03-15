# CUDA SGEMM Optimization: From Naive to Warp-Tiling

This repository contains a series of CUDA kernels optimized for Single-precision General Matrix Multiplication (SGEMM). The project documents the performance evolution on an **NVIDIA RTX 3050 (6GB)**, moving from a standard naive approach to advanced tiling techniques that maximize hardware utilization.

---

## 🚀 Performance Overview

The optimization process resulted in a significant reduction in execution time and a substantial increase in compute throughput. At a matrix dimension of **16,384 x 16,384**, the final optimized kernel achieved an **8.3x speedup** over the baseline.

### Key Metrics (RTX 3050)

| Kernel | Execution Time (16k) | Peak Throughput |
| --- | --- | --- |
| **Naive** | 20.91 seconds | ~0.45 TFLOPS |
| **Warp-tailing** | **2.51 seconds** | **3.50 TFLOPS** |

### Benchmark Visualizations

*Comparison of execution time across different matrix dimensions.*

*Compute throughput scaling, reaching a peak of 3.50 TFLOPS.*

---

## 🛠️ Optimization Stages

The project follows a systematic path to eliminate bottlenecks in the GPU pipeline:

### 1. Naive Implementation

The baseline kernel. Each thread calculates one element of the output matrix by reading directly from global memory. High global memory traffic and low data reuse lead to poor performance.

### 2. 2D Block-tiling

Introduces **Shared Memory** to cache data tiles. By loading blocks of matrices A and B into the scratchpad memory, we drastically reduce redundant global memory reads. This stage provides the most significant initial performance boost.

### 3. Vectorized Memory Access

Optimizes memory throughput by using `float4` vectorized loads (`LDG.E.128`). This allows the kernel to fetch 128 bits of data in a single instruction per thread, utilizing the memory controller more efficiently and reducing the total number of instructions issued.

### 4. Warp-level Tiling (Warp-tailing)

The final stage aligns the tiling strategy with the hardware's execution units (Warps). By managing work distribution at the warp level and optimizing register usage, we minimize synchronization overhead (`__syncthreads()`) and maximize the compute-to-memory ratio.

---

## 📂 Code Structure

The core implementation includes:

* **`matmul_naive`**: Baseline global memory implementation.
* **`matmultiled_2d`**: 2D shared memory tiling.
* **`vectorize`**: Tiled implementation using `float4` vectorized memory loads.
* **Benchmarking Suite**: A robust testing framework using `cudaStream_t` and asynchronous memory copies to measure kernel execution time across 20 iterations for accuracy.

### Compilation

To compile the kernel on a system with the CUDA Toolkit installed:

```bash
nvcc -O3 matmul_optimization.cu -o matmul_bench

```

---

## 📚 References & Credits

The optimizations in this repository are based on the excellent technical deep dives from:

* **Simon Boehm**: [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)
* **Elliot Arledge**: [CUDA Tutorial & Technical Analysis](https://www.youtube.com/watch?v=86FAWCzIe_4&t=13982s)

---

