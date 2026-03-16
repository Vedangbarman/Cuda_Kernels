# CUDA SGEMM Optimization: From Naive to Warp-Tiling

This repository contains a series of CUDA kernels optimized for Single-precision General Matrix Multiplication (SGEMM). The project documents the performance evolution on an **NVIDIA RTX 3050 (6GB)**, moving from a standard naive approach to advanced tiling and double-buffering techniques that maximize hardware utilization.

---

## 🚀 Performance Overview

The optimization process resulted in a massive reduction in execution time and a substantial increase in compute throughput. At a matrix dimension of **16,384 x 16,384**, the `Warptailing` kernel achieved an **8.3x speedup** over the baseline, and our `Double Buffering` implementation peaked at **3.88 TFLOPS** (at N=2048).

### 1. Custom Kernel Progression (Execution Time in µs)

This table tracks the performance of our hand-written kernels as we systematically removed bottlenecks. 

| Matrix Size | Naive | 2D Blocktiling | Vectorized | Warptailing |
| :--- | :--- | :--- | :--- | :--- |
| **256** | 401.30 | 339.81 | 327.05 | **270.51** |
| **512** | 951.68 | 548.32 | **254.00** | 267.41 |
| **1024** | 5,799.17 | 1,893.86 | 1,045.41 | **1,025.41** |
| **2048** | 37,181.13 | 10,046.31 | 6,507.35 | **6,143.37** |
| **4096** | 300,097.21 | 66,804.76 | 41,442.99 | **38,015.53** |
| **8192** | 2,612,513.94 | 563,063.18 | 323,278.86 | **311,290.23** |
| **16384** | 20,918,648.31 | 4,396,641.47 | 2,717,967.21 | **2,515,279.35** |

*Note: Execution time measured in microseconds (µs).*

### 2. cuBLAS Face-off & Double Buffering (Execution Time in µs)

This table compares our best Custom Double Buffering (DBuff) kernel against NVIDIA's proprietary `cuBLAS` library for both Single-Precision (SGEMM) and Half-Precision (HGEMM FP16) operations.

| Size | Naive (Baseline) | DBuff Kernel | cuBLAS SGEMM | cuBLAS HGEMM (FP16) | DBuff GFLOPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **256** | 171.50 | 134.04 | 129.67 | 70.09 | 250.3 |
| **512** | 972.98 | 350.35 | 163.85 | 87.70 | 766.2 |
| **1024** | 7,040.10 | 1,018.54 | 753.19 | 219.56 | 2,108.4 |
| **2048** | 40,604.76 | 4,419.68 | 4,587.54 | 1,156.75 | **3,887.1** |
| **4096** | 296,082.34 | 37,114.74 | 35,776.80 | 6,351.98 | 3,703.1 |
| **8192** | 2,444,402.00 | 299,730.25 | 275,095.86 | 49,005.49 | 3,668.3 |
| **16384** | 22,368,427.80 | 2,663,913.65 | 2,399,587.84 | 530,426.26 | 3,301.9 |

*(Note: The DBuff Kernel stays highly competitive with cuBLAS SGEMM, even beating it slightly at N=2048!)*

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
Aligns the tiling strategy with the hardware's execution units (Warps). By managing work distribution at the warp level and optimizing register usage, we minimize synchronization overhead (`__syncthreads()`) and maximize the compute-to-memory ratio.

### 5. Double Buffering (DBuff)
Hides memory latency by overlapping data fetching with computation. While the GPU computes the dot products for the current tile in registers, it simultaneously pre-fetches the next tile from global memory into a secondary shared memory buffer. 

---

## 📂 Code Structure

The core implementation includes:

* **`matmul_naive`**: Baseline global memory implementation.
* **`matmultiled_2d`**: 2D shared memory tiling.
* **`vectorize`**: Tiled implementation using `float4` vectorized memory loads.
* **`sgemmDoubleBuffered`**: Advanced double buffering implementation bridging the gap to cuBLAS.
* **Benchmarking Suite**: A robust testing framework measuring average kernel execution time across multiple iterations to ensure accuracy.

### Compilation

To compile the auto-tuned kernel on an Ampere architecture system (like the RTX 3050):

```bash
nvcc autotuned.cu -o sgemm -O3 -arch=sm_86
./sgemm
