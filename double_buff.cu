#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", 
                file, line, (uint)err, cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Kernels ---

__global__ void matmul_naive(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = alpha * sum + beta * C[row * n + col];
    }
}

__global__ void __launch_bounds__(256) sgemmDoubleBuffered(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    const uint tid = threadIdx.x;
    const uint threadCol = tid % (BN / TN);
    const uint threadRow = tid / (BN / TN);

    __shared__ __align__(16) float As[2][BK * BM];
    __shared__ __align__(16) float Bs[2][BK * BN];

    A += blockIdx.y * BM * k;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * n + blockIdx.x * BN;

    const uint iRA = tid / (BK / 4);
    const uint iCA = tid % (BK / 4);
    const uint iRB = tid / (BN / 4);
    const uint iCB = tid % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (uint offset = 0; offset < BM; offset += (256 * 4 / BK)) {
        float4 tmp = reinterpret_cast<float4 *>(&A[(iRA + offset) * k + iCA * 4])[0];
        As[0][(iCA * 4 + 0) * BM + iRA + offset] = tmp.x;
        As[0][(iCA * 4 + 1) * BM + iRA + offset] = tmp.y;
        As[0][(iCA * 4 + 2) * BM + iRA + offset] = tmp.z;
        As[0][(iCA * 4 + 3) * BM + iRA + offset] = tmp.w;
    }
    
    for (uint offset = 0; offset < BK; offset += (256 / (BN / 4))) {
        reinterpret_cast<float4 *>(&Bs[0][(iRB + offset) * BN + iCB * 4])[0] = 
            reinterpret_cast<float4 *>(&B[(iRB + offset) * n + iCB * 4])[0];
    }
    __syncthreads();

    int writeIdx = 1;
    int readIdx = 0;
    
    for (uint bkIdx = BK; bkIdx < k; bkIdx += BK) {
        A += BK; 
        B += BK * n;
        
        for (uint offset = 0; offset < BM; offset += (256 * 4 / BK)) {
            float4 tmp = reinterpret_cast<float4 *>(&A[(iRA + offset) * k + iCA * 4])[0];
            As[writeIdx][(iCA * 4 + 0) * BM + iRA + offset] = tmp.x;
            As[writeIdx][(iCA * 4 + 1) * BM + iRA + offset] = tmp.y;
            As[writeIdx][(iCA * 4 + 2) * BM + iRA + offset] = tmp.z;
            As[writeIdx][(iCA * 4 + 3) * BM + iRA + offset] = tmp.w;
        }
        
        for (uint offset = 0; offset < BK; offset += (256 / (BN / 4))) {
            reinterpret_cast<float4 *>(&Bs[writeIdx][(iRB + offset) * BN + iCB * 4])[0] = 
                reinterpret_cast<float4 *>(&B[(iRB + offset) * n + iCB * 4])[0];
        }
        
        #pragma unroll
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) regM[i] = As[readIdx][dotIdx * BM + threadRow * TM + i];
            for (uint i = 0; i < TN; ++i) regN[i] = Bs[readIdx][dotIdx * BN + threadCol * TN + i];
            
            for (uint iM = 0; iM < TM; ++iM) {
                for (uint iN = 0; iN < TN; ++iN) {
                    threadResults[iM * TN + iN] += regM[iM] * regN[iN];
                }
            }
        }
        __syncthreads();
        readIdx = writeIdx; 
        writeIdx = 1 - writeIdx;
    }
    
    #pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint i = 0; i < TM; ++i) regM[i] = As[readIdx][dotIdx * BM + threadRow * TM + i];
        for (uint i = 0; i < TN; ++i) regN[i] = Bs[readIdx][dotIdx * BN + threadCol * TN + i];
        
        for (uint iM = 0; iM < TM; ++iM) {
            for (uint iN = 0; iN < TN; ++iN) {
                threadResults[iM * TN + iN] += regM[iM] * regN[iN];
            }
        }
    }
    
    for (uint iM = 0; iM < TM; ++iM) {
        for (uint iN = 0; iN < TN; ++iN) {
            int cRow = threadRow * TM + iM;
            int cCol = threadCol * TN + iN;
            C[cRow * n + cCol] = alpha * threadResults[iM * TN + iN] + beta * C[cRow * n + cCol];
        }
    }
}

// --- Benchmark Logic ---

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void warmup(cublasHandle_t handle) {
    int size = 512;
    float *d_A; 
    half *d_Ah;
    
    CHECK_CUDA(cudaMalloc(&d_A, size * size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ah, size * size * sizeof(half)));
    
    float alpha = 1.0f, beta = 0.0f;
    half al_h = __float2half(1.0f), be_h = __float2half(0.0f);

    for(int i = 0; i < 20; i++) {
        matmul_naive<<<dim3(32, 32), dim3(16, 16)>>>(size, size, size, alpha, d_A, d_A, beta, d_A);
        sgemmDoubleBuffered<<<dim3(size/BN, size/BM), 256>>>(size, size, size, alpha, d_A, d_A, beta, d_A);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, d_A, size, d_A, size, &beta, d_A, size);
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &al_h, d_Ah, size, d_Ah, size, &be_h, d_Ah, size);
    }
    
    cudaDeviceSynchronize();
    cudaFree(d_A); 
    cudaFree(d_Ah);
}

void run_benchmark(int size, cublasHandle_t handle) {
    float alpha = 1.0f, beta = 0.0f;
    size_t bytes_f = (size_t)size * size * sizeof(float);
    size_t bytes_h = (size_t)size * size * sizeof(half);

    float *d_A, *d_B, *d_C;
    half *h_Ah, *h_Bh, *d_Ah, *d_Bh, *d_Ch;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_f)); 
    CHECK_CUDA(cudaMalloc(&d_B, bytes_f)); 
    CHECK_CUDA(cudaMalloc(&d_C, bytes_f));
    
    CHECK_CUDA(cudaMalloc(&d_Ah, bytes_h)); 
    CHECK_CUDA(cudaMalloc(&d_Bh, bytes_h)); 
    CHECK_CUDA(cudaMalloc(&d_Ch, bytes_h));
    
    h_Ah = (half*)malloc(bytes_h); 
    h_Bh = (half*)malloc(bytes_h);
    
    for (int i = 0; i < size * size; i++) { 
        h_Ah[i] = __float2half(0.1f); 
        h_Bh[i] = __float2half(0.1f); 
    }
    
    CHECK_CUDA(cudaMemcpy(d_Ah, h_Ah, bytes_h, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Bh, h_Bh, bytes_h, cudaMemcpyHostToDevice));

    double t_naive = 0, t_wrap = 0, t_sgemm = 0, t_hgemm = 0;
    const int iter = 10;

    for(int i = 0; i < iter; i++) { 
        double s = get_time(); 
        dim3 grid(CEIL_DIV(size, 16), CEIL_DIV(size, 16));
        dim3 block(16, 16);
        matmul_naive<<<grid, block>>>(size, size, size, alpha, d_A, d_B, beta, d_C); 
        cudaDeviceSynchronize(); 
        t_naive += get_time() - s; 
    }
    
    for(int i = 0; i < iter; i++) { 
        double s = get_time(); 
        dim3 grid(CEIL_DIV(size, BN), CEIL_DIV(size, BM));
        sgemmDoubleBuffered<<<grid, 256>>>(size, size, size, alpha, d_A, d_B, beta, d_C); 
        cudaDeviceSynchronize(); 
        t_wrap += get_time() - s; 
    }
    
    for(int i = 0; i < iter; i++) { 
        double s = get_time(); 
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, d_B, size, d_A, size, &beta, d_C, size)); 
        cudaDeviceSynchronize(); 
        t_sgemm += get_time() - s; 
    }
    
    half al_h = __float2half(1.0f), be_h = __float2half(0.0f);
    for(int i = 0; i < iter; i++) { 
        double s = get_time(); 
        CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &al_h, d_Bh, size, d_Ah, size, &be_h, d_Ch, size)); 
        cudaDeviceSynchronize(); 
        t_hgemm += get_time() - s; 
    }

    double gflops_wrap = (2.0 * size * size * size / ((t_wrap / iter) * 1e9));

    printf("%-7d | %-12.2f | %-12.2f | %-12.2f | %-12.2f | %-8.1f\n", 
           size, (t_naive/iter)*1e6, (t_wrap/iter)*1e6, (t_sgemm/iter)*1e6, (t_hgemm/iter)*1e6, gflops_wrap);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); 
    cudaFree(d_Ah); cudaFree(d_Bh); cudaFree(d_Ch);
    free(h_Ah); free(h_Bh);
}

int main() {
    cublasHandle_t handle; 
    CHECK_CUBLAS(cublasCreate(&handle));
    
    warmup(handle);
    
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    printf("%-7s | %-12s | %-12s | %-12s | %-12s | %-8s\n", 
           "Size", "Naive(us)", "DBuff(us)", "SGEMM(us)", "HGEMM(us)", "DB GFLOPS");
    printf("------------------------------------------------------------------------------------\n");
    
    for(int i = 0; i < 7; i++) {
        run_benchmark(sizes[i], handle);
    }
    
    cublasDestroy(handle);
    return 0;
}
