#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#define M 32768
#define K 32768
#define N 32768 // Number of columns in B and C
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define BLOCKSIZE 16
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define check_error(val) check((val), #val, __FILE__, __LINE__) 


template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

const int K9_NUM_THREADS = 256;

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}


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

__global__ void vectorize(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C){
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    int threadCol = threadIdx.x  % ( BN / TN );
    int threadRow = threadIdx.y / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    int innerRowA = threadIdx.x / (BK /4);
    int innerColA = threadIdx.x % (BK /4);
    int innerRowB = threadIdx.x / (BM / 4);
    int innerColB = threadIdx.x % (BM / 4);

    for(uint bkIdx = 0; bkIdx < K; bkIdx += BK){
        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) *BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;


        reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK*N; 

        for(uint dotIdx = 0; dotIdx < BK; ++ dotIdx){
            for(uint i = 0; i < TM; ++i){
                regM[i] = As[(dotIdx) * BM + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
            __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN];
        }
    }

}    

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) matmultiled_2d(int m, int n, int k, float alpha,const float *A, const float *B, float beta, float *C ){
    int tid = (threadIdx.y * blockDim.x) + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    int innerColA = tid % BK;
    int innerRowA = tid / BK;
    
    int strideA = numThreads / BK;

    int innerRowB = tid / BN;
    int innerColB = tid % BN;
    int strideB = numThreads / BN;


    A += (blockIdx.y * BM) * K;  
    B += (blockIdx.x * BN);
    
    float threadResults[TM * TN] = {0.0};
    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    
    // register caches for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
            A[(innerRowA + loadOffset) * K + innerColA];
    }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
            B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

            
    A += BK;        
    B += BK * N;    

    
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint i = 0; i < TM; ++i) {
            regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
        }
        for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
        }
        // perform outer product on register cache, accumulate
        // into threadResults
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[resIdxM * TN + resIdxN] +=
                regM[resIdxM] * regN[resIdxN];
        }
        }
    }
    __syncthreads();
    }

    
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN];
    }
  }

}


__global__ void __launch_bounds__(K9_NUM_THREADS) sgemmAutotuned(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C) {
    
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;


    constexpr int WMITER = CEIL_DIV(BM, WM);
    constexpr int WNITER = CEIL_DIV(BN, WN);


    const int threadCol = threadIdx.x % (WN / TN);
    const int threadRow = threadIdx.x / (WN / TN);


    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];


    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

 

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);


    float threadResults[WMITER * WNITER * TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK){
         for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
    

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB){
        reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = reinterpret_cast<float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();
        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
            // calculate per-thread results
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                                wnIdx * TN + resIdxN] +=
                    regM[resIdxM] * regN[resIdxN];
                }
            }
            }
        }
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * n + threadCol * TN + resIdxN];
    }
  }
}


double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(){
    // init variables   
    float *A, *B, *C;
    float *A_cpu, *B_cpu, *C_cpu;
    float alpha = 1.0f, beta = 0.0f;
    
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    dim3 naiveBlock(16, 16);
    dim3 naiveGrid(CEIL_DIV(N, 16), CEIL_DIV(M, 16));
    
    //create streams 
    cudaStream_t stream1;
    cudaStream_t stream2;

    //host memory
    check_error(cudaMallocHost((void**)&A_cpu, size_A));
    check_error(cudaMallocHost((void**)&B_cpu, size_B));
    check_error(cudaMallocHost((void**)&C_cpu, size_C));

    //init matrix
    srand(time(NULL));
    init_matrix(A_cpu, M, K);
    init_matrix(B_cpu, K, N);

    //device memory
    check_error(cudaMalloc((void **)&A, size_A));
    check_error(cudaMalloc((void **)&B, size_B));
    check_error(cudaMalloc((void **)&C, size_C));
      

    //create streams 
    check_error(cudaStreamCreate(&stream1));
    check_error(cudaStreamCreate(&stream2));

    check_error(cudaMemcpyAsync(A, A_cpu, size_A, cudaMemcpyHostToDevice,stream1));
    check_error(cudaMemcpyAsync(B, B_cpu, size_B, cudaMemcpyHostToDevice,stream2));

    //synchronize streams 
    check_error(cudaStreamSynchronize(stream2));

    //block dims and grid dims
    // 1D Block of threads. 256 threads total.
    dim3 blockDim((BM * BN) / (TM * TN)); 
    // 2D Grid mapping the output matrix
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); 

    vectorize<<<gridDim, blockDim, 0, stream1>>>(M, N, K, alpha, A, B, beta, C);

    //copy from gpu to ram
    check_error(cudaMemcpyAsync(C_cpu, C, size_C, cudaMemcpyDeviceToHost,stream1));

    


    //benchmarking
    printf("Benchmarking vectorized GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vectorize<<<gridDim, blockDim, 0, stream1>>>(M, N, K, alpha, A, B, beta, C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
        check_error(cudaGetLastError());
    }


    printf("Benchmarking Warptailing GPU implementation...\n");
    double w_gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        sgemmAutotuned<<<gridDim, blockDim, 0, stream1>>>(M, N, K, alpha, A, B, beta, C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        w_gpu_total_time += end_time - start_time;
        check_error(cudaGetLastError());
    }

    printf("Benchmarking 2D Blocktilling GPU implementation...\n");
    double two_d_gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmultiled_2d<<<gridDim, blockDim, 0, stream1>>>(M, N, K, alpha, A, B, beta, C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        two_d_gpu_total_time += end_time - start_time;
        check_error(cudaGetLastError());
    }

    // --- BENCHMARK NAIVE KERNEL ---
    printf("Benchmarking Naive GPU implementation...\n");
    double naive_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        // USING THE VARIABLES HERE!
        matmul_naive<<<naiveGrid, naiveBlock, 0, stream1>>>(M, N, K, alpha, A, B, beta, C);
        cudaDeviceSynchronize();
        double end_time = get_time();
        naive_total_time += end_time - start_time;
        check_error(cudaGetLastError());
    }

    double naive_avg_time = naive_total_time / 20.0;
    printf("Naive GPU average time: %f microseconds\n", (naive_avg_time * 1e6f));

    double two_d_gpu_avg_time = two_d_gpu_total_time / 20.0;
    printf("2D Blocktilling GPU average time: %f microseconds\n", (two_d_gpu_avg_time * 1e6f));

    double gpu_avg_time = gpu_total_time / 20.0;
    printf("Vectorrized GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));

    double w_gpu_avg_time = w_gpu_total_time / 20.0;
    printf("Warptailing GPU average time: %f microseconds\n", (w_gpu_avg_time * 1e6f));

    //free memory and destroy streams
    check_error(cudaFree(A));
    check_error(cudaFree(B));
    check_error(cudaFree(C));
    check_error(cudaStreamDestroy(stream1));
    check_error(cudaStreamDestroy(stream2));

    cudaFreeHost(A_cpu);
    cudaFreeHost(B_cpu);
    cudaFreeHost(C_cpu);
}