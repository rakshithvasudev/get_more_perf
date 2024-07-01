#include <stdio.h>
#include <cuda.h>
#include <chrono>

#define MATRIX_SIZE 1024
#define TILE_WIDTH 16

__global__ void matrixMulNaive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__ void matrixMulShared(float *A, float *B, float *C, int N) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0f;
    for (int m = 0; m < (N / TILE_WIDTH); ++m) {
        sharedA[ty][tx] = A[row * N + (m * TILE_WIDTH + tx)];
        sharedB[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

void initializeMatrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Your main function code
    float *A, *B, *C_naive, *C_shared;
    float *d_A, *d_B, *d_C_naive, *d_C_shared;
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C_naive = (float *)malloc(size);
    C_shared = (float *)malloc(size);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_naive, size);
    cudaMalloc(&d_C_shared, size);
    
    initializeMatrix(A, MATRIX_SIZE);
    initializeMatrix(B, MATRIX_SIZE);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(MATRIX_SIZE / TILE_WIDTH, MATRIX_SIZE / TILE_WIDTH);
    
    auto start_naive = std::chrono::high_resolution_clock::now();
    matrixMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_naive, MATRIX_SIZE);
    cudaDeviceSynchronize();
    auto end_naive = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_naive = end_naive - start_naive;
    
    auto start_shared = std::chrono::high_resolution_clock::now();
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_shared, MATRIX_SIZE);
    cudaDeviceSynchronize();
    auto end_shared = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_shared = end_shared - start_shared;
    
    cudaMemcpy(C_naive, d_C_naive, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(C_shared, d_C_shared, size, cudaMemcpyDeviceToHost);
    
    printf("Naive implementation time: %f seconds\n", duration_naive.count());
    printf("Shared memory implementation time: %f seconds\n", duration_shared.count());
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_shared);
    free(A);
    free(B);
    free(C_naive);
    free(C_shared);
    
    return 0;
}
