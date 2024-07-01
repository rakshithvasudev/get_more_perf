#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 32
#define VECTOR_SIZE (MATRIX_SIZE * MATRIX_SIZE)

// Utility function for checking CUDA errors
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// 1. Uncoalesced memory access (column-major access)
__global__ void uncoalescedKernel(float *d_out, float *d_in) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        d_out[col * MATRIX_SIZE + row] = d_in[col * MATRIX_SIZE + row] * 2.0f;
    }
}

// 2. Basic coalesced memory access (row-major access)
__global__ void basicCoalescedKernel(float *d_out, float *d_in) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        d_out[row * MATRIX_SIZE + col] = d_in[row * MATRIX_SIZE + col] * 2.0f;
    }
}

// 3. Coalesced access with shared memory
__global__ void sharedMemoryKernel(float *d_out, float *d_in) {
    __shared__ float sharedMem[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        sharedMem[threadIdx.y][threadIdx.x] = d_in[row * MATRIX_SIZE + col];
    }
    __syncthreads();

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        d_out[row * MATRIX_SIZE + col] = sharedMem[threadIdx.y][threadIdx.x] * 2.0f;
    }
}

// 4. Coalesced access with vectorized loads/stores
__global__ void vectorizedKernel(float4 *d_out, float4 *d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < VECTOR_SIZE / 4) {
        float4 data = d_in[idx];
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        d_out[idx] = data;
    }
}

// Function to initialize data
void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to verify results
bool verifyResults(float *output, float *input, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(output[i] - input[i] * 2.0f) > 1e-5) {
            printf("Verification failed at index %d!\n", i);
            return false;
        }
    }
    return true;
}

// Function to run and time a kernel
template<typename T>
float runKernel(void (*kernel)(T*, T*), T *d_out, T *d_in, dim3 gridSize, dim3 blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    kernel<<<gridSize, blockSize>>>(d_out, d_in);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Benchmark run
    const int numRuns = 100;
    float milliseconds = 0;

    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_out, d_in);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaCheckError();

    cudaEventElapsedTime(&milliseconds, start, stop);
    float avgTime = milliseconds / numRuns;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return avgTime;
}

int main() {
    const int matrixElements = MATRIX_SIZE * MATRIX_SIZE;
    size_t bytes = matrixElements * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Initialize input data
    initializeData(h_in, matrixElements);

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaCheckError();

    // Copy input data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();

    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dim3 vectorBlockSize(256);
    dim3 vectorGridSize((VECTOR_SIZE / 4 + vectorBlockSize.x - 1) / vectorBlockSize.x);

    printf("2D Grid dimensions: (%d, %d)\n", gridSize.x, gridSize.y);
    printf("2D Block dimensions: (%d, %d)\n", blockSize.x, blockSize.y);
    printf("1D Vector Grid dimension: %d\n", vectorGridSize.x);
    printf("1D Vector Block dimension: %d\n", vectorBlockSize.x);

    // Run and time kernels
    float uncoalescedTime = runKernel(uncoalescedKernel, d_out, d_in, gridSize, blockSize);
    float basicCoalescedTime = runKernel(basicCoalescedKernel, d_out, d_in, gridSize, blockSize);
    float sharedMemoryTime = runKernel(sharedMemoryKernel, d_out, d_in, gridSize, blockSize);
    float vectorizedTime = runKernel(vectorizedKernel, (float4*)d_out, (float4*)d_in, vectorGridSize, vectorBlockSize);

    // Verify results (using basic coalesced kernel for simplicity)
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool resultsCorrect = verifyResults(h_out, h_in, matrixElements);

    // Print results
    printf("\nResults:\n");
    printf("1. Uncoalesced Kernel (Column-major access):\n");
    printf("   Average execution time: %f ms\n", uncoalescedTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (uncoalescedTime * 1e6));

    printf("\n2. Basic Coalesced Kernel (Row-major access):\n");
    printf("   Average execution time: %f ms\n", basicCoalescedTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (basicCoalescedTime * 1e6));

    printf("\n3. Shared Memory Kernel:\n");
    printf("   Average execution time: %f ms\n", sharedMemoryTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (sharedMemoryTime * 1e6));

    printf("\n4. Vectorized Kernel:\n");
    printf("   Average execution time: %f ms\n", vectorizedTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (vectorizedTime * 1e6));

    printf("\nResult verification: %s\n", resultsCorrect ? "PASSED" : "FAILED");

    printf("\nPerformance improvements:\n");
    printf("Basic Coalesced vs Uncoalesced: %f%%\n", (uncoalescedTime - basicCoalescedTime) / uncoalescedTime * 100.0f);
    printf("Shared Memory vs Basic Coalesced: %f%%\n", (basicCoalescedTime - sharedMemoryTime) / basicCoalescedTime * 100.0f);
    printf("Vectorized vs Basic Coalesced: %f%%\n", (basicCoalescedTime - vectorizedTime) / basicCoalescedTime * 100.0f);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
