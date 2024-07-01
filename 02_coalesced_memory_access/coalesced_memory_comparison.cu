#include <cuda_runtime.h>
#include <stdio.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 32

// Utility function for checking CUDA errors
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Uncoalesced memory access kernel (column-major access)
__global__ void uncoalescedKernel(float *d_out, float *d_in) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        // Column-major access pattern (uncoalesced)
        d_out[col * MATRIX_SIZE + row] = d_in[col * MATRIX_SIZE + row] * 2.0f;
    }
}

// Coalesced memory access kernel (row-major access)
__global__ void coalescedKernel(float *d_out, float *d_in) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        // Row-major access pattern (coalesced)
        d_out[row * MATRIX_SIZE + col] = d_in[row * MATRIX_SIZE + col] * 2.0f;
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
float runKernel(void (*kernel)(float*, float*), float *d_out, float *d_in, dim3 gridSize, dim3 blockSize) {
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

    printf("Grid dimensions: (%d, %d)\n", gridSize.x, gridSize.y);
    printf("Block dimensions: (%d, %d)\n", blockSize.x, blockSize.y);

    // Run and time uncoalesced kernel
    float uncoalescedTime = runKernel(uncoalescedKernel, d_out, d_in, gridSize, blockSize);

    // Verify uncoalesced kernel results
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool uncoalescedResultCorrect = verifyResults(h_out, h_in, matrixElements);

    // Run and time coalesced kernel
    float coalescedTime = runKernel(coalescedKernel, d_out, d_in, gridSize, blockSize);

    // Verify coalesced kernel results
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool coalescedResultCorrect = verifyResults(h_out, h_in, matrixElements);

    // Print results
    printf("\nResults:\n");
    printf("Uncoalesced Kernel (Column-major access):\n");
    printf("  Average execution time: %f ms\n", uncoalescedTime);
    printf("  Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (uncoalescedTime * 1e6));
    printf("  Result verification: %s\n", uncoalescedResultCorrect ? "PASSED" : "FAILED");

    printf("\nCoalesced Kernel (Row-major access):\n");
    printf("  Average execution time: %f ms\n", coalescedTime);
    printf("  Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (coalescedTime * 1e6));
    printf("  Result verification: %s\n", coalescedResultCorrect ? "PASSED" : "FAILED");

    printf("\nPerformance improvement: %f%%\n", (uncoalescedTime - coalescedTime) / uncoalescedTime * 100.0f);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
