#include <cuda_runtime.h>
#include <stdio.h>

// Utility function for checking CUDA errors
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Basic kernel that may not fully utilize the GPU
__global__ void basicKernel(float *d_out, float *d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// Optimized kernel aiming for better occupancy
__global__ void optimizedKernel(float *d_out, float *d_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use grid-stride loop to handle multiple elements per thread
    for (; idx < size; idx += gridDim.x * blockDim.x) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

// Function to initialize data
void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(i);
    }
}

// Function to verify results
bool verifyResults(float *output, float *input, int size) {
    for (int i = 0; i < size; i++) {
        if (output[i] != input[i] * 2.0f) {
            printf("Verification failed at index %d!\n", i);
            return false;
        }
    }
    return true;
}

// Function to run and time a kernel
float runKernel(void (*kernel)(float*, float*, int), float *d_out, float *d_in, int size, int gridSize, int blockSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    kernel<<<gridSize, blockSize>>>(d_out, d_in, size);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Benchmark run
    const int numRuns = 100;
    float milliseconds = 0;

    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_out, d_in, size);
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
    const int N = 1000000;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Initialize input data
    initializeData(h_in, N);

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaCheckError();

    // Copy input data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();

    // Basic kernel configuration
    int basicBlockSize = 256;
    int basicGridSize = (N + basicBlockSize - 1) / basicBlockSize;

    // Optimized kernel configuration
    int optBlockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optBlockSize, optimizedKernel, 0, N);
    int optGridSize = (N + optBlockSize - 1) / optBlockSize;

    printf("Basic kernel configuration: gridSize = %d, blockSize = %d\n", basicGridSize, basicBlockSize);
    printf("Optimized kernel configuration: gridSize = %d, blockSize = %d\n", optGridSize, optBlockSize);

    // Run and time basic kernel
    float basicTime = runKernel(basicKernel, d_out, d_in, N, basicGridSize, basicBlockSize);

    // Verify basic kernel results
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool basicResultCorrect = verifyResults(h_out, h_in, N);

    // Run and time optimized kernel
    float optTime = runKernel(optimizedKernel, d_out, d_in, N, optGridSize, optBlockSize);

    // Verify optimized kernel results
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool optResultCorrect = verifyResults(h_out, h_in, N);

    // Print results
    printf("\nResults:\n");
    printf("Basic Kernel:\n");
    printf("  Average execution time: %f ms\n", basicTime);
    printf("  Effective bandwidth: %f GB/s\n", (N * sizeof(float) * 2) / (basicTime * 1e6));
    printf("  Result verification: %s\n", basicResultCorrect ? "PASSED" : "FAILED");

    printf("\nOptimized Kernel:\n");
    printf("  Average execution time: %f ms\n", optTime);
    printf("  Effective bandwidth: %f GB/s\n", (N * sizeof(float) * 2) / (optTime * 1e6));
    printf("  Result verification: %s\n", optResultCorrect ? "PASSED" : "FAILED");

    printf("\nPerformance improvement: %f%%\n", (basicTime - optTime) / basicTime * 100.0f);

    // Free memory
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
