/**
  Checks memory bandwidth of 
  GPU with respect to global memory
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

__global__ void bandwidth_kernel(float* d_a, float* d_b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        d_b[idx] = 2.0f * d_a[idx];
    }
}

void benchmark_memory(int size_mb)
{
    int n = size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = n * sizeof(float);

    float *h_a, *d_a, *d_b;
    h_a = (float*)malloc(bytes);
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));

    for (int i = 0; i < n; i++)
    {
        h_a[i] = 1.0f;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Warm-up
    bandwidth_kernel<<<gridSize, blockSize>>>(d_a, d_b, n);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    int num_iterations = 100;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++)
    {
        bandwidth_kernel<<<gridSize, blockSize>>>(d_a, d_b, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    float seconds = milliseconds / 1000.0f;
    float bandwidth = 2.0f * bytes * num_iterations / (seconds * 1e9);  // GB/s

    printf("Size: %d MB, Bandwidth: %.2f GB/s\n", size_mb, bandwidth);

    free(h_a);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

int main()
{
    int sizes[] = {1, 10, 100, 1000};  // MB
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Memory Bandwidth Test on GPU\n");
    printf("-----------------------------\n");

    for (int i = 0; i < num_sizes; i++)
    {
        benchmark_memory(sizes[i]);
    }

    return 0;
}
