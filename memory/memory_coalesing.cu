#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

/*
Memory coalescing is a technique used in GPU programming to optimize the memory access pattern of threads. 
The goal is to combine multiple memory access requests into a single transaction, 
reducing the number of memory transactions and increasing the overall memory bandwidth utilization.
This is particularly important in GPU architectures, where the memory bandwidth can be a critical bottleneck.

Autoregressive decoder LLMs are memory bandwidth limited. 

In a typical GPU, a warp consists of 32 threads. If memory accesses within a warp are not coalesced, each thread in the warp might generate a separate memory transaction. This can result in up to 32 transactions for a single warp, significantly reducing performance.

If memory accesses are coalesced, the entire warp can access memory in a single transaction (or a few large transactions), drastically improving memory access efficiency.

Practical Implications:
Performance Tuning: When optimizing GPU code, always check the memory access patterns. Tools like NVIDIA Nsight can help profile memory accesses.
Algorithm Design: Design algorithms that naturally align data in memory for coalesced access. This often involves rethinking data structures and access patterns.
Kernel Optimization: Use techniques like loop unrolling and thread-block reordering to ensure contiguous memory access within warps.
By understanding and applying memory coalescing principles, you can significantly improve the performance of GPU-accelerated applications.
*/

// Kernel for uncoalesced memory access
__global__ void uncoalesced_memory_access(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        //output[idx] = input[2 * idx];  // Non-contiguous access
        output[idx] = input[2 * idx];  // Non-contiguous access
    }
}

// Kernel for coalesced memory access
__global__ void coalesced_memory_access(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx];  // Contiguous access
    }
}

// Function to benchmark kernel execution time
void benchmark_memory_access(void (*kernel)(float*, float*, int), float *d_input, float *d_output, int N, int num_runs) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_runs; ++i) {
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Average execution time: " << (milliseconds / num_runs) << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int N = 1024 * 1024; // 1 million elements
    int num_runs = 1000;
    size_t size = N * sizeof(float);
    
    float *h_input = (float*)malloc(size * 2);
    float *h_output = (float*)malloc(size);
    
    for (int i = 0; i < 2 * N; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * 2);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size * 2, cudaMemcpyHostToDevice);
    
    std::cout << "Benchmarking uncoalesced memory access..." << std::endl;
    benchmark_memory_access(uncoalesced_memory_access, d_input, d_output, N, num_runs);
    
    std::cout << "Benchmarking coalesced memory access..." << std::endl;
    benchmark_memory_access(coalesced_memory_access, d_input, d_output, N, num_runs);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

