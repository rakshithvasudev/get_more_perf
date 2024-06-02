#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Kernel for basic memory access (contiguous but without unrolling)
__global__ void basic_kernel(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx];
    }
}

// Kernel with unrolled memory access
__global__ void unrolled_kernel(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 4) {
        output[4 * idx] = input[4 * idx];
        output[4 * idx + 1] = input[4 * idx + 1];
        output[4 * idx + 2] = input[4 * idx + 2];
        output[4 * idx + 3] = input[4 * idx + 3];
    }
}

// Kernel with reordered memory access
__global__ void reordered_kernel(float *input, float *output, int N) {
    int block_offset = blockIdx.x * (blockDim.x * 4);
    int idx = threadIdx.x * 4;

    if (block_offset + idx < N) {
        output[block_offset + idx] = input[block_offset + idx];
        output[block_offset + idx + 1] = input[block_offset + idx + 1];
        output[block_offset + idx + 2] = input[block_offset + idx + 2];
        output[block_offset + idx + 3] = input[block_offset + idx + 3];
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
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    std::cout << "Benchmarking basic kernel..." << std::endl;
    benchmark_memory_access(basic_kernel, d_input, d_output, N, num_runs);
    
    std::cout << "Benchmarking unrolled kernel..." << std::endl;
    benchmark_memory_access(unrolled_kernel, d_input, d_output, N, num_runs);
    
    std::cout << "Benchmarking reordered kernel..." << std::endl;
    benchmark_memory_access(reordered_kernel, d_input, d_output, N, num_runs);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

