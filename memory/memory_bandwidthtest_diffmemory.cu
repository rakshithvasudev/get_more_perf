/**
  Analysis:

Global Memory: These results look good and are likely accurate.

L2 Cache: The results are more reasonable now, showing higher bandwidth than global memory. This is closer to what we'd expect for L2 cache performance.

Shared Memory: The results for shared memory are problematic. The extreme increase with size and the unrealistically high bandwidth for larger sizes indicate that our benchmark is not correctly measuring shared memory performance.

The shared memory issue is likely due to:

Over-counting the actual memory operations performed.
The kernel possibly being optimized in unexpected ways by the compiler.
For larger sizes, we might be hitting some caching effects that aren't representative of true shared memory bandwidth.


  */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        exit(1);
    }
}

// Kernel for global memory bandwidth
__global__ void global_memory_kernel(float* input, float* output, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        output[i] = input[i] + 1.0f;
    }
}

// Kernel for L2 cache bandwidth
__global__ void l2_cache_kernel(float* input, float* output, size_t N, int repeat) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ARRAY_SIZE = 32;  // Adjust this size to fit in L2 cache
    float array[ARRAY_SIZE];
    
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        // Initialize array
        for (int j = 0; j < ARRAY_SIZE; j++) {
            array[j] = input[i % ARRAY_SIZE];
        }
        
        // Repeatedly access and modify the array
        for (int r = 0; r < repeat; r++) {
            for (int j = 0; j < ARRAY_SIZE; j++) {
                array[j] = array[(j + 1) % ARRAY_SIZE] + 1.0f;
            }
        }
        
        output[i] = array[0];  // Write back result
    }
}

// Kernel for shared memory bandwidth
__global__ void shared_memory_kernel(float* input, float* output, size_t N, int repeat) {
    extern __shared__ float shared[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        shared[tid] = input[i];
    }
    __syncthreads();

    // Repeatedly access and modify shared memory
    for (int r = 0; r < repeat; r++) {
        shared[tid] = shared[(tid + 1) % blockDim.x] + 1.0f;
        __syncthreads();
    }

    // Write back result
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        output[i] = shared[tid];
    }
}

void benchmark_memory(size_t size_mb, const char* memory_type) {
    size_t N = size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);

    float *h_input, *d_input, *d_output;
    h_input = (float*)malloc(bytes);
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));

    for (size_t i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = std::min((N + blockSize - 1) / blockSize, static_cast<size_t>(65535));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    int num_iterations = std::max(100, int(1e9 / N));
    int repeat = 100;  // Number of times to repeat the operation in L2 and shared memory kernels
    float milliseconds = 0;

    // Warm-up run
    if (strcmp(memory_type, "global") == 0) {
        global_memory_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    } else if (strcmp(memory_type, "L2") == 0) {
        l2_cache_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, repeat);
    } else if (strcmp(memory_type, "shared") == 0) {
        shared_memory_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N, repeat);
    }
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        if (strcmp(memory_type, "global") == 0) {
            global_memory_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
        } else if (strcmp(memory_type, "L2") == 0) {
            l2_cache_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, repeat);
        } else if (strcmp(memory_type, "shared") == 0) {
            shared_memory_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N, repeat);
        }
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    float seconds = milliseconds / 1000.0f;
    float bandwidth = 2.0f * bytes * num_iterations / (seconds * 1e9);  // GB/s
    if (strcmp(memory_type, "L2") == 0 || strcmp(memory_type, "shared") == 0) {
        bandwidth *= repeat;  // Adjust for repeated accesses
    }

    printf("%s Memory - Size: %zu MB, Bandwidth: %.2f GB/s\n", memory_type, size_mb, bandwidth);

    free(h_input);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

int main() {
    size_t sizes[] = {1, 10, 100, 1000};  // MB
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Memory Bandwidth Test on GPU\n");
    printf("-----------------------------\n");

    const char* memory_types[] = {"global", "L2", "shared"};
    int num_types = sizeof(memory_types) / sizeof(memory_types[0]);

    for (int j = 0; j < num_types; j++) {
        for (int i = 0; i < num_sizes; i++) {
            benchmark_memory(sizes[i], memory_types[j]);
        }
        printf("\n");
    }

    return 0;
}
