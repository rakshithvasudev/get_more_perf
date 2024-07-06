/**
Does it mean 100MB was fit at one shot or in different rounds?
The 100MB was not fit in shared memory all at once. Here's why:

The Hopper Tuning Guide states that the shared memory capacity per SM is 228 KB.
The maximum shared memory per thread block is 227 KB.
Our benchmark uses a fixed shared memory size of 16 KB per block (SHARED_MEM_SIZE).

So, the 100MB data is processed in multiple rounds:

The kernel processes data in chunks that fit into shared memory.
It uses a grid-stride loop to iterate over the entire 100MB dataset.
The bandwidth calculation takes into account the total amount of data processed and the time taken for all these iterations.


Why is the read bandwidth much smaller than write and L2?
There are several potential reasons for this:
a) Write Optimization: The Tensor Memory Accelerator (TMA) mentioned in the guide is particularly optimized for writes. It allows for efficient data movement and supports element-wise operations during writes, which could explain the higher write bandwidth.
b) Read Pattern: The read bandwidth might be lower due to the specific access pattern in our benchmark. If reads are not perfectly coalesced or if there's any branch divergence in the read loop, it could lower the effective bandwidth.
c) Caching Effects: For reads, the L2 cache might be more effective at serving requests, especially if there's any data reuse. This could explain why L2 bandwidth is higher than shared memory read bandwidth.
d) Benchmark Specifics: Our benchmark performs 100 read operations and 100 write operations in the kernel. The read operations accumulate a sum, which could introduce dependencies that slow down reads. The write operations might be more easily optimized by the hardware.
e) Hardware Specifics: The H100 architecture might have specific optimizations for shared memory writes that are not as effective for reads.

To further investigate this:

We could modify the benchmark to perform only reads or only writes to isolate their performance.
We could experiment with different access patterns to see if we can improve read performance.
We might want to use NVIDIA's profiling tools to get more detailed information about how the memory is being accessed and where potential bottlenecks are occurring.

Remember, these benchmarks give us a general idea of performance, but real-world application performance can vary based on specific access patterns and computational requirements.
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

#define SHARED_MEM_SIZE 16384  // 16 KB of shared memory per block

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

// Kernel for shared memory read bandwidth
__global__ void shared_memory_read_kernel(float* input, float* output, size_t N, int repeat) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t num_elements = blockDim.x;
    size_t grid_stride = gridDim.x * num_elements;

    for (size_t base = bid * num_elements; base < N; base += grid_stride) {
        // Load data into shared memory
        if (base + tid < N) {
            shared[tid] = input[base + tid];
        }
        __syncthreads();

        // Perform read operations in shared memory
        float sum = 0.0f;
        for (int i = 0; i < repeat; ++i) {
            sum += shared[(tid + i) % num_elements];
        }

        // Write back result
        if (base + tid < N) {
            output[base + tid] = sum;
        }
    }
}

// Kernel for shared memory write bandwidth
__global__ void shared_memory_write_kernel(float* input, float* output, size_t N, int repeat) {
    extern __shared__ float shared[];
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t num_elements = blockDim.x;
    size_t grid_stride = gridDim.x * num_elements;

    for (size_t base = bid * num_elements; base < N; base += grid_stride) {
        // Load data into shared memory
        if (base + tid < N) {
            shared[tid] = input[base + tid];
        }
        __syncthreads();

        // Perform write operations in shared memory
        for (int i = 0; i < repeat; ++i) {
            shared[tid] = shared[(tid + 1) % num_elements] + 1.0f;
            __syncthreads();
        }

        // Write back result
        if (base + tid < N) {
            output[base + tid] = shared[tid];
        }
    }
}

void benchmark_global_l2_memory(size_t size_mb, const char* memory_type) {
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
    int repeat = 100;  // Number of times to repeat the operation in L2 cache kernel
    float milliseconds = 0;

    // Warm-up run
    if (strcmp(memory_type, "global") == 0) {
        global_memory_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    } else if (strcmp(memory_type, "L2") == 0) {
        l2_cache_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, repeat);
    }
    cudaDeviceSynchronize();

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        if (strcmp(memory_type, "global") == 0) {
            global_memory_kernel<<<gridSize, blockSize>>>(d_input, d_output, N);
        } else if (strcmp(memory_type, "L2") == 0) {
            l2_cache_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, repeat);
        }
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    float seconds = milliseconds / 1000.0f;
    float bandwidth;
    if (strcmp(memory_type, "global") == 0) {
        bandwidth = 2.0f * bytes * num_iterations / (seconds * 1e9);  // GB/s
    } else if (strcmp(memory_type, "L2") == 0) {
        bandwidth = 2.0f * bytes * num_iterations * repeat / (seconds * 1e9);  // GB/s
    }

    printf("%s Memory - Size: %zu MB, Bandwidth: %.2f GB/s\n", memory_type, size_mb, bandwidth);

    free(h_input);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

void benchmark_shared_memory(size_t size_mb) {
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
    int num_elements_per_block = SHARED_MEM_SIZE / sizeof(float);
    int gridSize = std::min((N + num_elements_per_block - 1) / num_elements_per_block, static_cast<size_t>(65535));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    int num_iterations = std::max(1000, int(1e9 / N));  // Increased iterations for small sizes
    int repeat = 1000;  // Increased repeat count
    float milliseconds = 0;

    // Benchmark read operations
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        shared_memory_read_kernel<<<gridSize, blockSize, SHARED_MEM_SIZE>>>(d_input, d_output, N, repeat);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    float seconds = milliseconds / 1000.0f;
    float read_bandwidth = N * sizeof(float) * num_iterations * repeat / (seconds * 1e9);  // GB/s

    // Benchmark write operations
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        shared_memory_write_kernel<<<gridSize, blockSize, SHARED_MEM_SIZE>>>(d_input, d_output, N, repeat);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    seconds = milliseconds / 1000.0f;
    float write_bandwidth = N * sizeof(float) * num_iterations * repeat / (seconds * 1e9);  // GB/s

    printf("Shared Memory - Size: %zu MB, Read Bandwidth: %.2f GB/s, Write Bandwidth: %.2f GB/s\n", 
           size_mb, read_bandwidth, write_bandwidth);

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

    printf("Global Memory Benchmark:\n");
    for (int i = 0; i < num_sizes; i++) {
        benchmark_global_l2_memory(sizes[i], "global");
    }
    printf("\n");

    printf("L2 Cache Benchmark:\n");
    for (int i = 0; i < num_sizes; i++) {
        benchmark_global_l2_memory(sizes[i], "L2");
    }
    printf("\n");

    printf("Shared Memory Benchmark:\n");
    for (int i = 0; i < num_sizes; i++) {
        benchmark_shared_memory(sizes[i]);
    }

    return 0;
}
