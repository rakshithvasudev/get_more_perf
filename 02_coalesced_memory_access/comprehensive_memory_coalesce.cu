#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MATRIX_SIZE 5120
#define BLOCK_SIZE 32
#define VECTOR_SIZE (MATRIX_SIZE * MATRIX_SIZE)
#define PADDED_SIZE (MATRIX_SIZE + 1)  // Add padding to avoid bank conflicts

// Utility function for checking CUDA errors
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Global variable for texture object
cudaTextureObject_t texObj;


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
        int idx = row * MATRIX_SIZE + col;
        d_out[idx] = d_in[idx] * 2.0f;
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

// 5. Padded shared memory to avoid bank conflicts
__global__ void paddedSharedMemoryKernel(float *d_out, float *d_in) {
    __shared__ float sharedMem[BLOCK_SIZE][BLOCK_SIZE + 1];  // Add padding to avoid bank conflicts
    
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

// 6. Warp shuffle operations
__global__ void warpShuffleKernel(float *d_out, float *d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < VECTOR_SIZE) {
        float value = d_in[idx] * 2.0f;
        // Demonstrate warp shuffle without affecting the result
        value = __shfl_sync(0xffffffff, value, threadIdx.x % 32);
        d_out[idx] = value;
    }
}



// 7. Aligned memory access
__global__ void alignedAccessKernel(float4 *d_out, float4 *d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < VECTOR_SIZE / 4) {
        float4 data = __ldg(&d_in[idx]);  // Use __ldg for read-only data
        data.x *= 2.0f;
        data.y *= 2.0f;
        data.z *= 2.0f;
        data.w *= 2.0f;
        d_out[idx] = data;
    }
}

// 8. Loop unrolling
__global__ void loopUnrollingKernel(float *d_out, float *d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < VECTOR_SIZE) {
        float value = d_in[idx];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Demonstrate loop unrolling without changing the actual computation
            value = value * 2.0f / 2.0f;
        }
        d_out[idx] = value * 2.0f;
    }
}



// 9. Texture memory for read-only data
__global__ void textureMemoryKernel(float *d_out, cudaTextureObject_t texObj) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        float u = (float)col / MATRIX_SIZE;
        float v = (float)row / MATRIX_SIZE;
        float value = tex2D<float>(texObj, u, v);
        d_out[row * MATRIX_SIZE + col] = value * 2.0f;
    }
}

// 10. Cooperative Groups for flexible synchronization
__global__ void cooperativeGroupsKernel(float *d_out, float *d_in) {
    cg::thread_block block = cg::this_thread_block();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < VECTOR_SIZE) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
    // Demonstrate cooperative groups without affecting the result
    block.sync();
}



// Function to initialize data
void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verifyResults(float *output, float *input, int size, const char* kernelName) {
    const float epsilon = 1e-5;
    for (int i = 0; i < size; i++) {
        if (fabs(output[i] - input[i] * 2.0f) > epsilon) {
            printf("Verification failed for %s at index %d!\n", kernelName, i);
            printf("Expected: %f, Got: %f\n", input[i] * 2.0f, output[i]);
            printf("First 10 elements of input:\n");
            for (int j = 0; j < 10 && j < size; j++) {
                printf("%f ", input[j]);
            }
            printf("\nFirst 10 elements of output:\n");
            for (int j = 0; j < 10 && j < size; j++) {
                printf("%f ", output[j]);
            }
            printf("\n");
            return false;
        }
    }
    printf("Verification passed for %s\n", kernelName);
    return true;
}

// Function to run and time a kernel
template<typename KernelFunc, typename... Args>
float runKernel(KernelFunc kernel, dim3 gridSize, dim3 blockSize, Args... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up run
    kernel<<<gridSize, blockSize>>>(args...);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Benchmark run
    const int numRuns = 100;
    float milliseconds = 0;

    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
        kernel<<<gridSize, blockSize>>>(args...);
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
    dim3 block(256);
    dim3 grid((VECTOR_SIZE + block.x - 1) / block.x);

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

    // Setup for texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, MATRIX_SIZE, MATRIX_SIZE);
    //cudaMemcpy2DToArray(cuArray, 0, 0, h_in, bytes, cudaMemcpyHostToDevice);

    cudaCheckError();

    cudaMemcpy2DToArray(cuArray, 0, 0, h_in, MATRIX_SIZE * sizeof(float),
		                        MATRIX_SIZE * sizeof(float), MATRIX_SIZE,
					                        cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // In the main function, update the texture setup:
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);


    // Run and time kernels
    cudaMemset(d_out, 0, bytes);
    float uncoalescedTime = runKernel(uncoalescedKernel, gridSize, blockSize, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool uncoalescedCorrect = verifyResults(h_out, h_in, matrixElements, "Uncoalesced Kernel");

    cudaMemset(d_out, 0, bytes);
    float basicCoalescedTime = runKernel(basicCoalescedKernel, gridSize, blockSize, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool basicCoalescedCorrect = verifyResults(h_out, h_in, matrixElements, "Basic Coalesced Kernel");

    cudaMemset(d_out, 0, bytes);
    float sharedMemoryTime = runKernel(sharedMemoryKernel, gridSize, blockSize, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool sharedMemoryCorrect = verifyResults(h_out, h_in, matrixElements, "Shared Memory Kernel");

    cudaMemset(d_out, 0, bytes);
    float vectorizedTime = runKernel(vectorizedKernel, vectorGridSize, vectorBlockSize, (float4*)d_out, (float4*)d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool vectorizedCorrect = verifyResults(h_out, h_in, matrixElements, "Vectorized Kernel");

    cudaMemset(d_out, 0, bytes);
    float paddedSharedMemoryTime = runKernel(paddedSharedMemoryKernel, gridSize, blockSize, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool paddedSharedMemoryCorrect = verifyResults(h_out, h_in, matrixElements, "Padded Shared Memory Kernel");

    cudaMemset(d_out, 0, bytes);
    float warpShuffleTime = runKernel(warpShuffleKernel, grid, block, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool warpShuffleCorrect = verifyResults(h_out, h_in, matrixElements, "Warp Shuffle Kernel");


    cudaMemset(d_out, 0, bytes);
    float alignedAccessTime = runKernel(alignedAccessKernel, vectorGridSize, vectorBlockSize, (float4*)d_out, (float4*)d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool alignedAccessCorrect = verifyResults(h_out, h_in, matrixElements, "Aligned Access Kernel");

    cudaMemset(d_out, 0, bytes);
    float loopUnrollingTime = runKernel(loopUnrollingKernel, grid, block, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool loopUnrollingCorrect = verifyResults(h_out, h_in, matrixElements, "Loop Unrolling Kernel");

    cudaMemset(d_out, 0, bytes);
    float textureMemoryTime = runKernel(textureMemoryKernel, gridSize, blockSize, d_out, texObj);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool textureMemoryCorrect = verifyResults(h_out, h_in, matrixElements, "Texture Memory Kernel");

    cudaMemset(d_out, 0, bytes);
    float cooperativeGroupsTime = runKernel(cooperativeGroupsKernel, grid, block, d_out, d_in);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool cooperativeGroupsCorrect = verifyResults(h_out, h_in, matrixElements, "Cooperative Groups Kernel");



    // Print results
    printf("\nResults:\n");
    printf("1. Uncoalesced Kernel (Column-major access):\n");
    printf("   Average execution time: %f ms\n", uncoalescedTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (uncoalescedTime * 1e6));
    printf("   Verification: %s\n", uncoalescedCorrect ? "PASSED" : "FAILED");
    
    printf("\n2. Basic Coalesced Kernel (Row-major access):\n");
    printf("   Average execution time: %f ms\n", basicCoalescedTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (basicCoalescedTime * 1e6));
    printf("   Verification: %s\n", basicCoalescedCorrect ? "PASSED" : "FAILED");
    
    printf("\n3. Shared Memory Kernel:\n");
    printf("   Average execution time: %f ms\n", sharedMemoryTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (sharedMemoryTime * 1e6));
    printf("   Verification: %s\n", sharedMemoryCorrect ? "PASSED" : "FAILED");
    
    printf("\n4. Vectorized Kernel:\n");
    printf("   Average execution time: %f ms\n", vectorizedTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (vectorizedTime * 1e6));
    printf("   Verification: %s\n", vectorizedCorrect ? "PASSED" : "FAILED");
    
    printf("\n5. Padded Shared Memory Kernel:\n");
    printf("   Average execution time: %f ms\n", paddedSharedMemoryTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (paddedSharedMemoryTime * 1e6));
    printf("   Verification: %s\n", paddedSharedMemoryCorrect ? "PASSED" : "FAILED");
    
    printf("\n6. Warp Shuffle Kernel:\n");
    printf("   Average execution time: %f ms\n", warpShuffleTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (warpShuffleTime * 1e6));
    printf("   Verification: %s\n", warpShuffleCorrect ? "PASSED" : "FAILED");
    
    printf("\n7. Aligned Access Kernel:\n");
    printf("   Average execution time: %f ms\n", alignedAccessTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (alignedAccessTime * 1e6));
    printf("   Verification: %s\n", alignedAccessCorrect ? "PASSED" : "FAILED");
    
    printf("\n8. Loop Unrolling Kernel:\n");
    printf("   Average execution time: %f ms\n", loopUnrollingTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (loopUnrollingTime * 1e6));
    printf("   Verification: %s\n", loopUnrollingCorrect ? "PASSED" : "FAILED");
    
    printf("\n9. Texture Memory Kernel:\n");
    printf("   Average execution time: %f ms\n", textureMemoryTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (textureMemoryTime * 1e6));
    printf("   Verification: %s\n", textureMemoryCorrect ? "PASSED" : "FAILED");
    
    printf("\n10. Cooperative Groups Kernel:\n");
    printf("   Average execution time: %f ms\n", cooperativeGroupsTime);
    printf("   Effective bandwidth: %f GB/s\n", (matrixElements * sizeof(float) * 2) / (cooperativeGroupsTime * 1e6));
    printf("   Verification: %s\n", cooperativeGroupsCorrect ? "PASSED" : "FAILED");
    
    printf("\nPerformance improvements compared to basic coalesced access:\n");
    printf("Shared Memory: %f%%\n", (basicCoalescedTime - sharedMemoryTime) / basicCoalescedTime * 100.0f);
    printf("Vectorized: %f%%\n", (basicCoalescedTime - vectorizedTime) / basicCoalescedTime * 100.0f);
    printf("Padded Shared Memory: %f%%\n", (basicCoalescedTime - paddedSharedMemoryTime) / basicCoalescedTime * 100.0f);
    printf("Warp Shuffle: %f%%\n", (basicCoalescedTime - warpShuffleTime) / basicCoalescedTime * 100.0f);
    printf("Aligned Access: %f%%\n", (basicCoalescedTime - alignedAccessTime) / basicCoalescedTime * 100.0f);
    printf("Loop Unrolling: %f%%\n", (basicCoalescedTime - loopUnrollingTime) / basicCoalescedTime * 100.0f);
    printf("Texture Memory: %f%%\n", (basicCoalescedTime - textureMemoryTime) / basicCoalescedTime * 100.0f);
    printf("Cooperative Groups: %f%%\n", (basicCoalescedTime - cooperativeGroupsTime) / basicCoalescedTime * 100.0f);
 

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeArray(cuArray);
    cudaDestroyTextureObject(texObj);
    free(h_in);
    free(h_out);

    return 0;
}
