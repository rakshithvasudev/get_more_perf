# Coalesced Memory Access in CUDA

This example demonstrates the performance impact of coalesced vs. uncoalesced memory access patterns in CUDA.

## Overview

In CUDA programming, memory access patterns can significantly affect performance. This project includes a CUDA program that compares two memory access patterns:

1. Uncoalesced access (column-major traversal of a 2D matrix)
2. Coalesced access (row-major traversal of a 2D matrix)

The program benchmarks both patterns and reports their performance differences.

## Key Concepts

- **Coalesced Memory Access**: When threads in a warp access contiguous memory locations, allowing for efficient memory transactions.
- **Uncoalesced Memory Access**: When threads in a warp access scattered memory locations, resulting in multiple memory transactions and reduced efficiency.

## File Structure

- `coalesced_memory_comparison.cu`: The main CUDA source file containing both kernels and benchmarking code.
- `README.md`: This file, explaining the project and how to use it.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 8.0 or later recommended)
- C++ compiler compatible with your CUDA version

## Compilation

To compile the program, use the following command:

```
nvcc -O3 coalesced_memory_comparison.cu -o coalesced_memory_comparison
```

## Usage

After compilation, run the program with:

```
./coalesced_memory_comparison
```

The program will output:
1. Grid and block dimensions
2. Execution times for both uncoalesced and coalesced kernels
3. Effective bandwidth for each kernel
4. Verification results
5. Overall performance improvement

## Understanding the Results

- The uncoalesced kernel uses column-major access, which typically results in scattered memory accesses and lower performance.
- The coalesced kernel uses row-major access, which allows for efficient, coalesced memory accesses and typically higher performance.
- The performance improvement shows the benefit of using coalesced memory access patterns in CUDA programming.

## Customization

You can modify the `MATRIX_SIZE` and `BLOCK_SIZE` constants in the source code to experiment with different problem sizes and thread block configurations.

## Further Reading

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## License

This project is open source and available under the [MIT License](LICENSE).
