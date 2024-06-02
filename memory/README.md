# README

## Overview

This repository contains examples and compiled binary related to loop unrolling, memory coalescing, and other optimization techniques in C++, CUDA, and Python. Below are the instructions to compile and run these files.

## Files

- `cpp_loop_unrolling.cpp`: C++ code for loop unrolling
- `memory_coalesing.cu`: CUDA code for memory coalescing
- `numpy_unrolled.py`: Python code using NumPy for loop unrolling
- `unroll_reordering.cu`: CUDA code for unroll reordering
- `unrolled.py`: Python code for loop unrolling

## Prerequisites

Ensure you have the following installed:
- g++ (for compiling C++ files)
- nvcc (NVIDIA CUDA Compiler for compiling CUDA files)
- Python with NumPy (for running Python scripts)

## Compilation Instructions

### Compiling C++ Files

For the C++ file (`cpp_loop_unrolling.cpp`), you can use the following command with optimization flag `-O3`:

```sh
g++ -O3 -o cpp_loop_unrolling cpp_loop_unrolling.cpp
nvcc -o memory_coalesing memory_coalesing.cu
nvcc -o unroll_reordering unroll_reordering.cu
python numpy_unrolled.py
python unrolled.py
```
The output should print benchmark times.


