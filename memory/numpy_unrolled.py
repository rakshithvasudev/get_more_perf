import numpy as np
import timeit

# Standard Loop
def sum_array(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Loop Unrolling
def sum_array_unrolled(arr):
    total = 0
    n = len(arr)
    for i in range(0, n - n % 4, 4):
        total += arr[i] + arr[i+1] + arr[i+2] + arr[i+3]
    for i in range(n - n % 4, n):
        total += arr[i]
    return total

# Further optimized loop unrolling
def sum_array_unrolled_8(arr):
    total = 0
    n = len(arr)
    for i in range(0, n - n % 8, 8):
        total += arr[i] + arr[i+1] + arr[i+2] + arr[i+3] + arr[i+4] + arr[i+5] + arr[i+6] + arr[i+7]
    for i in range(n - n % 8, n):
        total += arr[i]
    return total

# Using NumPy
def sum_array_numpy(arr):
    return np.sum(arr)

# Create a large array for testing
arr = list(range(1000000))
arr_np = np.array(arr)

# Benchmarking
standard_loop_time = timeit.timeit('sum_array(arr)', globals=globals(), number=100)
unrolled_loop_time = timeit.timeit('sum_array_unrolled(arr)', globals=globals(), number=100)
#numpy_loop_time = timeit.timeit('sum_array_numpy(arr_np)', globals=globals(), number=100)
#numpy_loop_time = timeit.timeit('sum_array_numpy(arr_np)', globals=globals(), number=100)
unrolled_loop_time_8 = timeit.timeit('sum_array_unrolled_8(arr)', globals=globals(), number=100)


print(f"Standard Loop Time: {standard_loop_time} seconds")
print(f"Unrolled Loop Time: {unrolled_loop_time} seconds")
#print(f"NumPy Loop Time: {numpy_loop_time} seconds")
print(f"Further Unrolled Loop Time (8): {unrolled_loop_time_8} seconds")


