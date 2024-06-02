#def naive(n, idx, in, out):
#    i = idx
#    while i < n:
#        out[i] = in[i]
#   
#
#def unrolled_op(n, idx, in, out):
#    i = idx 
#    while i < n/4:
#        out[i] = in[i]
#        out[i+1] = in[i+1]
#        out[i+2] = in[i+2]
#        out[i+3] = in[i+3]
#
#        
#
#
#

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

# Create a large array for testing
arr = list(range(1000000))

# Benchmarking
standard_loop_time = timeit.timeit('sum_array(arr)', globals=globals(), number=100)
unrolled_loop_time = timeit.timeit('sum_array_unrolled(arr)', globals=globals(), number=100)

print(f"Standard Loop Time: {standard_loop_time} seconds")
print(f"Unrolled Loop Time: {unrolled_loop_time} seconds")



