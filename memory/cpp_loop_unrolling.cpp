#include <iostream>
#include <vector>
#include <chrono>

// Standard Loop
int sum_array(const std::vector<int>& arr) {
    int total = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        total += arr[i];
    }
    return total;
}

// Loop Unrolling
int sum_array_unrolled(const std::vector<int>& arr) {
    int total = 0;
    size_t n = arr.size();
    size_t i = 0;
    for (; i < n - n % 4; i += 4) {
        total += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
    }
    for (; i < n; ++i) {
        total += arr[i];
    }
    return total;
}

// Further optimized loop unrolling
int sum_array_unrolled_8(const std::vector<int>& arr) {
    int total = 0;
    size_t n = arr.size();
    size_t i = 0;
    for (; i < n - n % 8; i += 8) {
        total += arr[i] + arr[i+1] + arr[i+2] + arr[i+3] + arr[i+4] + arr[i+5] + arr[i+6] + arr[i+7];
    }
    for (; i < n; ++i) {
        total += arr[i];
    }
    return total;
}

// Benchmark function
template <typename Func>
double benchmark(Func func, const std::vector<int>& arr, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func(arr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

int main() {
    // Create a large array for testing
    std::vector<int> arr(1000000);
    for (int i = 0; i < 1000000; ++i) {
        arr[i] = i;
    }

    int iterations = 100;
    double standard_loop_time = benchmark(sum_array, arr, iterations);
    double unrolled_loop_time = benchmark(sum_array_unrolled, arr, iterations);
    double unrolled_loop_time_8 = benchmark(sum_array_unrolled_8, arr, iterations);

    std::cout << "Standard Loop Time: " << standard_loop_time << " seconds\n";
    std::cout << "Unrolled Loop Time: " << unrolled_loop_time << " seconds\n";
    std::cout << "Further Unrolled Loop Time (8): " << unrolled_loop_time_8 << " seconds\n";

    return 0;
}

