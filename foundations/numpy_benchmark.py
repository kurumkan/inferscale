import numpy as np
import csv
import time

def benchmark(size):
    print("-----------------------------------")
    print("Benchmark for arrays of size ", size)

    a = np.arange(size, dtype=np.float64)
    b = np.arange(size, dtype=np.float64) * 2


    def sum_array_loop():
        result = np.empty(size)
        for i, _ in enumerate(a):
            result[i] = a[i] + b[i]

        return result    

    def sum_array_vectors():
        result = a + b    
        return result    

    # checking correctness
    np.testing.assert_allclose(sum_array_loop(), sum_array_vectors())


    # warmap for loop
    sum_array_loop()
    sum_array_loop()
    sum_array_loop()

    times_loops = []
    for _ in range(5):
        start = time.perf_counter_ns()
        sum_array_loop()
        times_loops.append(time.perf_counter_ns() - start)
    print(times_loops)
    print("Average with loops:", np.mean(times_loops))


    # warmap for vectors
    sum_array_vectors()
    sum_array_vectors()
    sum_array_vectors()

    times_vectors = []
    for _ in range(5):
        start = time.perf_counter_ns()
        sum_array_vectors()
        times_vectors.append(time.perf_counter_ns() - start)
    print(times_vectors)
    print("Average with vectors:", np.mean(times_vectors))

    print("speedup", np.mean(times_loops) / np.mean(times_vectors))


    with open(f"benchmarks/numpy_add_{size}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "method", "run", "elapsed_ns"])

        for method, times in [
            ("python_loop", times_loops),
            ("numpy", times_vectors),
        ]:
            for run, elapsed in enumerate(times, 1):
                writer.writerow([size, method, run, elapsed])
    


benchmark(1000)
benchmark(1_000_000)
