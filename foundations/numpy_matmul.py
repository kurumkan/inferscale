import numpy as np
import time
import csv


A = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=np.float64)

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
], dtype=np.float64)

def matmul_manual(a, b):
    result = np.empty((a.shape[0], b.shape[1]), dtype=np.float64)

    for i, row_a in enumerate(a):
        for j in range(0, b.shape[1]):
            acc = 0
            for index in range(len(row_a)):
                acc += row_a[index] * b[index, j]

            result[i][j] = acc

    return result        

def matmul_np(a, b):
    return np.matmul(a, b)

np.testing.assert_allclose(matmul_manual(A, B), matmul_np(A, B))

def benchmark(n):
    print("-----------------------------------")
    print("Benchmark for arrays of size ", n, n)

    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    B = rng.random((n, n))
   
    # checking correctness
    np.testing.assert_allclose(matmul_np(A, B), matmul_manual(A, B))

    # warmap for manual
    matmul_manual(A, B)
    matmul_manual(A, B)
    matmul_manual(A, B)

    times_loops = []
    for _ in range(5):
        start = time.perf_counter_ns()
        matmul_manual(A, B)
        times_loops.append(time.perf_counter_ns() - start)
    print(times_loops)
    print("Average with manual approach:", np.mean(times_loops))


    # warmap for np version
    matmul_np(A, B)
    matmul_np(A, B)
    matmul_np(A, B)

    times_vectors = []
    for _ in range(5):
        start = time.perf_counter_ns()
        matmul_np(A, B)
        times_vectors.append(time.perf_counter_ns() - start)
    print(times_vectors)
    print("Average with np version:", np.mean(times_vectors))

    print("speedup", np.mean(times_loops) / np.mean(times_vectors))

    with open(f"benchmarks/matmul_{n}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "method", "run", "elapsed_ns"])

        for method, times in [
            ("manual", times_loops),
            ("numpy", times_vectors),
        ]:
            for run, elapsed in enumerate(times, 1):
                writer.writerow([n, method, run, elapsed])

benchmark(16)    
benchmark(64)    



    




