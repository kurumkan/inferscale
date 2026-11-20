from threading import Thread
from time import perf_counter
from threading import Lock

t = perf_counter()

lock = Lock()
stats = {"done": 0}

def worker(name, delay):
    for i in range(1_000_000):
        with lock:
            stats["done"] += 1
        
a = Thread(target=worker, args=("A", 1))

a.start()
a.join()

print(round(perf_counter() - t, 2), "секунд")

assert stats["done"] == 1_000_000
