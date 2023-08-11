from time import time
import torch

print(torch.cuda.device_count())

# decorator
def timeit(func):
    def cal_time(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        print(f"Elapsed: {time() - start}")

    return cal_time


@timeit
def barrier(n=1e6):
    for _ in range(int(n)):
        pass


if __name__ == "__main__":
    barrier(n=1e8)
