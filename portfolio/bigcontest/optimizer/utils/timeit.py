from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(time.time() - start_time)
        return func(*args, **kwargs)
    return wrapper
