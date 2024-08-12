import fcntl
import os
import sys
import time
import traceback
from functools import wraps
from multiprocessing import Process, Queue

import numpy as np


# https://gist.github.com/schlamar/2311116
def processify(func):
    """Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    """

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, "".join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + "processify_func"
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = "%s (in subprocess)\n%s" % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret

    return wrapper


class Locker:
    def __init__(self, lock_file):
        # Define the file path for the lock file
        self.lock_file = lock_file
        self.lock_file_fd = None
        # Attempt to create a lock file and write the process ID to it
        self.timeout = 1200

    def acquire(self):
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        fd = os.open(self.lock_file, open_mode)

        pid = os.getpid()
        lock_file_fd = None

        print(f"  {pid} waiting for lock")
        start_time = current_time = time.time()
        while current_time < start_time + self.timeout:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                pass
            else:
                lock_file_fd = fd
                break
            time.sleep(np.random.rand() * 10)
            current_time = time.time()
        if lock_file_fd is None:
            os.close(fd)

        self.lock_file_fd = lock_file_fd

    def release(self):
        fcntl.flock(self.lock_file_fd, fcntl.LOCK_UN)
        os.close(self.lock_file_fd)
        self.lock_file_fd = None
