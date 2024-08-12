import os
import sys
from contextlib import contextmanager

@contextmanager
def stderr_redirected(to=os.devnull):
    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stderr:
        with open(to, "w") as file:
            _redirect_stderr(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stderr(to=old_stderr)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

