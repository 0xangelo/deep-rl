import sys
import contextlib
from tqdm import tqdm as _tqdm, trange as _trange


ORIG_STD_OUT_ERR = (sys.stdout, sys.stderr)


class DummyTqdmFile(object):
    """
    Dummy file-like that will write to tqdm
    """

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            _tqdm.write(x, file=self.file)

    def read(self, x):
        raise RuntimeError("Can't read from DummyTqdmFile")

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


def std_out():
    return ORIG_STD_OUT_ERR[0]


@contextlib.contextmanager
def tqdm_out():
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, ORIG_STD_OUT_ERR)
        yield
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = ORIG_STD_OUT_ERR


def trange(*args, **kwargs):
    """
    A wrapper for tqdm.trange, automatically setting file=std_out() and
    dynamic_ncols=True if not specified.
    """
    if "file" not in kwargs:
        kwargs["file"] = std_out()
    if "dynamic_ncols" not in kwargs:
        kwargs["dynamic_ncols"] = True
    return _trange(*args, **kwargs)


def tqdm(*args, **kwargs):
    """
    A wrapper for tqdm.tqdm, automatically setting file=std_out() and
    dynamic_ncols=True if not specified.
    """
    if "file" not in kwargs:
        kwargs["file"] = std_out()
    if "dynamic_ncols" not in kwargs:
        kwargs["dynamic_ncols"] = True
    return _tqdm(*args, **kwargs)
