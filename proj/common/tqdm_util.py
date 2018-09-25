import sys, contextlib
from tqdm import tqdm, trange as _trange


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

ORIG_STD_OUT_ERR = (sys.stdout, sys.stderr)

def std_out():
    return ORIG_STD_OUT_ERR[0]

@contextlib.contextmanager
def tqdm_out():
    try:
        # sys.stdout = sys.stderr = DummyTqdmFile(orig_out_err[0])
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
    A replacement for tqdm.trange, automatically setting file=std_out() if not
    specified.
    """
    if 'file' in kwargs: return _trange(*args, **kwargs)
    return _trange(*args, file=std_out(), **kwargs)
