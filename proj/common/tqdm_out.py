import contextlib
import sys
from tqdm import tqdm

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

def term():
    return ORIG_STD_OUT_ERR[0]

@contextlib.contextmanager
def redirect():
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

