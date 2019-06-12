""" Utility functions to handle tqdm output"""
import sys
import contextlib
from tqdm import tqdm as _tqdm, trange as _trange


class DummyTqdmFile:
    """
    Dummy file-like that will write to tqdm
    """

    file = sys.stdout

    def __init__(self, file):
        self.file = file

    def write(self, output):
        # Avoid print() second call (useless \n)
        if output.rstrip():
            _tqdm.write(output, file=self.file)

    @staticmethod
    def read(_):
        raise RuntimeError("Can't read from DummyTqdmFile")

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


def tqdm_out():
    return contextlib.redirect_stdout(DummyTqdmFile(sys.stdout))


def trange(*args, **kwargs):
    """
    A wrapper for tqdm.trange, automatically setting file=sys.stdout and
    dynamic_ncols=True if not specified.
    """
    if "file" not in kwargs:
        kwargs["file"] = DummyTqdmFile.file
    if "dynamic_ncols" not in kwargs:
        kwargs["dynamic_ncols"] = True
    return _trange(*args, **kwargs)


def tqdm(*args, **kwargs):
    """
    A wrapper for tqdm.tqdm, automatically setting file=sys.stdout and
    dynamic_ncols=True if not specified.
    """
    if "file" not in kwargs:
        kwargs["file"] = DummyTqdmFile.file
    if "dynamic_ncols" not in kwargs:
        kwargs["dynamic_ncols"] = True
    return _tqdm(*args, **kwargs)
