# stdlib
import glob
import os
import shutil
from functools import wraps, lru_cache as lru_cache_orig
from copy import deepcopy
from math import log10
from time import time

import numpy as np


def lru_cache(maxsize=64, typed=False, copy=False):
    """Copying LRU cache - memoize results, return mutable copies

    Taken from:
    https://stackoverflow.com/questions/54909357/how-to-get-functools-lru-cache-to-return-new-instances
    :param maxsize:
    :param typed:
    :param copy:
    :return:
    """
    if not copy:
        return lru_cache_orig(maxsize, typed)

    def decorator(f):
        cached_func = lru_cache_orig(maxsize, typed)(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))

        return wrapper

    return decorator


def timing(f):
    """Time a function execution

    Use as a @timing decorator or timing(foo)(args)

    Taken from:
    https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

    :param f:
    :return:
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('func:{!r} '.format(f.__name__), end='')
        print('args:[{!r}, {!r}] :'.format(args, kwargs), end='')
        print('{:2.4f} sec'.format(end - start))
        return result

    return wrap


def safestr(*args):
    """Turn string into a filename
    :return: sanitized filename-safe string
    """
    string = str(args)
    keepcharacters = (' ', '.', '_', '-')
    return "".join(c for c in string if c.isalnum() or c in keepcharacters).rstrip().replace(' ', '_')


class DuplicateStream:
    """Make stream double-ended, outputting to stdout and a file

    Taken from:
    http://www.tentech.ca/2011/05/stream-tee-in-python-saving-stdout-to-file-while-keeping-the-console-alive/

    """

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        return callable1(*args, **kwargs)


def conv_dim_calc(w, k, d=1, p=0, s=1):
    """Calculate change of dimension size caused by a convolution layer

    out = (w - f + 2p) / s + 1
    f = 1 + (k - 1) * d

    :param w: width
    :param k: kernel size
    :param d: dilation rate
    :param p: padding
    :param s: stride
    :return: output size
    """
    return (w - (1 + (k - 1) * d) + 2 * p) // s + 1


def exp_form(val):
    """Get number formatted as 1eX = 10^X"""
    if np.isnan(val):
        return str(val)
    if val < 1:
        # inaccuracy tolerated
        return '0'

    # return '{:.2e}'.format(val)
    return '1e{:0.3g}'.format(log10(val))


def get_checkpoint_path(path='/tmp/model_checkpoints'):
    """Get clear directory for model checkpoints, deleting previous contents"""
    files = glob.glob(os.path.join(path, '*'))

    if len(files) > 0:
        print('Clearing model checkpoint directory:'
              '\tPath: {}'.format(path),
              '\tNumber of files: {}'.format(len(files)))

    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        else:
            shutil.rmtree(f)

    return os.path.join(path, 'checkpoint')


def inverse_indexing(arr, index):
    """arr - arr[index]

    `index` is a negative mask for selecting from `arr`
    """
    mask = np.ones(len(arr), dtype=np.bool)
    mask[index] = 0
    return arr[mask]  # could use a copy
