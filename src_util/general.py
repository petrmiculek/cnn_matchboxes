# stdlib
from functools import wraps, lru_cache as lru_cache_orig
from copy import deepcopy
from time import time


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

    Taken from
    https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

    :param f:
    :return:
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'func:{f.__name__!r} ', end='')
        # print(f'args:[{args!r}, {kwargs!r}] :', end='')
        print(f'{end - start:2.4f} sec')
        return result

    return wrap


def safestr(*args):
    """Turn string into a filename
    https://stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string
    :return: sanitized filename-safe string
    """
    string = str(args)
    keepcharacters = (' ', '.', '_', '-')
    return "".join(c for c in string if c.isalnum() or c in keepcharacters).rstrip().replace(' ', '_')


class DuplicateStream(object):
    """Make stream double-ended, outputting to stdout and a file

    http://www.tentech.ca/2011/05/stream-tee-in-python-saving-stdout-to-file-while-keeping-the-console-alive/
    Based on https://gist.github.com/327585 by Anand Kunal

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
