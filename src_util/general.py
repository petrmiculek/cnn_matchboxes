
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
