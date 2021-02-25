
def safestr(*args):
    """Turn string into a filename
    https://stackoverflow.com/questions/7406102/create-sane-safe-filename-from-any-unsafe-string
    :return: sanitized filename-safe string
    """
    string = str(args)
    keepcharacters = (' ', '.', '_')
    return "".join(c for c in string if c.isalnum() or c in keepcharacters).rstrip().replace(' ', '_')
