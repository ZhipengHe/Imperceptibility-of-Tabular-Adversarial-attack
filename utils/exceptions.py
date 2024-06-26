import argparse

class UnsupportedDataset(Exception):
    pass

class UnsupportedNorm(Exception):
    pass

class UnspportedNum(Exception):
    pass

def maybe_str_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if arg == "all":
        return arg
    raise argparse.ArgumentTypeError("x must be an int or 'all'")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')