import builtins
import numpy as np
import random
from string import ascii_lowercase

def rndstr(n=6):
    return ''.join(random.choices(ascii_lowercase, k=n))


def print(*args, **kwargs):
    builtins.print(*args, **kwargs, flush=True)
