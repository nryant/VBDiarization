"""Miscellaneous functions."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from math import ceil
import random

__all__ = ['make_dir', 'partition']


from math import ceil

def make_dir(dir_name):
    """Directory creation helper function.

    If directory already exists, nothing happens.
    """
    try:
        os.makedirs(dir_name)
    except OSError:
        pass


def partition(l, n, shuffle=False):
    """Partition a list ``l`` into ``n`` sublists."""
    l = list(l)
    if shuffle:
        random.shuffle(l)
    n = min(n, len(l))
    partitions = [[] for _ in range(n)]
    n_per = int(ceil(len(l) / float(n)))
    for ii, val in enumerate(l):
        partitions[ii // n_per].append(val)
    return partitions
