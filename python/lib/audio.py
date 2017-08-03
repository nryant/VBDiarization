"""Audio utilities."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import math
import os
import subprocess

import numpy as np

__all__ = ['af_to_array', 'get_bit_depth', 'get_dur', 'get_sr']

# TODO: Add check for SoX.
# TODO: Convert docstring format.


def get_bit_depth(af):
    """Return bit depth of audio file (None if unknown)."""
    try:
        cmd = ['soxi', '-b', af]
        with open(os.devnull, 'wb') as f:
            bit_depth = int(subprocess.check_output(cmd, stderr=f))
    except subprocess.CalledProcessError:
        raise IOError('Error opening: %s' % af)
    bit_depth = None if bit_depth == 0 else bit_depth
    return bit_depth


def get_dur(af):
    """Return duration in seconds of audio file."""
    try:
        cmd = ['soxi', '-D', af]
        with open(os.devnull, 'wb') as f:
            dur = float(subprocess.check_output(cmd, stderr=f))
    except subprocess.CalledProcessError:
        raise IOError('Error opening: %s' % af)
    return dur


def get_sr(af):
    """Return sample rate in Hz of audio file."""
    try:
        cmd = ['soxi', '-r', af]
        with open(os.devnull, 'wb') as f:
            sr = int(subprocess.check_output(cmd, stderr=f))
    except subprocess.CalledProcessError:
        raise IOError('Error opening: %s' % af)
    return sr


def af_to_array(af, target_sr=None, channel=1, start=0, end=None,
               remove_dc_offset=False, dbfs=None):
    """Return sample rate and data from audio file.

    Must be in a format understood by ``SoX``.

    Parameters
    ----------
    af : str
        Audio file.

    target_sr : int, optional
        Target sample rate (Hz) to which recording will be resampled. If None,
        do not resample.
        (Default: None)

    channel : int, optional
        Channel to extract (1-indexed).
        (Default: 1)

    start : float, optional
        Start time of trimmed recording in seconds.
        (Default: 0)

    end : float, optional
        End time of trimmed recording in seconds. If None, defaults to duration
        of recording.
        (Default: None)

    remove_dc_offset : bool, optional
        If True, mean center signal prior to any other processing.
        (Default: False)

    dbfs : float, optional
        A float specifying the target decibels relative to full scale (dBFS)
        for the signal. The gain of the signal will be adjusted to achieve this
        value. The formula for dBFS is given by
        :math:`\textrm{dBFS}=20*\log_{10}(\textrm{scale})`, so 0 dBFS
        corresponds to the maximum possible digital level at 16 bit precision
        (32678), -2.5 dBFS to 75% of this level, -6 dBFS to 50%, -12 dBFS to
        25%, etc. If None, then no normalization occurs.
        (Default: None)

    Returns
    -------
    sr : int
        Sample rate in Hz of (possibly resampled) recording.

    x : ndarray, (n_samples,)
        Audio samples.
    """
    # Set defaults.
    if target_sr is None:
        target_sr = get_sr(af)
    def ngm(m, n):
        return int(n*math.ceil(m/float(n)))
    bit_depth = get_bit_depth(af)
    bit_depth = 16 if bit_depth is None else bit_depth
    bit_depth = ngm(bit_depth, 8)
    if end is None:
        end = get_dur(af)

    # Resample and load into array.
    try:
        cmd = ['sox', af,
               '-b', str(bit_depth),
               '-e', 'signed-integer', # And linear PCM
               '-t', 'raw',   # Output file as type raw
               '-L', # Output little endian
               '-',  # Pipe to stdout
               'remix', str(channel), # Extract single channel.
               'trim', str(start), '=%s' % end,
               'rate', str(target_sr),
               ]
        if dbfs is not None:
            cmd = cmd[:1] + ['--norm=%d' % dbfs] + cmd[1:]
        with open(os.devnull, 'wb') as f:
            raw = subprocess.check_output(cmd, stderr=f)
        dtype = '<i%d' % (bit_depth/8)
        x = np.fromstring(raw, dtype)
        x = x.astype(dtype='float32', copy=False)
        sr = target_sr
    except subprocess.CalledProcessError:
        raise IOError('Error opening: %s' % af)

    # Remove dc offset.
    if remove_dc_offset:
        x -= x.mean()

    return sr, x
