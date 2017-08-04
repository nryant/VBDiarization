"""TODO"""
import errno
import os

import numpy as np


################################################################################
################################################################################

SOURCERATE = 1250
TARGETRATE = 100000
LOFREQ = 120
HIFREQ = 3800

ZMEANSOURCE = True
WINDOWSIZE = 250000.0
USEHAMMING = True
PREEMCOEF = 0.97
NUMCHANS = 24
CEPLIFTER = 22
NUMCEPS = 19
ADDDITHER = 1.0
RAWENERGY = True
ENORMALISE = True

deltawindow = accwindow = 2

cmvn_lc = 150
cmvn_rc = 150

fs = 1e7 / SOURCERATE


################################################################################
################################################################################

class NoVadException(Exception):
    """ No VAD exception - raised when there is no VAD definition for a file
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def row(v):
    return v.reshape((1, v.size))


def mkdir_p(path):
    """ mkdir 
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_ubm(fname):
    """ This function will load the UBM from the file and will return the
        parameters in three separate variables
    """
    gmm = np.load(fname)

    n_superdims = (gmm.shape[1] - 1) / 2

    weights = gmm[:, 0]
    means = gmm[:, 1:(n_superdims + 1)]
    covs = gmm[:, (n_superdims + 1):]

    return weights, means, covs


def load_vad_lab_as_bool_vec(lab_file):
    lab_cont = np.atleast_2d(np.loadtxt(lab_file, dtype=object))

    if lab_cont.shape[1] == 0:
        return np.empty(0), 0, 0

    # else:
    #     lab_cont = lab_cont.reshape((-1,lab_cont.shape[0]))

    if lab_cont.shape[1] == 3:
        lab_cont = lab_cont[lab_cont[:, 2] == 'sp', :][:, [0, 1]]

    n_regions = lab_cont.shape[0]
    ii = 0
    while True:
        try:
            start1, end1 = float(lab_cont[ii][0]), float(lab_cont[ii][1])
            jj = ii + 1
            start2, end2 = float(lab_cont[jj][0]), float(lab_cont[jj][1])
            if end1 >= start2:
                lab_cont = np.delete(lab_cont, ii, axis=0)
                ii -= 1
                lab_cont[jj - 1][0] = str(start1)
                lab_cont[jj - 1][1] = str(max(end1, end2))
            ii += 1
        except IndexError:
            break

    vad = np.round(np.atleast_2d(lab_cont).astype(np.float).T * 100).astype(np.int)
    vad[1] += 1  # Paja's bug!!!

    if not vad.size:
        return np.empty(0, dtype=bool)

    npc1 = np.c_[np.zeros_like(vad[0], dtype=bool), np.ones_like(vad[0], dtype=bool)]
    npc2 = np.c_[vad[0] - np.r_[0, vad[1, :-1]], vad[1] - vad[0]]

    out = np.repeat(npc1, npc2.flat)

    n_frames = sum(out)

    return out, n_regions, n_frames


def compute_vad(s, win_length=160, win_overlap=80):
    v = evad.compute_vad(s, win_length=win_length, win_overlap=win_overlap, n_realignment=10)

    n_frames = sum(v)
    n_regions = n_frames

    return v, n_regions, n_frames


def split_seq(seq, size):
    """ Split up seq in pieces of size """
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def normalize_stats(n, f, ubm_means, ubm_norm):
    """ Center the first-order UBM stats around UBM means and normalize 
        by the UBM covariance 
    """
    n_gauss = n.shape[0]
    n_superdim = f.shape[0]
    n_fdim = n_superdim / n_gauss

    f0 = f - ubm_means * np.kron(np.ones((n_fdim, 1), dtype=n.dtype), n).transpose()
    f0 = f0 * ubm_norm

    return n, f0
