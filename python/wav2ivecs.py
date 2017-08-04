#! /usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import math
import multiprocessing
import sys

from scipy import signal

from lib.audio import af_to_array, get_sr
from lib import features
from lib.raw2ivec import *
from lib.ivec import IvecSet
from lib.tools import loginfo, logwarning, Tools
from lib.utils import partition

from wav2ivec import init, get_ivec, get_vad, get_mfccs


def process_file(wav_dir, vad_dir, out_dir, file_name, model, min_size,
                 max_size, tolerance, wav_suffix='.wav', vad_suffix='.lab.gz'):
    """ Extract i-vectors from wav file.

        :param wav_dir: directory with wav files
        :type wav_dir: str
        :param vad_dir: directory with vad files
        :type vad_dir: str
        :param out_dir: output directory
        :type out_dir: str
        :param file_name: name of the file
        :type file_name: str
        :param model: input models for i-vector extraction
        :type model: tuple
        :param min_size: minimal size of window in ms
        :type min_size: int
        :param max_size: maximal size of window in ms
        :type max_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :param wav_suffix: suffix of wav files
        :type wav_suffix: str
        :param vad_suffix: suffix of vad files
        :type vad_suffix
    """
    loginfo('[wav2ivecs.process_file] Processing file {} ...'.format(file_name))
    ubm_weights, ubm_means, ubm_covs, ubm_norm, gmm_model, numg, dimf, v, mvvt = model
    if len(file_name.split()) > 1:
        file_name = file_name.split()[0]
    wav = os.path.join(wav_dir, file_name) + wav_suffix
    rate = get_sr(wav)
    #
    loginfo(wav)
    #
    if rate != 8000:
        logwarning('[wav2ivec.process_file] '
                   'The input file is expected to be in 8000 Hz, got {} Hz '
                   'instead, resampling.'.format(rate))
    rate, sig = af_to_array(wav, target_sr=8000)
    if ADDDITHER > 0.0:
        #loginfo('[wav2ivecs.process_file] Adding dither ...')
        sig = features.add_dither(sig, ADDDITHER)

    fea = get_mfccs(sig)
    vad, n_regions, n_frames = get_vad(vad_dir, file_name, vad_suffix, sig, fea)

    ivec_set = IvecSet()
    ivec_set.name = file_name
    for seg in get_segments(vad, min_size, max_size, tolerance):
        start, end = get_num_segments(seg[0]), get_num_segments(seg[1])
        w = get_ivec(fea[seg[0]:seg[1] + 1], numg, dimf, gmm_model, ubm_means, ubm_norm, v, mvvt)
        if w is None:
            continue
        ivec_set.add(w, start, end)
    Tools.mkdir_p(os.path.join(out_dir, os.path.dirname(file_name)))
    ivec_set.save(os.path.join(out_dir, '{}.pkl'.format(file_name)))


def _process_files(args):
    """TODO"""
    fns, ubm_file, v_file, kwargs = args
    models = init(ubm_file, v_file)
    for fn in fns:
        process_file(file_name=fn, model=models, **kwargs)


def process_files(fns, wav_dir, vad_dir, out_dir, ubm_file, v_file, min_size,
                  max_size, tolerance, wav_suffix='.wav', vad_suffix='.lab.gz',
                  n_jobs=1):
    """TODO"""
    kwargs = dict(wav_dir=wav_dir, vad_dir=vad_dir, out_dir=out_dir,
                  min_size=min_size, max_size=max_size,
                  wav_suffix=wav_suffix, vad_suffix=vad_suffix,
                  tolerance=tolerance)
    if n_jobs == 1:
        res = [_process_files((fns, ubm_file, v_file, kwargs))]
    else:
        pool = multiprocessing.Pool(n_jobs)
        res = pool.map(_process_files, ((part, ubm_file, v_file, kwargs)
                       for part in partition(fns, n_jobs)))



def get_segments(vad, min_size, max_size, tolerance):
    """ Return clustered speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param min_size: minimal size of window in ms
        :type min_size: int
        :param max_size: maximal size of window in ms
        :type max_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered segments
        :rtype: list
    """
    clusters = get_clusters(vad, get_num_frames(min_size), tolerance)
    segments = []
    max_frames = get_num_frames(max_size)
    for item in clusters.values():
        if item[1] - item[0] > max_frames:
            for ss in split_segment(item, max_frames):
                segments.append(ss)
        else:
            segments.append(item)
    return segments


def split_segment(segment, max_size):
    """ Split segment to more with adaptive size.

        :param segment: input segment
        :type segment: tuple
        :param max_size: maximal size of window in ms
        :type max_size: int
        :returns: splitted segment
        :rtype: list
    """
    size = segment[1] - segment[0]
    num_segments = int(math.ceil(size / max_size))
    size_segment = size / num_segments
    for ii in range(num_segments):
        yield (segment[0] + ii * size_segment, segment[0] + (ii + 1) * size_segment)


def get_num_frames(n):
    """ Get number of frames from ms.

        :param n: number of ms
        :type n: int
        :returns: number of frames
        :rtype: int
    """
    return int(1 + (n - WINDOWSIZE / 10000) / (TARGETRATE / 10000))


def get_num_segments(n):
    """ Get count of ms from number of frames.

        :param n: number of frames
        :type n: int
        :returns: number of ms
        :rtype: int
    """
    return int(n * (TARGETRATE / 10000) - (TARGETRATE / 10000) + (WINDOWSIZE / 10000))


def get_clusters(vad, min_size, tolerance=10):
    """ Cluster speech segments.

        :param vad: list with labels - voice activity detection
        :type vad: list
        :param min_size: minimal size of window in ms
        :type min_size: int
        :param tolerance: accept given number of frames as speech even when it is marked as silence
        :type tolerance: int
        :returns: clustered speech segments
        :rtype: list
    """
    num_prev = 0
    in_tolerance = 0
    num_clusters = 0
    clusters = {}
    for ii, frame in enumerate(vad):
        if frame:
            num_prev += 1
        else:
            in_tolerance += 1
            if in_tolerance > tolerance:
                if num_prev > min_size:
                    clusters[num_clusters] = (ii - num_prev, ii)
                    num_clusters += 1
                num_prev = 0
                in_tolerance = 0
    return clusters


# TODO:
# - move code in wav2ivec to other files and remove dependency on that script
# - consider making required non-positional arguments into positional arguments
# - allow float input for min/max window size


def main(argv):
    parser = argparse.ArgumentParser(
        description='Extract i-vectors used for diarization from audio wav files.',
        add_help=True, usage='%(prog)s [options]')
    parser.add_argument(
        '-l', '--input-list', dest='input_list', nargs=None, required=True,
        help='list of input files without suffix')
    parser.add_argument(
        '--audio-dir', dest='audio_dir', required=True,
        help='directory with audio files')
    parser.add_argument(
        '--audio-ext', dest='audio_ext', nargs=None, default='.wav',
        help='audio file extension (Default: %(default)s)')
    parser.add_argument(
        '--sad-dir', dest='sad_dir', nargs=None, required=True,
        help='directory with SAD label files')
    parser.add_argument(
        '--sad-ext', dest='sad_ext', nargs=None, default='.lab.gz',
        help='label file extension (Default: %(default)s)')
    parser.add_argument(
        '--out-dir', dest='out_dir', nargs=None, required=True,
        help='output directory for i-vectors')
    parser.add_argument(
        '--ubm-file', dest='ubm_file', nargs=None, required=True,
        help='Universal Background Model file')
    parser.add_argument(
        '--v-file', dest='v_file', nargs=None, required=True,
        help='V Model file')
    parser.add_argument(
        '--min-window-size', dest='min_window_size', type=int, nargs=None, default=1000,
        help='minimal window size (ms) for i-vector extraction (Default: %(default)s ms)')
    parser.add_argument(
        '--max-window-size', dest='max_window_size', type=int, nargs=None, default=2000,
        help='maximal window size (ms) for i-vector extraction (Default: %(default)s ms)')
    parser.add_argument(
        '--sad-tolerance', dest='sad_tolerance', type=int, nargs=None, default=5,
        help='silence tolerance in ms (Default: %(default)s ms)')
    parser.add_argument(
        '-j', '--num-threads', dest='num_threads', type=int, nargs=None, default=1,
        help='number of threads to use (Default: %(default)s)')
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    
    # Load script file.
    with open(args.input_list, 'rb') as f:
        files = [line.rstrip('\n') for line in f]

    # Extract i-vectors in parallel.
    process_files(
        files, args.audio_dir, args.sad_dir, args.out_dir, args.ubm_file,
        args.v_file, args.min_window_size, args.max_window_size,
        args.sad_tolerance, args.audio_ext, args.sad_ext, args.num_threads)


if __name__ == '__main__':
    main(sys.argv[1:])
