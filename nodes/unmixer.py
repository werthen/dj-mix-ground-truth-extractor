"""
Unmixing node
"""

from itertools import tee
import logging

import numpy as np
import scipy as sp

import librosa
from librosa import resample

import cvxpy as cvx

N_FFT = 4410


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    fst, snd = tee(iterable)

    next(snd, None)
    return zip(fst, snd)


def preprocess(x, xsr, y, ysr, z, zsr):
    nsr = 44100
    n_fft = N_FFT
    hop_length = int(2 * n_fft)

    assert len(x) == len(y) == len(z)

    D_x = librosa.stft(resample(x, xsr, nsr), hop_length=hop_length, n_fft=n_fft)
    D_y = librosa.stft(resample(y, ysr, nsr), hop_length=hop_length, n_fft=n_fft)
    D_z = librosa.stft(resample(z, zsr, nsr), hop_length=hop_length, n_fft=n_fft)

    return D_x, D_y, D_z


def generate_problem(nD_x, nD_y, nD_z, tv, dist):
    # Problem data.
    m, n = nD_x.shape

    X = np.abs(nD_x) ** 2 / N_FFT
    Y = np.abs(nD_y) ** 2 / N_FFT
    Z = np.abs(nD_z) ** 2 / N_FFT

    a = cvx.Variable(n)
    b = cvx.Variable(n)

    tv_coeff = cvx.Parameter(nonneg=True, value=tv, name='tv')
    dist_coeff = cvx.Parameter(nonneg=True, value=dist, name='dist')

    combination = X * cvx.diag(a) + Y * cvx.diag(b)
    objective = cvx.Minimize(
        cvx.norm(Z - combination, 'fro') +
        dist_coeff * (cvx.sum_squares(a) + cvx.sum_squares(b)) +
        tv_coeff * (cvx.tv(a) + cvx.tv(b))
    )

    # constraints = []
    constraints = [
        0 <= a, a <= 1,
        0 <= b, b <= 1
    ]

    for i in range(n - 1):
        constraints.append(a[i] >= a[i + 1])
        constraints.append(b[i] <= b[i + 1])

    prob = cvx.Problem(objective, constraints)

    return prob


def generate_problem_fade_constrained(nD_x, nD_y, nD_z, tv, dist):
    # Problem data.
    m, n = nD_x.shape

    X = (np.abs(nD_x) ** 2) / N_FFT
    Y = (np.abs(nD_y) ** 2) / N_FFT
    Z = (np.abs(nD_z) ** 2) / N_FFT

    a = cvx.Variable(n)

    tv_coeff = cvx.Parameter(nonneg=True, value=tv, name='tv')
    dist_coeff = cvx.Parameter(nonneg=True, value=dist, name='dist')

    combination = X * cvx.diag(a) + Y * cvx.diag(1 - a)
    objective = cvx.Minimize(
        cvx.norm(combination - Z, 'fro') +
        dist_coeff * cvx.sum_squares(a) +
        tv_coeff * cvx.tv(a)
    )

    constraints = [0 <= a, a <= 1]

    for i in range(n - 1):
        constraints.append(a[i] >= a[i + 1])

    prob = cvx.Problem(objective, constraints)

    return prob


def unmix_convex_cf(x, xsr, y, ysr, z, zsr, tv=0, dist=0):
    D_x, D_y, D_z = preprocess(x, xsr, y, ysr, z, zsr)
    problem = generate_problem(D_x, D_y, D_z, tv=tv, dist=dist)

    problem.solve()

    transform = lambda x: np.sqrt(np.clip(x.value, 0, 1))
    return transform(problem.variables()[0]), transform(problem.variables()[1])


def unmix_convex(x, xsr, y, ysr, z, zsr, tv=0, dist=0):
    D_x, D_y, D_z = preprocess(x, xsr, y, ysr, z, zsr)
    problem = generate_problem_fade_constrained(D_x, D_y, D_z, tv=tv, dist=dist)

    problem.solve()

    a_hat = np.clip(problem.variables()[0].value, 0, 1)

    a_track = np.sqrt(a_hat)
    b_track = np.sqrt(1 - a_hat)

    return a_track, b_track


# TODO: remove sample rates, they're for compatibility purposes
def unmix_linear_cf(x, _xsr, y, _ysr, z, _zsr):
    xf = np.matrix(np.abs(librosa.stft(x)) ** 2 / 2048)
    yf = np.matrix(np.abs(librosa.stft(y)) ** 2 / 2048)
    zf = np.matrix(np.abs(librosa.stft(z)) ** 2 / 2048)

    res = []

    for i in range(yf.shape[1]):
        x_min = yf[:, i] - xf[:, i]

        term1 = np.multiply(yf[:, i], x_min)
        term2 = np.multiply(zf[:, i], x_min)
        denom = np.power(x_min, 2)

        x = np.sum(term1 - term2) / np.sum(denom)

        res.append(x)

    mf = lambda x: sp.signal.medfilt(x, 201)
    a_hat = mf(np.clip(np.nan_to_num(res), 0, 1))

    return np.sqrt(a_hat), np.sqrt(1 - a_hat)


# TODO: remove sample rates, they're for compatibility purposes
def unmix_linear(x, _xsr, y, _ysr, z, _zsr):
    F = 101

    N_FFT = 4092

    xf = np.matrix(np.abs(librosa.stft(x, n_fft=N_FFT)) ** 2 / N_FFT)
    yf = np.matrix(np.abs(librosa.stft(y, n_fft=N_FFT)) ** 2 / N_FFT)
    zf = np.matrix(np.abs(librosa.stft(z, n_fft=N_FFT)) ** 2 / N_FFT)

    res = None

    for i in np.arange(0, zf.shape[1]):
        Yn = zf[:, i]
        Xn = np.concatenate((xf[:, i], yf[:, i]), axis=1)
        A_hat = np.linalg.pinv(Xn) * Yn

        if res is None:
            res = A_hat
        else:
            res = np.concatenate((res, A_hat), axis=1)

    mf = lambda x: sp.signal.medfilt(x, F)
    transform = lambda x: np.sqrt(mf(np.clip(x, 0, 1)))

    return transform(res.tolist()[0]), transform(res.tolist()[1])


def stretch_clip(audio_data, stretch):
    """Stretch the audio in order to pitch up or pitch down the audio"""
    rng = np.arange(0, len(audio_data))

    # Stretcher function, interpolates a signal
    stretcher = lambda x: np.interp(x, rng, audio_data)

    # Amount of elements in stretched signal
    n_elems_s = int(audio_data.shape[0] / stretch)

    return stretcher(np.arange(n_elems_s) * stretch)


def calc_offset(a, x, b):
    return a * x + b


class Unmixer:
    provides = ['faders']
    needs = ['xfades', 'src_track_paths']

    METHODS = {
        'linear': unmix_linear,
        'linear_cf': unmix_linear_cf,
        'convex': unmix_convex,
        'convex_cf': unmix_convex_cf,
    }

    def __init__(self, method='linear'):
        self.method = method

    def process(self, state):
        xfades = state['xfades']
        prefix = state['src_track_paths']

        faders = []

        for xf in xfades:
            logging.info('unmixing %s', ', '.join(map(str, xf['src_tracks'])))
            tr1, tr2 = xf['src_tracks']
            bounds = xf['mix']
            duration = bounds[1] - bounds[0]

            off1 = calc_offset(tr1.slope, bounds[0], tr1.offset)
            off2 = calc_offset(tr2.slope, bounds[0], tr2.offset)
            x, xsr = librosa.load(prefix + '/' + xf['src_tracks'][0].path.split('/')[-1], offset=off1,
                                  duration=duration / tr1.slope)
            y, ysr = librosa.load(prefix + '/' + xf['src_tracks'][1].path.split('/')[-1], offset=off2,
                                  duration=duration / tr2.slope)
            z, zsr = librosa.load(state['mix_path'], offset=bounds[0], duration=duration)

            if xsr != ysr != zsr:
                raise Exception('Sample rates do not match')

            x = stretch_clip(x, tr1.slope)
            y = stretch_clip(y, tr2.slope)

            x = np.pad(x, (0, len(z) - len(x)), 'constant')
            y = np.pad(y, (len(z) - len(y), 0), 'constant')

            # track1 = xf['src_tracks'][0].path.split('/')[-1]
            # track2 = xf['src_tracks'][1].path.split('/')[-1]

            # librosa.output.write_wav('mix_output_' + track1 + '_0.wav', np.vstack((x, z)), xsr)
            # librosa.output.write_wav('mix_output_' + track2 + '_1.wav', np.vstack((y, z)), xsr)

            faders.append(self.METHODS[self.method](x, xsr, y, ysr, z, zsr))

        return {**state, 'faders': faders}
