"""Split a .ser video into two independent halves A and B (blocks of consecutive frames, Thue-Morse order),
optionally cropped to a square that follows the drift. The halves are raw .ser files (Bayer kept) for any stacker.

Block length: frames closer than the correlation time tau_c share seeing and must stay in the same half.
tau_c is estimated from the correlation of the frame residuals (band-pass, slow trends removed):
    rho(D) = < corr(r_i, r_{i+D}) >,   tau_c = first D with rho(D) < rho(1)/e ;   block = max(2 tau_c, 20).
Thue-Morse: block j goes to half s_j = (-1)^{t_j}, t_j = parity of the binary digits of j.
On 2^p blocks  sum_j s_j j^m = 0  for m < p: slow polynomial drifts cancel between A and B.
"""
import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
from .ser import SER, write_ser
from . import core


def thue_morse(n):
    return np.array([bin(j).count('1') % 2 for j in range(n)])


def ranges(spec, n):
    if not spec:
        return np.arange(n)
    out = []
    for r in spec.split(','):
        a, b = (r.split('-') + [None])[:2]
        out.append(np.arange(int(a), (int(b) if b else int(a)) + 1))
    k = np.concatenate(out)
    return k[(k >= 0) & (k < n)]


def drift(S, frames, binning, log):
    """global shift (dy, dx) of every frame w.r.t. a reference, in full-resolution pixels (median-filtered)."""
    ref = np.mean([S.luminance(k, binning) for k in frames[len(frames) // 2: len(frames) // 2 + 10]], 0)
    w = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    Fr = np.conj(np.fft.fft2((ref - ref.mean()) * w))
    sh = np.zeros((len(frames), 2)); t0 = time.time()
    for j, k in enumerate(frames):
        a = S.luminance(k, binning)
        c = np.real(np.fft.ifft2(np.fft.fft2((a - a.mean()) * w) * Fr))
        i, jj = np.unravel_index(c.argmax(), c.shape)
        H, W = c.shape
        sh[j] = [(i - H if i > H // 2 else i), (jj - W if jj > W // 2 else jj)]
        if j % 500 == 0:
            log('  drift: frame %d/%d  (%.0f s)' % (j, len(frames), time.time() - t0))
    sh = np.stack([nd.median_filter(sh[:, c], 9, mode='nearest') for c in (0, 1)], 1)   # remove isolated misfires
    return sh * binning, ref


def choose_center(ref, binning, shifts, side, H, W):
    """centre (full-res, reference frame) of the most detailed square that stays inside every frame."""
    g = np.hypot(*np.gradient(nd.gaussian_filter(ref, 1)))
    half = side / 2 / binning
    E = nd.uniform_filter(g, int(2 * half) + 1)
    bright = nd.uniform_filter((ref > 0.25 * np.percentile(ref, 99)).astype(float), int(2 * half) + 1)
    ys, xs = np.mgrid[:ref.shape[0], :ref.shape[1]]
    lo_y = half - shifts[:, 0].min() / binning + 4; hi_y = ref.shape[0] - half - shifts[:, 0].max() / binning - 4
    lo_x = half - shifts[:, 1].min() / binning + 4; hi_x = ref.shape[1] - half - shifts[:, 1].max() / binning - 4
    ok = (ys > lo_y) & (ys < hi_y) & (xs > lo_x) & (xs < hi_x) & (bright > 0.98)
    if not ok.any():
        sys.exit('the crop does not fit inside all frames: use a smaller --crop or --frames')
    E = np.where(ok, E, -1); i, j = np.unravel_index(E.argmax(), E.shape)
    return int(i * binning + binning // 2), int(j * binning + binning // 2)


def correlation_time(S, frames, crops, log, nmax=600):
    """tau_c (frames) from band-pass residual correlations of consecutive frames (up to nmax frames)."""
    sel = frames[:nmax]; b = 2 if S.bayer else 1
    F = []
    for k, (y0, x0, h, w) in zip(sel, crops[:nmax]):
        a = np.asarray(S.frames[k][y0:y0 + h, x0:x0 + w], dtype=np.float32)
        if a.ndim == 3: a = a.mean(-1)
        if b > 1: a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean((1, 3))
        a = a / a.mean()
        F.append(nd.gaussian_filter(a, 1.0) - nd.gaussian_filter(a, 4.0))      # band k ~ 0.04-0.16 cycles/px
    F = np.array(F); F -= nd.uniform_filter1d(F, 101, axis=0, mode='nearest')    # remove slow trends
    F = F.reshape(len(F), -1); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    rho = np.array([np.mean(np.sum(F[:-d] * F[d:], 1)) for d in range(1, 61)])
    below = np.where(rho < rho[0] / np.e)[0]
    tau = int(below[0]) + 1 if len(below) else 60
    log('  correlation of residuals rho(D), D=1,2,4,8,16,32: ' + ' '.join('%.2f' % rho[d - 1] for d in (1, 2, 4, 8, 16, 32)))
    return tau, rho


def main(a, log=print):
    S = SER(a.ser); t0 = time.time()
    log('%s: %d frames %dx%d, %d bit, %s' % (os.path.basename(a.ser), S.n, S.width, S.height, S.depth,
                                              'Bayer ' + S.bayer if S.bayer else ('RGB' if S.channels == 3 else 'mono')))
    frames = ranges(a.frames, S.n)
    side = a.crop if a.crop else 0
    if side:
        side = side // 2 * 2
        binning = 4 if max(S.width, S.height) > 1000 else 2
        log('estimating drift...')
        shifts, ref = drift(S, frames, binning, log)
        log('  drift range: dy %.0f..%.0f  dx %.0f..%.0f px' % (shifts[:, 0].min(), shifts[:, 0].max(), shifts[:, 1].min(), shifts[:, 1].max()))
        cy, cx = a.center if a.center else choose_center(ref, binning, shifts, side, S.height, S.width)
        crops = []
        for dy, dx in shifts:
            y0 = int(round(cy - side / 2 + dy)); x0 = int(round(cx - side / 2 + dx))     # content of frame k sits at ref + (dy, dx)
            y0 -= y0 % 2; x0 -= x0 % 2                       # keep the Bayer phase
            y0 = min(max(y0, 0), S.height - side); x0 = min(max(x0, 0), S.width - side)
            crops.append((y0, x0, side, side))
        log('crop %dx%d centred at (y %d, x %d) of the reference frame, following the drift' % (side, side, cy, cx))
    else:
        crops = [(0, 0, S.height, S.width)] * len(frames)
    if a.block == 'auto':
        tau, _ = correlation_time(S, frames, crops, log)
        block = max(2 * tau, 20)
        log('correlation time tau_c = %d frames -> blocks of %d frames' % (tau, block))
    else:
        block = int(a.block)
    nb = int(np.ceil(len(frames) / block))
    sign = thue_morse(nb) if a.scheme == 'thue-morse' else np.arange(nb) % 2
    half = sign[np.arange(len(frames)) // block].copy()
    # same number of frames in A and B (some stackers, e.g. AutoStakkert! in batch, choke on different counts):
    # the surplus frames at the end of the larger half are left out (marked -1 in the csv)
    nA, nB = (half == 0).sum(), (half == 1).sum()
    big = 0 if nA > nB else 1
    extra = abs(nA - nB)
    if extra:
        half[np.where(half == big)[0][-extra:]] = -1
        log('%d frames left out so that A and B have the same number of frames' % extra)
    out = a.out or os.path.splitext(a.ser)[0] + ('_crop%d' % side if side else '')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for h, nm in ((0, 'A'), (1, 'B')):
        idx = np.where(half == h)[0]
        write_ser(out + '_%s.ser' % nm, S, frames[idx], [crops[i] for i in idx])
        log('written %s_%s.ser: %d frames' % (out, nm, len(idx)))
    np.savetxt(out + '_split.csv', np.c_[frames, half, np.array(crops)[:, :2]], fmt='%d', delimiter=',',
               header='frame,half(0=A 1=B -1=unused),crop_y0,crop_x0', comments='')
    log('%s split, %d blocks of %d frames (%.0f s). Now stack %s_A.ser and %s_B.ser with the SAME settings.'
        % (a.scheme, nb, block, time.time() - t0, os.path.basename(out), os.path.basename(out)))
