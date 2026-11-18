"""Tests with synthetic data whose answer is known in advance.
Run with  `python -m pytest tests`  or simply  `python tests/test_lucky_snr.py`."""
import os
import struct
import sys
import tempfile
import gc
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lucky_snr import core, split                       # noqa: E402
from lucky_snr.ser import SER                           # noqa: E402


def texture(n, rng, slope=1.5):
    """smooth 'lunar-like' random texture with power spectrum ~ k^(-2 slope), mean 1."""
    ky, kx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing='ij')
    k = np.hypot(kx, ky); k[0, 0] = 1
    t = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * k ** -slope))
    return 1 + 0.05 * t / t.std()


def true_snr(S, sigma, border=50):
    """exact SNR(k) of the full stack (A+B)/2 when A = S + n_A, B = S + n_B, white noise of std sigma per half."""
    a = S[border:-border, border:-border]
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fft2((a - a.mean()) * w)
    K = np.hypot(*np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing='ij'))
    noise_full = sigma ** 2 * (w ** 2).sum() / 2
    kb = np.arange(0.02, 0.5, 0.004)
    return kb, np.array([(abs(F[abs(K - k0) < 0.004]) ** 2).mean() / noise_full for k0 in kb])


def test_snr_from_halves_matches_truth():
    rng = np.random.default_rng(1)
    S = texture(512, rng); sigma = 0.01
    A = S + sigma * rng.standard_normal(S.shape)
    B = S + sigma * rng.standard_normal(S.shape)
    kb, snr, kc, _ = core.snr_halves(A, B, grid_free=False)
    kt, st = true_snr(S, sigma)
    for k in (0.04, 0.08, 0.12, 0.16):
        ratio = np.interp(k, kb, snr) / np.interp(k, kt, st)
        assert 0.75 < ratio < 1.35, 'SNR at k=%.2f off by a factor %.2f' % (k, ratio)
    k_true = kt[np.argmax((st < 1) & (kt > 0.05))]
    assert abs(kc - k_true) < 0.02, 'cutoff %.3f, expected %.3f' % (kc, k_true)


def test_more_noise_means_lower_cutoff():
    rng = np.random.default_rng(2)
    S = texture(512, rng)
    cut = []
    for sigma in (0.005, 0.02):
        A = S + sigma * rng.standard_normal(S.shape); B = S + sigma * rng.standard_normal(S.shape)
        cut.append(core.snr_halves(A, B)[2])
    assert cut[0] > cut[1] + 0.03, cut


def test_identical_halves_are_flagged():
    """a common nuisance (A == B up to tiny noise) must push the cutoff to the end of the band: compare warns above 0.48"""
    rng = np.random.default_rng(3)
    S = texture(256, rng) + 0.01 * rng.standard_normal((256, 256))
    kc = core.snr_halves(S, S + 1e-5 * rng.standard_normal(S.shape))[2]
    assert kc > 0.48, kc


def write_test_ser(path, frames):
    """mono 16-bit SER with the standard 178-byte header."""
    n, h, w = frames.shape
    hdr = bytearray(178)
    hdr[:14] = b'LUCAM-RECORDER'
    struct.pack_into('<7i', hdr, 14, 0, 0, 0, w, h, 16, n)
    with open(path, 'wb') as f:
        f.write(bytes(hdr)); f.write(frames.astype('<u2').tobytes())


def test_split_equal_halves_thue_morse_and_drift():
    rng = np.random.default_rng(4)
    big = texture(256, rng); N, side = 160, 64
    drift = np.c_[np.linspace(0, 20, N), np.linspace(0, -15, N)].round().astype(int)     # (dy, dx) per frame
    fr = np.array([np.roll(big, tuple(d), (0, 1))[60:188, 60:188] for d in drift])        # 128 x 128 frames
    fr = (fr / fr.max() * 30000 + rng.normal(0, 50, fr.shape)).clip(0, 65535)
    # ignore_cleanup_errors: on Windows a file still mapped in memory cannot be deleted
    with tempfile.TemporaryDirectory(**({"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {})) as tmp:
        src = os.path.join(tmp, 'test.ser'); write_test_ser(src, fr)
        out = os.path.join(tmp, 'out', 'test')
        args = types.SimpleNamespace(ser=src, out=out, crop=side, center=None, block='20', scheme='thue-morse', frames=None)
        split.main(args, log=lambda *a: None)
        A, B = SER(out + '_A.ser'), SER(out + '_B.ser')
        assert A.n == B.n and A.n + B.n >= N - 1
        assert (A.width, A.height) == (side, side)
        tab = np.loadtxt(out + '_split.csv', delimiter=',', skiprows=1).astype(int)
        half = tab[:, 1]
        for j, s in enumerate(split.thue_morse(N // 20)):        # blocks of 20 frames follow the Thue-Morse sequence
            blk = half[j * 20:(j + 1) * 20]
            assert np.all(blk[blk >= 0] == s), 'block %d' % j
        # the crop follows the drift: the same scene in the first and the last frame of a half
        a0, a1 = np.asarray(A.frames[0], float), np.asarray(A.frames[A.n - 1], float)
        assert np.corrcoef(a0.ravel(), a1.ravel())[0, 1] > 0.95
        del A, B, a0, a1                 # release the memory-mapped files before the folder is removed (Windows)
        gc.collect()


def test_dark_area_is_not_a_broken_stack():
    """a featureless dark region (night side of the terminator) must not raise the 'broken stack' warning,
    while a misaligned quarter (structure in A and B, but different) must."""
    import tifffile
    from lucky_snr import compare
    rng = np.random.default_rng(5)
    S = texture(600, rng); S[:, 300:] = 0.02                                   # right half: night side
    A = S + 0.003 * rng.standard_normal(S.shape)
    B = S + 0.003 * rng.standard_normal(S.shape)
    Bbad = B.copy(); Bbad[:300, :300] = np.roll(B[:300, :300], 25, axis=1)    # top-left quarter shifted
    with tempfile.TemporaryDirectory(**({"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {})) as tmp:
        files = {}
        for nm, im in (('A', A), ('B', B), ('Bbad', Bbad)):
            files[nm] = os.path.join(tmp, nm + '.tif'); tifffile.imwrite(files[nm], im.astype(np.float32))
        for bname, expect in (('B', False), ('Bbad', True)):
            lines = []
            res, _ = compare.run([('test', files['A'], files[bname])], log=lambda *a: None)
            compare.report(res, (.04, .08, .12), log=lines.append)
            flagged = any('WARNING' in l for l in lines)
            assert flagged == expect, (bname, [l for l in lines if 'WARNING' in l or 'field' in l])


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok ', name)
    print('all tests passed')
