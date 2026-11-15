"""Image I/O, registration of B on A, and SNR(k) from two independent half-stacks.

SNR of the full stack, per ring of spatial frequency k (B registered on A, Hann window):
    SNR(k) = Re<F_A F_B*>_k / ( <|F_A - F_B|^2>_k / 4 )
The cutoff k_c is the first k (> 0.05 cycles/px) where SNR < 1.
"""
import numpy as np
import scipy.ndimage as nd
from numpy.lib.stride_tricks import sliding_window_view as swv


# ---------------------------------------------------------------- I/O
def read_fits(path):
    b = open(path, 'rb').read(); i = 0; cards = {}
    while True:
        blk = b[i:i + 2880].decode('ascii', 'ignore'); i += 2880; end = False
        for j in range(0, 2880, 80):
            c = blk[j:j + 80]
            if c[:8].strip() == 'END':
                end = True; break
            if c[8:10] == '= ':
                cards[c[:8].strip()] = c[10:].split('/')[0].strip()
        if end:
            break
    bp = int(cards['BITPIX']); W = int(cards['NAXIS1']); H = int(cards['NAXIS2'])
    nz = int(cards.get('NAXIS3', 1)) if int(cards['NAXIS']) > 2 else 1
    dt = {-32: '>f4', -64: '>f8', 16: '>i2', 32: '>i4', 8: 'u1'}[bp]
    a = np.frombuffer(b[i:i + W * H * nz * np.dtype(dt).itemsize], dt).astype(float)
    a = a * float(cards.get('BSCALE', 1)) + float(cards.get('BZERO', 0))
    return a.reshape(nz, H, W).mean(0)


def read_image(path):
    """2-D float image (colour -> mean of channels), normalised to mean 1."""
    p = path.lower()
    if p.endswith(('.fit', '.fits', '.fts')):
        a = read_fits(path)
    elif p.endswith(('.tif', '.tiff')):
        import tifffile
        a = tifffile.imread(path)
    else:
        from PIL import Image
        a = np.asarray(Image.open(path))
    a = np.asarray(a, dtype=float)
    if a.ndim == 3:
        a = a.mean(-1) if a.shape[-1] in (3, 4) else a.mean(0)
    return a / a.mean()


# ---------------------------------------------------------------- registration
def smoothreg(g, r, B=64, ST=16, sm=2.0):
    """register r on g with a smooth displacement field (64-px blocks, step 16)."""
    H, W = g.shape; w = np.outer(np.hanning(B), np.hanning(B))
    a = swv(nd.gaussian_filter(g, sm), (B, B))[::ST, ::ST]; b = swv(nd.gaussian_filter(r, sm), (B, B))[::ST, ::ST]
    a = (a - a.mean(axis=(2, 3), keepdims=True)) * w; b = (b - b.mean(axis=(2, 3), keepdims=True)) * w
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))), axes=(2, 3))
    ny, nx = c.shape[:2]; fl = c.reshape(ny, nx, -1).argmax(-1)
    i = np.clip(fl // B, 1, B - 2); j = np.clip(fl % B, 1, B - 2)
    I, J = np.meshgrid(range(ny), range(nx), indexing='ij'); z = c[I, J, i, j]
    py = 0.5 * (c[I, J, i - 1, j] - c[I, J, i + 1, j]) / (c[I, J, i - 1, j] - 2 * z + c[I, J, i + 1, j] + 1e-12)
    px = 0.5 * (c[I, J, i, j - 1] - c[I, J, i, j + 1]) / (c[I, J, i, j - 1] - 2 * z + c[I, J, i, j + 1] + 1e-12)
    D = np.stack([i - B // 2 + np.clip(py, -1, 1), j - B // 2 + np.clip(px, -1, 1)])
    D = np.stack([nd.gaussian_filter(nd.median_filter(x, 3), 3.0) for x in D])
    gy, gx = np.meshgrid((np.arange(H) - B // 2) / ST, (np.arange(W) - B // 2) / ST, indexing='ij')
    U = np.stack([nd.map_coordinates(x, [gy, gx], order=3, mode='nearest') for x in D])
    Y, X = np.mgrid[:H, :W].astype(float)
    return nd.map_coordinates(r, [Y - U[0], X - U[1]], order=3, mode='nearest')


def sub_shift(a, b):
    """sub-pixel shift s with a(x) ~ b(x - s) (correlation peak + parabola)."""
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2((a - a.mean()) * w) * np.conj(np.fft.fft2((b - b.mean()) * w)))))
    i, j = np.unravel_index(c.argmax(), c.shape)
    i = int(np.clip(i, 1, c.shape[0] - 2)); j = int(np.clip(j, 1, c.shape[1] - 2))
    py = 0.5 * (c[i - 1, j] - c[i + 1, j]) / (c[i - 1, j] - 2 * c[i, j] + c[i + 1, j])
    px = 0.5 * (c[i, j - 1] - c[i, j + 1]) / (c[i, j - 1] - 2 * c[i, j] + c[i, j + 1])
    return i - a.shape[0] // 2 + py, j - a.shape[1] // 2 + px


def fourier_shift(b, s):
    ky, kx = np.meshgrid(np.fft.fftfreq(b.shape[0]), np.fft.fftfreq(b.shape[1]), indexing='ij')
    return np.real(np.fft.ifft2(np.fft.fft2(b) * np.exp(-2j * np.pi * (ky * s[0] + kx * s[1]))))


def register_fine(a, b, rounds=3):
    """B on A: repeated sub-pixel global shift (Fourier, no blur), then a smooth field for slow deformations."""
    for _ in range(rounds):
        s = sub_shift(a, b)
        if abs(s[0]) < 0.01 and abs(s[1]) < 0.01:
            break
        b = fourier_shift(b, s)
    return smoothreg(a, b)


def int_shift(ref, img):
    """integer (dy, dx) with ref(y, x) ~ img(y - dy, x - dx)."""
    H, W = max(ref.shape[0], img.shape[0]) * 2, max(ref.shape[1], img.shape[1]) * 2
    c = np.real(np.fft.ifft2(np.fft.fft2(ref - ref.mean(), (H, W)) * np.conj(np.fft.fft2(img - img.mean(), (H, W)))))
    i, j = np.unravel_index(c.argmax(), c.shape)
    return (i - H if i > H // 2 else i), (j - W if j > W // 2 else j)


def _ncc(ref, img, dy, dx):
    y0, x0 = max(0, dy), max(0, dx); y1, x1 = min(ref.shape[0], img.shape[0] + dy), min(ref.shape[1], img.shape[1] + dx)
    if y1 - y0 < 50 or x1 - x0 < 50:
        return -1
    a = ref[y0:y1, x0:x1]; b = img[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return np.corrcoef(a[::3, ::3].ravel(), b[::3, ::3].ravel())[0, 1]


ORIENT = {'id': lambda z: z, 'flipud': lambda z: z[::-1], 'fliplr': lambda z: z[:, ::-1], 'rot180': lambda z: z[::-1, ::-1]}


def align_to(ref, img):
    """best orientation and integer shift of img on ref: (ncc, orientation, dy, dx)."""
    best = None
    for nm, f in ORIENT.items():
        z = f(img); dy, dx = int_shift(ref, z); r = _ncc(ref, z, dy, dx)
        if best is None or r > best[0]:
            best = (r, nm, dy, dx)
    return best


# ---------------------------------------------------------------- SNR from halves
def snr_halves(A, B, border=50, dk=0.004, grid_free=True, grid_halfwidth=7.5):
    """SNR(k) of the full stack from the halves (B already roughly aligned): returns k, SNR, cutoff k_c, modes per ring.
    grid_free: leave out the Fourier modes within +-grid_halfwidth degrees of the pixel axes, where fixed column/row
    patterns of the sensor (common to both halves) live; costs ~1/6 of the modes, makes the estimate robust to them."""
    Br = smoothreg(A, B); A = A / A.mean(); Br = Br / Br.mean()
    bd = min(border, min(A.shape) // 8)
    a = A[bd:-bd, bd:-bd]; b = Br[bd:-bd, bd:-bd]
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    FA = np.fft.fft2((a - a.mean()) * w); FB = np.fft.fft2((b - b.mean()) * w)
    K = np.hypot(*np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing='ij'))
    kb = np.arange(0, 0.5, dk); S = np.zeros(len(kb)); n = np.zeros(len(kb))
    keep = np.ones_like(K, bool)
    if grid_free:
        ky_, kx_ = np.meshgrid(np.fft.fftfreq(a.shape[0]), np.fft.fftfreq(a.shape[1]), indexing='ij')
        ang = np.degrees(np.arctan2(ky_, kx_)) % 90
        keep = (ang > grid_halfwidth) & (ang < 90 - grid_halfwidth)
    for i, k0 in enumerate(kb):
        m = (abs(K - k0) < dk) & keep; n[i] = ((abs(K - k0) < dk / 2) & keep).sum() / 2
        if m.sum():
            S[i] = max(np.real(FA[m] * np.conj(FB[m])).mean() / ((abs(FA[m] - FB[m]) ** 2).mean() / 4), 0)          # = SNR of the full stack (A+B)/2
    S = np.convolve(S, np.ones(3) / 3, 'same'); S[0] = S[1]
    below = (S < 1) & (kb > 0.05); icut = np.argmax(below) if below.any() else len(kb) - 1
    S[icut:] = 0
    return kb, S, kb[icut], n


def information(kb, S, n):
    """information content of the stack in bits: sum over modes of log2(1 + SNR) (Shannon capacity)."""
    return float(np.sum(n * np.log2(1 + S)))


def directional_check(A, B, kmin=0.25, kmax=0.45, nsec=12, border=40):
    """ratio max/median of the SNR in angular sectors at high k (B registered on A).
    A fixed pattern shared by both halves (sensor column/row pattern, interpolation moire) shows up as a strong
    excess along one direction: values >> 3 mean the SNR at high k is inflated by a common artefact."""
    a, b = A[border:-border, border:-border], B[border:-border, border:-border]; n0, n1 = a.shape
    w = np.outer(np.hanning(n0), np.hanning(n1))
    FA = np.fft.fft2((a - a.mean()) * w); FB = np.fft.fft2((b - b.mean()) * w)
    ky, kx = np.meshgrid(np.fft.fftfreq(n0), np.fft.fftfreq(n1), indexing='ij'); k = np.hypot(ky, kx)
    th = (np.arctan2(ky, kx) + np.pi / nsec / 2) % np.pi      # sector 0 centred on 0 deg
    m = (k > kmin) & (k < kmax); C = np.real(FA * np.conj(FB)); D = abs(FA - FB) ** 2 / 4
    sec = []
    for j in range(nsec):
        mm = m & (np.floor(th / (np.pi / nsec)) == j)
        sec.append(max(C[mm].sum() / max(D[mm].sum(), 1e-30), 0) * 2)
    sec = np.array(sec)
    # sensor patterns are aligned with the pixel grid: compare the sectors centred on 0 and 90 deg with the others
    # (the Moon itself can be strongly anisotropic, e.g. grazing light, but not specifically along the grid)
    grid = [0, nsec // 2]
    other = np.delete(sec, grid); med = max(np.median(other), 0.3)
    j = grid[int(np.argmax(sec[grid]))]
    return float(sec[j] / med), float(np.degrees(j * np.pi / nsec)), sec
