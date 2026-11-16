"""Compare stacks made from the two halves A and B of the same video: SNR(k), cutoff, information, ranking."""
import csv
import glob
import os
import re
import numpy as np
import scipy.ndimage as nd
from . import core, as3

IMG = ('.tif', '.tiff', '.fit', '.fits', '.fts', '.png')
HALF = re.compile(r'(?:(?<=[_\-. ])|^)(?:meta|half)?([AB])(?=$|[_\-. ])', re.I)


def half_of(name):
    """('A' or 'B', stem without the half token) from a video or stack name, or (None, name)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    ms = list(HALF.finditer(stem))
    if not ms:
        return None, stem
    m = ms[-1]
    return m.group(1).upper(), stem[:m.start()] + '#' + stem[m.end():]


def discover(folders):
    """AS! stacks in the folders, paired A/B by identical settings and video stem."""
    groups = {}
    for fo in folders:
        for p in sorted(glob.glob(os.path.join(fo, '**', '*'), recursive=True)):
            if not p.lower().endswith(IMG):
                continue
            if os.path.basename(p).startswith('lucky_snr'):
                continue
            d = as3.describe(p)
            if d is not None:                                   # AutoStakkert!: settings from the .as3 file
                h, stem = half_of(d['source'] or os.path.basename(p))
                if h is None:
                    continue
                key = ('AS', stem, d['ap_size'], d['select'], d['drizzle'], d['quality'])
            else:                                               # any other stacker: pair by the A/B token in the name
                h, stem = half_of(p)
                if h is None:
                    continue
                sub = os.path.relpath(os.path.dirname(p), fo)
                lab = stem.replace('#', '').strip('_- .')
                d = dict(stack=p, label=(sub + ': ' + lab) if sub != '.' else lab, sharpened=False)
                key = ('name', os.path.dirname(p), stem)
            groups.setdefault(key, {})[h] = d
    pairs, lonely = [], []
    for key, g in groups.items():
        if 'A' in g and 'B' in g:
            pairs.append((g['A'].get('label') or as3.label(g['A']), g['A']['stack'], g['B']['stack'], g['A']['sharpened'] or g['B']['sharpened']))
        else:
            lonely.extend(x['stack'] for x in g.values())
    def order(p):
        m = re.search(r'AP(\S+) (\d+)', p[0])
        return (int(m.group(1)) if m and m.group(1).isdigit() else 0, int(m.group(2)) if m else 0, p[0])
    return sorted(pairs, key=order), lonely


def _scale(img, ref):
    r = np.sqrt(img.size / ref.size)
    for s in (1.0, 1.5, 2.0, 3.0):
        if abs(r - s) / s < 0.08:
            return s
    return r


def run(pairs, side=512, ks=(.04, .08, .12, .16, .20, .24, .28), log=print):
    """pairs: list of (name, A path, B path[, sharpened]). Returns list of result dicts."""
    rows, ref = [], None
    for p in pairs:
        name, fa, fb = p[:3]
        A, B = core.read_image(fa), core.read_image(fb)
        if ref is None:
            ref = A
        s = _scale(A, ref)
        As = nd.zoom(A, 1 / s, order=1) if s != 1 else A
        r, oa, dya, dxa = core.align_to(ref, As); A = core.ORIENT[oa](A)
        Bs = nd.zoom(core.ORIENT[oa](B), 1 / s, order=1) if s != 1 else core.ORIENT[oa](B)
        rb, ob, dyb, dxb = core.align_to(ref, Bs)
        B = core.ORIENT[ob](core.ORIENT[oa](B))
        log('%-28s scale %.1f  A: %-6s ncc %.3f   B: %-6s ncc %.3f' % (name, s, oa, r, ob, rb))
        if min(r, rb) < 0.5:
            log('   -> SKIPPED: does not match the first stack (a different video or target?)')
            continue
        problems = []
        ra = A.shape[0] * A.shape[1]; rb = B.shape[0] * B.shape[1]
        if max(ra, rb) > 1.1 * min(ra, rb):
            problems.append('A and B have different sizes (%dx%d vs %dx%d)' % (A.shape + B.shape))
        for nm_, im in (('A', A), ('B', B)):
            z = (im <= 0).mean()
            if z > 0.005:
                problems.append('%s has %.1f%% empty (black) pixels' % (nm_, 100 * z))
        rows.append(dict(name=name, A=A, B=B, s=s, sa=(dya, dxa), sb=(dyb, dxb), sharpened=(p[3] if len(p) > 3 else False),
                         problems=problems))
    # common square, in reference pixels
    lim = []
    for r in rows:
        for img, (dy, dx) in ((r['A'], r['sa']), (r['B'], r['sb'])):
            lim.append((max(0, dy), max(0, dx), dy + img.shape[0] / r['s'], dx + img.shape[1] / r['s']))
    Y0 = max(l[0] for l in lim); X0 = max(l[1] for l in lim); Y1 = min(l[2] for l in lim); X1 = min(l[3] for l in lim)
    L = int(min(side, Y1 - Y0, X1 - X0)) // 2 * 2
    cy, cx = (Y0 + Y1) / 2, (X0 + X1) / 2
    log('common region: %d x %d px (reference pixels)' % (L, L))
    out = []
    for r in rows:
        s = r['s']; Ls = int(round(L * s)) // 2 * 2
        def cut(img, sh):
            y = int(round((cy - L / 2 - sh[0]) * s)); x = int(round((cx - L / 2 - sh[1]) * s))
            return img[y:y + Ls, x:x + Ls]
        a_, b_ = cut(r['A'], r['sa']), cut(r['B'], r['sb'])
        b_ = core.register_fine(a_, b_)
        kb, S, kc, n = core.snr_halves(a_, b_)
        aniso, ang, _ = core.directional_check(a_, b_)
        kb_ref = kb * s                          # cycles per reference pixel
        sub = []
        for qy in (0, Ls // 2):
            for qx in (0, Ls // 2):
                sa_, sb_ = a_[qy:qy + Ls // 2, qx:qx + Ls // 2], b_[qy:qy + Ls // 2, qx:qx + Ls // 2]
                kq, sq, _, _ = core.snr_halves(sa_, sb_, border=16)
                sub.append([np.interp(k / s, kq, sq) for k in ks])
        out.append(dict(name=r['name'], k=kb_ref, snr=S, cutoff=kc * s, bits=core.information(kb, S, n) / s ** 2,
                        at=[np.interp(k / s, kb, S) for k in ks], sub=np.array(sub), sharpened=r['sharpened'],
                        aniso=aniso, ang=ang, problems=r['problems']))
    return out, L


def equivalent(res, best):
    """rows whose median SNR ratio to the best (over the table frequencies) is within the scatter between
    sub-squares (at least 5%)."""
    out = []
    for r in res:
        if r is best:
            continue
        at, bt = np.array(r['at']), np.array(best['at'])
        ok = bt > 5
        if not ok.any():
            continue
        q = np.median(at[ok] / bt[ok])
        m = [np.median(r['sub'][s_][best['sub'][s_] > 5] / best['sub'][s_][best['sub'][s_] > 5])
             for s_ in range(len(r['sub'])) if (best['sub'][s_] > 5).any()]
        sd = np.std(m, ddof=1) if len(m) > 1 else 0.0
        if q >= 1 - max(0.05, sd):
            out.append(r)
    return out


def advice(best, eq, optics, unreliable=False):
    """plain-language suggestions from the cutoff of the best stack."""
    kc = best['cutoff']; L = ['', 'what this means']
    if unreliable:
        L.append('  * the cutoff of the best stack is not reliable (see the warning): no advice on sampling or sharpening limits.')
        if eq:
            L.append('  * several settings are statistically equivalent at low/medium k: choose freely among them.')
        return L
    L.append('  * finest detail supported by the data: period 1/k_c = %.1f px (features ~%.1f px across).'
             % (1 / kc, 0.5 / kc))
    L.append('    Above k_c = %.3f cycles/px the two halves no longer agree: linear sharpening (wavelets, Wiener,'
             ' Richardson-Lucy) there amplifies noise only.' % kc)
    L.append('    Learned tools (e.g. neural-network deconvolution) may render detail beyond k_c, but that detail'
             ' cannot be verified from these data.')
    if kc > 0.42:
        L.append('  * k_c is close to the sampling limit (0.5 cycles/px): the image is UNDERSAMPLED. More focal length'
                 ' (Barlow) or drizzle would recover finer detail.')
    elif kc < 0.22:
        L.append('  * k_c is well below the sampling limit (0.5): the image is OVERSAMPLED for this seeing. A shorter'
                 ' focal length or 2x2 binning would give more signal per pixel with no loss of detail.')
    else:
        L.append('  * sampling is well matched to the detail actually recorded (k_c between 0.22 and 0.42 cycles/px).')
    if optics:
        sc = optics['scale']
        L.append('  * pixel scale %.3f arcsec/px: finest detail %.2f arcsec (period).' % (sc, sc / kc))
        if optics.get('fc'):
            r = kc / optics['fc']
            if r > 1.05:
                L.append('  * WARNING: k_c is above the diffraction cutoff %.3f cycles/px computed from --pixel/--focal/'
                         '--aperture/--wavelength: no real detail can exist there. Check those values (focal length with'
                         ' the Barlow?) or the stacks (sharpened? drizzled?).' % optics['fc'])
            else:
                L.append('  * diffraction cutoff %.3f cycles/px (lambda %.0f nm): the stack reaches %.0f%% of it%s'
                         % (optics['fc'], optics['lam'], 100 * r, ' - close to the optical limit.' if r > 0.8 else
                            ' - limited by seeing/stacking, not by the optics.'))
    if eq:
        L.append('  * several settings are statistically equivalent: choose freely among them (e.g. the one with more'
                 ' frames for a smoother result).')
    return L


def _g(x):
    return '%.0f' % x if x < 1000 else '%.0fk' % (x / 1e3)


def _rng(lo, hi):
    return _g(lo) + '-' + _g(hi)


def report(res, ks, path=None, log=print, optics=None):
    w = max(max(len(r['name']) for r in res), 17)
    lines = ['%-*s ' % (w, 'k (cycles/px)') + ''.join('%9.2f' % k for k in ks) + '   cutoff   info(kbit)',
             '%-*s ' % (w, 'period (px)') + ''.join('%9.1f' % (1 / k) for k in ks),
             '%-*s ' % (w, 'SNR of the stack:')]
    for r in res:
        lines.append('%-*s ' % (w, r['name']) + ''.join('%9.1f' % x for x in r['at']) + '   %.3f   %8.1f' % (r['cutoff'], r['bits'] / 1e3)
                     + ('   (SHARPENED stack: not comparable)' if r['sharpened'] else ''))
    lines.append('\nuncertainty (min-max over 4 sub-squares)')
    for r in res:
        lines.append('%-*s ' % (w, r['name']) + ''.join('%9s' % _rng(lo, hi) for lo, hi in zip(r['sub'].min(0), r['sub'].max(0))))
    best = max(res, key=lambda r: r['bits'])
    eq = equivalent(res, best)
    lines.append('\nbest (most information): %s   cutoff %.3f cycles/px' % (best['name'], best['cutoff']))
    if eq:
        lines.append('equivalent within the uncertainty: ' + ', '.join(r['name'] for r in eq))
    lines.append('ratio to the best:')
    for r in res:
        lines.append('%-*s ' % (w, r['name']) + ''.join(('%9.2f' % (x / y)) if y > 0.5 else '        -' for x, y in zip(r['at'], best['at'])))
    grid = [r for r in res if r['aniso'] > 4]
    if grid:
        lines.append('')
        lines.append('note: a %s pattern of the sensor is visible in both halves (excess x%.0f along the pixel grid);'
                     % ('column' if grid[0]['ang'] < 45 else 'row', max(r['aniso'] for r in grid)))
        lines.append('      those Fourier directions are left out of the SNR, so the table is not affected.')
    warn = [r for r in res if r['cutoff'] > 0.48]
    if warn:
        lines.append('')
        lines.append('WARNING - the SNR never drops below 1 before the sampling limit for: ' + ', '.join(r['name'] for r in warn))
        lines.append('  Something common to both halves inflates it, so these rows are NOT reliable. Usual causes:')
        lines.append('  * the stacker reused the alignment reference of one half for the other (AutoStakkert!: close and reopen')
        lines.append('    the two files for every AP size, instead of changing the AP size on the files already open);')
        lines.append('  * the two halves are not independent (same frames, or stacks of the same file).')
    bad = []
    for r in res:
        pr = list(r.get('problems', []))
        lo = r['sub'][:, 0].min(); hi = r['sub'][:, 0].max()       # lowest tabulated k
        if hi > 0 and lo < 0.05 * hi:
            pr.append('part of the field shows almost no common signal even at k=%.2f' % ks[0])
        if pr:
            bad.append((r['name'], pr))
    if bad:
        lines.append('')
        lines.append('WARNING - these stacks look broken (a failed alignment in the stacker?), so their rows are NOT reliable:')
        for nm_, pr in bad:
            lines.append('  * %s: %s' % (nm_, '; '.join(pr)))
        lines.append('  Look at the A and B stacks: they should show the same field, with no doubled features and no black borders.')
        lines.append('  If not, stack that half again with the same alignment settings as the other one.')
    lines += advice(best, eq, optics, unreliable=(best in warn) or any(nm_ == best['name'] for nm_, _ in bad))
    txt = '\n'.join(lines); log(txt)
    if path:
        open(path + '.txt', 'w').write(txt + '\n')
        with open(path + '.csv', 'w', newline='') as f:
            wr = csv.writer(f); wr.writerow(['name'] + ['k%.2f' % k for k in ks] + ['cutoff', 'info_bits'])
            for r in res:
                wr.writerow([r['name']] + ['%.4g' % x for x in r['at']] + ['%.4f' % r['cutoff'], '%.0f' % r['bits']])
    return best


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best = max(res, key=lambda r: r['bits'])
    kmax = max(r['cutoff'] for r in res) * 1.05
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw=dict(height_ratios=(3, 2)))
    # colour by AP size and shade by frame percentage when the names allow it (AS! stacks), else one colour each
    import re
    parsed = [re.search(r'AP(\d+)\D+(\d+)%', r['name']) for r in res]
    if all(parsed):
        aps = sorted({int(m.group(1)) for m in parsed}); pcs = sorted({int(m.group(2)) for m in parsed})
        cmaps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens, plt.cm.Purples, plt.cm.Reds, plt.cm.Greys]
        styles = []
        for m in parsed:
            cm = cmaps[aps.index(int(m.group(1))) % len(cmaps)]
            shade = 0.45 + 0.5 * (pcs.index(int(m.group(2))) / max(len(pcs) - 1, 1))
            styles.append(dict(color=cm(shade)))
    else:
        styles = [dict(color=plt.cm.tab20(i % 20)) for i in range(len(res))]
    for r, st in zip(res, styles):
        m = r['snr'] > 0
        lw = 2.4 if r is best else 1.4
        ax[0].semilogy(r['k'][m], r['snr'][m], label=r['name'] + ('  (best)' if r is best else ''), lw=lw, **st)
        q = np.interp(r['k'], best['k'], best['snr'])
        mm = m & (q > 1) & (r['k'] > 0.02)
        ax[1].plot(r['k'][mm], r['snr'][mm] / q[mm], lw=lw, **st)
    ax[0].axhline(1, color='k', lw=0.7, ls='--')
    ax[0].set_ylabel('SNR of the full stack'); ax[0].set_title('SNR from independent halves')
    ax[0].grid(alpha=0.3, which='both'); ax[0].legend(fontsize=8)
    ax[1].axhline(1, color='k', lw=0.7); ax[1].set_ylabel('SNR / SNR of "%s"' % best['name'])
    ax[1].set_xlabel('spatial frequency k (cycles/pixel)'); ax[1].grid(alpha=0.3)
    ax[1].set_xlim(0, kmax); ax[1].set_ylim(0, 1.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
