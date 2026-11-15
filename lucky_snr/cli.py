"""lucky-snr command line:  split | compare"""
import argparse
import os
import sys


def cmd_compare(a):
    from . import compare
    pairs = []
    if a.folders:
        found, lonely = compare.discover(a.folders)
        pairs += found
        if lonely:
            print('stacks without a partner half (ignored):\n  ' + '\n  '.join(lonely))
    for name, fa, fb in (a.pair or []):
        pairs.append((name, fa, fb))
    if not pairs:
        sys.exit('no A/B pairs found. Give folders with AutoStakkert! stacks (+ .as3 files) or use --pair NAME A B')
    print('%d pairs' % len(pairs))
    ks = [float(v) for v in a.k.split(',')]
    res, L = compare.run(pairs, side=a.side, ks=ks)
    out = a.out or os.path.join(a.folders[0] if a.folders else '.', 'lucky_snr')
    optics = None
    if a.pixel and a.focal:
        optics = dict(scale=206.265 * a.pixel / a.focal, lam=a.wavelength)
        if a.aperture:
            optics['fc'] = a.pixel * 1e-6 / (a.wavelength * 1e-9 * a.focal / a.aperture)
    compare.report(res, ks, out, optics=optics)
    compare.plot(res, out + '.png')
    print('\nwritten: %s.txt, %s.csv, %s.png' % (out, out, out))


def cmd_split(a):
    from . import split
    split.main(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lucky-snr', description=__doc__)
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('compare', help='SNR(k) of stacks made from the halves A and B')
    c.add_argument('folders', nargs='*', help='folders with AutoStakkert! stacks and their .as3 files')
    c.add_argument('--pair', nargs=3, action='append', metavar=('NAME', 'A', 'B'), help='any other stacker: name, stack of half A, stack of half B')
    c.add_argument('--side', type=int, default=512, help='side of the common square (px), default 512')
    c.add_argument('--k', default='.04,.08,.12,.16,.20,.24,.28', help='frequencies for the table (cycles/px)')
    c.add_argument('--out', help='output prefix (default: <first folder>/lucky_snr)')
    c.add_argument('--pixel', type=float, help='pixel size in micron (optional, for arcsec and diffraction)')
    c.add_argument('--focal', type=float, help='effective focal length in mm (with Barlow)')
    c.add_argument('--aperture', type=float, help='aperture in mm')
    c.add_argument('--wavelength', type=float, default=550, help='wavelength in nm (default 550; e.g. 700 for IR-pass)')
    c.set_defaults(func=cmd_compare)
    s = sp.add_parser('split', help='split a .ser video into two independent halves A and B')
    s.add_argument('ser')
    s.add_argument('--out', help='output prefix (default: next to the input)')
    s.add_argument('--crop', type=int, default=512, help='side of the crop that follows the drift (0 = full frame)')
    s.add_argument('--center', nargs=2, type=int, metavar=('Y', 'X'), help='crop centre in the reference frame (default: automatic)')
    s.add_argument('--block', default='auto', help='frames per block (default: auto = 2 x correlation time)')
    s.add_argument('--scheme', choices=['thue-morse', 'alternate'], default='thue-morse')
    s.add_argument('--frames', help='frame ranges to use, e.g. 0-775,1270-2900 (default: all)')
    s.set_defaults(func=cmd_split)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
