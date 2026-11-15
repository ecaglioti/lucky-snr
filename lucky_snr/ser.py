"""Minimal reader/writer for SER videos (mono, Bayer or RGB; 8 or 16 bit)."""
import struct
import numpy as np

HEADER = 178
BAYER = {8: 'RGGB', 9: 'GRBG', 10: 'GBRG', 11: 'BGGR'}


class SER:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            h = f.read(HEADER)
        self.header = h
        (self.lu_id, self.color_id, self.little_endian, self.width, self.height,
         self.depth, self.n) = struct.unpack('<7i', h[14:42])
        self.channels = 3 if self.color_id in (100, 101) else 1
        self.bpp = 1 if self.depth <= 8 else 2
        # most capture programs write LittleEndian = 0 although the data ARE little endian (common SER quirk)
        dt = np.uint8 if self.bpp == 1 else np.dtype('<u2')
        shape = (self.n, self.height, self.width) + ((3,) if self.channels == 3 else ())
        self.frames = np.memmap(path, dtype=dt, mode='r', offset=HEADER, shape=shape)
        self.frame_bytes = self.width * self.height * self.channels * self.bpp
        self.bayer = BAYER.get(self.color_id)

    def __len__(self):
        return self.n

    def timestamps(self):
        """per-frame timestamps (int64) if the trailer is present, else None."""
        import os
        start = HEADER + self.n * self.frame_bytes
        if os.path.getsize(self.path) >= start + 8 * self.n:
            return np.fromfile(self.path, dtype='<i8', count=self.n, offset=start)
        return None

    def luminance(self, k, binning=1):
        """frame k as float32 luminance (colour: mean of channels; Bayer: 2x2 superpixel), optionally binned."""
        a = np.asarray(self.frames[k], dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(-1)
        b = binning
        if self.bayer and b < 2:
            b = 2                      # never treat a raw Bayer mosaic as luminance at full resolution
        if b > 1:
            H, W = a.shape[0] // b * b, a.shape[1] // b * b
            a = a[:H, :W].reshape(H // b, b, W // b, b).mean((1, 3))
        return a


def write_ser(path, src, frames, crops, observer=b'lucky-snr'):
    """write the selected `frames` of SER `src`, each cropped to crops[j] = (y0, x0, h, w) (same h, w for all)."""
    h, w = crops[0][2], crops[0][3]
    hdr = bytearray(src.header)
    struct.pack_into('<7i', hdr, 14, src.lu_id, src.color_id, src.little_endian, w, h, src.depth, len(frames))
    hdr[42:82] = observer[:40].ljust(40, b'\0')
    ts = src.timestamps()
    with open(path, 'wb') as f:
        f.write(bytes(hdr))
        for k, (y0, x0, _, _) in zip(frames, crops):
            f.write(np.ascontiguousarray(src.frames[k][y0:y0 + h, x0:x0 + w]).tobytes())
        if ts is not None:
            f.write(ts[np.asarray(frames)].astype('<i8').tobytes())
