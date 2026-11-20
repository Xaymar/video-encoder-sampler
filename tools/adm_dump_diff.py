# Compares two VES_ADM_DUMP files (frame 1's ADM scale 0 bands and sums, from two GPUs): which block first differs, where, and by how much.
# usage: python adm_dump_diff.py A.bin B.bin
import struct, sys
NAMES = ['RA', 'RH', 'RV', 'RD', 'DA', 'DH', 'DV', 'DD', 'XH', 'XV', 'XD', 'AH', 'AV', 'AD', 'CAH', 'CAV', 'CAD', 'CFH', 'CFV', 'CFD']

def load(path):
    b = open(path, 'rb').read()
    bw, bh, nblk, ncm = struct.unpack_from('<4i', b, 0)
    n = bw * bh
    cur = list(struct.unpack_from('<%di' % (nblk * n), b, 16))
    # scale 0's bands are packed: block k (0..3) holds the reference's band k in its low 16 bits and the distorted frame's in its high ones
    for k in range(4):
        for i in range(n):
            w = cur[k * n + i]
            lo = w & 0xffff
            cur[k * n + i] = lo - 0x10000 if lo >= 0x8000 else lo
            cur[(k + 4) * n + i] = w >> 16
    off = 16 + 4 * nblk * n
    cm = struct.unpack_from('<%dq' % (ncm // 8), b, off)
    off += ncm
    den = struct.unpack_from('<%dq' % ((len(b) - off) // 8), b, off)
    return bw, bh, cur, cm, den

aw, ah, ac, acm, aden = load(sys.argv[1])
bw, bh, bc, bcm, bden = load(sys.argv[2])
assert (aw, ah) == (bw, bh), 'different sizes'
n = aw * ah
print('bands %dx%d' % (aw, ah))
for k, name in enumerate(NAMES):
    diffs = [i for i in range(n) if ac[k * n + i] != bc[k * n + i]]
    if diffs:
        ex = ', '.join('(%d,%d) %d vs %d' % (i % aw, i // aw, ac[k * n + i], bc[k * n + i]) for i in diffs[:6])
        print('%-4s %8d of %d differ; first: %s' % (name, len(diffs), n, ex))
    else:
        print('%-4s identical' % name)
for label, x, y in (('cm rows', acm, bcm), ('den rows', aden, bden)):
    d = [i for i in range(min(len(x), len(y))) if x[i] != y[i]]
    print('%s: %d of %d differ%s' % (label, len(d), len(x), ('; first row %d band %d: %d vs %d' % (d[0] // 3, d[0] % 3, x[d[0]], y[d[0]])) if d else ''))
