"""Regenerates the images and libjxl `ssimulacra2` scores of tests/reference.rs."""
import subprocess
from PIL import Image
def base(kind, w, h):
    px = []
    for y in range(h):
        for x in range(w):
            if kind == 'texture':
                px.append(((x*3 + y*5 + (x*y) % 17) % 200 + 20, (x*7 + y*11) % 200 + 20, (x*13 + y*2) % 200 + 20))
            else:
                px.append((x*255//(w-1), y*255//(h-1), ((x//8 + y//8) % 2)*120 + 60))
    return px
def clamp(v): return max(0, min(255, v))
def distort(px, w, h, kind, amount):
    out = []
    for y in range(h):
        for x in range(w):
            p = px[y*w + x]
            if kind == 'noise':
                out.append(tuple(clamp(v + ((x*19 + y*23 + c*7) % 11 - 5)*amount) for c, v in enumerate(p)))
            elif kind == 'shift':
                out.append((p[0], clamp(p[1] + amount), p[2]))
            elif kind == 'posterize':
                out.append(tuple((v // amount)*amount for v in p))
            elif kind == 'blur':
                acc = [0, 0, 0]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = px[min(max(y+dy, 0), h-1)*w + min(max(x+dx, 0), w-1)]
                        for c in range(3): acc[c] += q[c]
                out.append(tuple(a // 9 for a in acc))
    return out
cases = []
for (bk, w, h) in [('texture', 97, 83), ('smooth', 256, 200), ('smooth', 31, 17), ('texture', 9, 8)]:
    ref = base(bk, w, h)
    Image.frombytes('RGB', (w, h), bytes(v for p in ref for v in p)).save('ref.png')
    for (dk, amt) in [('noise', 1), ('noise', 4), ('shift', 3), ('posterize', 16), ('blur', 0)]:
        t = distort(ref, w, h, dk, amt)
        Image.frombytes('RGB', (w, h), bytes(v for p in t for v in p)).save('t.png')
        s = float(subprocess.check_output(['ssimulacra2', 'ref.png', 't.png']))
        cases.append((bk, w, h, dk, amt, s))
for c in cases:
    print(f'    ("{c[0]}", {c[1]}, {c[2]}, "{c[3]}", {c[4]}, {c[5]:.6f}),')
