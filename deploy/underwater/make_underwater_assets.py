"""Procedural assets for the realistic underwater render (render_rov_se3_realistic.py). Visual only."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

OUT = Path(__file__).resolve().parent / "assets" / "scene"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def fbm(n, octaves, base, seed):
    """fractal value noise on an n x n grid, in [0, 1]"""
    r = np.random.default_rng(seed); acc = np.zeros((n, n)); amp, tot = 1.0, 0.0
    for o in range(octaves):
        k = base * 2 ** o
        g = r.random((k + 1, k + 1))
        img = Image.fromarray((g * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
        acc += amp * np.asarray(img, float) / 255; tot += amp; amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min())


N = 512
x, y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
dunes = fbm(N, 4, 3, 1)
warp = fbm(N, 3, 4, 2)
ripples = 0.5 + 0.5 * np.sin(2 * np.pi * (38 * x + 9 * y + 3.0 * warp))
h = 0.82 * dunes + 0.18 * ripples
Image.fromarray((255 * (h - h.min()) / (h.max() - h.min())).astype(np.uint8)).save(OUT / "seabed_h.png")

T = 2048
X, Y = np.meshgrid(np.linspace(0, 1, T), np.linspace(0, 1, T))
grain = np.random.default_rng(3).random((T, T))
grain = np.asarray(Image.fromarray((grain * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), float) / 255
tone = fbm(T, 5, 6, 4)
rip = 0.5 + 0.5 * np.sin(2 * np.pi * (150 * X + 36 * Y + 6.0 * fbm(T, 3, 8, 5)))
c = np.zeros((T, T))
for k, (fx, fy, ph) in enumerate([(23, 7, 0.0), (-11, 19, 1.3), (5, -17, 2.1), (17, 13, 0.7)]):
    w = fbm(T, 3, 5, 10 + k)
    c += np.sin(2 * np.pi * (fx * X + fy * Y + 2.2 * w) + ph)
caus = np.exp(-np.abs(c) * 3.2)
caus = np.asarray(Image.fromarray((caus * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), float) / 255
sand = np.array([0.63, 0.57, 0.44])
img = sand[None, None, :] * (0.72 + 0.18 * tone[..., None] + 0.10 * grain[..., None]) * (0.88 + 0.12 * rip[..., None])
img += np.array([0.30, 0.38, 0.36])[None, None, :] * (caus[..., None] ** 1.5) * 0.55
pebbles = np.random.default_rng(9).random((T, T)) > 0.9993
img[pebbles] *= 0.55
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT / "seabed_tex.png")


def icosphere(sub):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(sub):
        cache, nf = {}, []
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = v[a] + v[b]; v.append(p / np.linalg.norm(p)); cache[key] = len(v) - 1
            return cache[key]
        for a, b, c_ in f:
            ab, bc, ca = mid(a, b), mid(b, c_), mid(c_, a)
            nf += [(a, ab, ca), (b, bc, ab), (c_, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


for k in range(4):
    v, f = icosphere(3)
    r = np.random.default_rng(20 + k)
    dirs = r.normal(size=(6, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    disp = 1 + sum(0.18 * r.random() * np.maximum(v @ d, 0) ** 3 for d in dirs) + 0.06 * r.normal(size=len(v))
    v = v * disp[:, None] * np.array([1.0, 0.8 + 0.3 * r.random(), 0.55 + 0.2 * r.random()])
    v[:, 2] -= v[:, 2].min() * 0.6
    with open(OUT / f"rock_{k}.obj", "w") as fh:
        fh.writelines(f"v {a:.5f} {b:.5f} {c_:.5f}\n" for a, b, c_ in v)
        fh.writelines(f"f {a + 1} {b + 1} {c_ + 1}\n" for a, b, c_ in f)
print("wrote", sorted(p.name for p in OUT.iterdir()))
