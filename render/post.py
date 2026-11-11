import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size, mono=False):
    try:
        return ImageFont.truetype(MONO if mono else FONT, size)
    except OSError:
        return ImageFont.load_default()


class Grade:
    def __init__(self, w, h, vignette=0.55, bloom=0.35, thresh=200, tint=(1.0, 1.0, 1.0), lift=(0, 0, 0)):
        y, x = np.mgrid[0:h, 0:w]
        r = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2))
        self.vig = (1 - vignette * np.clip(r - 0.35, 0, None) ** 1.6)[..., None]
        self.bloom, self.thresh = bloom, thresh
        self.tint, self.lift = np.array(tint), np.array(lift)

    def __call__(self, rgb):
        f = rgb.astype(np.float32)
        bright = np.clip(f - self.thresh, 0, None) * (255.0 / (255 - self.thresh))
        glow = np.asarray(Image.fromarray(bright.astype(np.uint8)).filter(ImageFilter.GaussianBlur(18)), np.float32)
        f = f + self.bloom * glow
        f = f * self.tint + self.lift
        f = f * self.vig
        return np.clip(f, 0, 255).astype(np.uint8)


class Snow:
    def __init__(self, w, h, n=150, seed=0):
        r = np.random.default_rng(seed)
        self.w, self.h = w, h
        self.p = np.c_[r.uniform(0, w, n), r.uniform(0, h, n)]
        self.z = r.uniform(0.2, 1.0, n)
        self.v = np.c_[r.normal(0.15, 0.25, n), r.uniform(0.15, 0.6, n)] * self.z[:, None]

    def __call__(self, img, t):
        self.p += self.v
        self.p[:, 0] %= self.w
        self.p[:, 1] %= self.h
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for (x, y), z in zip(self.p, self.z):
            s = 0.6 + 2.2 * z ** 2
            a = int(25 + 95 * z)
            d.ellipse([x - s, y - s, x + s, y + s], fill=(200, 235, 230, a))
        layer = layer.filter(ImageFilter.GaussianBlur(0.8))
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def god_rays(w, h, t, strength=0.22):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(x - w * 0.35, y + h * 0.6)
    rays = 0.5 + 0.5 * np.sin(ang * 38 + 0.6 * np.sin(t * 0.7)) * np.sin(ang * 17 - t * 0.35)
    fall = np.clip(1 - y / (h * 0.95), 0, 1) ** 1.7
    return (strength * 255 * rays * fall)[..., None] * np.array([0.55, 0.85, 0.9], np.float32)


def fade(frames, n=12):
    out = list(frames)
    for k in range(min(n, len(out))):
        a = k / n
        out[k] = (out[k].astype(np.float32) * a).astype(np.uint8)
        out[-1 - k] = (out[-1 - k].astype(np.float32) * a).astype(np.uint8)
    return out


def hud(img, title, rows, v_ratio, accent=(163, 230, 53)):
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    x0, y0, pw, ph = 28, 26, 330, 128
    d.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], 16, fill=(8, 12, 24, 150), outline=(255, 255, 255, 40))
    d.text((x0 + 18, y0 + 14), title, font=font(19), fill=(240, 242, 248))
    for k, (a, b) in enumerate(rows):
        d.text((x0 + 18, y0 + 46 + 22 * k), a, font=font(14, True), fill=(170, 178, 196))
        d.text((x0 + 160, y0 + 46 + 22 * k), b, font=font(14, True), fill=(240, 242, 248))
    bx, by, bw = x0 + 18, y0 + ph - 22, pw - 36
    d.rounded_rectangle([bx, by, bx + bw, by + 8], 4, fill=(255, 255, 255, 40))
    d.rounded_rectangle([bx, by, bx + max(8, int(bw * min(1, v_ratio))), by + 8], 4, fill=accent + (255,))
    msg = "inside the estimated region  V ≤ ρ" if v_ratio <= 1 else "outside the estimated region"
    f = font(14, True)
    d.text((W - 28 - d.textlength(msg, font=f), H - 40), msg, font=f, fill=accent + (230,))
    return img


def write_video(frames, out, fps=30, gif=False, gif_width=420, gif_fps=12):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out.with_suffix(".mp4"))], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    Image.fromarray(frames[len(frames) // 3]).save(out.with_suffix(".jpg"), quality=88)
    if not gif:
        return
    vf = f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out.with_suffix(".mp4")), "-vf", vf, str(out.with_suffix(".gif"))], check=True)
