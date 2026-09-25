"""Shared constants, colour helpers, easing, noise and timing for the music video."""
import math
import os

import numpy as np
import skia

W, H = 1920, 1080
FPS = 30
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
BUILD = os.path.join(ROOT, "build")
TAU = math.tau


# ----------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------
def hx(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


PAL = {k: hx(v) for k, v in dict(
    black="#120b08", ink="#1b0f0a", white="#fff8ec", linen="#f3e9d2", linen2="#d9ccb0",
    gold="#f2c14e", gold2="#d99a1e", gold3="#8a5a12", amber="#ffb347",
    lapis="#1f3f95", lapis2="#15296a", lapis3="#0b1740", turq="#2ec4b6", turq2="#1a8f86",
    red="#c0392b", red2="#8e1f16", carnelian="#e0603a", ochre="#cc7a2f", sand="#e8c58a",
    sand2="#c99a5b", sand3="#9c6b3a", sand4="#6b4424", night="#0a0f2c", night2="#1b1446",
    purple="#4b1d6e", magenta="#ff2e88", pink="#ff7eb6", cyan="#00e5ff", lime="#b8ff3c",
    skin1="#b5653a", skin2="#d08b55", skin3="#8a4b2b", skin4="#e3a871", green="#3c9d5d",
    leaf="#2f7d4a", leaf2="#1d5433", nile="#1b5f7a", nile2="#0f3a52",
).items()}


def C(c):
    """Accept '#hex', (r,g,b) floats or palette key."""
    if isinstance(c, str):
        return PAL[c] if c in PAL else hx(c)
    return c


def col(c, a=1.0):
    r, g, b = C(c)
    a = min(1.0, max(0.0, a))
    return skia.Color(int(clamp(r) * 255 + 0.5), int(clamp(g) * 255 + 0.5), int(clamp(b) * 255 + 0.5), int(a * 255 + 0.5))


def mix(c1, c2, t):
    a, b = C(c1), C(c2)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def scale_c(c, k):
    r, g, b = C(c)
    return (r * k, g * k, b * k)


def hsv(h, s, v):
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


# ----------------------------------------------------------------------------
# paint helpers
# ----------------------------------------------------------------------------
BM = skia.BlendMode


def P(c="black", a=1.0, stroke=None, blur=0.0, blend=None, shader=None, cap="round", aa=True):
    p = skia.Paint(AntiAlias=aa)
    p.setColor(col(c, a))
    if stroke is not None:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap({"round": skia.Paint.kRound_Cap, "butt": skia.Paint.kButt_Cap,
                        "square": skia.Paint.kSquare_Cap}[cap])
        p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if blend is not None:
        p.setBlendMode(blend)
    if shader is not None:
        p.setShader(shader)
    return p


def lin_grad(p0, p1, stops, mode=skia.TileMode.kClamp):
    cols = [col(c, a) for (_, c, a) in stops]
    pos = [s for (s, _, _) in stops]
    return skia.GradientShader.MakeLinear([skia.Point(*p0), skia.Point(*p1)], cols, pos, mode)


def rad_grad(center, r, stops, mode=skia.TileMode.kClamp):
    cols = [col(c, a) for (_, c, a) in stops]
    pos = [s for (s, _, _) in stops]
    return skia.GradientShader.MakeRadial(skia.Point(*center), max(r, 1e-3), cols, pos, mode)


def poly(pts, close=True):
    path = skia.Path()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    if close:
        path.close()
    return path


def svg(d):
    return skia.Path.FromSVGString(d) if hasattr(skia.Path, "FromSVGString") else skia.SkParsePath.FromSVGString(d)


_font_cache = {}


def font(name, size):
    tf = _font_cache.get(name)
    if tf is None:
        tf = skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, name))
        _font_cache[name] = tf
    f = skia.Font(tf, size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    return f


def text_width(s, f):
    return f.measureText(s)


def draw_text_c(canvas, s, x, y, f, paint):
    w = f.measureText(s)
    canvas.drawString(s, x - w / 2, y, f, paint)


# ----------------------------------------------------------------------------
# maths / easing
# ----------------------------------------------------------------------------
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a, b, x):
    return clamp((x - a) / (b - a)) if b != a else 0.0


def remap(x, a, b, c, d):
    return c + (d - c) * inv_lerp(a, b, x)


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def smoother(t):
    t = clamp(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out(t, p=3):
    t = clamp(t)
    return 1 - (1 - t) ** p


def ease_in(t, p=3):
    t = clamp(t)
    return t ** p


def ease_io(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_back(t, s=1.7):
    t = clamp(t) - 1
    return 1 + t * t * ((s + 1) * t + s)


def ease_elastic(t):
    t = clamp(t)
    if t in (0.0, 1.0):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * TAU / 3) + 1


def hit(u, snap=0.28, back=1.4):
    """Beat-snap curve: goes 0->1 in the first `snap` of the beat with a little overshoot."""
    return ease_back(u / snap, back) if u < snap else 1.0


def pulse(u, k=6.0):
    return math.exp(-k * max(0.0, u))


def tri(x):
    x = x % 1.0
    return 1 - abs(2 * x - 1)


def rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def fract(x):
    return x - math.floor(x)


def hash1(n):
    n = (int(n) * 374761393 + 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return (n ^ (n >> 16)) / 4294967295.0


def vnoise(x, seed=0):
    i = math.floor(x)
    f = x - i
    a, b = hash1(i + seed * 1013), hash1(i + 1 + seed * 1013)
    return lerp(a, b, f * f * (3 - 2 * f)) * 2 - 1


def fbm(x, seed=0, oct=3):
    v, amp, fr = 0.0, 0.5, 1.0
    for o in range(oct):
        v += amp * vnoise(x * fr, seed + o * 17)
        amp *= 0.5
        fr *= 2.03
    return v


# ----------------------------------------------------------------------------
# audio-driven timing
# ----------------------------------------------------------------------------
class Audio:
    def __init__(self, path):
        d = np.load(path)
        self.grid = d["grid"]
        self.ft = d["ft"]
        self.env = {k: d[k] for k in ("low", "mid", "high", "rms", "onset", "kick", "snare", "hat")}
        self.spec = d["spec"]
        self.duration = float(d["duration"])
        self.period = float(np.median(np.diff(self.grid)))

    def beat(self, t):
        g = self.grid
        if t < g[0]:
            return (t - g[0]) / self.period
        if t > g[-1]:
            return len(g) - 1 + (t - g[-1]) / self.period
        return float(np.interp(t, g, np.arange(len(g))))

    def beat_time(self, b):
        g = self.grid
        if b < 0:
            return g[0] + b * self.period
        if b > len(g) - 1:
            return g[-1] + (b - len(g) + 1) * self.period
        return float(np.interp(b, np.arange(len(g)), g))

    def bar_time(self, bar):
        return self.beat_time(bar * 4)

    def e(self, name, t):
        return float(np.interp(t, self.ft, self.env[name]))

    def bands(self, t):
        i = int(np.clip(np.searchsorted(self.ft, t), 0, len(self.ft) - 1))
        return self.spec[i]


_audio = None


def audio():
    global _audio
    if _audio is None:
        _audio = Audio(os.path.join(BUILD, "features.npz"))
    return _audio
