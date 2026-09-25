"""The paper-craft toolkit: paper stock, hand-cut and torn edges, cast shadows, split pins, cut-paper
letters, a parallax camera, and baked (pre-rendered) scenery layers.

Every piece is a closed outline in its own local coordinates. Outlines are sampled densely and nudged
along their normals by smooth noise, so edges look scissor-cut (small, slow wobble) or torn (larger,
fractal, with a white fibrous rim where the coloured face tore off the paper core). Pieces are filled
with the colour blended (overlay) with a tileable paper texture that is attached to the piece, so the
fibres move with it. Each piece drops a soft shadow on whatever is below; static scenery is baked into
images once per scene so a frame is mostly blits plus the animated cast.
"""
import math
import os
from functools import lru_cache

import numpy as np
import skia

from mv.core import TAU, clamp, ease_back, ease_io, ease_out, hash1, lerp, smooth, smoother  # noqa: F401
from .core import FONT_DIR, H, W

# ----------------------------------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------------------------------
PAL = dict(
    paper="#fbf6ea", core="#fffaf0", ink="#2b2522", shadow="#2a1f1a",
    sky_grey="#b9c3c8", sky_grey2="#d5dad9", sky_blue="#8fd0f2", sky_blue2="#d8f1fb",
    cloud_grey="#9da7ae", cloud_grey2="#b3bcc2", cloud="#ffffff",
    grass="#74b64f", grass2="#5c9e3d", grass3="#8cc65f", leaf="#4f9a3f", leaf2="#3d7f31", moss="#9ac46a",
    hill="#9cc184", hill2="#86ad70", hill3="#b3d09a", trunk="#8a5a3b", bark="#6e4630",
    water="#5aa7dc", water2="#4290cc", water3="#3378b5", water4="#7cc0ea", foam="#e9f6fc", puddle="#8fb3c7",
    sea="#3f8fd0", sea2="#2f78bd", sea3="#2465a6", sea4="#66abe0",
    sand="#f0d9a4", rock="#a4a7aa", rock2="#8a8e92",
    red="#e8413c", orange="#f7931e", yellow="#ffd23f", green="#58b947", blue="#2f7fd8", purple="#8e5bd6",
    pink="#f7a1c4", rose="#ef6f9a", cream="#fff3d6", gold="#e8b84a", brown="#8a5a3b", white="#ffffff",
    black="#262322", cheek="#ff9aa8", lilac="#c9a7f0", lily="#f9c6dc",
)
RAINBOW = ["red", "orange", "yellow", "green", "blue", "purple"]


def rgb(c):
    if isinstance(c, str):
        h = PAL.get(c, c).lstrip("#")
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
    return c


def col(c, a=1.0):
    r, g, b = rgb(c)
    return skia.Color(int(clamp(r) * 255 + .5), int(clamp(g) * 255 + .5), int(clamp(b) * 255 + .5),
                      int(clamp(a) * 255 + .5))


def mixc(c1, c2, t):
    a, b = rgb(c1), rgb(c2)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def shade(c, k):
    """k<0 darker, k>0 lighter."""
    return mixc(c, (0, 0, 0) if k < 0 else (1, 1, 1), abs(k))


# ----------------------------------------------------------------------------------------------------
# paper stock
# ----------------------------------------------------------------------------------------------------
TEX = 512


def _fft_noise(rng, n, beta):
    f = np.fft.fft2(rng.standard_normal((n, n)))
    k = np.sqrt(np.fft.fftfreq(n)[None, :] ** 2 + np.fft.fftfreq(n)[:, None] ** 2)
    k[0, 0] = 1
    x = np.real(np.fft.ifft2(f / k ** beta))
    return (x - x.mean()) / (x.std() + 1e-9)


@lru_cache(None)
def texture():
    """Tileable paper stock around mid-grey (neutral for overlay): mottling, faint fibres and grain."""
    rng = np.random.default_rng(11)
    v = 0.5 + 0.022 * _fft_noise(rng, TEX, 1.7) + 0.010 * _fft_noise(rng, TEX, 0.4)
    surf = skia.Surface(TEX, TEX)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeCap=skia.Paint.kRound_Cap)
    for i in range(700):
        x, y = rng.uniform(0, TEX, 2)
        a = rng.uniform(0, TAU)
        ln = rng.uniform(3, 11)
        bend = rng.uniform(-0.4, 0.4)
        light = rng.random() < 0.55
        p.setColor(skia.Color(255, 255, 255, 255) if light else skia.Color(0, 0, 0, 255))
        p.setStrokeWidth(rng.uniform(0.5, 1.0))
        p.setAlphaf(rng.uniform(0.25, 0.6) if light else rng.uniform(0.15, 0.35))
        for ox in (-TEX, 0, TEX):
            for oy in (-TEX, 0, TEX):
                path = skia.Path()
                path.moveTo(x + ox, y + oy)
                mx, my = x + ox + math.cos(a) * ln / 2 + math.cos(a + 1.57) * bend * ln, \
                    y + oy + math.sin(a) * ln / 2 + math.sin(a + 1.57) * bend * ln
                path.quadTo(mx, my, x + ox + math.cos(a) * ln, y + oy + math.sin(a) * ln)
                c.drawPath(path, p)
    fib = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType).astype(np.float32) / 255
    a = fib[..., 3]
    lum = fib[..., 0] / np.maximum(a, 1e-4)
    v = v + a * (lum - 0.5) * 0.09
    g = (np.clip(v, 0, 1) * 255).astype(np.uint8)
    arr = np.dstack([g, g, g, np.full_like(g, 255)])
    return skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType)


SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear)


def _tex_matrix(seed):
    m = skia.Matrix()
    m.setRotate(90 * (int(hash1(seed * 7 + 1) * 4)))
    m.postTranslate(hash1(seed * 13 + 5) * TEX, hash1(seed * 17 + 3) * TEX)
    return m


_BLEND = {}


def paper_paint(color, seed=0, alpha=1.0):
    """Colour overlaid with the paper stock. Building a blend shader around an image is slow in skia,
    so there is one per (quantised) colour and each piece gets a cheap local-matrix copy of it."""
    r, g, b = rgb(color)
    key = (int(r * 63 + .5), int(g * 63 + .5), int(b * 63 + .5))
    sh = _BLEND.get(key)
    if sh is None:
        base = skia.Shaders.Color(skia.Color(int(key[0] * 255 / 63 + .5), int(key[1] * 255 / 63 + .5),
                                             int(key[2] * 255 / 63 + .5)))
        tx = texture().makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, SAMPLING, skia.Matrix())
        sh = _BLEND[key] = skia.Shaders.Blend(skia.BlendMode.kOverlay, base, tx)
    p = skia.Paint(AntiAlias=True, Shader=sh.makeWithLocalMatrix(_tex_matrix(seed)))
    if alpha < 1:
        p.setAlphaf(clamp(alpha))
    return p


def texture_overlay(c, path_or_rect, seed=0, strength=1.0):
    """Lay the paper stock over whatever is already drawn (overlay: mid-grey is neutral)."""
    p = skia.Paint(AntiAlias=True, BlendMode=skia.BlendMode.kOverlay,
                   Shader=texture().makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, SAMPLING,
                                               _tex_matrix(seed)))
    if strength < 1:
        p.setAlphaf(strength)
    if isinstance(path_or_rect, skia.Rect):
        c.drawRect(path_or_rect, p)
    else:
        c.drawPath(path_or_rect, p)


# ----------------------------------------------------------------------------------------------------
# outlines
# ----------------------------------------------------------------------------------------------------
def _loop_noise(u, seed, octaves=3, base=6):
    """Smooth periodic noise on u in [0,1): sum of random-phase sines."""
    rng = np.random.default_rng(int(seed * 1000) % (2 ** 31))
    out = np.zeros_like(u)
    amp = 1.0
    for o in range(octaves):
        k = base * 2 ** o
        for j in range(2):
            kk = k + j * (k // 2 + 1)
            out += amp * 0.5 * np.sin(TAU * kk * u + rng.uniform(0, TAU))
        amp *= 0.5
    return out


def ellipse(cx, cy, rx, ry, n=None, a0=0.0):
    n = n or max(24, int((rx + ry) * 0.6))
    a = a0 + np.linspace(0, TAU, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def resample(pts, step=4.0, closed=True):
    p = np.asarray(pts, float)
    q = np.vstack([p, p[:1]]) if closed else p
    seg = np.sqrt(((q[1:] - q[:-1]) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    n = max(8, int(total / step))
    t = np.linspace(0, total, n, endpoint=not closed)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def from_path(path, step=4.0):
    """Sample a skia.Path's first contour into points."""
    pm = skia.PathMeasure(path, True)
    L = pm.getLength()
    n = max(8, int(L / step))
    out = []
    for i in range(n):
        pos, _ = pm.getPosTan(L * i / n)
        out.append((pos.x(), pos.y()))
    return np.array(out)


def normals(pts, closed=True):
    p = np.asarray(pts, float)
    nxt = np.roll(p, -1, 0) if closed else np.vstack([p[1:], p[-1:] + (p[-1:] - p[-2:-1])])
    prv = np.roll(p, 1, 0) if closed else np.vstack([p[:1] - (p[1:2] - p[:1]), p[:-1]])
    d = nxt - prv
    nrm = np.stack([d[:, 1], -d[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    # make them point outward (positive signed area -> clockwise in y-down coords)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return nrm if area > 0 else -nrm


def cut(pts, seed=0, amp=1.1, step=5.0):
    """Scissor-cut edge: gentle wobble plus tiny facets."""
    p = resample(pts, step)
    u = np.arange(len(p)) / len(p)
    L = np.sqrt(((np.roll(p, -1, 0) - p) ** 2).sum(1)).sum()
    base = max(3, int(L / 140))
    d = _loop_noise(u, seed + 0.5, 2, base) * amp + (np.random.default_rng(int(seed * 77) % 2 ** 31)
                                                     .uniform(-0.35, 0.35, len(p))) * amp * 0.5
    return p + normals(p) * d[:, None]


def torn(pts, seed=0, amp=3.0, step=3.0):
    p = resample(pts, step)
    u = np.arange(len(p)) / len(p)
    L = np.sqrt(((np.roll(p, -1, 0) - p) ** 2).sum(1)).sum()
    base = max(4, int(L / 60))
    rng = np.random.default_rng(int(seed * 131) % 2 ** 31)
    d = _loop_noise(u, seed + 0.25, 4, base) * amp + rng.uniform(-0.5, 0.5, len(p)) * amp * 0.45
    return p + normals(p) * d[:, None]


def grow(pts, d):
    p = np.asarray(pts, float)
    return p + normals(p) * d


def path_of(pts, closed=True):
    return skia.Path.Polygon([skia.Point(float(x), float(y)) for x, y in pts], closed)


def spline(pts, per=10, closed=True):
    """Points along a Catmull-Rom spline through pts (closed by default)."""
    p = np.asarray(pts, float)
    n = len(p)
    out = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = p[(i - 1) % n] if (closed or i > 0) else p[i]
        p1, p2 = p[i], p[(i + 1) % n]
        p3 = p[(i + 2) % n] if (closed or i + 2 < n) else p2
        for k in range(per):
            u = k / per
            u2, u3 = u * u, u * u * u
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    if not closed:
        out.append(p[-1])
    return np.array(out)


_CUT = {}


def cpath(pts, seed=0, amp=0.8, soft=False):
    """Cached scissor-cut outline for a static local shape (soft: spline through pts first)."""
    key = (tuple((round(float(x), 2), round(float(y), 2)) for x, y in pts), seed, amp, soft)
    p = _CUT.get(key)
    if p is None:
        if len(_CUT) > 4000:
            _CUT.clear()
        q = cut(spline(pts, 12) if soft else pts, seed, amp)
        p = _CUT[key] = path_of(q)
    return p


def smooth_path(pts, closed=True, k=1.0):
    """Catmull-Rom through pts."""
    p = [tuple(map(float, q)) for q in pts]
    n = len(p)
    path = skia.Path()
    path.moveTo(*p[0])
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = p[(i - 1) % n] if (closed or i > 0) else p[i]
        p1, p2 = p[i], p[(i + 1) % n]
        p3 = p[(i + 2) % n] if (closed or i + 2 < n) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) * k / 6, p1[1] + (p2[1] - p0[1]) * k / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) * k / 6, p2[1] - (p3[1] - p1[1]) * k / 6)
        path.cubicTo(*c1, *c2, *p2)
    if closed:
        path.close()
    return path


def union(paths):
    out = paths[0]
    for q in paths[1:]:
        r = skia.Op(out, q, skia.PathOp.kUnion_PathOp)
        out = r if r is not None else out
    return out


# ----------------------------------------------------------------------------------------------------
# drawing pieces
# ----------------------------------------------------------------------------------------------------
_SHADOW_CACHE = {}


def shadow_paint(sigma, alpha):
    key = (round(sigma, 1), round(alpha, 3))
    p = _SHADOW_CACHE.get(key)
    if p is None:
        p = skia.Paint(AntiAlias=True, Color=col("shadow", alpha))
        if sigma > 0.2:
            p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma))
        _SHADOW_CACHE[key] = p
    return p


def drop(c, path, lift=1.0, alpha=0.30):
    """Shadow of a piece `lift` units above the page (light from the upper left)."""
    if lift <= 0:
        return
    c.save()
    c.translate(2.2 * lift, 3.2 * lift)
    c.drawPath(path, shadow_paint(1.2 + 2.6 * lift, alpha * min(1.0, 0.55 + 0.25 * lift)))
    c.restore()


def piece(c, path, color, seed=0, lift=1.0, alpha=1.0, rim=0.35, shadow=0.30, tex=True, grad=0.10):
    """A paper piece: cast shadow, textured colour, light top-left falloff and a pale cut rim."""
    if not isinstance(path, skia.Path):
        path = path_of(path)
    if shadow > 0:
        drop(c, path, lift, shadow * alpha)
    c.drawPath(path, paper_paint(color, seed, alpha) if tex else skia.Paint(AntiAlias=True, Color=col(color, alpha)))
    if grad > 0:
        b = path.getBounds()
        g = skia.GradientShader.MakeLinear(
            [skia.Point(b.left(), b.top()), skia.Point(b.right(), b.bottom())],
            [skia.Color(255, 255, 255, int(255 * grad * alpha)), skia.Color(255, 255, 255, 0),
             skia.Color(0, 0, 0, int(255 * grad * 0.8 * alpha))], [0.0, 0.5, 1.0])
        c.drawPath(path, skia.Paint(AntiAlias=True, Shader=g))
    if rim > 0:
        c.drawPath(path, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.3,
                                    Color=col("core", rim * alpha)))
    return path


def torn_piece(c, pts, color, seed=0, lift=1.0, rim_w=3.5, amp=3.0, alpha=1.0, shadow=0.28, grad=0.08):
    """Torn paper: a white fibrous rim peeks out beyond the coloured face."""
    outer = path_of(torn(grow(resample(pts, 3.0), rim_w * 0.6), seed + 0.7, amp * 1.1))
    inner = path_of(torn(pts, seed, amp))
    if shadow > 0:
        drop(c, outer, lift, shadow * alpha)
    c.drawPath(outer, paper_paint(mixc("core", color, 0.06), seed + 3, alpha))
    piece(c, inner, color, seed, lift=0, alpha=alpha, rim=0, shadow=0, grad=grad)
    return inner


def brad(c, x, y, r=7.0, lift=0.6):
    """Split pin holding a joint."""
    c.drawCircle(x + 1.2 * lift, y + 1.8 * lift, r, shadow_paint(1.4, 0.35))
    g = skia.GradientShader.MakeRadial(skia.Point(x - r * 0.35, y - r * 0.4), r * 1.4,
                                       [col("#fff4c2"), col("#e2b24b"), col("#9c6c1c")], [0.0, 0.45, 1.0])
    c.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Shader=g))
    c.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.8,
                                     Color=col("#7a520f", 0.6)))


def stroke(c, pts, width, color="ink", alpha=1.0, closed=False, cap="round"):
    """A felt-pen line."""
    path = smooth_path(pts, closed) if len(pts) > 2 else path_of(pts, closed)
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, Color=col(color, alpha),
                   StrokeCap=skia.Paint.kRound_Cap if cap == "round" else skia.Paint.kButt_Cap,
                   StrokeJoin=skia.Paint.kRound_Join)
    c.drawPath(path, p)


def thread(c, x0, y0, x1, y1, sag=6.0, alpha=0.8):
    path = skia.Path()
    path.moveTo(x0, y0)
    path.quadTo((x0 + x1) / 2 + sag * 0.3, (y0 + y1) / 2 + sag, x1, y1)
    c.drawPath(path, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.6,
                                Color=col("#f7f2e6", alpha)))
    c.drawPath(path, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=0.6,
                                Color=col("#8d8577", alpha * 0.6)))


# ----------------------------------------------------------------------------------------------------
# cut-paper letters
# ----------------------------------------------------------------------------------------------------
_TF = {}


def typeface(name="TitanOne.ttf"):
    tf = _TF.get(name)
    if tf is None:
        tf = skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, name))
        _TF[name] = tf
    return tf


@lru_cache(512)
def glyph_outline(ch, size, fontname="TitanOne.ttf"):
    f = skia.Font(typeface(fontname), size)
    g = f.textToGlyphs(ch)
    path = f.getPath(g[0])
    adv = f.getWidths(g)[0]
    return path, adv


def word_layout(text, size, fontname="TitanOne.ttf", track=0.02):
    """-> [(char, x_offset_of_centre, advance)], total width."""
    out, x = [], 0.0
    for ch in text:
        path, adv = glyph_outline(ch, size, fontname)
        out.append((ch, x + adv / 2, adv))
        x += adv + size * track
    return out, x - size * track


def letter(c, ch, x, y, size, color, seed=0, rot=0.0, scale=1.0, lift=1.4, border=None, alpha=1.0,
           fontname="TitanOne.ttf"):
    """One sticker-style cut-paper letter centred at x, baseline-ish centre at y."""
    if ch == " ":
        return
    path, adv = glyph_outline(ch, size, fontname)
    b = path.getBounds()
    border = size * 0.075 if border is None else border
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(scale, scale)
    c.translate(-(b.left() + b.right()) / 2, -(b.top() + b.bottom()) / 2)
    back = skia.Path()
    skia.Paint(Style=skia.Paint.kStroke_Style, StrokeWidth=border * 2,
               StrokeJoin=skia.Paint.kRound_Join).getFillPath(path, back)
    back = union([back, path])
    drop(c, back, lift, 0.32 * alpha)
    c.drawPath(back, paper_paint("paper", seed + 1, alpha))
    piece(c, path, color, seed, lift=0, alpha=alpha, rim=0.0, shadow=0, grad=0.12)
    c.restore()


# ----------------------------------------------------------------------------------------------------
# camera + baked layers
# ----------------------------------------------------------------------------------------------------
class Cam:
    """World -> screen with parallax: depth p=1 is the action plane, p<1 far, p>1 near."""

    def __init__(self, x=W / 2, y=H / 2, zoom=1.0, rot=0.0):
        self.x, self.y, self.zoom, self.rot = x, y, zoom, rot

    def apply(self, c, p=1.0):
        z = 1.0 + (self.zoom - 1.0) * p
        c.translate(W / 2, H / 2)
        if self.rot:
            c.rotate(self.rot * p)
        c.scale(z, z)
        c.translate(-(W / 2 + (self.x - W / 2) * p), -(H / 2 + (self.y - H / 2) * p))

    def to_screen(self, x, y, p=1.0):
        z = 1.0 + (self.zoom - 1.0) * p
        return W / 2 + (x - (W / 2 + (self.x - W / 2) * p)) * z, H / 2 + (y - (H / 2 + (self.y - H / 2) * p)) * z


class Layer:
    """Static scenery pre-rendered once: draw_fn(canvas) in world units inside rect (l, t, r, b)."""

    def __init__(self, rect, draw_fn, blur=0.0, shadow=(0.0, 0.0), res=1.0):
        self.l, self.t, self.r, self.b = rect
        self.draw_fn, self.blur, self.shadow, self.res = draw_fn, blur, shadow, res
        self._img = None

    def image(self):
        if self._img is None:
            w = int(math.ceil((self.r - self.l) * self.res))
            h = int(math.ceil((self.b - self.t) * self.res))
            surf = skia.Surface(w, h)
            c = surf.getCanvas()
            c.clear(skia.ColorTRANSPARENT)
            lift, alpha = self.shadow
            filt = None
            if lift > 0:
                filt = skia.ImageFilters.Merge([
                    skia.ImageFilters.DropShadowOnly(3.0 * lift * self.res, 4.5 * lift * self.res,
                                                     (2 + 4 * lift) * self.res, (2 + 4 * lift) * self.res,
                                                     col("shadow", alpha)), None])
            if self.blur > 0:
                bl = skia.ImageFilters.Blur(self.blur * self.res, self.blur * self.res, skia.TileMode.kDecal, filt)
                filt = bl
            if filt is not None:
                c.saveLayer(None, skia.Paint(ImageFilter=filt))
            c.scale(self.res, self.res)
            c.translate(-self.l, -self.t)
            self.draw_fn(c)
            if filt is not None:
                c.restore()
            self._img = surf.makeImageSnapshot()
        return self._img

    def draw(self, c, dx=0.0, dy=0.0, alpha=1.0):
        img = self.image()
        p = skia.Paint(AntiAlias=True)
        if alpha < 1:
            p.setAlphaf(alpha)
        c.save()
        c.translate(self.l + dx, self.t + dy)
        if self.res != 1.0:
            c.scale(1 / self.res, 1 / self.res)
        c.drawImage(img, 0, 0, SAMPLING, p)
        c.restore()


# ----------------------------------------------------------------------------------------------------
# finishing
# ----------------------------------------------------------------------------------------------------
@lru_cache(None)
def _light_overlay():
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    g = skia.GradientShader.MakeRadial(skia.Point(W * 0.34, H * 0.18), W * 0.95,
                                       [skia.Color(255, 244, 222, 46), skia.Color(255, 244, 222, 0)], [0.0, 1.0])
    c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=g))
    v = skia.GradientShader.MakeRadial(skia.Point(W / 2, H * 0.5), W * 0.72,
                                       [skia.Color(0, 0, 0, 0), skia.Color(0, 0, 0, 0), skia.Color(38, 24, 10, 120)],
                                       [0.0, 0.62, 1.0])
    c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=v))
    return surf.makeImageSnapshot()


@lru_cache(None)
def _grain():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(100)
    n = gaussian_filter(rng.normal(0, 1, (512, 512)), 0.8, mode="wrap")
    a = np.clip(n / n.std() * 30 + 128, 0, 255).astype(np.uint8)
    arr = np.dstack([a, a, a, np.full_like(a, 255)])
    return skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType)


def finish(c, frame, grain=0.0):
    c.resetMatrix()
    c.drawImage(_light_overlay(), 0, 0)
    if grain > 0:
        m = skia.Matrix()
        m.setTranslate(hash1(frame * 3 + 1) * 512, hash1(frame * 5 + 2) * 512)
        p = skia.Paint(BlendMode=skia.BlendMode.kOverlay,
                       Shader=_grain().makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat,
                                                  skia.SamplingOptions(skia.FilterMode.kNearest), m))
        p.setAlphaf(grain)
        c.drawRect(skia.Rect.MakeWH(W, H), p)
