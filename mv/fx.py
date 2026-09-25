"""Full-frame post effects: bloom, chromatic aberration, grade, grain, vignette, flashes."""
import math

import numpy as np
import skia

from .core import BM, H, P, W, col, lin_grad, rad_grad

_small = None
_small2 = None
_grain = None
_vig = None
SAMP = skia.SamplingOptions(skia.FilterMode.kLinear)


def _surfaces():
    global _small, _small2
    if _small is None:
        _small = skia.Surface(W // 4, H // 4)
        _small2 = skia.Surface(W // 4, H // 4)
    return _small, _small2


def bloom(surface, amount=0.6, threshold=0.55, radius=10.0, tint=None):
    if amount <= 0.01:
        return
    img = surface.makeImageSnapshot()
    s1, s2 = _surfaces()
    k = 1.0 / (1.0 - threshold)
    m = [k, 0, 0, 0, -threshold * k,
         0, k, 0, 0, -threshold * k,
         0, 0, k, 0, -threshold * k,
         0, 0, 0, 1, 0]
    p = skia.Paint()
    p.setColorFilter(skia.ColorFilters.Matrix(m))
    c1 = s1.getCanvas()
    c1.clear(skia.ColorBLACK)
    c1.drawImageRect(img, skia.Rect.MakeWH(W, H), skia.Rect.MakeWH(W // 4, H // 4), SAMP, p)
    c2 = s2.getCanvas()
    c2.clear(skia.ColorBLACK)
    bp = skia.Paint()
    bp.setImageFilter(skia.ImageFilters.Blur(radius, radius, skia.TileMode.kClamp))
    c2.drawImage(s1.makeImageSnapshot(), 0, 0, SAMP, bp)
    # a second, wider pass for a soft halo
    c1.clear(skia.ColorBLACK)
    bp2 = skia.Paint()
    bp2.setImageFilter(skia.ImageFilters.Blur(radius * 3, radius * 3, skia.TileMode.kClamp))
    c1.drawImage(s2.makeImageSnapshot(), 0, 0, SAMP, bp2)
    canvas = surface.getCanvas()
    ap = skia.Paint()
    ap.setBlendMode(BM.kPlus)
    ap.setAlphaf(min(1.0, amount))
    if tint is not None:
        ap.setColorFilter(skia.ColorFilters.Blend(col(tint), BM.kModulate))
    canvas.drawImageRect(s2.makeImageSnapshot(), skia.Rect.MakeWH(W, H), SAMP, ap)
    ap.setAlphaf(min(1.0, amount * 0.8))
    canvas.drawImageRect(s1.makeImageSnapshot(), skia.Rect.MakeWH(W, H), SAMP, ap)


def grade(surface, sat=1.0, contrast=1.0, bright=0.0, tint=(1, 1, 1), lift=(0, 0, 0)):
    img = surface.makeImageSnapshot()
    lr, lg, lb = 0.2126, 0.7152, 0.0722
    s = sat
    sm = [
        [lr * (1 - s) + s, lg * (1 - s), lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s) + s, lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s), lb * (1 - s) + s],
    ]
    m = []
    off = 0.5 * (1 - contrast) + bright
    for i in range(3):
        row = [sm[i][j] * contrast * tint[i] for j in range(3)]
        m += row + [0, off + lift[i]]
    m += [0, 0, 0, 1, 0]
    p = skia.Paint()
    p.setColorFilter(skia.ColorFilters.Matrix(m))
    c = surface.getCanvas()
    c.save()
    c.resetMatrix()
    p.setBlendMode(BM.kSrc)
    c.drawImage(img, 0, 0, SAMP, p)
    c.restore()


def chroma(surface, shift=6.0, angle=0.0):
    if shift < 0.4:
        return
    img = surface.makeImageSnapshot()
    c = surface.getCanvas()
    c.save()
    c.resetMatrix()
    c.clear(skia.ColorBLACK)
    dx, dy = math.cos(angle) * shift, math.sin(angle) * shift
    for (mask, ox, oy) in (((1, 0, 0), dx, dy), ((0, 1, 0), 0, 0), ((0, 0, 1), -dx, -dy)):
        m = [mask[0], 0, 0, 0, 0,
             0, mask[1], 0, 0, 0,
             0, 0, mask[2], 0, 0,
             0, 0, 0, 1, 0]
        p = skia.Paint()
        p.setColorFilter(skia.ColorFilters.Matrix(m))
        p.setBlendMode(BM.kPlus)
        c.drawImage(img, ox, oy, SAMP, p)
    c.restore()


def _grain_tiles():
    global _grain
    if _grain is None:
        rng = np.random.default_rng(3)
        _grain = []
        for i in range(6):
            n = rng.normal(128, 40, (256, 256)).clip(0, 255).astype(np.uint8)
            rgba = np.dstack([n, n, n, np.full_like(n, 255)])
            _grain.append(skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType))
    return _grain


def grain(canvas, frame, amount=0.07):
    tiles = _grain_tiles()
    img = tiles[frame % len(tiles)]
    ox, oy = (frame * 97) % 256, (frame * 61) % 256
    mat = skia.Matrix()
    mat.setScale(1.6, 1.6)
    mat.postTranslate(ox, oy)
    sh = img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, SAMP, mat)
    p = skia.Paint(Shader=sh)
    p.setBlendMode(BM.kOverlay)
    p.setAlphaf(amount)
    canvas.save()
    canvas.resetMatrix()
    canvas.drawRect(skia.Rect.MakeWH(W, H), p)
    canvas.restore()


def vignette(canvas, strength=0.55, color="#000000"):
    canvas.save()
    canvas.resetMatrix()
    canvas.drawRect(skia.Rect.MakeWH(W, H),
                    P(shader=rad_grad((W / 2, H / 2), W * 0.72, [(0, color, 0), (0.55, color, 0.0),
                                                                  (1, color, strength)])))
    canvas.restore()


def flash(canvas, color="#ffffff", a=1.0, blend=None):
    if a <= 0.004:
        return
    canvas.save()
    canvas.resetMatrix()
    canvas.drawRect(skia.Rect.MakeWH(W, H), P(color, a, blend=blend))
    canvas.restore()


def letterbox(canvas, k):
    if k <= 0:
        return
    h = H * 0.12 * k
    canvas.save()
    canvas.resetMatrix()
    canvas.drawRect(skia.Rect.MakeLTRB(0, 0, W, h), P("#000000"))
    canvas.drawRect(skia.Rect.MakeLTRB(0, H - h, W, H), P("#000000"))
    canvas.restore()


def scanlines(canvas, a=0.12, t=0.0):
    canvas.save()
    canvas.resetMatrix()
    p = P("#000000", a, stroke=2)
    off = int(t * 60) % 4
    for y in range(off, H, 4):
        canvas.drawLine(0, y, W, y, p)
    canvas.restore()


def speed_lines(canvas, t, a=0.4, n=40, color="#ffffff", cx=W / 2, cy=H / 2):
    canvas.save()
    canvas.resetMatrix()
    rng = np.random.default_rng(int(t * 24))
    for i in range(n):
        ang = rng.uniform(0, math.tau)
        r0 = rng.uniform(0.35, 0.6) * W
        r1 = r0 + rng.uniform(0.15, 0.5) * W
        canvas.drawLine(cx + math.cos(ang) * r0, cy + math.sin(ang) * r0, cx + math.cos(ang) * r1,
                        cy + math.sin(ang) * r1, P(color, a * rng.uniform(0.3, 1), stroke=rng.uniform(1.5, 5)))
    canvas.restore()


def iris(canvas, k, cx=W / 2, cy=H / 2):
    """Circle-wipe: k=1 fully open, k=0 fully black."""
    if k >= 1:
        return
    r = math.hypot(W, H) * 0.6 * max(0.0, k)
    path = skia.Path()
    path.addRect(skia.Rect.MakeLTRB(-10, -10, W + 10, H + 10))
    path.addCircle(cx, cy, r)
    path.setFillType(skia.PathFillType.kEvenOdd)
    canvas.save()
    canvas.resetMatrix()
    canvas.drawPath(path, P("#000000"))
    canvas.restore()
