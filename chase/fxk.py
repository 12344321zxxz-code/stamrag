"""Anime / manga effects: lightning, speed lines, halftone, SFX lettering, shockwaves, impact frames, glitch."""
import math

import numpy as np
import skia

from mv import fx as mvfx
from .core import (BM, H, TAU, W, C, P, clamp, col, ease_back, ease_out, font, hash1, lerp, lin_grad, mix, path_from,
                   poly, rad_grad, scale_c)

SAMP = skia.SamplingOptions(skia.FilterMode.kLinear)


# ----------------------------------------------------------------------------
# lightning
# ----------------------------------------------------------------------------
def bolt_points(p0, p1, seed, jag=0.22, depth=6):
    rng = np.random.default_rng(seed)
    pts = [p0, p1]
    disp = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * jag
    for _ in range(depth):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) or 1
            o = rng.uniform(-1, 1) * disp
            out += [(mx - dy / L * o, my + dx / L * o), b]
        pts = out
        disp *= 0.55
    return pts


def lightning(c, p0, p1, seed=0, width=6.0, color="#9af6ff", core="#ffffff", a=1.0, branches=3, jag=0.22, glow=True):
    pts = bolt_points(p0, p1, seed, jag)
    path = path_from(pts, close=False)
    if glow:
        c.drawPath(path, P(color, 0.35 * a, stroke=width * 6, blur=width * 3))
        c.drawPath(path, P(color, 0.8 * a, stroke=width * 2.2, blur=width * 0.8))
    c.drawPath(path, P(core, a, stroke=width))
    rng = np.random.default_rng(seed + 99)
    for k in range(branches):
        i = int(rng.uniform(0.15, 0.8) * len(pts))
        a0 = pts[i]
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * rng.uniform(0.15, 0.35)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) + rng.uniform(-0.9, 0.9)
        b0 = (a0[0] + math.cos(ang) * L, a0[1] + math.sin(ang) * L)
        bp = path_from(bolt_points(a0, b0, seed + k * 7, jag, 4), close=False)
        if glow:
            c.drawPath(bp, P(color, 0.5 * a, stroke=width * 1.5, blur=width))
        c.drawPath(bp, P(core, 0.9 * a, stroke=width * 0.45))
    return pts


# ----------------------------------------------------------------------------
# speed lines
# ----------------------------------------------------------------------------
def radial_lines(c, cx, cy, t, n=110, inner=0.32, color="#ffffff", a=0.9, fps_hold=12, seed=0, width=1.0):
    """Manga focus lines: tapered wedges from the frame edge toward the centre (re-rolled on 'twos')."""
    k = int(t * fps_hold)
    rng = np.random.default_rng(seed * 1000 + k)
    R = math.hypot(W, H)
    for i in range(n):
        ang = rng.uniform(0, TAU)
        r0 = R * rng.uniform(inner, inner + 0.25)
        wdt = rng.uniform(0.002, 0.012) * width
        p = poly([(cx + math.cos(ang - wdt) * R, cy + math.sin(ang - wdt) * R),
                  (cx + math.cos(ang) * r0, cy + math.sin(ang) * r0),
                  (cx + math.cos(ang + wdt) * R, cy + math.sin(ang + wdt) * R)])
        c.drawPath(p, P(color, a * rng.uniform(0.5, 1.0)))


def hlines(c, t, n=40, color="#ffffff", a=0.5, y0=0, y1=H, speed=3000, seed=0, direction=-1, width=3):
    rng = np.random.default_rng(seed)
    for i in range(n):
        y = rng.uniform(y0, y1)
        L = rng.uniform(150, 700)
        sp = speed * rng.uniform(0.7, 1.3)
        x = (rng.uniform(0, W + L) + direction * t * sp) % (W + L * 2) - L
        c.drawLine(x, y, x + L, y, P(color, a * rng.uniform(0.3, 1), stroke=width * rng.uniform(0.5, 1.5)))


# ----------------------------------------------------------------------------
# screentone / halftone
# ----------------------------------------------------------------------------
_tone = {}


def tone_shader(cell=12, dot=0.42, color="#000000", angle=45):
    key = (cell, dot, color, angle)
    if key not in _tone:
        s = skia.Surface(cell, cell)
        cc = s.getCanvas()
        cc.clear(skia.ColorTRANSPARENT)
        cc.drawCircle(cell / 2, cell / 2, cell * dot / 2 * 1.4, P(color))
        img = s.makeImageSnapshot()
        m = skia.Matrix()
        m.setRotate(angle)
        _tone[key] = img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, SAMP, m)
    return _tone[key]


def halftone(c, path_or_rect, color="#000000", a=0.35, cell=12, dot=0.42, angle=45):
    p = skia.Paint(AntiAlias=True, Shader=tone_shader(cell, dot, color, angle))
    p.setAlphaf(a)
    if isinstance(path_or_rect, skia.Rect):
        c.drawRect(path_or_rect, p)
    else:
        c.drawPath(path_or_rect, p)


# ----------------------------------------------------------------------------
# lettering
# ----------------------------------------------------------------------------
def outlined_text(c, s, x, y, size, fnt="Bangers.ttf", fill="#ffffff", stroke="#120a18", sw=None, a=1.0,
                  shadow=True, align="center", skew=0.0, shader=None, glow=None, extra_stroke=None):
    f = font(fnt, size)
    w = f.measureText(s)
    x0 = x - w / 2 if align == "center" else (x - w if align == "right" else x)
    sw = sw if sw is not None else size * 0.12
    c.save()
    if skew:
        c.translate(x, y)
        c.skew(skew, 0)
        c.translate(-x, -y)
    if glow:
        c.drawString(s, x0, y, f, P(glow, 0.8 * a, blur=size * 0.18))
    if shadow:
        c.drawString(s, x0 + size * 0.06, y + size * 0.08, f, P("#000000", 0.55 * a, stroke=sw))
        c.drawString(s, x0 + size * 0.06, y + size * 0.08, f, P("#000000", 0.55 * a))
    if extra_stroke:
        c.drawString(s, x0, y, f, P(extra_stroke, a, stroke=sw * 2.2))
    c.drawString(s, x0, y, f, P(stroke, a, stroke=sw))
    if shader is not None:
        p = skia.Paint(AntiAlias=True, Shader=shader)
        p.setAlphaf(a)
        c.drawString(s, x0, y, f, p)
    else:
        c.drawString(s, x0, y, f, P(fill, a))
    c.restore()
    return w


def sfx(c, text, x, y, size, age, color="#ffd21f", stroke="#120a18", angle=-8, fnt="Bangers.ttf", life=0.9, a=1.0,
        jitter=4.0, seed=0):
    """Manga sound-effect lettering that pops in, shakes, then fades."""
    if age < 0 or age > life:
        return
    k = ease_back(clamp(age / 0.12), 2.5)
    fade = clamp((life - age) / 0.25)
    jx = math.sin(age * 60 + seed) * jitter * (1 - clamp(age / 0.4))
    jy = math.cos(age * 55 + seed * 2) * jitter * (1 - clamp(age / 0.4))
    c.save()
    c.translate(x + jx, y + jy)
    c.rotate(angle)
    c.scale(k * (1 + 0.1 * clamp(age / life)), k * (1 + 0.1 * clamp(age / life)))
    outlined_text(c, text, 0, size * 0.35, size, fnt, fill=color, stroke=stroke, a=a * fade, extra_stroke="#ffffff")
    c.restore()


def shockwave(c, x, y, age, rmax=900, color="#ffffff", a=1.0, width=40, life=0.6):
    if age < 0 or age > life:
        return
    f = age / life
    r = rmax * ease_out(f, 2)
    c.drawCircle(x, y, r, P(color, a * (1 - f) * 0.8, stroke=width * (1 - f) + 2))


def burst(c, x, y, r, n=14, color="#ffd21f", a=1.0, inner=0.55, rot=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n * 2):
        ang = rot + i / (n * 2) * TAU
        rr = r * (1 if i % 2 == 0 else inner * rng.uniform(0.85, 1.1))
        pts.append((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
    p = poly(pts)
    c.drawPath(p, P(color, a))
    return p


def sparks(c, x, y, age, seed=0, n=18, color="#fff39a", speed=600, a=1.0, life=0.5, size=3.0):
    if age < 0 or age > life:
        return
    rng = np.random.default_rng(seed)
    f = age / life
    for i in range(n):
        ang = rng.uniform(0, TAU)
        sp = speed * rng.uniform(0.4, 1.0)
        d = sp * age * (1 - 0.5 * f)
        px, py = x + math.cos(ang) * d, y + math.sin(ang) * d + 400 * age * age
        qx, qy = x + math.cos(ang) * d * 0.8, y + math.sin(ang) * d * 0.8 + 400 * age * age * 0.8
        c.drawLine(qx, qy, px, py, P(color, a * (1 - f), stroke=size))


# ----------------------------------------------------------------------------
# post effects
# ----------------------------------------------------------------------------
def impact(surface, amount=1.0, invert=True):
    """High-contrast 'impact frame': threshold to ink/paper and invert."""
    if amount <= 0:
        return
    img = surface.makeImageSnapshot()
    lr, lg, lb = 0.2126, 0.7152, 0.0722
    k, th = 6.0, 0.45
    # out = clamp(k * (L - th)), or its inverse 1 - k * (L - th)
    s, off = (-k, 1.0 + k * th) if invert else (k, -k * th)
    row = [s * lr, s * lg, s * lb, 0, off]
    m = row + row + row + [0, 0, 0, 1, 0]
    p = skia.Paint()
    p.setColorFilter(skia.ColorFilters.Matrix(m))
    p.setAlphaf(amount)
    c = surface.getCanvas()
    c.save()
    c.resetMatrix()
    c.drawImage(img, 0, 0, SAMP, p)
    c.restore()


def glitch(surface, amount, seed=0):
    if amount <= 0.01:
        return
    img = surface.makeImageSnapshot()
    c = surface.getCanvas()
    rng = np.random.default_rng(seed)
    c.save()
    c.resetMatrix()
    n = int(4 + 10 * amount)
    for i in range(n):
        y0 = rng.uniform(0, H)
        h = rng.uniform(8, 80) * amount + 4
        dx = rng.uniform(-80, 80) * amount
        c.save()
        c.clipRect(skia.Rect.MakeLTRB(0, y0, W, y0 + h))
        p = skia.Paint()
        if i % 3 == 0:
            p.setColorFilter(skia.ColorFilters.Matrix([1, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 0, 1, 0]))
        c.drawImage(img, dx, 0, SAMP, p)
        c.restore()
    c.restore()


def zoom_blur(surface, amount, cx=W / 2, cy=H / 2, steps=5):
    """Cheap radial motion blur by stacking scaled copies."""
    if amount <= 0.005:
        return
    img = surface.makeImageSnapshot()
    c = surface.getCanvas()
    c.save()
    c.resetMatrix()
    for i in range(1, steps + 1):
        s = 1 + amount * i / steps
        p = skia.Paint()
        p.setAlphaf(0.5 / (i + 0.5))
        c.save()
        c.translate(cx, cy)
        c.scale(s, s)
        c.translate(-cx, -cy)
        c.drawImage(img, 0, 0, SAMP, p)
        c.restore()
    c.restore()


def motion_smear(surface, dx, dy, steps=4):
    if abs(dx) + abs(dy) < 1:
        return
    img = surface.makeImageSnapshot()
    c = surface.getCanvas()
    c.save()
    c.resetMatrix()
    for i in range(1, steps + 1):
        p = skia.Paint()
        p.setAlphaf(0.45 / (i + 0.3))
        c.drawImage(img, -dx * i / steps, -dy * i / steps, SAMP, p)
    c.restore()


bloom = mvfx.bloom
grade = mvfx.grade
chroma = mvfx.chroma
grain = mvfx.grain
vignette = mvfx.vignette
flash = mvfx.flash
letterbox = mvfx.letterbox
scanlines = mvfx.scanlines
iris = mvfx.iris
