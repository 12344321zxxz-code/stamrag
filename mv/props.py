"""Scenery and props: skies, pyramids, dunes, palms, sphinx, boats, symbols, particles."""
import math

import numpy as np
import skia

from .core import (BM, C, H, P, TAU, W, clamp, col, ease_out, fbm, font, hash1, lerp, lin_grad,
                   mix, poly, rad_grad, scale_c, smooth, vnoise)

_cache = {}


def cached(key, fn):
    v = _cache.get(key)
    if v is None:
        v = fn()
        _cache[key] = v
    return v


# ----------------------------------------------------------------------------
# sky & celestial
# ----------------------------------------------------------------------------
def sky(c, stops, y0=0, y1=H, x0=-400, x1=W + 400):
    c.drawRect(skia.Rect.MakeLTRB(x0, y0 - 2000, x1, y1 + 2000),
               P(shader=lin_grad((0, y0), (0, y1), [(s, cc, 1.0) for s, cc in stops])))


def _stars(seed, n, w, h):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n) ** 1.0
    sz = rng.power(4, n) * -1 + 1
    sz = 0.6 + (1 - rng.power(1.0 / 0.25, n)) * 0.0 + rng.exponential(0.6, n)
    ph = rng.uniform(0, TAU, n)
    return xs, ys, np.clip(sz, 0.5, 4.0), ph


def stars(c, t, alpha=1.0, seed=1, n=700, w=W, h=H, x0=0, y0=0, band=None, tw=1.0):
    xs, ys, sz, ph = cached(("stars", seed, n, w, h), lambda: _stars(seed, n, w, h))
    if alpha <= 0.01:
        return
    for bucket in range(4):
        lo, hi = [(0, 1.0), (1.0, 1.6), (1.6, 2.4), (2.4, 9)][bucket]
        m = (sz >= lo) & (sz < hi)
        if not m.any():
            continue
        r = [0.7, 1.1, 1.6, 2.2][bucket]
        idx = np.nonzero(m)[0]
        tws = 0.55 + 0.45 * np.sin(t * (1.5 + (idx % 7) * 0.4) * tw + ph[idx])
        for group in range(3):
            g = idx[(tws >= group / 3) & (tws < (group + 1) / 3 + (0.01 if group == 2 else 0))]
            if len(g) == 0:
                continue
            a = alpha * (0.35 + 0.65 * (group + 0.5) / 3)
            pts = [skia.Point(x0 + xs[i], y0 + ys[i]) for i in g]
            p = P("#fff6e0", a, stroke=r * 2)
            c.drawPoints(skia.Canvas.PointMode.kPoints_PointMode, pts, p)
    big = np.nonzero(sz > 3.0)[0]
    for i in big[:40]:
        x, y = x0 + xs[i], y0 + ys[i]
        a = alpha * (0.5 + 0.5 * math.sin(t * 2.2 + ph[i]))
        L = 5 + sz[i] * 3
        p = P("#fff2d0", a * 0.8, stroke=1.2)
        c.drawLine(x - L, y, x + L, y, p)
        c.drawLine(x, y - L, x, y + L, p)
        c.drawCircle(x, y, 6, P("#ffe9b0", a * 0.3, blur=5))


def glow(c, x, y, r, color, a=1.0):
    c.drawCircle(x, y, r, P(shader=rad_grad((x, y), r, [(0, color, a), (0.35, color, a * 0.35), (1, color, 0)])))


def sun(c, x, y, r, core="#fff4c2", edge="#ff9a1f", glow_c="#ffb03a", glow_a=0.6, stripes=0, stripe_t=0.0,
        stripe_col=None, rays=0.0, ray_t=0.0):
    glow(c, x, y, r * 3.2, glow_c, glow_a)
    if rays > 0:
        c.save()
        c.translate(x, y)
        c.rotate(ray_t * 8)
        for i in range(24):
            a = i * 15
            c.save()
            c.rotate(a)
            path = poly([(r * 1.02, -r * 0.06), (r * 2.6, -r * 0.16), (r * 2.6, r * 0.16), (r * 1.02, r * 0.06)])
            c.drawPath(path, P(glow_c, 0.18 * rays))
            c.restore()
        c.restore()
    c.save()
    if stripes:
        for i in range(stripes):
            f = (i + stripe_t % 1.0) / stripes
            yy = y + r * (0.05 + f * 0.95)
            hh = r * (0.02 + 0.12 * f)
            c.clipRect(skia.Rect.MakeLTRB(x - r - 2, yy, x + r + 2, yy + hh), skia.ClipOp.kDifference, True)
    c.drawCircle(x, y, r, P(shader=lin_grad((x, y - r), (x, y + r), [(0, core, 1), (0.55, mix(core, edge, 0.6), 1),
                                                                      (1, edge, 1)])))
    c.restore()


def aten_rays(c, x, y, r, t, n=13, a=1.0, color="#ffd35a"):
    """Sun rays ending in little open hands (Amarna style), fanning downward."""
    for i in range(n):
        ang = math.radians(lerp(20, 160, i / (n - 1)))
        L = r * (2.2 + 0.25 * math.sin(t * 3 + i))
        x1, y1 = x + math.cos(ang) * L, y + math.sin(ang) * L
        c.drawLine(x + math.cos(ang) * r * 1.05, y + math.sin(ang) * r * 1.05, x1, y1, P(color, 0.55 * a, stroke=r * 0.04))
        c.save()
        c.translate(x1, y1)
        c.rotate(math.degrees(ang) - 90)
        hs = r * 0.09
        hand = poly([(-hs * 0.6, 0), (hs * 0.6, 0), (hs * 0.7, hs * 1.3), (-hs * 0.7, hs * 1.3)])
        c.drawPath(hand, P(color, 0.75 * a))
        c.restore()


def moon(c, x, y, r, a=1.0, glow_a=0.5):
    glow(c, x, y, r * 4, "#9fb8ff", glow_a * a)
    c.drawCircle(x, y, r, P(shader=rad_grad((x - r * 0.3, y - r * 0.3), r * 1.4,
                                            [(0, "#fffdf2", a), (0.7, "#e7e2d0", a), (1, "#b9b3a0", a)])))
    rng = np.random.default_rng(7)
    for _ in range(9):
        cx, cy = rng.uniform(-0.6, 0.6, 2)
        rr = rng.uniform(0.06, 0.2)
        if cx * cx + cy * cy < 0.55:
            c.drawCircle(x + cx * r, y + cy * r, rr * r, P("#a8a28e", 0.35 * a))


# ----------------------------------------------------------------------------
# pyramids, dunes, landscape
# ----------------------------------------------------------------------------
def pyramid(c, x, y, w, h, lit="#f0c070", shade="#a86a32", light=1, neon=None, neon_a=1.0, cap=0.0,
            courses=True, alpha=1.0, edge_x=0.18):
    ax, ay = x, y - h
    ex = x + w * edge_x * light
    left = poly([(x - w / 2, y), (ax, ay), (ex, y)])
    right = poly([(ex, y), (ax, ay), (x + w / 2, y)])
    lc, rc = (shade, lit) if light > 0 else (lit, shade)
    c.drawPath(left, P(shader=lin_grad((0, ay), (0, y), [(0, mix(lc, "white", 0.08), alpha), (1, scale_c(lc, 0.85), alpha)])))
    c.drawPath(right, P(shader=lin_grad((0, ay), (0, y), [(0, mix(rc, "white", 0.12), alpha), (1, scale_c(rc, 0.88), alpha)])))
    if courses and h > 60:
        n = int(clamp(h / 14, 6, 40))
        c.save()
        c.clipPath(poly([(x - w / 2, y), (ax, ay), (x + w / 2, y)]), skia.ClipOp.kIntersect, True)
        for i in range(1, n):
            yy = ay + h * i / n
            c.drawLine(x - w / 2, yy, x + w / 2, yy, P("black", 0.07 * alpha, stroke=max(1, h / 300)))
        c.restore()
    if cap > 0:
        ch = h * 0.09
        cw = w * 0.09
        cp = poly([(ax - cw / 2, ay + ch), (ax, ay), (ax + cw / 2, ay + ch)])
        c.drawPath(cp, P("#ffd84a", alpha))
        glow(c, ax, ay + ch * 0.4, w * 0.25 * cap, "#ffd35a", 0.8 * cap * alpha)
    if neon:
        edge = poly([(x - w / 2, y), (ax, ay), (x + w / 2, y)])
        c.drawPath(edge, P(neon, 0.45 * neon_a, stroke=10, blur=8))
        c.drawPath(edge, P(mix(neon, "white", 0.5), neon_a, stroke=2.5))
        c.drawLine(ax, ay, ex, y, P(neon, 0.6 * neon_a, stroke=2))


def ridge(xw, base, amp, seed, freq=1.0):
    return base + amp * (0.6 * math.sin(xw * 0.0021 * freq + seed) + 0.3 * math.sin(xw * 0.0053 * freq + seed * 2.1)
                         + 0.15 * math.sin(xw * 0.011 * freq + seed * 3.7))


def dune(c, scroll, base, amp, seed, top, bottom, rim=None, freq=1.0, x0=-60, x1=W + 60, step=16, bottom_y=H + 40,
         alpha=1.0):
    path = skia.Path()
    path.moveTo(x0, bottom_y)
    ymin = 1e9
    for x in range(int(x0), int(x1) + step, step):
        y = ridge(x + scroll, base, amp, seed, freq)
        ymin = min(ymin, y)
        path.lineTo(x, y)
    path.lineTo(x1, bottom_y)
    path.close()
    c.drawPath(path, P(shader=lin_grad((0, ymin), (0, bottom_y), [(0, top, alpha), (1, bottom, alpha)])))
    if rim:
        rp = skia.Path()
        first = True
        for x in range(int(x0), int(x1) + step, step):
            y = ridge(x + scroll, base, amp, seed, freq)
            if first:
                rp.moveTo(x, y)
                first = False
            else:
                rp.lineTo(x, y)
        c.drawPath(rp, P(rim, 0.55 * alpha, stroke=3))


def palm(c, x, y, h, t=0.0, lean=0.1, color=None, trunk="#6b4a2b", frond="#2f7d4a", sil=False, seed=0, a=1.0):
    """Palm tree rooted at (x,y), height h."""
    if sil:
        trunk = frond = color or "#120b08"
    sway = math.sin(t * 1.3 + seed) * 0.03
    tx = x + h * (lean + sway)
    ty = y - h
    # trunk
    tp = skia.Path()
    w0, w1 = h * 0.045, h * 0.028
    cx, cy = x + h * lean * 0.25, y - h * 0.5
    tp.moveTo(x - w0, y)
    tp.quadTo(cx - w1, cy, tx - w1, ty)
    tp.lineTo(tx + w1, ty)
    tp.quadTo(cx + w1, cy, x + w0, y)
    tp.close()
    c.drawPath(tp, P(trunk, a))
    if not sil:
        for i in range(1, 12):
            f = i / 12
            px = (1 - f) ** 2 * x + 2 * (1 - f) * f * cx + f * f * tx
            py = (1 - f) ** 2 * y + 2 * (1 - f) * f * cy + f * f * ty
            ww = lerp(w0, w1, f)
            c.drawLine(px - ww, py + 2, px + ww, py - 3, P(scale_c(trunk, 0.7), a, stroke=2))
    # fronds
    n = 9
    for i in range(n):
        ang = lerp(-160, -20, i / (n - 1)) + math.sin(t * 1.7 + i + seed) * 4
        L = h * (0.42 + 0.08 * hash1(i + seed * 13))
        droop = 0.55
        ex = tx + math.cos(math.radians(ang)) * L
        ey = ty + math.sin(math.radians(ang)) * L * 0.55 + L * droop * 0.55
        mx = tx + math.cos(math.radians(ang)) * L * 0.55
        my = ty + math.sin(math.radians(ang)) * L * 0.55 - L * 0.12
        fp = skia.Path()
        wd = h * 0.035
        fp.moveTo(tx, ty)
        fp.quadTo(mx, my - wd, ex, ey)
        fp.quadTo(mx, my + wd, tx, ty)
        fp.close()
        fc = frond if sil else mix(frond, "#0e2a18", 0.3 * (i % 2))
        c.drawPath(fp, P(fc, a))
        if not sil:
            c.drawPath(fp, P(scale_c(frond, 0.6), a * 0.7, stroke=1.2))
    if not sil:
        for k in range(3):
            c.drawCircle(tx + (k - 1) * h * 0.03, ty + h * 0.03, h * 0.022, P("#8a4b1e", a))


def obelisk(c, x, y, w, h, lit="#e9b870", shade="#a86d3a", glyph_col="#7a4a22", a=1.0, t=0.0, glow_top=0.0):
    body = poly([(x - w / 2, y), (x - w * 0.36, y - h), (x + w * 0.36, y - h), (x + w / 2, y)])
    c.drawPath(body, P(lit, a))
    c.drawPath(poly([(x, y), (x, y - h), (x + w * 0.36, y - h), (x + w / 2, y)]), P(shade, a))
    tip = poly([(x - w * 0.36, y - h), (x, y - h - w * 0.6), (x + w * 0.36, y - h)])
    c.drawPath(tip, P("#ffd65a", a))
    if glow_top > 0:
        glow(c, x, y - h - w * 0.3, w * 2.5, "#ffd35a", glow_top * a)
    f = font("Hiero.ttf", w * 0.42)
    glyphs = cached(("obglyph",), lambda: glyph_list(40, 3))
    for i in range(int(h / (w * 0.5)) - 1):
        g = glyphs[i % len(glyphs)]
        c.drawString(g, x - w * 0.2, y - h + w * 0.7 + i * w * 0.5, f, P(glyph_col, 0.8 * a))


def camel(c, x, y, s, t, color="#1a0f0a", a=1.0):
    """Silhouette camel walking right, feet at y."""
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    p = P(color, a)
    body = skia.Path()
    body.addOval(skia.Rect.MakeLTRB(-0.55, -1.25, 0.45, -0.75))
    c.drawPath(body, p)
    hump = skia.Path()
    hump.addOval(skia.Rect.MakeLTRB(-0.35, -1.5, 0.15, -1.0))
    c.drawPath(hump, p)
    neck = poly([(0.3, -1.1), (0.62, -1.55), (0.72, -1.5), (0.45, -0.95)])
    c.drawPath(neck, p)
    head = skia.Path()
    head.addOval(skia.Rect.MakeLTRB(0.58, -1.66, 0.9, -1.48))
    c.drawPath(head, p)
    for i, (lx, ph) in enumerate(((-0.4, 0), (-0.3, 0.5), (0.25, 0.5), (0.35, 0))):
        sw = math.sin(t * 5 + ph * TAU) * 0.12
        c.drawLine(lx, -0.9, lx + sw, 0, P(color, a, stroke=0.07))
    c.drawLine(-0.55, -1.0, -0.68, -0.6, P(color, a, stroke=0.04))
    c.restore()


# ----------------------------------------------------------------------------
# hieroglyphs & symbols
# ----------------------------------------------------------------------------
_GLYPHS = None


def glyph_list(n=200, seed=0):
    global _GLYPHS
    if _GLYPHS is None:
        f = font("Hiero.ttf", 40)
        ok = []
        for cp in range(0x13000, 0x1342F):
            ch = chr(cp)
            gid = f.textToGlyphs(ch)
            if len(gid) and gid[0] != 0:
                ok.append(ch)
        _GLYPHS = ok
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(_GLYPHS), n)
    return [_GLYPHS[i] for i in idx]


FAV = [chr(c) for c in (0x13080, 0x132F9, 0x131A3, 0x131F3, 0x13153, 0x1313F, 0x13171, 0x13193, 0x131CB,
                        0x13216, 0x13250, 0x132AA, 0x133CF, 0x1340D, 0x130C0, 0x1308B, 0x13143, 0x1319B,
                        0x132BD, 0x13079, 0x13000, 0x13050, 0x130ED, 0x13191)]


def glyph_column(c, x, y, h, size, color, seed=0, a=1.0, t=0.0, lit=0.0, lit_col="#fff0a0"):
    f = font("Hiero.ttf", size)
    gl = cached(("gcol", seed), lambda: glyph_list(60, seed))
    n = int(h / (size * 1.08))
    for i in range(n):
        g = gl[i % len(gl)]
        yy = y + (i + 1) * size * 1.08
        cc = color
        if lit > 0:
            k = clamp(1 - abs(((t * 1.5 + seed * 0.13) % 1.0) * n - i) / 3.0) * lit
            cc = mix(color, lit_col, k)
        c.drawString(g, x, yy, f, P(cc, a))


def glyph_row(c, x, y, w, size, color, seed=0, a=1.0):
    f = font("Hiero.ttf", size)
    gl = cached(("grow", seed), lambda: glyph_list(80, seed + 100))
    n = int(w / (size * 1.05))
    for i in range(n):
        c.drawString(gl[i % len(gl)], x + i * size * 1.05, y, f, P(color, a))


def eye_of_horus(c, x, y, s, color="#1a0f0a", a=1.0, fill_iris=True, stroke_k=1.0, open_=1.0, pupil_col=None,
                 glow_col=None):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    sw = 0.13 * stroke_k
    p = P(color, a, stroke=sw)
    if glow_col:
        gp = P(glow_col, 0.5 * a, stroke=sw * 3, blur=0.15)
    brow = skia.Path()
    brow.moveTo(-1.05, -0.55)
    brow.quadTo(0.0, -0.95, 1.25, -0.62)
    eye = skia.Path()
    h1, h2 = 0.5 * open_, 0.32 * open_
    eye.moveTo(-0.95, 0.02)
    eye.quadTo(-0.1, -h1 * 1.6, 0.8, -0.02)
    eye.quadTo(-0.1, h2 * 1.4, -0.95, 0.02)
    eye.close()
    ext = skia.Path()
    ext.moveTo(0.78, -0.02)
    ext.lineTo(1.55, -0.02)
    drop = skia.Path()
    drop.moveTo(-0.15, 0.3)
    drop.quadTo(-0.12, 0.7, -0.3, 1.15)
    spiral = skia.Path()
    spiral.moveTo(0.25, 0.28)
    spiral.cubicTo(0.35, 0.8, 1.05, 1.15, 1.2, 0.75)
    spiral.cubicTo(1.3, 0.45, 0.95, 0.4, 0.92, 0.62)
    for path in (brow, eye, ext, drop, spiral):
        if glow_col:
            c.drawPath(path, gp)
    if fill_iris:
        c.drawPath(eye, P("#fbf3dc", a * 0.9))
    for path in (brow, eye, ext, drop, spiral):
        c.drawPath(path, p)
    if open_ > 0.15:
        c.save()
        c.clipPath(eye, skia.ClipOp.kIntersect, True)
        c.drawCircle(-0.08, -0.05, 0.3, P(pupil_col or color, a))
        c.restore()
    brow_fill = skia.Path()
    brow_fill.moveTo(-1.05, -0.55)
    brow_fill.quadTo(0.0, -0.95, 1.25, -0.62)
    brow_fill.lineTo(1.2, -0.52)
    brow_fill.quadTo(0.0, -0.82, -1.0, -0.45)
    brow_fill.close()
    c.drawPath(brow_fill, P(color, a))
    c.restore()


def ankh(c, x, y, s, color="#f2c14e", a=1.0, stroke=None):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    loop = skia.Path()
    loop.addOval(skia.Rect.MakeLTRB(-0.32, -1.0, 0.32, -0.25))
    if stroke:
        p = P(color, a, stroke=stroke)
        c.drawPath(loop, p)
        c.drawLine(-0.6, -0.18, 0.6, -0.18, p)
        c.drawLine(0, -0.25, 0, 1.0, p)
    else:
        c.drawPath(loop, P(color, a, stroke=0.16))
        c.drawPath(poly([(-0.62, -0.26), (0.62, -0.26), (0.62, -0.08), (-0.62, -0.08)]), P(color, a))
        c.drawPath(poly([(-0.1, -0.2), (0.1, -0.2), (0.16, 1.0), (-0.16, 1.0)]), P(color, a))
    c.restore()


def ankh_points(n=60):
    pts = []
    for i in range(n):
        f = i / n
        if f < 0.45:
            a = f / 0.45 * TAU
            pts.append((math.sin(a) * 0.32, -0.62 - math.cos(a) * 0.38))
        elif f < 0.7:
            k = (f - 0.45) / 0.25
            pts.append((lerp(-0.62, 0.62, k), -0.17))
        else:
            k = (f - 0.7) / 0.3
            pts.append((0.0, lerp(-0.2, 1.0, k)))
    return pts


def scarab(c, x, y, s, ang=0.0, wings=0.0, body="#1f5a6b", shell="#2ec4b6", gold=False, a=1.0, t=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    c.scale(s, s)
    if gold:
        body, shell = "#b07a12", "#f2c14e"
    lp = P("#140c08", a, stroke=0.07)
    for side in (-1, 1):
        for j, oy in enumerate((-0.25, 0.05, 0.35)):
            k = math.sin(t * 20 + j * 2 + side) * 0.1
            c.drawLine(0.25 * side, oy, (0.7 + k) * side, oy + 0.15 + j * 0.05, lp)
    if wings > 0:
        for side in (-1, 1):
            c.save()
            c.rotate(side * (25 + 55 * wings))
            wp = skia.Path()
            wp.addOval(skia.Rect.MakeLTRB(-0.28 + side * 0.4, -0.2, 0.28 + side * 0.4, 1.25))
            c.drawPath(wp, P("#bff4ff" if not gold else "#fff0b0", 0.45 * a * wings))
            c.restore()
    head = skia.Path()
    head.addOval(skia.Rect.MakeLTRB(-0.22, -0.78, 0.22, -0.45))
    c.drawPath(head, P(body, a))
    pron = skia.Path()
    pron.addOval(skia.Rect.MakeLTRB(-0.38, -0.6, 0.38, -0.12))
    c.drawPath(pron, P(body, a))
    el = skia.Path()
    el.addOval(skia.Rect.MakeLTRB(-0.45, -0.22, 0.45, 0.75))
    c.drawPath(el, P(shader=rad_grad((-0.12, 0.05), 0.8, [(0, mix(shell, "white", 0.45), a), (0.5, shell, a),
                                                          (1, body, a)])))
    c.drawLine(0, -0.2, 0, 0.74, P(body, a, stroke=0.05))
    for yy in (-0.62, -0.72):
        pass
    c.restore()


def lotus(c, x, y, s, open_=1.0, color="#ff8fb8", tip="#ffd6e7", a=1.0, pad=True):
    if pad:
        pp = skia.Path()
        pp.addOval(skia.Rect.MakeLTRB(x - s * 1.4, y - s * 0.28, x + s * 1.4, y + s * 0.28))
        c.drawPath(pp, P("#2f7d4a", a))
        c.drawPath(poly([(x, y), (x + s * 1.5, y - s * 0.1), (x + s * 1.5, y + s * 0.1)]), P("#0e2a2e", a))
    n = 7
    for i in range(n):
        f = (i / (n - 1)) * 2 - 1
        ang = f * (15 + 55 * open_)
        c.save()
        c.translate(x, y)
        c.rotate(ang)
        L = s * (1.1 - abs(f) * 0.25)
        pet = skia.Path()
        pet.moveTo(0, 0)
        pet.quadTo(-s * 0.35, -L * 0.55, 0, -L)
        pet.quadTo(s * 0.35, -L * 0.55, 0, 0)
        c.drawPath(pet, P(shader=lin_grad((0, 0), (0, -L), [(0, color, a), (1, tip, a)])))
        c.drawPath(pet, P(scale_c(color, 0.6), 0.5 * a, stroke=max(1, s * 0.04)))
        c.restore()
    c.drawCircle(x, y - s * 0.25, s * 0.18 * open_, P("#ffd23a", a * open_))


# ----------------------------------------------------------------------------
# big set pieces
# ----------------------------------------------------------------------------
def sphinx(c, x, y, s, eye_glow=0.0, lit="#e3b06a", shade="#a8743e", a=1.0, eye_col="#ff2e88"):
    """Lying sphinx facing right, (x,y) = ground under the chest, s = scale (height ~ s)."""
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    body = skia.Path()
    body.moveTo(-3.2, 0)
    body.cubicTo(-3.3, -0.5, -3.0, -0.75, -2.5, -0.78)
    body.lineTo(-0.6, -0.72)
    body.cubicTo(-0.2, -0.75, 0.05, -0.95, 0.1, -1.05)
    body.lineTo(0.55, -1.05)
    body.lineTo(0.62, -0.3)
    body.lineTo(1.9, -0.28)
    body.lineTo(1.95, 0)
    body.close()
    c.drawPath(body, P(shader=lin_grad((0, -1.1), (0, 0), [(0, lit, a), (1, shade, a)])))
    # rear haunch and front paw details
    haunch = skia.Path()
    haunch.addOval(skia.Rect.MakeLTRB(-3.1, -0.72, -2.1, 0.02))
    c.drawPath(haunch, P(scale_c(shade, 0.95), 0.55 * a))
    paw = poly([(0.35, -0.3), (1.95, -0.28), (1.98, 0), (0.35, 0)])
    c.drawPath(paw, P(mix(lit, "white", 0.1), a))
    for i in range(3):
        c.drawLine(1.7 + i * 0.08, -0.25, 1.72 + i * 0.08, 0, P(shade, 0.8 * a, stroke=0.02))
    tail = skia.Path()
    tail.moveTo(-3.1, -0.1)
    tail.quadTo(-2.4, -0.05, -2.1, -0.2)
    c.drawPath(tail, P(shade, a, stroke=0.06))
    # head with striped headcloth
    head = skia.Path()
    head.moveTo(-0.25, -0.95)
    head.cubicTo(-0.3, -1.9, 0.2, -2.25, 0.62, -2.2)
    head.cubicTo(0.95, -2.15, 1.05, -1.95, 1.02, -1.75)
    head.lineTo(1.1, -1.55)
    head.lineTo(1.0, -1.5)
    head.lineTo(1.02, -1.35)
    head.cubicTo(0.95, -1.18, 0.8, -1.12, 0.62, -1.15)
    head.lineTo(0.55, -0.9)
    head.close()
    c.drawPath(head, P(lit, a))
    cloth = skia.Path()
    cloth.moveTo(0.72, -1.9)
    cloth.cubicTo(0.6, -2.35, -0.1, -2.35, -0.3, -1.95)
    cloth.lineTo(-0.55, -1.0)
    cloth.lineTo(0.1, -0.85)
    cloth.lineTo(0.35, -1.55)
    cloth.quadTo(0.5, -1.85, 0.72, -1.9)
    cloth.close()
    c.drawPath(cloth, P("#d8a24e", a))
    c.save()
    c.clipPath(cloth, skia.ClipOp.kIntersect, True)
    for i in range(12):
        yy = -2.4 + i * 0.14
        c.drawLine(-1, yy + 0.3, 1, yy - 0.1, P("#27407f", 0.85 * a, stroke=0.06))
    c.restore()
    c.drawPath(cloth, P(shade, a, stroke=0.02))
    # face details
    c.drawLine(0.62, -1.72, 0.9, -1.74, P("#3a2210", a, stroke=0.03))
    eye = skia.Path()
    eye.moveTo(0.62, -1.62)
    eye.quadTo(0.76, -1.72, 0.9, -1.63)
    eye.quadTo(0.76, -1.56, 0.62, -1.62)
    c.drawPath(eye, P("#1c1008", a))
    if eye_glow > 0:
        c.drawCircle(0.8, -1.63, 0.25, P(eye_col, 0.6 * eye_glow * a, blur=0.12))
        c.drawCircle(0.8, -1.63, 0.06, P("white", eye_glow * a))
    beard = poly([(0.78, -1.15), (0.9, -1.16), (0.88, -0.85), (0.8, -0.84)])
    c.drawPath(beard, P("#8a5a26", a))
    c.restore()
    return (x + 0.8 * s, y - 1.63 * s)  # eye position


def felucca(c, x, y, s, t=0.0, sail="#f3e9d2", hull="#6b3f1f", a=1.0, rock=0.0):
    """Boat with lateen sail, waterline at (x, y)."""
    c.save()
    c.translate(x, y)
    c.rotate(rock)
    c.scale(s, s)
    hp = skia.Path()
    hp.moveTo(-1.6, -0.35)
    hp.quadTo(-1.2, 0.25, 0.0, 0.25)
    hp.quadTo(1.3, 0.25, 1.8, -0.45)
    hp.lineTo(1.5, -0.18)
    hp.lineTo(-1.4, -0.18)
    hp.close()
    c.drawPath(hp, P(shader=lin_grad((0, -0.45), (0, 0.25), [(0, mix(hull, "white", 0.15), a), (1, scale_c(hull, 0.6), a)])))
    c.drawLine(-1.45, -0.2, 1.52, -0.2, P("#f2c14e", a, stroke=0.05))
    c.drawLine(0.1, -0.2, 0.1, -3.0, P("#3b2412", a, stroke=0.06))
    sp = skia.Path()
    bil = math.sin(t * 1.3) * 0.08
    sp.moveTo(-1.4, -0.55)
    sp.lineTo(1.2, -3.6)
    sp.quadTo(0.9 + bil, -1.8, 0.35, -0.55)
    sp.close()
    c.drawPath(sp, P(shader=lin_grad((-1.4, -0.5), (1.2, -3.6), [(0, sail, a), (1, mix(sail, "#ffd9a0", 0.5), a)])))
    c.drawPath(sp, P(scale_c(sail, 0.7), a, stroke=0.03))
    c.drawLine(-1.5, -0.45, 1.3, -3.7, P("#3b2412", a, stroke=0.05))
    c.restore()


def barque(c, x, y, s, t=0.0, a=1.0, sun_r=0.55, gold="#f2c14e"):
    """Solar barque: papyrus boat carrying a sun disk."""
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    hp = skia.Path()
    hp.moveTo(-2.2, -1.1)
    hp.cubicTo(-2.0, -0.2, -1.4, 0.2, 0.0, 0.2)
    hp.cubicTo(1.4, 0.2, 2.0, -0.2, 2.3, -1.2)
    hp.quadTo(2.45, -1.45, 2.2, -1.35)
    hp.cubicTo(1.8, -0.45, 1.2, -0.2, 0.0, -0.2)
    hp.cubicTo(-1.2, -0.2, -1.7, -0.45, -2.05, -1.25)
    hp.quadTo(-2.25, -1.4, -2.2, -1.1)
    hp.close()
    c.drawPath(hp, P(shader=lin_grad((0, -1.4), (0, 0.2), [(0, mix(gold, "white", 0.3), a), (1, "#a8650e", a)])))
    for i in range(9):
        xx = -1.6 + i * 0.4
        c.drawLine(xx, -0.18, xx + 0.05, 0.15, P("#8a5210", 0.7 * a, stroke=0.035))
    shrine = poly([(-0.55, -0.2), (-0.5, -0.75), (0.5, -0.75), (0.55, -0.2)])
    c.drawPath(shrine, P("#1f3f95", a))
    c.drawPath(poly([(-0.6, -0.75), (0.6, -0.75), (0.6, -0.85), (-0.6, -0.85)]), P(gold, a))
    for i in range(4):
        c.drawLine(-0.4 + i * 0.27, -0.25, -0.4 + i * 0.27, -0.7, P("#2ec4b6", a, stroke=0.05))
    # oars
    for i in range(3):
        sw = math.sin(t * 3 + i) * 0.25
        c.drawLine(-1.2 + i * 0.3, -0.1, -1.6 + i * 0.3 + sw, 0.9, P("#6b3f1f", a, stroke=0.05))
    sun(c, 0, -0.85 - sun_r * 1.05, sun_r, core="#fff1b0", edge="#ff5a1f", glow_c="#ff9a3a", glow_a=0.55)
    c.restore()


def column(c, x, y, w, h, t=0.0, lit="#d9a765", shade="#8f5f30", paint_col=("#1f3f95", "#c0392b", "#2ec4b6"), a=1.0,
           glyph_seed=0):
    """Temple column with papyrus capital, base at (x,y)."""
    sh = poly([(x - w / 2, y), (x - w * 0.45, y - h * 0.8), (x + w * 0.45, y - h * 0.8), (x + w / 2, y)])
    c.drawPath(sh, P(shader=lin_grad((x - w / 2, 0), (x + w / 2, 0), [(0, shade, a), (0.35, lit, a), (0.7, lit, a),
                                                                      (1, scale_c(shade, 0.8), a)])))
    for i in range(3):
        yy = y - h * (0.8 - 0.03 * i)
        c.drawRect(skia.Rect.MakeLTRB(x - w * 0.46, yy - h * 0.012, x + w * 0.46, yy), P(paint_col[i % 3], a))
    cap = skia.Path()
    cap.moveTo(x - w * 0.45, y - h * 0.8)
    cap.cubicTo(x - w * 0.5, y - h * 0.88, x - w * 0.95, y - h * 0.93, x - w * 0.8, y - h * 0.97)
    cap.lineTo(x + w * 0.8, y - h * 0.97)
    cap.cubicTo(x + w * 0.95, y - h * 0.93, x + w * 0.5, y - h * 0.88, x + w * 0.45, y - h * 0.8)
    cap.close()
    c.drawPath(cap, P(lit, a))
    c.save()
    c.clipPath(cap, skia.ClipOp.kIntersect, True)
    for i in range(7):
        xx = x - w * 0.8 + i * w * 0.27
        c.drawLine(x + (xx - x) * 0.55, y - h * 0.8, xx, y - h * 0.97, P(paint_col[i % 3], 0.8 * a, stroke=w * 0.07))
    c.restore()
    c.drawRect(skia.Rect.MakeLTRB(x - w * 0.55, y - h, x + w * 0.55, y - h * 0.97), P(shade, a))
    f = font("Hiero.ttf", w * 0.34)
    gl = cached(("colg", glyph_seed), lambda: glyph_list(30, glyph_seed + 7))
    for i in range(int(h * 0.62 / (w * 0.38))):
        c.drawString(gl[i % len(gl)], x - w * 0.17, y - h * 0.72 + (i + 1) * w * 0.38, f, P("#5a3414", 0.55 * a))


def torch(c, x, y, s, t, seed=0, a=1.0):
    c.drawLine(x, y, x, y - s * 1.2, P("#3b2412", a, stroke=s * 0.12))
    bowl = poly([(x - s * 0.3, y - s * 1.2), (x + s * 0.3, y - s * 1.2), (x + s * 0.18, y - s * 1.0), (x - s * 0.18, y - s * 1.0)])
    c.drawPath(bowl, P("#f2c14e", a))
    fl = 1 + 0.25 * fbm(t * 6 + seed, seed)
    glow(c, x, y - s * 1.55, s * 3.5 * fl, "#ff9a3a", 0.35 * a)
    for k, (colr, sc) in enumerate((("#ff5a1f", 1.0), ("#ffb03a", 0.7), ("#fff2b0", 0.4))):
        w = s * 0.34 * sc
        hh = s * 1.2 * sc * fl
        sway = vnoise(t * 5 + seed + k, seed) * s * 0.12
        fp = skia.Path()
        by = y - s * 1.2
        fp.moveTo(x - w, by)
        fp.cubicTo(x - w, by - hh * 0.5, x + sway, by - hh * 0.7, x + sway * 1.5, by - hh)
        fp.cubicTo(x + sway, by - hh * 0.6, x + w, by - hh * 0.5, x + w, by)
        fp.close()
        c.drawPath(fp, P(colr, a))


# ----------------------------------------------------------------------------
# particles
# ----------------------------------------------------------------------------
def firework(c, x, y, age, seed=0, color="#ffd35a", n=70, speed=520, shape="burst", a=1.0, size=1.0):
    if age < 0 or age > 2.2:
        return
    rng = np.random.default_rng(seed)
    g = 260.0
    fade = clamp(1 - age / 2.1) ** 1.4
    if shape == "ankh":
        base = ankh_points(n)
        dirs = [(px * 1.6, py * 1.6) for px, py in base]
    elif shape == "eye":
        dirs = []
        for i in range(n):
            u = i / n * TAU
            dirs.append((math.cos(u) * 1.1, math.sin(u) * 0.45 * (1 if math.sin(u) < 0 else 0.8)))
    else:
        dirs = []
        for i in range(n):
            u = i / n * TAU + rng.uniform(-0.05, 0.05)
            m = rng.uniform(0.75, 1.0)
            dirs.append((math.cos(u) * m, math.sin(u) * m))
    dec = 1 - math.exp(-age * 2.8)
    for i, (dx, dy) in enumerate(dirs):
        px = x + dx * speed * size * dec / 2.8
        py = y + dy * speed * size * dec / 2.8 + 0.5 * g * age * age * 0.5
        dec2 = 1 - math.exp(-(age - 0.06) * 2.8) if age > 0.06 else 0
        qx = x + dx * speed * size * dec2 / 2.8
        qy = y + dy * speed * size * dec2 / 2.8 + 0.5 * g * max(0, age - 0.06) ** 2 * 0.5
        tw = 0.6 + 0.4 * math.sin(age * 40 + i)
        cc = color if i % 5 else "#ffffff"
        c.drawLine(qx, qy, px, py, P(cc, a * fade * tw, stroke=2.2 * size))
        c.drawCircle(px, py, 2.4 * size, P(cc, a * fade))
    if age < 0.35:
        glow(c, x, y, 160 * size, color, (1 - age / 0.35) * 0.8 * a)


def rocket(c, x0, y0, x1, y1, f, color="#ffd35a", a=1.0):
    f = clamp(f)
    e = ease_out(f, 2)
    x, y = lerp(x0, x1, e), lerp(y0, y1, e)
    c.drawLine(x, y, lerp(x0, x1, max(0, e - 0.12)), lerp(y0, y1, max(0, e - 0.12)), P(color, 0.6 * a, stroke=3))
    c.drawCircle(x, y, 4, P("white", a))


def confetti(c, t, seed=0, n=160, a=1.0, colors=("#f2c14e", "#2ec4b6", "#1f3f95", "#c0392b", "#fff8ec"), t0=0.0, wind=40):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-200, W + 200, n)
    ph = rng.uniform(0, TAU, n)
    sp = rng.uniform(90, 220, n)
    st = rng.uniform(0, 3, n)
    for i in range(n):
        age = t - t0 - st[i] * 0.3
        if age < 0:
            continue
        y = -30 + sp[i] * age
        if y > H + 40:
            y = (y + 40) % (H + 80) - 40
        x = xs[i] + math.sin(age * 2 + ph[i]) * 40 + wind * age
        x = (x + 200) % (W + 400) - 200
        c.save()
        c.translate(x, y)
        c.rotate(age * 200 + ph[i] * 57)
        sx = abs(math.cos(age * 6 + ph[i]))
        c.drawRect(skia.Rect.MakeLTRB(-7 * sx, -4, 7 * sx, 4), P(colors[i % len(colors)], a))
        c.restore()


def dust_puff(c, x, y, age, s=1.0, color="#e8c58a", a=1.0):
    if age < 0 or age > 0.8:
        return
    f = age / 0.8
    for k in range(4):
        dx = (k - 1.5) * 14 * s * (0.4 + f)
        r = s * (8 + 22 * ease_out(f)) * (0.7 + 0.1 * k)
        c.drawCircle(x + dx, y - r * 0.5 - f * 8 * s, r, P(color, a * 0.45 * (1 - f)))


def cloud(c, x, y, w, h, color="#ffffff", a=0.5, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(7):
        cx = x + (i / 6 - 0.5) * w * 0.8 + rng.uniform(-0.05, 0.05) * w
        r = h * (0.35 + 0.35 * math.sin(i / 6 * math.pi)) * rng.uniform(0.8, 1.2)
        c.drawCircle(cx, y - r * 0.3, r, P(color, a, blur=r * 0.25))


def grid_floor(c, horizon, t, speed=1.0, color="#ff2e88", a=1.0, spacing=120, n_lines=28, glow_c=None, x_shift=0.0):
    vx = W / 2 + x_shift
    c.save()
    c.clipRect(skia.Rect.MakeLTRB(0, horizon, W, H), skia.ClipOp.kIntersect, True)
    c.drawRect(skia.Rect.MakeLTRB(0, horizon, W, H), P(shader=lin_grad((0, horizon), (0, H), [(0, "#1a0630", a), (1, "#07010f", a)])))
    gp = P(glow_c or color, 0.35 * a, stroke=7, blur=5)
    lp = P(color, a, stroke=2)
    for i in range(-n_lines, n_lines + 1):
        xb = vx + i * spacing * 4.2
        c.drawLine(vx + i * 6, horizon, xb, H + 400, gp)
        c.drawLine(vx + i * 6, horizon, xb, H + 400, lp)
    off = (t * speed) % 1.0
    for j in range(22):
        z = (j + 1 - off)
        if z <= 0.05:
            continue
        yy = horizon + (H - horizon) * 1.25 / z * 0.55
        if yy > H + 10:
            continue
        al = clamp((yy - horizon) / 120)
        c.drawLine(0, yy, W, yy, P(glow_c or color, 0.35 * a * al, stroke=7, blur=5))
        c.drawLine(0, yy, W, yy, P(color, a * al, stroke=max(1.0, 2.5 * (yy - horizon) / (H - horizon))))
    c.restore()
    c.drawRect(skia.Rect.MakeLTRB(0, horizon - 60, W, horizon + 50),
               P(shader=lin_grad((0, horizon - 60), (0, horizon + 50), [(0, color, 0), (0.55, color, 0.35 * a), (1, color, 0)])))
