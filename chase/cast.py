"""Supporting cast (all original): the pink 'Cringe' blob, AMPY the amp-bot, the 200 trophy, crowds."""
import math

import numpy as np
import skia

from .core import (TAU, C, P, clamp, ease_out, font, hash1, lerp, lin_grad, mix, path_from, poly, rad_grad, scale_c)

BLOB, BLOB_HI, BLOB_SH, BLOB_INK = "#ff5fa8", "#ffc2e0", "#d8338a", "#5a0d3a"


# ----------------------------------------------------------------------------
# the blob
# ----------------------------------------------------------------------------
def blob_path(cx, cy, r, t, sx=1.0, sy=1.0, wob=1.0, seed=0, ground=None, n=44, spike=0.0):
    pts = []
    for i in range(n):
        a = i / n * TAU
        rr = r * (1 + wob * (0.055 * math.sin(3 * a + t * 3.1 + seed) + 0.04 * math.sin(5 * a - t * 4.3 + seed * 2)
                             + 0.022 * math.sin(8 * a + t * 6.2)) + spike * 0.25 * max(0, math.sin(7 * a + t * 2)) ** 8)
        x = cx + math.cos(a) * rr * sx
        y = cy + math.sin(a) * rr * sy
        if ground is not None and y > ground:
            y = ground + (y - ground) * 0.08
        pts.append((x, y))
    return path_from(pts, close=True, smooth_k=1.0)


def draw_blob(c, cx, cy, r, t, look=(1.0, 0.0), mood="panic", sx=1.0, sy=1.0, wob=1.0, seed=0, ground=None,
              arms_up=0.0, alpha=1.0, glow=0.0, lw=None, crack=0.0):
    lw = lw or max(1.5, r * 0.045)
    body = blob_path(cx, cy, r, t, sx, sy, wob, seed, ground)
    if glow > 0:
        c.drawPath(body, P("#ff7ac0", 0.5 * glow * alpha, blur=r * 0.25))
    # arms (pseudopods) raised to hold the trophy
    if arms_up > 0:
        for sd in (-1, 1):
            base = (cx + sd * r * 0.55 * sx, cy - r * 0.55 * sy)
            tip = (cx + sd * r * 0.32 * sx, cy - r * (0.95 + 0.55 * arms_up) * sy + math.sin(t * 7 + sd) * r * 0.05)
            arm = skia.Path()
            arm.moveTo(base[0] - sd * r * 0.2, base[1] + r * 0.1)
            arm.quadTo(base[0] + sd * r * 0.2, (base[1] + tip[1]) / 2, tip[0] + sd * r * 0.1, tip[1])
            arm.quadTo(tip[0] - sd * r * 0.12, tip[1] + r * 0.05, base[0] - sd * r * 0.05, base[1] + r * 0.25)
            arm.close()
            c.drawPath(arm, P(BLOB_INK, alpha, stroke=lw * 2))
            c.drawPath(arm, P(BLOB, alpha))
    c.drawPath(body, P(BLOB_INK, alpha, stroke=lw * 2))
    c.drawPath(body, P(shader=rad_grad((cx - r * 0.3 * sx, cy - r * 0.4 * sy), r * 1.5,
                                       [(0, BLOB_HI, alpha), (0.45, BLOB, alpha), (1, BLOB_SH, alpha)])))
    # gloss
    c.drawOval(skia.Rect.MakeLTRB(cx - r * 0.62 * sx, cy - r * 0.72 * sy, cx - r * 0.15 * sx, cy - r * 0.38 * sy),
               P("#ffffff", 0.55 * alpha))
    c.drawCircle(cx - r * 0.05 * sx, cy - r * 0.62 * sy, r * 0.06, P("#ffffff", 0.7 * alpha))
    if crack > 0:
        rng = np.random.default_rng(seed + 5)
        for k in range(9):
            a = rng.uniform(0, TAU)
            p = skia.Path()
            p.moveTo(cx, cy)
            x, y = cx, cy
            for j in range(5):
                a += rng.uniform(-0.6, 0.6)
                L = r * 0.22 * crack
                x, y = x + math.cos(a) * L * sx, y + math.sin(a) * L * sy
                p.lineTo(x, y)
            c.drawPath(p, P("#fff7c0", alpha * crack, stroke=lw * 1.6))
            c.drawPath(p, P("#fff7c0", 0.5 * alpha * crack, stroke=lw * 5, blur=lw * 2))
    _blob_face(c, cx, cy, r, t, look, mood, sx, sy, alpha, lw)


def _blob_face(c, cx, cy, r, t, look, mood, sx, sy, a, lw):
    ex = r * 0.3 * sx
    ey = cy - r * 0.12 * sy
    er = r * 0.2
    lx, ly = look
    for sd in (-1, 1):
        x = cx + sd * ex + lx * r * 0.08
        c.drawCircle(x, ey, er, P("#ffffff", a))
        c.drawCircle(x, ey, er, P(BLOB_INK, a, stroke=lw))
        if mood == "dizzy":
            sp = skia.Path()
            for k in range(30):
                ang = k * 0.5 + t * 8 * sd
                rr = er * 0.85 * k / 30
                (sp.moveTo if k == 0 else sp.lineTo)(x + math.cos(ang) * rr, ey + math.sin(ang) * rr)
            c.drawPath(sp, P(BLOB_INK, a, stroke=lw * 0.8))
        elif mood == "happy":
            arc = skia.Path()
            arc.moveTo(x - er * 0.6, ey + er * 0.2)
            arc.quadTo(x, ey - er * 0.7, x + er * 0.6, ey + er * 0.2)
            c.drawCircle(x, ey, er * 1.02, P("#ff8ac4", a))
            c.drawPath(arc, P(BLOB_INK, a, stroke=lw * 1.2))
        else:
            pr = er * (0.35 if mood in ("panic", "shock") else 0.5)
            px = x + lx * er * 0.45
            py = ey + ly * er * 0.45 + (math.sin(t * 20 + sd) * er * 0.08 if mood == "panic" else 0)
            c.drawCircle(px, py, pr, P("#1a0612", a))
            c.drawCircle(px + pr * 0.35, py - pr * 0.35, pr * 0.3, P("#ffffff", a))
    # brows
    for sd in (-1, 1):
        x = cx + sd * ex
        b = skia.Path()
        if mood in ("smug", "angry"):
            b.moveTo(x - sd * er * 0.9, ey - er * 1.1)
            b.lineTo(x + sd * er * 0.9, ey - er * 1.45)
        else:
            b.moveTo(x - sd * er * 0.9, ey - er * 1.5)
            b.lineTo(x + sd * er * 0.8, ey - er * 1.2)
        c.drawPath(b, P(BLOB_INK, a, stroke=lw * 1.1))
    # blush lines
    for sd in (-1, 1):
        bx = cx + sd * r * 0.52 * sx
        for k in range(3):
            c.drawLine(bx - r * 0.07 + k * r * 0.055, ey + er * 1.3, bx - r * 0.1 + k * r * 0.055, ey + er * 1.65,
                       P("#c0145a", 0.8 * a, stroke=lw * 0.6))
    # mouth
    my = cy + r * 0.28 * sy
    m = skia.Path()
    if mood in ("panic", "shock"):
        m.addOval(skia.Rect.MakeLTRB(cx - r * 0.1, my - r * 0.06, cx + r * 0.1, my + r * 0.12))
        c.drawPath(m, P("#5a0d3a", a))
    elif mood == "smug":
        m.moveTo(cx - r * 0.18, my)
        m.quadTo(cx, my + r * 0.12, cx + r * 0.2, my - r * 0.06)
        c.drawPath(m, P(BLOB_INK, a, stroke=lw))
    elif mood == "happy":
        m.moveTo(cx - r * 0.18, my - r * 0.03)
        m.quadTo(cx, my + r * 0.2, cx + r * 0.18, my - r * 0.03)
        m.close()
        c.drawPath(m, P("#5a0d3a", a))
    else:
        m.moveTo(cx - r * 0.2, my)
        for k in range(1, 7):
            m.lineTo(cx - r * 0.2 + k * r * 0.4 / 6, my + (r * 0.04 if k % 2 else -r * 0.02))
        c.drawPath(m, P(BLOB_INK, a, stroke=lw * 0.9))
    # sweat drop
    if mood in ("panic", "normal", "shock"):
        sx0 = cx + r * 0.72 * sx
        sy0 = cy - r * 0.55 * sy + ((t * 1.2) % 1.0) * r * 0.2
        d = skia.Path()
        d.moveTo(sx0, sy0 - r * 0.14)
        d.quadTo(sx0 + r * 0.1, sy0 + r * 0.02, sx0, sy0 + r * 0.06)
        d.quadTo(sx0 - r * 0.1, sy0 + r * 0.02, sx0, sy0 - r * 0.14)
        c.drawPath(d, P("#bfeaff", 0.9 * a))
        c.drawPath(d, P("#3a6a9a", 0.9 * a, stroke=lw * 0.5))


# ----------------------------------------------------------------------------
# trophy
# ----------------------------------------------------------------------------
def draw_trophy(c, x, y, s, t=0.0, a=1.0, shine=0.0, spin=0.0):
    """Trophy with its base centred at (x, y); s ~ total height."""
    c.save()
    c.translate(x, y)
    c.scale(s * (1 if spin == 0 else math.cos(spin)), s)
    ink = P("#3a2206", a, stroke=0.025)
    gold = lin_grad((-0.3, 0), (0.3, 0), [(0, "#b8741a", a), (0.35, "#ffe27a", a), (0.55, "#ffcf3a", a), (1, "#a8620e", a)])
    base = poly([(-0.26, 0.0), (0.26, 0.0), (0.22, -0.14), (-0.22, -0.14)])
    c.drawPath(base, ink)
    c.drawPath(base, P("#3b2415", a))
    c.drawRect(skia.Rect.MakeLTRB(-0.16, -0.11, 0.16, -0.035), P("#ffd96a", a))
    stem = poly([(-0.06, -0.14), (0.06, -0.14), (0.035, -0.32), (-0.035, -0.32)])
    c.drawPath(stem, ink)
    c.drawPath(stem, P(shader=gold))
    for sd in (-1, 1):
        h = skia.Path()
        h.addOval(skia.Rect.MakeLTRB(sd * 0.28 - 0.1, -0.78, sd * 0.28 + 0.1, -0.52))
        c.drawPath(h, P("#3a2206", a, stroke=0.075))
        c.drawPath(h, P("#ffcf3a", a, stroke=0.04))
    cup = skia.Path()
    cup.moveTo(-0.3, -0.86)
    cup.lineTo(0.3, -0.86)
    cup.cubicTo(0.3, -0.5, 0.18, -0.34, 0.0, -0.32)
    cup.cubicTo(-0.18, -0.34, -0.3, -0.5, -0.3, -0.86)
    cup.close()
    c.drawPath(cup, ink)
    c.drawPath(cup, P(shader=gold))
    c.drawOval(skia.Rect.MakeLTRB(-0.3, -0.9, 0.3, -0.82), P("#fff0a8", a))
    c.drawOval(skia.Rect.MakeLTRB(-0.3, -0.9, 0.3, -0.82), P("#3a2206", a, stroke=0.015))
    c.save()
    c.scale(0.01, 0.01)
    f = font("Bangers.ttf", 24)
    w = f.measureText("200")
    c.drawString("200", -w / 2, -52, f, P("#8a4e0a", a))
    c.restore()
    c.drawRect(skia.Rect.MakeLTRB(-0.2, -0.82, -0.14, -0.4), P("#ffffff", 0.45 * a))
    c.restore()
    if shine > 0:
        for k in range(4):
            ang = t * 2 + k * 1.7
            sx_ = x + math.cos(ang) * s * 0.4
            sy_ = y - s * 0.6 + math.sin(ang * 1.3) * s * 0.3
            sparkle(c, sx_, sy_, s * 0.08 * (0.6 + 0.4 * math.sin(t * 6 + k)), "#fff6c0", a * shine)


def sparkle(c, x, y, r, color="#ffffff", a=1.0):
    p = skia.Path()
    p.moveTo(x, y - r)
    p.quadTo(x, y, x + r, y)
    p.quadTo(x, y, x, y + r)
    p.quadTo(x, y, x - r, y)
    p.quadTo(x, y, x, y - r)
    c.drawPath(p, P(color, a))


# ----------------------------------------------------------------------------
# AMPY
# ----------------------------------------------------------------------------
def draw_ampy(c, x, y, s, t=0.0, bass=0.0, mood="happy", hop=0.0, a=1.0, lean=0.0):
    """Amp-bot standing with feet at (x, y); s ~ total height."""
    c.save()
    c.translate(x, y - hop * s * 0.3)
    c.rotate(lean)
    c.scale(s, s)
    ink = P("#120c16", a, stroke=0.035)
    for sd in (-1, 1):
        lg = skia.Path()
        lg.moveTo(sd * 0.2, -0.22)
        lg.lineTo(sd * 0.22 + math.sin(t * 8 + sd) * 0.02 * hop, -0.02)
        c.drawPath(lg, P("#120c16", a, stroke=0.07))
        c.drawRoundRect(skia.Rect.MakeLTRB(sd * 0.22 - 0.1, -0.05, sd * 0.22 + 0.1, 0.02), 0.04, 0.04, P("#ec3441", a))
    body = skia.Path()
    body.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-0.42, -0.95, 0.42, -0.2), 0.08, 0.08))
    c.drawPath(body, ink)
    c.drawPath(body, P(shader=lin_grad((0, -0.95), (0, -0.2), [(0, "#3b3646", a), (1, "#1e1a26", a)])))
    c.drawRect(skia.Rect.MakeLTRB(-0.42, -0.95, 0.42, -0.8), P("#f2c14e", a))
    for k in range(5):
        c.drawCircle(-0.3 + k * 0.15, -0.875, 0.035, P("#1a1520", a))
        c.drawLine(-0.3 + k * 0.15, -0.875, -0.3 + k * 0.15 + 0.025 * math.cos(t * 3 + k), -0.9, P("#ffffff", a, stroke=0.012))
    # speaker
    sc = (0.0, -0.5)
    rr = 0.24 * (1 + 0.08 * bass)
    c.drawCircle(sc[0], sc[1], 0.27, P("#0f0c14", a))
    c.drawCircle(sc[0], sc[1], rr, P(shader=rad_grad(sc, rr, [(0, "#5a5470", a), (0.6, "#2a2636", a), (1, "#141018", a)])))
    c.drawCircle(sc[0], sc[1], rr * 0.35, P("#6a6488", a))
    for k in range(3):
        c.drawCircle(sc[0], sc[1], rr * (0.5 + k * 0.17), P("#0a0810", 0.6 * a, stroke=0.01))
    # LED eyes on the top panel strip
    for sd in (-1, 1):
        ex, ey = sd * 0.16, -0.755
        if mood == "happy":
            e = skia.Path()
            e.moveTo(ex - 0.06, ey + 0.02)
            e.quadTo(ex, ey - 0.05, ex + 0.06, ey + 0.02)
            c.drawPath(e, P("#6afcff", a, stroke=0.03))
        elif mood == "shock":
            c.drawCircle(ex, ey, 0.04, P("#6afcff", a))
        else:
            c.drawRoundRect(skia.Rect.MakeLTRB(ex - 0.06, ey - 0.02, ex + 0.06, ey + 0.02), 0.01, 0.01, P("#6afcff", a))
        c.drawCircle(ex, ey, 0.09, P("#6afcff", 0.25 * a, blur=0.04))
    # antenna
    c.drawLine(0.25, -0.95, 0.32, -1.18, P("#120c16", a, stroke=0.03))
    c.drawCircle(0.32, -1.2, 0.05, P("#ff3d6a", a))
    c.drawCircle(0.32, -1.2, 0.12, P("#ff3d6a", 0.35 * a * (0.5 + 0.5 * math.sin(t * 6)), blur=0.05))
    c.restore()


# ----------------------------------------------------------------------------
# crowd
# ----------------------------------------------------------------------------
def crowd(c, x0, x1, y, t, b, n=30, size=60, color="#0c0716", lights=True, seed=0, a=1.0, cheer=1.0):
    rng = np.random.default_rng(seed)
    xs = np.sort(rng.uniform(x0, x1, n))
    for i, x in enumerate(xs):
        s = size * rng.uniform(0.8, 1.2)
        ph = rng.uniform(0, 1)
        bob = abs(math.sin((b + ph * 0.5) * math.pi)) * s * 0.12 * cheer
        yy = y - bob
        c.drawCircle(x, yy - s * 0.95, s * 0.28, P(color, a))
        c.drawPath(path_from([(x - s * 0.45, yy + 5), (x - s * 0.38, yy - s * 0.55), (x, yy - s * 0.7), (x + s * 0.38, yy - s * 0.55),
                              (x + s * 0.45, yy + 5)], close=True, smooth_k=0.5), P(color, a))
        if i % 3 == 0:
            arm_up = cheer * (0.5 + 0.5 * math.sin((b + ph) * math.pi))
            hx, hy = x + s * 0.3, yy - s * (0.6 + 0.8 * arm_up)
            c.drawLine(x + s * 0.25, yy - s * 0.5, hx, hy, P(color, a, stroke=s * 0.14))
            if lights:
                c.drawRect(skia.Rect.MakeLTRB(hx - s * 0.06, hy - s * 0.18, hx + s * 0.06, hy), P("#e8f4ff", a))
                c.drawCircle(hx, hy - s * 0.1, s * 0.3, P("#bfe6ff", 0.35 * a, blur=s * 0.12))
