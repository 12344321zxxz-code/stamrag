"""Scenery built from paper: skies, torn clouds and hills, water strips, plants, props and the rainbow.

All functions draw in world units onto a canvas; static ones are baked into Layers by the scenes.
"""
import math

import numpy as np
import skia

from .craft import (RAINBOW, TAU, brad, clamp, col, cpath, cut, ellipse, from_path, hash1, mixc, paper_paint, path_of,
                    piece, shade, spline, stroke, texture_overlay, torn_piece, union)
from .cast import cheeks, eye, mouth


def rnd(seed, k=0):
    return hash1(int(seed * 7919 + k * 104729))


# ----------------------------------------------------------------------------------------------------
# sky and clouds
# ----------------------------------------------------------------------------------------------------
def sky(c, l, t, r, b, top, bottom, seed=1):
    rect = skia.Rect.MakeLTRB(l, t, r, b)
    g = skia.GradientShader.MakeLinear([skia.Point(0, t), skia.Point(0, b)], [col(top), col(bottom)], [0.0, 1.0])
    c.drawRect(rect, skia.Paint(Shader=g))
    texture_overlay(c, rect, seed)


def cloud_pts(x, y, w, h, seed=0, lobes=5):
    circles = []
    for i in range(lobes):
        u = (i + 0.5) / lobes
        cx = x - w / 2 + w * u
        r = h * (0.42 + 0.3 * math.sin(u * math.pi)) * (0.85 + 0.3 * rnd(seed, i))
        cy = y + h * 0.18 - r * 0.55
        circles.append(path_of(ellipse(cx, cy, r * 1.05, r, 36)))
    circles.append(path_of(ellipse(x, y + h * 0.12, w * 0.5, h * 0.28, 48)))
    return from_path(union(circles), 5)


def cloud(c, x, y, w, h, color="cloud", seed=0, lift=1.5, alpha=1.0):
    pts = cloud_pts(x, y, w, h, seed)
    return torn_piece(c, pts, color, seed, lift=lift, amp=2.4, alpha=alpha)


# ----------------------------------------------------------------------------------------------------
# land
# ----------------------------------------------------------------------------------------------------
def band(x0, x1, y, amp, period, seed, bottom=None, step=18, phase=0.0, harm=0.35):
    xs = np.arange(x0, x1 + step, step, dtype=float)
    ph = rnd(seed, 1) * TAU + phase
    top = y - amp * np.sin(xs / period * TAU + ph) - amp * harm * np.sin(xs / period * TAU * 2.3 + ph * 1.7)
    bottom = y + 2000 if bottom is None else bottom
    return np.vstack([np.stack([xs, top], 1), [[x1, bottom], [x0, bottom]]])


def hills(c, x0, x1, y, amp, period, color, seed, lift=1.6, rim=3.5, amp_t=3.0):
    return torn_piece(c, band(x0, x1, y, amp, period, seed), color, seed, lift=lift, rim_w=rim, amp=amp_t)


# ----------------------------------------------------------------------------------------------------
# water
# ----------------------------------------------------------------------------------------------------
def wave_band(x0, x1, y, amp, period, seed, depth=900, scallop=False):
    step = 12
    xs = np.arange(x0, x1 + step, step, dtype=float)
    ph = rnd(seed, 3) * TAU
    if scallop:
        u = (xs / period + ph) % 1.0
        top = y - amp * (1 - (2 * u - 1) ** 2) * 1.2 + amp * 0.3
    else:
        top = y - amp * np.sin(xs / period * TAU + ph) - amp * 0.3 * np.sin(xs / period * TAU * 2.1 + ph * 2)
    return np.vstack([np.stack([xs, top], 1), [[x1, y + depth], [x0, y + depth]]])


def water_strip(c, x0, x1, y, amp, period, color, seed, lift=1.2, depth=900, scallop=False, rim=3.0,
                shine=True):
    pts = wave_band(x0, x1, y, amp, period, seed, depth, scallop)
    p = torn_piece(c, pts, color, seed, lift=lift, rim_w=rim, amp=2.0, grad=0.0)
    if shine:
        # a few pale cut-paper glints riding the strip
        for k in range(int((x1 - x0) / 260)):
            gx = x0 + 120 + k * 260 + 80 * rnd(seed, k)
            gy = y + 26 + 30 * rnd(seed, k + 50)
            gl = cut([(gx - 34, gy), (gx + 34, gy - 3), (gx + 30, gy + 4), (gx - 30, gy + 6)], seed + k, 0.6)
            piece(c, gl, mixc(color, "white", 0.35), seed + k + 7, lift=0.3, alpha=0.8, rim=0, shadow=0.1,
                  grad=0)
    return p


# ----------------------------------------------------------------------------------------------------
# plants
# ----------------------------------------------------------------------------------------------------
def leaf_pts(L, W_, bend=0.0):
    pts = []
    n = 16
    for i in range(n + 1):
        u = i / n
        w = W_ * math.sin(math.pi * u) ** 0.9 * (1 - 0.15 * u)
        pts.append((L * u, -w + bend * L * u * u))
    for i in range(n - 1, 0, -1):
        u = i / n
        w = W_ * math.sin(math.pi * u) ** 0.9 * (1 - 0.15 * u)
        pts.append((L * u, w + bend * L * u * u))
    return pts


def leaf(c, x, y, L, W_, rot, color="leaf", seed=0, lift=1.4, veins=True, bend=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    p = cpath(leaf_pts(L, W_, bend), seed, 0.8)
    piece(c, p, color, seed, lift=lift, grad=0.12)
    if veins:
        vc = mixc(color, "white", 0.35)
        stroke(c, [(L * u, bend * L * u * u) for u in np.linspace(0.02, 0.92, 8)], max(1.5, W_ * 0.07), vc, 0.8)
        for k in range(1, 5):
            u = k / 5.5
            bx, by = L * u, bend * L * u * u
            for sgn in (-1, 1):
                stroke(c, [(bx, by), (bx + L * 0.12, by + sgn * W_ * 0.55 * math.sin(math.pi * u))],
                       max(1.0, W_ * 0.04), vc, 0.7)
    c.restore()


def grass_tuft(c, x, y, s, color="grass2", seed=0, blades=7, lift=1.2, sway=0.0):
    for i in range(blades):
        u = (i + 0.5) / blades - 0.5
        h = s * (0.7 + 0.5 * rnd(seed, i)) * (1 - abs(u) * 0.8)
        lean = u * 60 + (rnd(seed, i + 9) - 0.5) * 20 + sway
        wdt = s * 0.09
        base = x + u * s * 0.5
        a = math.radians(lean - 90)
        tip = (base + math.cos(a) * h, y + math.sin(a) * h)
        mid = (base + math.cos(a) * h * 0.5 + wdt * 0.5, y + math.sin(a) * h * 0.5)
        pts = [(base - wdt, y + 2), (mid[0] - wdt * 0.6, mid[1]), tip, (mid[0] + wdt * 0.6, mid[1]), (base + wdt, y + 2)]
        cc = shade(color, 0.08 * (rnd(seed, i + 3) - 0.5) * 2)
        piece(c, cut(pts, seed + i, 0.5), cc, seed + i, lift=lift, grad=0.1)


def flower(c, x, y, s, petal="pink", seed=0, stem=True, petals=6, center="yellow", lift=1.3, rot=0.0):
    if stem:
        piece(c, cut([(x - 3 * s / 40, y), (x - 2, y + 160 * s / 40), (x + 2, y + 160 * s / 40), (x + 3 * s / 40, y)],
                     seed, 0.4), "leaf2", seed, lift=0.8, grad=0)
        leaf(c, x, y + 90 * s / 40, 44 * s / 40, 12 * s / 40, -30, "leaf", seed + 3, lift=0.8, veins=False)
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    for k in range(petals):
        a = 360 * k / petals
        c.save()
        c.rotate(a)
        piece(c, cpath(ellipse(s * 0.55, 0, s * 0.5, s * 0.3, 24), seed + k, 0.6), petal, seed + k, lift=lift,
              grad=0.12)
        c.restore()
    piece(c, cpath(ellipse(0, 0, s * 0.32, s * 0.32, 28), seed + 20, 0.5), center, seed + 20, lift=lift + 0.3)
    brad(c, 0, 0, s * 0.14)
    c.restore()


def tree(c, x, y, s, seed=0, canopy="leaf", trunk="trunk", round_=True, fruit=None):
    tw = 26 * s
    piece(c, cut([(x - tw, y), (x - tw * 0.6, y - 230 * s), (x + tw * 0.6, y - 230 * s), (x + tw, y)], seed, 0.8),
          trunk, seed, lift=1.4, grad=0.12)
    blobs = [(-90, -250, 95), (0, -320, 115), (90, -250, 95), (-40, -210, 80), (50, -200, 80)]
    pts = from_path(union([path_of(ellipse(x + dx * s, y + dy * s, r * s, r * s * 0.95, 40)) for dx, dy, r in blobs]),
                    6)
    torn_piece(c, pts, canopy, seed + 1, lift=2.0, amp=3.0)
    for k, (dx, dy, r) in enumerate([(-50, -290, 46), (46, -250, 40), (-10, -220, 34)]):
        pts = cloud_pts(x + dx * s, y + dy * s, r * 2.2 * s, r * 1.1 * s, seed + 5 + k, 3)
        torn_piece(c, pts, shade(canopy, -0.08 - 0.03 * k), seed + 5 + k, lift=0.8, amp=2.0, rim_w=2.5)
    if fruit:
        for k in range(6):
            fx = x + (rnd(seed, k) - 0.5) * 220 * s
            fy = y - 200 * s - rnd(seed, k + 10) * 150 * s
            piece(c, cpath(ellipse(fx, fy, 13 * s, 13 * s, 20), seed + 30 + k, 0.5), fruit, seed + 30 + k, lift=1.0)


def bush(c, x, y, w, h, color="leaf", seed=0, lift=1.6):
    pts = cloud_pts(x, y - h * 0.4, w, h, seed, 4)
    torn_piece(c, pts, color, seed, lift=lift, amp=2.6)


def fence(c, x0, x1, y, h=150, seed=0, color="#f6efdf"):
    rail = cut([(x0, y - h * 0.62), (x1, y - h * 0.62), (x1, y - h * 0.5), (x0, y - h * 0.5)], seed, 0.6)
    piece(c, rail, shade(color, -0.05), seed, lift=1.0)
    x = x0 + 10
    k = 0
    while x < x1:
        hh = h * (0.95 + 0.1 * rnd(seed, k))
        pts = [(x, y), (x, y - hh + 14), (x + 17, y - hh), (x + 34, y - hh + 14), (x + 34, y)]
        piece(c, cut(pts, seed + k, 0.7), color, seed + k, lift=1.3, grad=0.1)
        x += 58
        k += 1


def reeds(c, x, y, s, seed=0, n=5, cattails=True, color="#6f9e3f", lift=1.2, sway=0.0):
    for i in range(n):
        u = (i + 0.5) / n - 0.5
        h = s * (0.75 + 0.45 * rnd(seed, i))
        lean = u * 18 + sway * (0.6 + rnd(seed, i + 4))
        a = math.radians(lean - 90)
        bx = x + u * s * 0.35
        tip = (bx + math.cos(a) * h, y + math.sin(a) * h)
        w = s * 0.03
        pts = [(bx - w, y + 2), (tip[0] - w * 0.3, tip[1] + 10), tip, (tip[0] + w * 0.3, tip[1] + 10), (bx + w, y + 2)]
        piece(c, cut(pts, seed + i, 0.4), shade(color, 0.1 * (rnd(seed, i + 2) - 0.5)), seed + i, lift=lift,
              grad=0.08)
        if cattails and i % 2 == 0:
            cx, cy = bx + math.cos(a) * h * 0.78, y + math.sin(a) * h * 0.78
            c.save()
            c.translate(cx, cy)
            c.rotate(lean)
            piece(c, cpath(ellipse(0, 0, s * 0.035, s * 0.11, 24), seed + 40 + i, 0.5), "#8a5a3b", seed + 40 + i,
                  lift=lift + 0.3, grad=0.15)
            c.restore()


def lily_pad(c, x, y, rx, ry, seed=0, color="#5bb84a", lift=1.0, rot=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    pts = []
    for a in np.linspace(0.18, TAU - 0.18, 60):
        pts.append((rx * math.cos(a), ry * math.sin(a)))
    pts.append((0, 0))
    p = cpath(pts, seed, 0.8)
    piece(c, p, color, seed, lift=lift, grad=0.12)
    vp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1.8, Color=col(mixc(color, "white", 0.3), 0.8))
    for a in np.linspace(0.6, TAU - 0.6, 7):
        c.drawLine(0, 0, rx * 0.85 * math.cos(a), ry * 0.85 * math.sin(a), vp)
    c.restore()


def water_lily(c, x, y, s, seed=0, color="lily"):
    for k, a in enumerate((-60, -30, 0, 30, 60)):
        c.save()
        c.translate(x, y)
        c.rotate(a)
        piece(c, cpath([(0, 0), (-s * 0.22, -s * 0.5), (0, -s * 0.85), (s * 0.22, -s * 0.5)], seed + k, 0.5),
              shade(color, 0.05 * (k % 2)), seed + k, lift=1.0, grad=0.15)
        c.restore()
    piece(c, cpath(ellipse(x, y - s * 0.12, s * 0.16, s * 0.1, 18), seed + 9, 0.4), "yellow", seed + 9, lift=1.2)


def rock(c, x, y, rx, ry, color="rock", seed=0, lift=1.4):
    pts = [(x + rx * math.cos(a) * (0.85 + 0.25 * rnd(seed, k)), y + ry * math.sin(a) * (0.85 + 0.2 * rnd(seed, k + 5)))
           for k, a in enumerate(np.linspace(math.pi, TAU, 9))] + [(x + rx, y + ry * 0.2), (x - rx, y + ry * 0.2)]
    piece(c, cpath(pts, seed, 1.0, soft=True), color, seed, lift=lift, grad=0.16)


def willow(c, x, y, s, seed=0):
    piece(c, cut([(x - 30 * s, y), (x - 18 * s, y - 300 * s), (x + 18 * s, y - 300 * s), (x + 30 * s, y)], seed, 1.0),
          "bark", seed, lift=1.4, grad=0.12)
    dome = [(x - 230 * s, y - 300 * s), (x - 200 * s, y - 420 * s), (x - 100 * s, y - 490 * s), (x, y - 510 * s),
            (x + 100 * s, y - 490 * s), (x + 200 * s, y - 420 * s), (x + 230 * s, y - 300 * s)]
    torn_piece(c, spline(dome, 10), "#7fae5a", seed + 1, lift=1.8, amp=2.6)
    for k in range(16):
        u = k / 15 - 0.5
        bx = x + u * 420 * s
        L = (180 + 120 * rnd(seed, k)) * s * (1 - abs(u))
        pts = [(bx - 14 * s, y - 330 * s), (bx + 14 * s, y - 330 * s), (bx + 8 * s, y - 330 * s + L),
               (bx, y - 320 * s + L + 20 * s), (bx - 8 * s, y - 330 * s + L)]
        piece(c, cut(pts, seed + 10 + k, 0.6), shade("#8cbc62", 0.08 * (k % 3 - 1)), seed + 10 + k, lift=1.0,
              grad=0.08)


def palm(c, x, y, s, seed=0, sway=0.0):
    pts = []
    for i in range(9):
        u = i / 8
        pts.append((x + 40 * s * math.sin(u * 1.8) + sway * u, y - 260 * s * u))
    trunk = [(px - 11 * s * (1 - 0.4 * i / 8), py) for i, (px, py) in enumerate(pts)] + \
            [(px + 11 * s * (1 - 0.4 * i / 8), py) for i, (px, py) in enumerate(pts)][::-1]
    piece(c, cut(trunk, seed, 0.6), "#b07a4a", seed, lift=1.2, grad=0.12)
    tx, ty = pts[-1]
    for k, a in enumerate((-160, -120, -60, -20, 30, 150)):
        leaf(c, tx, ty, 150 * s, 30 * s, a, "#4f9a3f", seed + 5 + k, lift=1.4, veins=True, bend=0.35 * (1 if a > -90 else -1))


def island(c, x, y, s, seed=0):
    mound = [(x - 260 * s, y + 10), (x - 180 * s, y - 40 * s), (x, y - 70 * s), (x + 200 * s, y - 36 * s),
             (x + 280 * s, y + 10)]
    torn_piece(c, spline(mound + [(x, y + 60 * s)], 8), "sand", seed, lift=1.2, amp=2.0)
    palm(c, x - 40 * s, y - 50 * s, s, seed + 3)


def lighthouse(c, x, y, s, seed=0, beam=0.0):
    body = [(x - 48 * s, y), (x - 32 * s, y - 280 * s), (x + 32 * s, y - 280 * s), (x + 48 * s, y)]
    piece(c, cut(body, seed, 0.6), "white", seed, lift=1.4, grad=0.14)
    c.save()
    c.clipPath(path_of(body), doAntiAlias=True)
    for k in range(3):
        yy = y - 40 * s - k * 85 * s
        c.drawRect(skia.Rect.MakeLTRB(x - 60 * s, yy - 38 * s, x + 60 * s, yy), paper_paint("red", seed + 3 + k))
    c.restore()
    piece(c, cut([(x - 40 * s, y - 280 * s), (x + 40 * s, y - 280 * s), (x + 40 * s, y - 296 * s),
                  (x - 40 * s, y - 296 * s)], seed + 7, 0.5), "#3d3a38", seed + 7, lift=1.0)
    piece(c, cut([(x - 26 * s, y - 296 * s), (x + 26 * s, y - 296 * s), (x + 26 * s, y - 340 * s),
                  (x - 26 * s, y - 340 * s)], seed + 8, 0.5), "#ffe89a", seed + 8, lift=0.8)
    piece(c, cut([(x - 34 * s, y - 340 * s), (x, y - 378 * s), (x + 34 * s, y - 340 * s)], seed + 9, 0.5), "red",
          seed + 9, lift=1.0)


def gull(c, x, y, s, t, seed=0):
    f = math.sin(t * 7 + seed) * 0.5
    stroke(c, [(x - 30 * s, y + (8 - 14 * f) * s), (x - 14 * s, y - 6 * s), (x, y + 3 * s),
               (x + 14 * s, y - 6 * s), (x + 30 * s, y + (8 - 14 * f) * s)], 4.5 * s, "#f7f4ee")
    stroke(c, [(x - 30 * s, y + (8 - 14 * f) * s), (x - 14 * s, y - 6 * s), (x, y + 3 * s),
               (x + 14 * s, y - 6 * s), (x + 30 * s, y + (8 - 14 * f) * s)], 1.2 * s, "#8a8580", 0.6)


def dragonfly(c, x, y, s, t, seed=0, rot=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    flap = math.sin(t * 60 + seed)
    for side in (-1, 1):
        for k, (dx, L) in enumerate(((-4, 46), (10, 40))):
            c.save()
            c.translate(dx, 0)
            c.rotate(side * (70 + 18 * flap) + (k - 0.5) * 20 * side)
            wp = path_of(ellipse(L / 2, 0, L / 2, 8, 20))
            c.drawPath(wp, skia.Paint(AntiAlias=True, Color=col("#e8f4fb", 0.7)))
            c.drawPath(wp, skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=1,
                                      Color=col("#8fb0c4", 0.8)))
            c.restore()
    piece(c, cpath([(-60, -3), (0, -5), (14, -7), (22, 0), (14, 7), (0, 5), (-60, 3)], seed, 0.4), "#3fb8c9", seed,
          lift=1.0, grad=0.2)
    eye(c, 18, -2, 5, (0.3, 0), 0, "happy", seed + 1, lift=0.2)
    c.restore()


# ----------------------------------------------------------------------------------------------------
# sun, drops, rainbow, tracker
# ----------------------------------------------------------------------------------------------------
def sun(c, x, y, r, t, seed=0, face=True, mouth_open=0.0, spin=1.0):
    c.save()
    c.translate(x, y)
    c.rotate(t * 12 * spin)
    for k in range(12):
        c.save()
        c.rotate(30 * k)
        L = r * (0.55 if k % 2 else 0.75)
        piece(c, cpath([(r * 0.8, -r * 0.16), (r * 0.8 + L, 0), (r * 0.8, r * 0.16)], seed + k, 0.5),
              "#ffb627" if k % 2 else "#ffcf3d", seed + k, lift=1.0, grad=0.1)
        c.restore()
    c.restore()
    piece(c, cpath(ellipse(x, y, r, r, 60), seed + 20, 0.8), "#ffd23f", seed + 20, lift=1.6, grad=0.16)
    if face:
        eye(c, x - r * 0.3, y - r * 0.1, r * 0.13, (0, 0), 0, "joy", seed + 21)
        eye(c, x + r * 0.3, y - r * 0.1, r * 0.13, (0, 0), 0, "joy", seed + 22)
        cheeks(c, x, y + r * 0.18, r * 0.45, r * 0.14, 0.7)
        mouth(c, x, y + r * 0.25, r * 0.4, mouth_open)


def raindrop(c, x, y, s, seed=0, color="#8fc6ea", lift=1.2):
    pts = [(0, -30), (12, -8), (15, 6), (8, 16), (0, 18), (-8, 16), (-15, 6), (-12, -8)]
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    piece(c, cpath(pts, seed, 0.5, soft=True), color, seed, lift=lift, grad=0.2)
    c.drawCircle(-5, 2, 3.2, skia.Paint(AntiAlias=True, Color=col("white", 0.75)))
    c.restore()


def arc_pts(cx, cy, r0, r1, a0=180.0, a1=360.0, n=90):
    outer = [(cx + r1 * math.cos(math.radians(a)), cy + r1 * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]
    inner = [(cx + r0 * math.cos(math.radians(a)), cy + r0 * math.sin(math.radians(a))) for a in np.linspace(a1, a0, n)]
    return outer + inner


def rainbow_band(c, cx, cy, r, w, color, seed, sweep=1.0, lift=1.6, alpha=1.0):
    if sweep <= 0.001:
        return
    a1 = 180 + 180 * clamp(sweep)
    pts = arc_pts(cx, cy, r - w / 2, r + w / 2, 180, a1, max(12, int(90 * sweep)))
    torn_piece(c, np.array(pts), color, seed, lift=lift, rim_w=3.0, amp=2.2, alpha=alpha)


def tracker(c, x, y, s, fills, pops, t, alpha=1.0):
    """The HUD: a small paper badge with six rainbow arcs, filled as colours are collected.
    fills[i] 0..1 (fill amount), pops[i] >= 0 seconds since filled (for a bounce)."""
    if alpha <= 0.01:
        return
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    badge = cpath([(-130, 40), (-128, -40), (-110, -86), (-60, -112), (0, -120), (60, -112), (110, -86), (128, -40),
                   (130, 40)], 901, 1.0, soft=True)
    piece(c, badge, "cream", 901, lift=2.0 * alpha, alpha=alpha, grad=0.08)
    for i in range(6):
        r = 100 - i * 14
        pts = arc_pts(0, 28, r - 6, r + 6, 180, 360, 50)
        path = path_of(pts)
        f = fills[i]
        if f < 0.99:
            dash = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.4,
                              Color=col(mixc(RAINBOW[i], "#a89c86", 0.55), 0.95 * alpha),
                              PathEffect=skia.DashPathEffect.Make([7.0, 5.0], 0.0))
            c.drawPath(path, dash)
        if f > 0.01:
            bounce = 1.0 + 0.25 * math.exp(-pops[i] * 8) * math.sin(pops[i] * 30) if pops[i] >= 0 else 1.0
            c.save()
            c.translate(0, 28)
            c.scale(bounce, bounce)
            c.translate(0, -28)
            piece(c, path_of(arc_pts(0, 28, r - 6, r + 6, 180, 180 + 180 * f, 50)), RAINBOW[i], 910 + i, lift=0.8,
                  alpha=alpha, grad=0.05, rim=0.2)
            c.restore()
    c.restore()


def splash_drops(c, x, y, u, n=9, spread=90, height=110, seed=0, color="#bfe3f7", s=1.0):
    """Paper droplets thrown up at u=0 falling back by u=1."""
    if u <= 0 or u >= 1:
        return
    for k in range(n):
        a = math.radians(-90 + (rnd(seed, k) - 0.5) * 120)
        v = (0.6 + 0.6 * rnd(seed, k + 20))
        dx = math.cos(a) * spread * v * u
        dy = math.sin(a) * height * v * u + 260 * u * u * v
        if dy > 4:
            continue
        sz = (7 + 6 * rnd(seed, k + 40)) * (1 - 0.5 * u) * s
        c.save()
        c.translate(x + dx * s, y + dy * s)
        c.rotate(math.degrees(math.atan2(dy + 1e-3, dx + 1e-3)) + 90)
        piece(c, cpath([(0, -sz * 1.4), (sz * 0.8, 0), (0, sz), (-sz * 0.8, 0)], seed + k, 0.3, soft=True),
              color, seed + k, lift=0.8, rim=0.2, grad=0.1)
        c.restore()


def confetti(c, t, seed=0, n=90, w=1920, h=1080, t0=0.0):
    if t < t0:
        return
    u = t - t0
    for k in range(n):
        x0 = rnd(seed, k) * w
        vy = 160 + 140 * rnd(seed, k + 1)
        y = -40 + u * vy - 200 * rnd(seed, k + 2)
        if y < -40 or y > h + 40:
            continue
        x = x0 + 40 * math.sin(u * (1.5 + rnd(seed, k + 3) * 2) + k)
        colr = RAINBOW[k % 6] if k % 7 else "white"
        c.save()
        c.translate(x, y)
        c.rotate(u * 200 * (rnd(seed, k + 4) - 0.5) + k * 37)
        c.scale(1.0, abs(math.cos(u * (3 + 3 * rnd(seed, k + 5)) + k)) + 0.15)
        if k % 3 == 0:
            p = path_of(ellipse(0, 0, 8, 8, 12))
        else:
            p = path_of([(-10, -5), (10, -5), (10, 5), (-10, 5)])
        piece(c, p, colr, seed + k, lift=0.6, rim=0, grad=0, shadow=0.2)
        c.restore()
